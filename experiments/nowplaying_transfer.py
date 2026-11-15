"""Offline listening-log transfer; see NOWPLAYING_PROTOCOL.md."""
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from experiments.controlled_study_v2 import infer, stamp, digest

ROOT=Path(__file__).resolve().parents[1]
WORDS=[('happy','happiness','joy','joyful','excited','enjoy','enjoying','love','loved','lovely'),
 ('sad','sadness','unhappy','depressed','crying','heartbroken','lonely'),
 ('angry','anger','mad','furious','rage'),('neutral','calm','relaxed','relaxing','chill','chilling')]
LABELS=['happy','sad','angry','neutral']

def events(frame):
    lookup={word:i for i,words in enumerate(WORDS) for word in words}
    frame=frame.copy()
    frame['emotion']=frame.hashtag.astype(str).str.lower().str.lstrip('#').map(lookup)
    keys=['user_id','track_id','created_at']
    groups=frame.groupby(keys,sort=True).emotion.agg(lambda x: sorted(set(x.dropna().astype(int))))
    out=groups.reset_index()
    out['emotion']=out.emotion.map(lambda x:x[0] if len(x)==1 else -1)
    return out

def main():
    out=ROOT/'results/nowplaying_transfer.json'
    lock=ROOT/'results/nowplaying_transfer.started.json'
    if out.exists() or lock.exists(): raise SystemExit('Refusing repeat execution without deviation log.')
    archive=ROOT.parent/'recovered_study/data/nowplaying/nowplaying_rs_train_test.zip'
    pred=ROOT/'results/meld_checkpoint_complete_train_predictions.csv'
    meta=dict(started_utc=stamp(),protocol_sha256=digest(ROOT/'NOWPLAYING_PROTOCOL.md'),source_sha256=digest(Path(__file__)),data_sha256=digest(archive),meld_predictions_sha256=digest(pred),numpy=np.__version__,pandas=pd.__version__)
    with lock.open('x') as f: json.dump(meta,f,indent=2)
    assert hashlib.md5(archive.read_bytes()).hexdigest()=='32d00b55dbeb2b617fd041c8156d3e86'
    with zipfile.ZipFile(archive) as z:
        train=events(pd.read_csv(z.open('Train-Test Splits/Context/train.csv'),dtype={'user_id':str}))
        test=events(pd.read_csv(z.open('Train-Test Splits/Context/test.csv'),dtype={'user_id':str}))
    tracks=sorted(train.track_id.unique()); track_index={t:i for i,t in enumerate(tracks)}
    known=test.track_id.isin(track_index)
    selected=test[known & (test.emotion>=0) & test.user_id.isin(set(train.user_id))].copy().reset_index(drop=True)
    if len(selected)==0: raise RuntimeError('No eligible proxy-labelled events.')
    users=sorted(selected.user_id.unique()); user_index={u:i for i,u in enumerate(users)}
    user_ids=np.array([user_index[u] for u in selected.user_id]); nuser=len(users)
    counts=np.bincount(user_ids,minlength=nuser)
    pop=train.track_id.value_counts().reindex(tracks,fill_value=0).to_numpy(float); pop/=pop.sum()
    base={}
    for user in users:
        hist=train[train.user_id==user].track_id.value_counts().reindex(tracks,fill_value=0).to_numpy(float)
        base[user]=.5*hist/hist.sum()+.5*pop
    affinity=np.ones((len(tracks),4))
    for (track,emotion),count in train[train.emotion>=0].groupby(['track_id','emotion']).size().items():
        affinity[track_index[track],int(emotion)]+=count
    affinity/=affinity.sum(1,keepdims=True)
    donors=pd.read_csv(pred); donors=donors[(donors.split=='test') & donors.true_label_4.isin(LABELS)].reset_index(drop=True)
    truth=donors.true_label_4.map({x:i for i,x in enumerate(LABELS)}).to_numpy()
    branches=[np.array([json.loads(x) for x in donors[col]]) for col in ('text_probs_4_calibrated','speech_probs_4_calibrated')]
    pools=[np.flatnonzero(truth==i) for i in range(4)]
    def branch_metrics(t,probs):
        return [dict(accuracy=float(accuracy_score(t,p.argmax(1))),macro_f1=float(f1_score(t,p.argmax(1),labels=range(4),average='macro',zero_division=0))) for p in probs]
    def retrieve(scores,target):
        rank=1+int(np.sum(scores>scores[target]))+int(np.sum((scores[:target]==scores[target])))
        return np.array([1/np.log2(rank+1) if rank<=10 else 0.,float(rank<=10),1/rank if rank<=10 else 0.])
    def average_user(values):
        sums=np.zeros((nuser,3)); np.add.at(sums,user_ids,values);return sums/counts[:,None]
    targets=np.array([track_index[t] for t in selected.track_id]); actual=selected.emotion.to_numpy(int)
    base_events=np.array([retrieve(base[u],t) for u,t in zip(selected.user_id,targets)])
    base_users=average_user(base_events)
    saved={}; records=[]; realized=[]
    for seed in range(2026,2036):
        rng=np.random.default_rng(seed)
        draw=np.array([rng.choice(pools[e]) for e in actual])
        probs=[p[draw] for p in branches]
        realized.append(dict(seed=seed,branches=branch_metrics(actual,probs)))
        for fusion in ('text','rescaled'):
            labels,c,q=infer(probs,fusion); low=c<.6
            for handling in ('ungated','relabel','abstain','downweight'):
                values=[]
                for j,(user,target) in enumerate(zip(selected.user_id,targets)):
                    if handling=='abstain' and low[j]: scores=base[user]
                    elif handling=='downweight': scores=base[user]*(4*(affinity @ q[j]))
                    else:
                        label=3 if handling=='relabel' and low[j] else labels[j]
                        scores=base[user]*(4*affinity[:,label])
                    values.append(retrieve(scores,target))
                uv=average_user(np.array(values)); saved.setdefault((fusion,handling),[]).append(uv)
                records.append(dict(seed=seed,fusion=fusion,handling=handling,ndcg_at_10=float(uv[:,0].mean()),recall_at_10=float(uv[:,1].mean()),mrr_at_10=float(uv[:,2].mean()),fallback_rate=float(low.mean()) if handling=='relabel' else 0.,abstention_rate=float(low.mean()) if handling=='abstain' else 0.,coverage=float((~low).mean()) if handling in ('relabel','abstain') else None))
        print('nowplaying seed',seed,'complete',flush=True)
    boot=np.random.default_rng(0).integers(0,nuser,(10000,nuser))
    contrasts=[]
    for fusion in ('text','rescaled'):
        for a,b in [('abstain','relabel'),('downweight','relabel'),('downweight','abstain')]+[(h,'base') for h in ('ungated','relabel','abstain','downweight')]:
            x=np.array(saved[fusion,a]); y=np.repeat(base_users[None,:,:],10,axis=0) if b=='base' else np.array(saved[fusion,b])
            d=(x-y).mean(0); delta_seed=(x-y).mean(1)
            cis=np.quantile(d[boot].mean(1),[.025,.975],axis=0)
            contrasts.append(dict(fusion=fusion,left=a,right=b,metrics={m:dict(mean=float(d[:,i].mean()),ci95=cis[:,i].tolist(),positive_seeds=int((delta_seed[:,i]>0).sum()),negative_seeds=int((delta_seed[:,i]<0).sum()),seed_differences=delta_seed[:,i].tolist()) for i,m in enumerate(('ndcg_at_10','recall_at_10','mrr_at_10'))}))
    meta['finished_utc']=stamp()
    result=dict(metadata=meta,scope='Observed listening targets; weak hashtag mood proxies; injected MELD errors; no causal evaluation',counts=dict(train_events=len(train),test_events=len(test),test_unknown_catalog=int((~known).sum()),test_unlabelled_or_conflicting=int((test.emotion<0).sum()),eligible_events=len(selected),eligible_users=nuser,catalog_tracks=len(tracks),proxy_classes={LABELS[i]:int((actual==i).sum()) for i in range(4)}),meld_pool=branch_metrics(truth,branches),realized_errors=realized,base_metrics=dict(zip(('ndcg_at_10','recall_at_10','mrr_at_10'),base_users.mean(0).tolist())),runs=records,contrasts=contrasts)
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    np.savez_compressed(ROOT/'results/nowplaying_user_metrics.npz',base=base_users,**{f'{f}_{h}':np.array(v) for (f,h),v in saved.items()})
    print(json.dumps(result['counts']),flush=True)

if __name__=='__main__':main()
