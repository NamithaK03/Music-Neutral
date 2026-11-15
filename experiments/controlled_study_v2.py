"""Frozen Amendment 1 experiment; refuses to overwrite existing output."""
import hashlib
import itertools
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sonar.controlled_study import scenarios
from sonar.recommender import LinUCB

ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(2026, 2036))
HANDLING = ('ungated', 'relabel', 'abstain', 'downweight')
BOOT = np.random.default_rng(0).integers(0, 10, (10000, 10))

def stamp():
    return datetime.now(timezone.utc).isoformat()

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def softmax(x):
    exp = np.exp(x-x.max(axis=1, keepdims=True))
    return exp/exp.sum(axis=1, keepdims=True)

def calibrate():
    rng = np.random.default_rng(9999)
    truth = rng.integers(4, size=10000)
    result = {}
    for branch, target in [('text', .685), ('speech', .578)]:
        errors = rng.normal(size=(10000, 4))
        grid = []
        for sd in np.round(np.arange(.1, 3.001, .05), 2):
            accuracy = float(np.mean((2*np.eye(4)[truth]+sd*errors).argmax(1)==truth))
            grid.append(dict(sd=float(sd), accuracy=accuracy))
        best = min(grid, key=lambda r: (abs(r['accuracy']-target), r['sd']))
        result[branch] = dict(target=target, selected=best, grid=grid)
    return result

def scenario(seed, noise, preference_sd):
    # Replay the original stream, replacing only the two noise multipliers.
    data = list(scenarios(seed, 1.0, preference_sd))
    rng = np.random.default_rng(seed)
    for g in data[1]:
        rng.dirichlet(np.eye(4)[g]*3.5+.5)
    truth = rng.integers(4, size=len(data[2]))
    assert np.array_equal(truth, data[2])
    data[3] = [softmax(2*np.eye(4)[truth]+sd*rng.normal(size=(len(truth), 4))) for sd in noise]
    return data

def infer(branches, fusion):
    p, s = branches
    m, n = p.max(1), s.max(1)
    a, b = p.argmax(1), s.argmax(1)
    if fusion == 'text':
        return a, m, p
    same = a == b
    return np.where(same | (m > n), a, b), np.where(same, np.maximum(m,n), (m+n)/2), (m[:,None]*p+n[:,None]*s)/(m+n)[:,None]

def evaluate(data, fusion, handling, bandit, threshold=.6):
    affinities, genres, truth, branches, completion, uniforms = data
    labels, confidence, q = infer(branches, fusion)
    low = confidence < threshold
    emitted = np.where(low, 3, labels) if handling == 'relabel' else labels
    scores = affinities[:,emitted].T.copy()
    context = np.column_stack((np.eye(4)[emitted], confidence, np.ones(len(truth))))
    if handling == 'abstain':
        scores[low] = .25
        context[low,:4] = 0
    elif handling == 'downweight':
        scores = q @ affinities.T
        context = np.column_stack((q, q.max(1), np.ones(len(truth))))
    lists = np.stack([np.flatnonzero(genres==g)[np.argsort(-scores[:,genres==g], axis=1, kind='stable')[:,:10]] for g in range(4)], axis=1)
    list_scores = np.take_along_axis(scores[:,None,:], lists, axis=2).mean(2)
    actions = tuple(str(i) for i in range(4))
    policy = LinUCB(actions, dimension=6, exploration=.5, decay=0)
    chosen = list_scores.argmax(1)
    if bandit:
        for t, x in enumerate(context):
            chosen[t] = int(policy.select(x, actions))
            first = lists[t,chosen[t],0]
            policy.update(str(chosen[t]), x, 1.0 if uniforms[t,first]<completion[t,first] else -1.0)
    playlists = lists[np.arange(len(truth)),chosen]
    first = playlists[:,0]
    discount = 1/np.log2(np.arange(2,12))
    ideal = np.sort(completion,axis=1)[:,-10:][:,::-1] @ discount
    gated = handling in ('relabel','abstain')
    order = np.argsort(-confidence,kind='stable')
    errors = labels != truth
    return dict(ndcg_at_10=float(np.mean((np.take_along_axis(completion, playlists, axis=1) @ discount)/ideal)),
        skip_rate=float(np.mean(uniforms[np.arange(len(truth)),first]>=completion[np.arange(len(truth)),first])),
        expected_completion=float(np.mean(completion[np.arange(len(truth)),first])),
        fallback_rate=float(low.mean()) if handling=='relabel' else 0.0,
        abstention_rate=float(low.mean()) if handling=='abstain' else 0.0,
        emotion_accuracy=float(np.mean(emitted==truth)) if handling in ('relabel','ungated') else None,
        coverage=float((~low).mean()) if gated else None,
        covered_accuracy=float(np.mean(labels[~low]==truth[~low])) if gated and (~low).any() else None,
        risk_coverage_area=float(np.mean(np.cumsum(errors[order])/np.arange(1,len(truth)+1))) if gated else None)

def interval(values):
    valid = [x for x in values if x is not None]
    if len(valid)!=10:
        return dict(mean=None, ci95=None, valid_seeds=len(valid), positive=None, negative=None, zero=None)
    a = np.asarray(valid)
    return dict(mean=float(a.mean()), ci95=np.quantile(a[BOOT].mean(1), [.025,.975], method='linear').tolist(),
        valid_seeds=10, positive=int((a>0).sum()), negative=int((a<0).sum()), zero=int((a==0).sum()))

def comparisons(records):
    index = {(r['noise'],r['preference_sd'],r['fusion'],r['bandit'],r['handling'],r['seed']):r for r in records}
    result = []
    for noise, pref, fusion in itertools.product(('low','high','asymmetric'),(0.0,.5),('rescaled','text')):
        contrasts = [(b,b,a,z) for b in (False,True) for a,z in [('abstain','relabel'),('downweight','relabel'),('downweight','abstain')]]
        contrasts += [(True,False,h,h) for h in HANDLING]
        for ba,bz,a,z in contrasts:
            pairs = [(index[noise,pref,fusion,ba,a,s]['metrics'], index[noise,pref,fusion,bz,z,s]['metrics']) for s in SEEDS]
            metrics = {k:interval([None if x[k] is None or y[k] is None else x[k]-y[k] for x,y in pairs]) for k in pairs[0][0]}
            result.append(dict(noise=noise,preference_sd=pref,fusion=fusion,left=dict(handling=a,bandit=ba),right=dict(handling=z,bandit=bz),metrics=metrics))
    return result

def main():
    path = ROOT/'results/controlled_study_v2.json'
    lock = ROOT/'results/controlled_study_v2.started.json'
    if path.exists() or lock.exists():
        raise SystemExit('Existing execution: do not rerun without a dated deviation.')
    metadata = dict(started_utc=stamp(), python=platform.python_version(), numpy=np.__version__, seeds=SEEDS,
        protocol_sha256=digest(ROOT/'CONTROLLED_PROTOCOL_AMENDMENT_1.md'),
        source_sha256=digest(Path(__file__)), original_source_sha256=digest(ROOT/'sonar/controlled_study.py'))
    with lock.open('x') as f:
        json.dump(metadata,f,indent=2)
    calibration = calibrate()
    noise_levels = {'low':(.5,.5),'high':(1.5,1.5),'asymmetric':tuple(calibration[b]['selected']['sd'] for b in ('text','speech'))}
    print('Calibration:',calibration['text']['selected'],calibration['speech']['selected'],flush=True)
    records, sweeps = [], []
    for noise,pref,seed in itertools.product(noise_levels,(0.0,.5),SEEDS):
        data = scenario(seed,noise_levels[noise],pref)
        for fusion,handling,bandit in itertools.product(('rescaled','text'),HANDLING,(False,True)):
            row = dict(noise=noise,noise_sd=noise_levels[noise],preference_sd=pref,seed=seed,fusion=fusion,handling=handling,bandit=bandit,threshold=.6)
            row['metrics'] = evaluate(data,fusion,handling,bandit)
            records.append(row)
            if not bandit and handling in ('relabel','abstain'):
                sweeps.append(dict(row))
                for threshold in (.4,.5,.7,.8):
                    sweeps.append(dict(row,threshold=threshold,metrics=evaluate(data,fusion,handling,False,threshold)))
        print(noise,pref,seed,'completed',len(records),flush=True)
    metadata['finished_utc'] = stamp()
    result = dict(metadata=metadata,calibration=calibration,runs=records,threshold_sweep=sweeps,contrasts=comparisons(records))
    with path.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    print('Saved',len(records),'core runs and',len(sweeps),'sweep rows',flush=True)

if __name__ == '__main__':
    main()
