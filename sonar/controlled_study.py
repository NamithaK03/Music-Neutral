"""Paired synthetic pipeline experiment. See CONTROLLED_PROTOCOL.md."""
import json
from pathlib import Path
import numpy as np
from sonar.recommender import LinUCB


def scenarios(seed, noise, preference_sd, n=1000):
    rng = np.random.default_rng(seed)
    genres = np.repeat(np.arange(4), 20)
    affinities = np.array([rng.dirichlet(np.eye(4)[g]*3.5+0.5) for g in genres])
    truth = rng.integers(4, size=n)
    probabilities = []
    for _ in range(2):
        logits = 2*np.eye(4)[truth] + noise*rng.normal(size=(n,4))
        exp = np.exp(logits-logits.max(axis=1,keepdims=True))
        probabilities.append(exp/exp.sum(axis=1,keepdims=True))
    preferences = preference_sd*rng.normal(size=4)
    utility = affinities[:,truth].T + preferences[genres][None,:] - 0.5
    completion = 0.1 + 0.8/(1+np.exp(-2*utility))
    uniforms = rng.random(size=(n,len(genres)))
    return affinities, genres, truth, probabilities, completion, uniforms


def evaluate(data, fusion, bandit):
    affinities, genres, truth, branches, completion, uniforms = data
    actions = tuple(str(i) for i in range(4))
    policy = LinUCB(actions=actions, dimension=6, exploration=0.5, decay=0)
    accum = {k: [] for k in ('ndcg_at_10','skip_rate','fallback_rate','emotion_accuracy','expected_completion')}
    discount = 1/np.log2(np.arange(2,12))
    for t, actual in enumerate(truth):
        labels = [int(p[t].argmax()) for p in branches]
        scores = [float(p[t].max()) for p in branches]
        if fusion == 'text':
            label, score = labels[0], scores[0]
        else:
            weights = np.array(scores)*(0.7 if fusion == 'fixed' else 1.0)
            label = labels[0] if labels[0] == labels[1] or weights[0] > weights[1] else labels[1]
            score = float(weights.max() if labels[0] == labels[1] else weights.mean())
        fallback = score < 0.6
        if fallback:
            label = 3
        context = np.r_[np.eye(4)[label],score,1.0]
        lists = []
        for g in range(4):
            indices = np.flatnonzero(genres == g)
            lists.append(indices[np.argsort(-affinities[indices,label],kind='stable')[:10]])
        if bandit:
            g = int(policy.select(context, actions))
        else:
            g = int(np.argmax([affinities[items,label].mean() for items in lists]))
        playlist = lists[g]
        first = playlist[0]
        completed = uniforms[t,first] < completion[t,first]
        if bandit:
            policy.update(str(g),context,1.0 if completed else -1.0)
        ideal = np.sort(completion[t])[::-1][:10] @ discount
        accum['ndcg_at_10'].append(float(completion[t,playlist] @ discount / ideal))
        accum['skip_rate'].append(float(not completed))
        accum['fallback_rate'].append(float(fallback))
        accum['emotion_accuracy'].append(float(label == actual))
        accum['expected_completion'].append(float(completion[t,first]))
    return {k: float(np.mean(v)) for k,v in accum.items()}


def main():
    records = []
    for noise in (0.5,1.5):
        for preference_sd in (0.0,0.5):
            for seed in range(2026,2036):
                data = scenarios(seed,noise,preference_sd)
                for fusion in ('fixed','rescaled','text'):
                    for bandit in (False,True):
                        records.append(dict(seed=seed,noise=noise,preference_sd=preference_sd,
                            fusion=fusion,bandit=bandit,metrics=evaluate(data,fusion,bandit)))
    summary = []
    paired = []
    for noise in (0.5,1.5):
        for preference_sd in (0.0,0.5):
            for fusion in ('fixed','rescaled','text'):
                selected=[r for r in records if (r['noise'],r['preference_sd'],r['fusion'])==(noise,preference_sd,fusion)]
                for bandit in (False,True):
                    runs=[r for r in selected if r['bandit']==bandit]
                    metrics={k:{'mean':float(np.mean([r['metrics'][k] for r in runs])),
                                'sd':float(np.std([r['metrics'][k] for r in runs],ddof=1))} for k in runs[0]['metrics']}
                    summary.append(dict(noise=noise,preference_sd=preference_sd,fusion=fusion,bandit=bandit,metrics=metrics))
                for seed in range(2026,2036):
                    a=next(r for r in selected if r['seed']==seed and not r['bandit'])
                    b=next(r for r in selected if r['seed']==seed and r['bandit'])
                    paired.append(dict(noise=noise,preference_sd=preference_sd,fusion=fusion,seed=seed,
                        delta={k:b['metrics'][k]-a['metrics'][k] for k in a['metrics']}))
    result={'scope':'new controlled synthetic follow-up; not human evidence', 'protocol':'CONTROLLED_PROTOCOL.md',
            'runs':records,'summary':summary,'paired_bandit_minus_deterministic':paired}
    path=Path(__file__).resolve().parent.parent/'results/controlled_study.json'
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('Saved',len(records),'runs to',path)


if __name__ == '__main__':
    main()
