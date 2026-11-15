"""Read-only integrity checks; does not generate scenarios or execute policies."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
new=json.loads((root/'results/controlled_study_v2.json').read_text())
old=json.loads((root/'results/controlled_study.json').read_text())
assert len(new['runs'])==960 and len(new['threshold_sweep'])==1200
cells=Counter(tuple(r[k] for k in ('noise','preference_sd','fusion','handling','bandit')) for r in new['runs'])
assert len(cells)==96 and set(cells.values())=={10}
for r in new['runs']+new['threshold_sweep']:
    assert all(v is None or 0<=v<=1 for v in r['metrics'].values())
original={(r['noise'],r['preference_sd'],r['fusion'],r['bandit'],r['seed']):r for r in old['runs']}
matched=0
for r in new['runs']:
    if r['handling']=='relabel' and r['noise'] in ('low','high'):
        before=original[.5 if r['noise']=='low' else 1.5,r['preference_sd'],r['fusion'],r['bandit'],r['seed']]
        for metric,value in before['metrics'].items():
            assert abs(value-r['metrics'][metric])<1e-12,(r,metric,value)
        matched+=1
assert matched==160
frozen=root/'protocol_archive/frozen_amendment_1'
manifest=json.loads((frozen/'MANIFEST.json').read_text())
assert manifest['archived_utc']<new['metadata']['started_utc']<new['metadata']['finished_utc']
for name in ('CONTROLLED_PROTOCOL.md','CONTROLLED_PROTOCOL_AMENDMENT_1.md','experiments/controlled_study_v2.py','sonar/controlled_study.py','sonar/recommender.py'):
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==manifest['files'][name],name
assert new['metadata']['protocol_sha256']==manifest['files']['CONTROLLED_PROTOCOL_AMENDMENT_1.md']
assert new['metadata']['source_sha256']==manifest['files']['experiments/controlled_study_v2.py']
transfer=json.loads((root/'results/nowplaying_transfer.json').read_text())
assert len(transfer['runs'])==80 and len(transfer['contrasts'])==14
assert sum(transfer['counts']['proxy_classes'].values())==transfer['counts']['eligible_events']
users=np.load(root/'results/nowplaying_user_metrics.npz')
for fusion in ('text','rescaled'):
    for handling in ('ungated','relabel','abstain','downweight'):
        arrays=users[f'{fusion}_{handling}'];assert arrays.shape==(10,53,3)
        for i,seed in enumerate(range(2026,2036)):
            row=next(r for r in transfer['runs'] if (r['fusion'],r['handling'],r['seed'])==(fusion,handling,seed))
            np.testing.assert_allclose(arrays[i].mean(0),[row[m] for m in ('ndcg_at_10','recall_at_10','mrr_at_10')],atol=1e-14)
print('PASS: 96 complete core cells; 1200 sweep rows; 160 original relabel runs match; frozen hashes and archive chronology; all 80 listening-log runs match user arrays.')
