"""Build a descriptive results table directly from the controlled-study record."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
data=json.loads((root/'results/controlled_study.json').read_text())
lines=['# Controlled synthetic follow-up results', '',
       'Mean (sample SD) across ten seeds; 1,000 interactions per seed. All variants share scenarios and all four genre actions.', '',
       '| Noise | Preference SD | Fusion | Policy | NDCG@10 | Skip rate | Fallback |',
       '|---|---|---|---|---|---|---|']
for r in data['summary']:
    values=[f"{r['metrics'][k]['mean']:.3f} ({r['metrics'][k]['sd']:.3f})" for k in ['ndcg_at_10','skip_rate','fallback_rate']]
    lines.append('| '+ ' | '.join([str(r['noise']),str(r['preference_sd']),r['fusion'],'LinUCB' if r['bandit'] else 'deterministic']+values)+' |')
lines += ['', 'These are new simulator measurements, not replications of legacy values. Shared scenarios enable within-generator comparisons; neither human benefit nor algorithmic novelty follows from these results. Graded NDCG uses completion probabilities and is not comparable with the legacy binary-relevance NDCG.', '', 'All seed-level metrics and paired differences are retained in `results/controlled_study.json`.']
(root/'CONTROLLED_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
