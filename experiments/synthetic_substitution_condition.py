"""Exploratory: evaluates the Neutral-substitution accuracy condition in the
Amendment 1 generator by replaying the frozen inference streams (no policy
execution). Cross-checks the identity against recorded v2 emotion accuracy."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from controlled_study_v2 import scenario, infer, SEEDS

v2 = json.load(open(ROOT / 'results/controlled_study_v2.json'))
cal = v2['calibration']
levels = {'low': (.5, .5), 'high': (1.5, 1.5),
          'asymmetric': (cal['text']['selected']['sd'], cal['speech']['selected']['sd'])}
recorded = {(r['noise'], r['preference_sd'], r['fusion'], r['handling'], r['seed']): r['metrics']
            for r in v2['runs'] if not r['bandit']}
rows, max_err = [], 0.0
for noise, sds in levels.items():
    for fusion in ('rescaled', 'text'):
        lowc = corr = neut = n = 0
        for seed in SEEDS:
            data = scenario(seed, sds, 0.0)
            truth = data[2]
            label, score, _ = infer(data[3], fusion)
            low = score < 0.6
            lowc += low.sum(); n += len(truth)
            corr += (label[low] == truth[low]).sum(); neut += (truth[low] == 3).sum()
            acc_u = np.mean(label == truth); acc_r = np.mean(np.where(low, 3, label) == truth)
            for pref in (0.0, 0.5):  # emotion stream is identical across preference SD
                mu = recorded[(noise, pref, fusion, 'ungated', seed)]['emotion_accuracy']
                mr = recorded[(noise, pref, fusion, 'relabel', seed)]['emotion_accuracy']
                max_err = max(max_err, abs(mu - acc_u), abs(mr - acc_r))
        rows.append(dict(noise=noise, fusion=fusion, low_share=lowc / n,
                         p_label_correct_low=corr / lowc, p_true_neutral_low=neut / lowc,
                         delta_acc=lowc / n * (neut - corr) / lowc))
out = dict(status='exploratory; devised after inspecting v2 and MELD results', threshold=0.6,
           max_abs_diff_vs_recorded_accuracy=max_err, rows=rows)
json.dump(out, open(ROOT / 'results/synthetic_substitution_condition.json', 'w'), indent=2)
print(json.dumps(out, indent=1))
