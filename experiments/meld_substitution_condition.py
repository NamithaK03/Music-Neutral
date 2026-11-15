"""Exploratory analysis: when does substituting Neutral for low-confidence predictions help?

Uses only the saved MELD checkpoint predictions (results/meld_checkpoint_predictions.csv).
Devised after the v2 and MELD results were inspected; it is exploratory, not prespecified.

Identity checked numerically for every row:
  Acc(relabel) - Acc(ungated) = pi_L * (P(Y=neutral | L) - P(Yhat=Y | L)),
where L is the set of inputs whose confidence score is below the threshold.
"""
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
C4 = ['happy', 'sad', 'angry', 'neutral']
NEUTRAL = 3
THRESHOLDS = [round(t, 2) for t in np.arange(0.40, 0.8001, 0.05)]


def macro_f1(y, p):
    f = []
    for c in range(4):
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f))


def fuse(pt, ps, rule):
    """Return (label, score) for text-only, rescaled and fixed (0.7-capped) trust rules."""
    if rule == 'text-only':
        return pt.argmax(1), pt.max(1)
    scale = 0.7 if rule == 'fixed' else 1.0
    lt, ls = pt.argmax(1), ps.argmax(1)
    wt, ws = scale * pt.max(1), scale * ps.max(1)
    agree = lt == ls
    label = np.where(agree, lt, np.where(wt > ws, lt, ls))  # ties favour speech, as in the reference rule
    score = np.where(agree, np.maximum(wt, ws), (wt + ws) / 2)
    return label, score


def main():
    rows = [r for r in csv.DictReader(open(ROOT / 'results/meld_checkpoint_predictions.csv', encoding='utf-8'))
            if r['true_label_4']]
    out = {'status': 'exploratory; devised after inspecting v2 and MELD results',
           'source': 'results/meld_checkpoint_predictions.csv', 'thresholds': THRESHOLDS, 'rows': [],
           'composition': []}
    for split in ('dev', 'test'):
        part = [r for r in rows if r['split'] == split]
        y = np.array([C4.index(r['true_label_4']) for r in part])
        pt = np.array([json.loads(r['text_probs_4_calibrated']) for r in part])
        ps = np.array([json.loads(r['speech_probs_4_calibrated']) for r in part])
        for rule in ('text-only', 'rescaled', 'fixed'):
            label, score = fuse(pt, ps, rule)
            acc_u = float(np.mean(label == y)); f1_u = macro_f1(y, label)
            for t in THRESHOLDS:
                low = score < t
                relabel = np.where(low, NEUTRAL, label)
                pi = float(low.mean())
                p_correct = float(np.mean(label[low] == y[low])) if low.any() else None
                p_neutral = float(np.mean(y[low] == NEUTRAL)) if low.any() else None
                acc_r = float(np.mean(relabel == y))
                predicted = pi * (p_neutral - p_correct) if low.any() else 0.0
                assert abs((acc_r - acc_u) - predicted) < 1e-12
                out['rows'].append(dict(split=split, rule=rule, threshold=t, n=int(len(y)), low_share=pi,
                                        p_label_correct_low=p_correct, p_true_neutral_low=p_neutral,
                                        acc_ungated=acc_u, acc_relabel=acc_r, delta_acc=acc_r - acc_u,
                                        macro_f1_ungated=f1_u, macro_f1_relabel=macro_f1(y, relabel)))
            if rule in ('text-only', 'rescaled'):
                low = score < 0.6
                out['composition'].append(dict(split=split, rule=rule, threshold=0.6,
                    overall={C4[c]: float(np.mean(y == c)) for c in range(4)},
                    low_confidence={C4[c]: float(np.mean(y[low] == c)) for c in range(4)},
                    predicted_neutral_low=float(np.mean(label[low] == NEUTRAL))))
    path = ROOT / 'results/meld_substitution_condition.json'
    path.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print('identity verified for', len(out['rows']), 'rows; wrote', path)


if __name__ == '__main__':
    main()
