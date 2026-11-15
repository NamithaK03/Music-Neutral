"""Writes manuscript tables directly from saved result files (no hand transcription)."""
import ast, csv, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
T = ROOT / 'tables'
f3 = lambda x: f'{x:.3f}'
f4 = lambda x: f'{x:.4f}'.replace('-', '$-$')
ci = lambda c: f'[{f4(c[0])}, {f4(c[1])}]'

meld = json.load(open(ROOT / 'results/meld_substitution_condition.json'))
for split in ('test', 'dev'):
    rows = []
    for rule in ('text-only', 'rescaled', 'fixed'):
        for r in meld['rows']:
            if r['split'] == split and r['rule'] == rule and r['threshold'] in (0.4, 0.6, 0.8):
                rows.append(' & '.join([rule, f"{r['threshold']:.1f}", f3(r['low_share']), f3(r['p_label_correct_low']),
                                        f3(r['p_true_neutral_low']), f"{f3(r['acc_ungated'])} $\\to$ {f3(r['acc_relabel'])}",
                                        f"{f3(r['macro_f1_ungated'])} $\\to$ {f3(r['macro_f1_relabel'])}"]) + r' \\')
        if rule != 'fixed': rows.append(r'\midrule')
    (T / f'meld_condition_{split}_rows.tex').write_text('\n'.join(rows) + '\n')

comp = []
for c in meld['composition']:
    o, l = c['overall'], c['low_confidence']
    comp.append(' & '.join([c['split'], c['rule'], 'all'] + [f3(o[k]) for k in ('happy', 'sad', 'angry', 'neutral')] + ['--']) + r' \\')
    comp.append(' & '.join(['', '', r'$c<0.6$'] + [f3(l[k]) for k in ('happy', 'sad', 'angry', 'neutral')] + [f3(c['predicted_neutral_low'])]) + r' \\')
(T / 'meld_composition_rows.tex').write_text('\n'.join(comp) + '\n')

syn = json.load(open(ROOT / 'results/synthetic_substitution_condition.json'))
expl = json.load(open(ROOT / 'results/controlled_v2_exploratory_contrasts.json'))['contrasts']
srows = []
for r in syn['rows']:
    ur = [next(c for c in expl if (c['noise'], c['fusion'], c['preference_sd'], c['left'], c['right'], c['bandit'])
               == (r['noise'], r['fusion'], p, 'ungated', 'relabel', False))['metrics']['ndcg_at_10']['mean'] for p in (0.0, 0.5)]
    srows.append(' & '.join([r['noise'], 'text-only' if r['fusion'] == 'text' else r['fusion'], f3(r['low_share']), f3(r['p_label_correct_low']), f3(r['p_true_neutral_low']),
                             f3(-r['delta_acc']), f4(ur[0]), f3(ur[0] / -r['delta_acc']), f4(ur[1]), f3(ur[1] / -r['delta_acc'])]) + r' \\')
(T / 'synthetic_condition_rows.tex').write_text('\n'.join(srows) + '\n')

wr = {}
for r in csv.DictReader(open(ROOT / 'results/controlled_v2_all_contrasts.csv')):
    L, R = ast.literal_eval(r['left']), ast.literal_eval(r['right'])
    if r['metric'] == 'ndcg_at_10' and L['handling'] == 'downweight' and R['handling'] == 'relabel' and L['bandit'] == R['bandit']:
        wr[(r['noise'], float(r['preference_sd']), r['fusion'], L['bandit'])] = (float(r['mean']), ast.literal_eval(r['ci95']), int(r['positive']))
wrows, count = [], 0
for n in ('low', 'high', 'asymmetric'):
    for p in (0.0, 0.5):
        for fu in ('rescaled', 'text'):
            cells = []
            for b in (False, True):
                m, c, pos = wr[(n, p, fu, b)]
                star = '$^{\\ast}$' if c[0] > 0 else ''
                count += c[0] > 0
                cells.append(f'{f4(m)} {ci(c)}{star} & {pos}')
            wrows.append(' & '.join([n, f'{p:.1f}', 'text-only' if fu == 'text' else fu] + cells) + r' \\')
(T / 'wr_all_rows.tex').write_text('\n'.join(wrows) + '\n')
assert count == 13, count
print('tables written; W-R zero-excluding positive cells =', count)

# Combined test/development table (Table 4): differences relabel - ungated
dash = lambda x: f'{x:.3f}'.replace('-', '$-$')
both = []
for rule in ('text-only', 'rescaled', 'fixed'):
    for t in (0.4, 0.6, 0.8):
        cells = [rule, f'{t:.1f}']
        for split in ('test', 'dev'):
            r = next(r for r in meld['rows'] if r['split'] == split and r['rule'] == rule and r['threshold'] == t)
            cells += [f3(r['low_share']), f3(r['p_label_correct_low']), f3(r['p_true_neutral_low']),
                      dash(r['acc_relabel'] - r['acc_ungated']), dash(r['macro_f1_relabel'] - r['macro_f1_ungated'])]
        both.append(' & '.join(cells) + r' \\')
    if rule != 'fixed':
        both.append(r'\midrule')
(T / 'meld_condition_both_rows.tex').write_text('\n'.join(both) + '\n')
print('combined MELD condition table written')
