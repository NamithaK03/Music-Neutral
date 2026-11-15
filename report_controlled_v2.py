"""Reporting only: reads saved results, never executes a policy."""
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from experiments.controlled_study_v2 import interval

root = Path(__file__).resolve().parent
data = json.loads((root/'results/controlled_study_v2.json').read_text())
metrics = list(data['runs'][0]['metrics'])
keys = ['noise','preference_sd','fusion','handling','bandit','threshold']
summaries = []
for source, records in [('core',data['runs']),('sweep',data['threshold_sweep'])]:
    groups = {}
    for r in records:
        groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    for key, rows in groups.items():
        row = dict(zip(keys,key),source=source,n=len(rows))
        for m in metrics:
            values = [r['metrics'][m] for r in rows]
            row[m] = None if any(x is None for x in values) else float(np.mean(values))
            row[m+'_sd'] = None if row[m] is None else float(np.std(values,ddof=1))
        summaries.append(row)
with (root/'results/controlled_v2_all_cells.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(summaries[0])); w.writeheader(); w.writerows(summaries)
flat = []
for r in data['contrasts']:
    for m, v in r['metrics'].items():
        flat.append(dict(noise=r['noise'],preference_sd=r['preference_sd'],fusion=r['fusion'],left=str(r['left']),right=str(r['right']),metric=m,**v))
with (root/'results/controlled_v2_all_contrasts.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
lines=['# Amendment 1: every core condition','','Mean across ten paired seeds. Full metrics and threshold sweeps: `results/controlled_v2_all_cells.csv`; all intervals: `results/controlled_v2_all_contrasts.csv`.','','|Noise|Preference SD|Fusion|Handling|Bandit|NDCG@10|Skip|','|---|---|---|---|---|---|---|']
for r in summaries:
    if r['source']=='core':
        lines.append('|'+ '|'.join(str(r[k]) for k in keys[:5])+f"|{r['ndcg_at_10']:.4f}|{r['skip_rate']:.4f}|")
lines += ['','## Paired NDCG contrasts','','|Noise|Preference|Fusion|Left minus right|Delta [95% CI]|Positive seeds|','|---|---|---|---|---|---|']
for r in data['contrasts']:
    v=r['metrics']['ndcg_at_10']; lo,hi=v['ci95']
    label=lambda x:x['handling']+(' LinUCB' if x['bandit'] else ' deterministic')
    lines.append(f"|{r['noise']}|{r['preference_sd']}|{r['fusion']}|{label(r['left'])} minus {label(r['right'])}|{v['mean']:.4f} [{lo:.4f}, {hi:.4f}]|{v['positive']}/10|")
(root/'CONTROLLED_V2_RESULTS.md').write_text('\n'.join(lines)+'\n')
old=json.loads((root/'results/controlled_study.json').read_text())
old_rows=[]
for noise,pref,fusion in itertools.product((.5,1.5),(0.,.5),('fixed','rescaled','text')):
    values=[r['delta']['ndcg_at_10'] for r in old['paired_bandit_minus_deterministic'] if (r['noise'],r['preference_sd'],r['fusion'])==(noise,pref,fusion)]
    old_rows.append(dict(noise=noise,preference_sd=pref,fusion=fusion,**interval(values)))
(root/'results/controlled_original_paired_intervals.json').write_text(json.dumps(old_rows,indent=2))
print('Original intervals:')
for r in old_rows: print(r)
print('Extension policy contrast counts (positive / negative / includes zero):')
for a,b in [('abstain','relabel'),('downweight','relabel'),('downweight','abstain')]:
    rows=[r for r in data['contrasts'] if r['left']['handling']==a and r['right']['handling']==b]
    cis=[r['metrics']['ndcg_at_10']['ci95'] for r in rows]
    print(a,b,len(rows),sum(lo>0 for lo,hi in cis),sum(hi<0 for lo,hi in cis),sum(lo<=0<=hi for lo,hi in cis))
# Complete machine-generated supplement with all means and all metric contrasts.
tex=[r'\documentclass{article}',r'\usepackage[margin=1.5cm]{geometry}',r'\usepackage{longtable,booktabs}',r'\begin{document}',r'\section*{Amendment 1: complete results}',r'Each cell averages ten seeds. Intervals are paired seed bootstrap, 10,000 resamples, pointwise and unadjusted. NA denotes an undefined or inapplicable metric.']
for m in metrics:
    tex += [r'\subsection*{'+m.replace('_',r'\_')+'}',r'\begin{longtable}{lllllll}',r'Noise & Pref. & Fusion & Handling & Bandit & Mean & SD \\ \hline']
    for r in summaries:
        if r['source']=='core':
            val='NA' if r[m] is None else f"{r[m]:.5f}"
            sd='NA' if r[m+'_sd'] is None else f"{r[m+'_sd']:.5f}"
            tex.append(f"{r['noise']} & {r['preference_sd']} & {r['fusion']} & {r['handling']} & {int(r['bandit'])} & {val} & {sd}"+r' \\')
    tex.append(r'\end{longtable}')
tex += [r'\section*{All paired contrasts}',r'Policy abbreviations: U=ungated, R=relabel, A=abstain, W=downweight. Suffix 0=deterministic, 1=LinUCB. Counts are positive/negative/zero seed differences.']
ab={'ungated':'U','relabel':'R','abstain':'A','downweight':'W'}
for m in metrics:
    tex += [r'\subsection*{'+m.replace('_',r'\_')+'}',r'\begin{longtable}{llllrrl}',r'Noise & Pref. & Fusion & Contrast & Mean & 95\% CI & Counts \\ \hline']
    for r in data['contrasts']:
        v=r['metrics'][m]; name=lambda x:ab[x['handling']]+str(int(x['bandit']))
        val='NA' if v['mean'] is None else f"{v['mean']:.5f}"
        ci='NA' if v['ci95'] is None else f"[{v['ci95'][0]:.5f}, {v['ci95'][1]:.5f}]"
        counts='NA' if v['positive'] is None else f"{v['positive']}/{v['negative']}/{v['zero']}"
        tex.append(f"{r['noise']} & {r['preference_sd']} & {r['fusion']} & {name(r['left'])}-{name(r['right'])} & {val} & {ci} & {counts}"+r' \\')
    tex.append(r'\end{longtable}')
tex.append(r'\section*{All deterministic threshold sweeps}')
for m in metrics:
    tex += [r'\subsection*{'+m.replace('_',r'\_')+'}',r'\begin{longtable}{llllrrr}',r'Noise & Pref. & Fusion & Handling & Threshold & Mean & SD \\ \hline']
    for r in summaries:
        if r['source']=='sweep':
            val='NA' if r[m] is None else f"{r[m]:.5f}"
            sd='NA' if r[m+'_sd'] is None else f"{r[m+'_sd']:.5f}"
            tex.append(f"{r['noise']} & {r['preference_sd']} & {r['fusion']} & {r['handling']} & {r['threshold']} & {val} & {sd}"+r' \\')
    tex.append(r'\end{longtable}')
tex.append(r'\end{document}')
(root/'supplement_all_conditions.tex').write_text('\n'.join(tex)+'\n')
