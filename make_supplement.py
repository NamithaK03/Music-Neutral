"""Write the supplementary report (Document S1) as LaTeX, directly from the saved result files."""
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
OUT = ROOT / 'supplement'
OUT.mkdir(exist_ok=True)

NOISE = ('low', 'high', 'asymmetric')
NOISE_LABEL = {'low': 'low', 'high': 'high', 'asymmetric': 'asym.'}
PREFS = (0.0, 0.5)
RULES = ('text', 'rescaled')
RULE_LABEL = {'text': 'text-only', 'rescaled': 'rescaled', 'fixed': 'fixed', 'text-only': 'text-only'}
POL = ('ungated', 'relabel', 'abstain', 'downweight')
POL_SHORT = {'ungated': 'U', 'relabel': 'R', 'abstain': 'A', 'downweight': 'W'}
SEL = ((False, 'deterministic'), (True, 'LinUCB'))
METRICS = [
    ('ndcg_at_10', 'Graded NDCG@10', 3),
    ('skip_rate', 'First-track skip rate', 3),
    ('expected_completion', 'Expected first-track completion', 3),
    ('emotion_accuracy', 'Emitted-label accuracy', 3),
    ('fallback_rate', 'Fallback rate', 3),
    ('abstention_rate', 'Abstention rate', 3),
    ('coverage', 'Coverage', 3),
    ('covered_accuracy', 'Covered accuracy', 3),
    ('risk_coverage_area', 'Risk--coverage area', 3),
]
DASH = '---'


def num(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return DASH
    s = f'{x:.{d}f}'
    if s.startswith('-'):
        s = '$-$' + s[1:]
    if s in ('$-$' + '0.' + '0' * d,):
        s = '0.' + '0' * d
    return s


def ci(lo, hi, d=4):
    return f'[{num(lo, d)}, {num(hi, d)}]'


tex = []
w = tex.append

w(r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{newpxtext,newpxmath}
\usepackage{booktabs,longtable,array,multirow}
\usepackage[font=small,labelfont=bf,labelsep=period]{caption}
\usepackage[hidelinks]{hyperref}
\usepackage{titlesec}
\titleformat{\section}{\large\bfseries}{\thesection.}{0.5em}{}
\titleformat{\subsection}{\normalsize\itshape}{\thesubsection.}{0.5em}{}
\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\thetable}{S\arabic{table}}
\setlength{\LTcapwidth}{\textwidth}
\setlength{\tabcolsep}{4.5pt}
\renewcommand{\arraystretch}{1.08}
\setlength{\parindent}{0pt}\setlength{\parskip}{4pt}
\begin{document}
\begin{center}
{\Large\bfseries Supplementary Materials}\\[6pt]
{\large Neutral Relabeling of Uncertain Emotion Predictions in Music Recommendation}\\[6pt]
K. Namitha and Harinarayanan U\\[2pt]
{\small Tables S1--S22: complete results of the controlled study, the MELD substitution analysis and the listening-log study}
\end{center}
\vspace{4pt}
''')

w(r'''\section{Overview and Conventions}
This document reports every condition, contrast and secondary metric that the main text summarizes. All values are written by a script (\texttt{make\_supplement.py}) directly from the saved result files in the public repository; no value is typed by hand.

\textbf{Main experiment.} Four policies (U, ungated; R, relabel; A, abstain; W, down-weight) are crossed with two inference rules (text-only, rescaled fusion), two genre selectors (deterministic, LinUCB), three noise settings (low, $\sigma=0.5$; high, $\sigma=1.5$; asymmetric, text and speech noise SDs 1.40 and 1.90) and two preference SDs (0.0, 0.5): 96 conditions with ten paired seeds each, at threshold $\tau=0.6$. A \emph{cell} is one combination of noise, preference SD, inference rule and selector.

\textbf{Metrics.} Graded NDCG@10 uses completion probabilities as relevance. Skip rate is the share of interactions whose first track is not completed. Expected completion is the mean completion probability of the first track. Emitted-label accuracy is the share of emitted labels that equal the true emotion (defined for U and R). Fallback and abstention rates are the shares of inputs with $c<\tau$ under R and A. Coverage is the share of inputs with $c\ge\tau$, covered accuracy is the accuracy of predicted labels on those inputs, and the risk--coverage area is the mean error rate among the $k$ most confident inputs, averaged over $k$ (all three defined for R and A). A dash marks a metric that is undefined for a policy.

\textbf{Intervals.} Paired contrasts report the mean seed difference, a percentile 95\% interval from 10,000 bootstrap resamples of the ten seed differences, and the numbers of positive and negative seeds. Intervals are pointwise and unadjusted for multiple comparisons; an asterisk marks an interval that excludes zero.
''')

# ------------------------------------------------------------------ S2 condition means
runs = json.load(open(RES / 'controlled_study_v2.json'))
grp = defaultdict(list)
for r in runs['runs']:
    grp[(r['noise'], r['preference_sd'], r['fusion'], r['bandit'], r['handling'])].append(r['metrics'])

w(r'\section{Main Experiment: Condition Means}' + '\n')
w('Tables S1--S9 give the mean across ten seeds for each of the 96 conditions, with the sample SD in parentheses.\n')
for key, label, d in METRICS:
    w(r'\begin{longtable}{llll' + 'c' * 4 + '}')
    w(r'\caption{' + label + r' for all main-experiment conditions: mean (SD) across ten seeds.}\\')
    w(r'\toprule Noise & Pref.\ SD & Rule & Selector & U & R & A & W \\ \midrule \endfirsthead')
    w(r'\toprule Noise & Pref.\ SD & Rule & Selector & U & R & A & W \\ \midrule \endhead')
    w(r'\bottomrule \endfoot')
    first = True
    for n in NOISE:
        if not first:
            w(r'\midrule')
        first = False
        for p in PREFS:
            for f in RULES:
                for b, bl in SEL:
                    cells = []
                    for h in POL:
                        vals = [m[key] for m in grp[(n, p, f, b, h)] if m[key] is not None]
                        if not vals:
                            cells.append(DASH)
                        else:
                            cells.append(f'{num(np.mean(vals), d)} ({num(np.std(vals, ddof=1), d)})')
                    w(f'{NOISE_LABEL[n]} & {p:.1f} & {RULE_LABEL[f]} & {bl} & ' + ' & '.join(cells) + r' \\')
    w(r'\end{longtable}')

# ------------------------------------------------------------------ S3 contrasts
rows = list(csv.DictReader(open(RES / 'controlled_v2_all_contrasts.csv')))
CON = [  # (left handling, left bandit, right handling, right bandit, label)
    ('abstain', False, 'relabel', False, r'A$-$R, det.'),
    ('downweight', False, 'relabel', False, r'W$-$R, det.'),
    ('downweight', False, 'abstain', False, r'W$-$A, det.'),
    ('abstain', True, 'relabel', True, r'A$-$R, LinUCB'),
    ('downweight', True, 'relabel', True, r'W$-$R, LinUCB'),
    ('downweight', True, 'abstain', True, r'W$-$A, LinUCB'),
    ('ungated', True, 'ungated', False, r'U: LinUCB$-$det.'),
    ('relabel', True, 'relabel', False, r'R: LinUCB$-$det.'),
    ('abstain', True, 'abstain', False, r'A: LinUCB$-$det.'),
    ('downweight', True, 'downweight', False, r'W: LinUCB$-$det.'),
]
idx = {}
for r in rows:
    L, R = ast.literal_eval(r['left']), ast.literal_eval(r['right'])
    idx[(r['noise'], float(r['preference_sd']), r['fusion'], L['handling'], L['bandit'], R['handling'], R['bandit'], r['metric'])] = r

w(r'\section{Main Experiment: Paired Contrasts}' + '\n')
w('Tables S10--S18 list all ten paired contrasts in each of the 12 noise, preference and rule combinations, one table per metric. '
  'The prespecified down-weight-minus-relabel contrasts (W$-$R) are those in Table~7 of the main text. Contrasts that are undefined for a metric are omitted.\n')
for key, label, d in METRICS:
    w(r'\begin{longtable}{lllrlr}')
    w(r'\caption{Paired contrasts in ' + label[0].lower() + label[1:] + r' (main experiment): mean difference, 95\% interval and positive/negative seeds.}\\')
    hdr = r'\toprule Noise & Pref.\ SD & Rule & \multicolumn{1}{l}{Contrast} & Mean [95\% interval] & $+/-$ \\ \midrule'
    w(hdr + r' \endfirsthead')
    w(hdr + r' \endhead')
    w(r'\bottomrule \endfoot')
    first = True
    for n in NOISE:
        for p in PREFS:
            for f in RULES:
                block = []
                for lh, lb, rh, rb, cl in CON:
                    r = idx.get((n, p, f, lh, lb, rh, rb, key))
                    if r is None or int(r['valid_seeds']) == 0 or r['mean'] in ('', 'None', 'nan'):
                        continue
                    lo, hi = ast.literal_eval(r['ci95'])
                    star = r'$^{\ast}$' if (lo > 0 or hi < 0) else ''
                    block.append(f'{cl} & {num(float(r["mean"]), 4)} {ci(lo, hi)}{star} & {r["positive"]}/{r["negative"]}')
                if not block:
                    continue
                if not first:
                    w(r'\midrule')
                first = False
                for i, b in enumerate(block):
                    lead = f'{NOISE_LABEL[n]} & {p:.1f} & {RULE_LABEL[f]}' if i == 0 else ' & & '
                    w(lead + ' & ' + b + r' \\')
    w(r'\end{longtable}')

# ------------------------------------------------------------------ S4 threshold sweep
sw = defaultdict(list)
for r in runs['threshold_sweep']:
    sw[(r['noise'], r['preference_sd'], r['fusion'], r['threshold'], r['handling'])].append(r['metrics'])
w(r'\section{Main Experiment: Threshold Sweep}' + '\n')
w('The descriptive sweep evaluates relabeling and abstention with the deterministic selector at five thresholds (1,200 runs). '
  r'Table~S19 gives means across ten seeds. The fallback rate under R equals the abstention rate under A, because both apply the same threshold to the same confidence scores.' + '\n')
w(r'\begin{longtable}{lllcccccc}')
w(r'\caption{Threshold sweep with the deterministic selector: mean across ten seeds. NDCG: graded NDCG@10 under R and A; rate: share of inputs with $c<\tau$; Acc.: emitted-label accuracy under R; Cov.\ acc.: accuracy of predicted labels on covered inputs.}\\')
hdr = r'\toprule Noise & Pref.\ SD & Rule & $\tau$ & NDCG (R) & NDCG (A) & Rate & Acc.\ (R) & Cov.\ acc. \\ \midrule'
w(hdr + r' \endfirsthead'); w(hdr + r' \endhead'); w(r'\bottomrule \endfoot')
taus = sorted({r['threshold'] for r in runs['threshold_sweep']})
first = True
for n in NOISE:
    for p in PREFS:
        for f in RULES:
            if not first:
                w(r'\midrule')
            first = False
            for i, t in enumerate(taus):
                R_ = sw[(n, p, f, t, 'relabel')]; A_ = sw[(n, p, f, t, 'abstain')]
                mean = lambda xs, k: np.mean([x[k] for x in xs if x[k] is not None]) if any(x[k] is not None for x in xs) else None
                lead = f'{NOISE_LABEL[n]} & {p:.1f} & {RULE_LABEL[f]}' if i == 0 else ' & & '
                w(lead + f' & {t:.1f} & {num(mean(R_, "ndcg_at_10"))} & {num(mean(A_, "ndcg_at_10"))} & '
                  f'{num(mean(R_, "fallback_rate"))} & {num(mean(R_, "emotion_accuracy"))} & {num(mean(R_, "covered_accuracy"))}' + r' \\')
w(r'\end{longtable}')

# ------------------------------------------------------------------ S5 MELD all thresholds
meld = json.load(open(RES / 'meld_substitution_condition.json'))
w(r'\section{MELD Substitution Condition at All Thresholds}' + '\n')
w(r'Table~S20 extends Table~4 of the main text to all nine thresholds, both splits and all three inference rules (test $n=2{,}211$; development $n=897$). '
  r'$\pi_L$ is the share of inputs with $c<\tau$; Corr.$|L$ and Neut.$|L$ are the shares of those inputs whose predicted label is correct and whose true label is Neutral. Exploratory analysis.' + '\n')
w(r'\begin{longtable}{llccccccc}')
w(r'\caption{Neutral substitution on MELD at all splits, inference rules and thresholds (four-class subset). $\Delta$Acc and $\Delta$F1: relabel minus ungated.}\\')
hdr = r'\toprule Split & Rule & $\tau$ & $\pi_L$ & Corr.$|L$ & Neut.$|L$ & Acc.\ (U) & $\Delta$Acc & $\Delta$F1 \\ \midrule'
w(hdr + r' \endfirsthead'); w(hdr + r' \endhead'); w(r'\bottomrule \endfoot')
order = {'test': 0, 'dev': 1}; rorder = {'text-only': 0, 'rescaled': 1, 'fixed': 2}
mrows = sorted(meld['rows'], key=lambda r: (order[r['split']], rorder[r['rule']], r['threshold']))
prev = None
for r in mrows:
    g = (r['split'], r['rule'])
    if prev is not None and g != prev:
        w(r'\midrule')
    lead = f'{"test" if r["split"] == "test" else "development"} & {r["rule"]}' if g != prev else ' & '
    prev = g
    w(lead + f' & {r["threshold"]:.2f} & {num(r["low_share"])} & {num(r["p_label_correct_low"])} & {num(r["p_true_neutral_low"])} & '
      f'{num(r["acc_ungated"])} & {num(r["delta_acc"])} & {num(r["macro_f1_relabel"] - r["macro_f1_ungated"])}' + r' \\')
w(r'\end{longtable}')

# ------------------------------------------------------------------ S6 listening log
now = json.load(open(RES / 'nowplaying_transfer.json'))
NP = {'downweight': 'down-weight', 'base': 'base', 'ungated': 'ungated', 'relabel': 'relabel', 'abstain': 'abstain'}
w(r'\section{Listening-Log Study: All Paired Contrasts}' + '\n')
w(r'Table~S21 lists all 14 paired contrasts of the listening-log study for the three ranking metrics. Intervals resample the 53 users (10,000 resamples) after averaging over the ten corruption seeds; the seed counts give the number of seeds with a positive or negative difference.' + '\n')
w(r'\begin{longtable}{llrrr}')
w(r'\caption{Paired contrasts in the listening-log study: mean difference [95\% user-bootstrap interval] and positive/negative seeds.}\\')
hdr = r'\toprule Rule & Contrast & NDCG@10 & Recall@10 & RR@10 \\ \midrule'
w(hdr + r' \endfirsthead'); w(hdr + r' \endhead'); w(r'\bottomrule \endfoot')
prev = None
for c in now['contrasts']:
    if prev is not None and c['fusion'] != prev:
        w(r'\midrule')
    lead = RULE_LABEL[c['fusion']] if c['fusion'] != prev else ''
    prev = c['fusion']
    cells = []
    for k in ('ndcg_at_10', 'recall_at_10', 'mrr_at_10'):
        m = c['metrics'][k]; lo, hi = m['ci95']
        star = r'$^{\ast}$' if (lo > 0 or hi < 0) else ''
        cells.append(r'\begin{tabular}[t]{@{}r@{}}' + f'{num(m["mean"], 4)}{star}' + r'\\[-1pt]{\scriptsize ' + ci(lo, hi) + f' {m["positive_seeds"]}/{m["negative_seeds"]}' + r'}\end{tabular}')
    w(f'{lead} & {NP[c["left"]]} $-$ {NP[c["right"]]} & ' + ' & '.join(cells) + r' \\[2pt]')
w(r'\end{longtable}')

# ------------------------------------------------------------------ S7 preliminary experiment
pre = json.load(open(RES / 'controlled_study.json'))
w(r'\section{Preliminary Experiment: All Conditions}' + '\n')
w(r'The preliminary experiment applied relabeling at $\tau=0.6$ in all 24 conditions. Table~S22 gives the mean (SD) across ten seeds; Table~9 of the main text shows its rescaled-fusion rows for NDCG@10.' + '\n')
w(r'\begin{longtable}{lllccccc}')
w(r'\caption{All conditions of the preliminary experiment (relabeling at $\tau=0.6$): mean (SD) across ten seeds.}\\')
hdr = r'\toprule Noise SD & Pref.\ SD & Rule & Selector & NDCG@10 & Skip rate & Fallback rate & Label acc. \\ \midrule'
w(hdr + r' \endfirsthead'); w(hdr + r' \endhead'); w(r'\bottomrule \endfoot')
srt = sorted(pre['summary'], key=lambda s: (s['noise'], s['preference_sd'], ['fixed', 'rescaled', 'text'].index(s['fusion']), s['bandit']))
prev = None
for s in srt:
    g = (s['noise'], s['preference_sd'])
    if prev is not None and g != prev:
        w(r'\midrule')
    prev = g
    m = s['metrics']
    f = lambda k: f'{num(m[k]["mean"])} ({num(m[k]["sd"])})'
    w(f'{s["noise"]:.1f} & {s["preference_sd"]:.1f} & {RULE_LABEL[s["fusion"]]} & {"LinUCB" if s["bandit"] else "deterministic"} & '
      f'{f("ndcg_at_10")} & {f("skip_rate")} & {f("fallback_rate")} & {f("emotion_accuracy")}' + r' \\')
w(r'\end{longtable}')

w(r'\end{document}')
(OUT / 'Supplementary_S1.tex').write_text('\n'.join(tex) + '\n')
print('supplement written:', OUT / 'Supplementary_S1.tex')
