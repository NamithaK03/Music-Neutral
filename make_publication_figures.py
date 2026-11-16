"""Publication figures for the manuscript (scientific-plot style).

Every figure is drawn from saved result files only. Style follows conventional
scientific plotting: serif (STIX/Times) type with matching math, fully boxed
axes, inward major and minor ticks on all four sides, framed legends, distinct
marker shapes per series, error bars with caps, and no decorative shading.
Figure widths equal the manuscript text width (394.36 pt) so type prints at
its true size.
"""
import ast
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
R = lambda name: json.loads((ROOT / 'results' / name).read_text(encoding='utf-8'))

FULL = 394.36 / 72.27                       # text width in inches
K = 'black'
# Nature Publishing Group (NPG) palette, as distributed in ggsci; widely used in journal figures
BLUE, RED, GREEN, ORANGE, PURPLE, GREY, SKY = '#3C5488', '#E64B35', '#00A087', '#F39B7F', '#8491B4', '#7E6148', '#4DBBD5'
RULE = {'text': BLUE, 'text-only': BLUE, 'rescaled': RED, 'fixed': GREY}

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5, 'legend.fontsize': 7.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.6, 'axes.edgecolor': K, 'axes.labelcolor': K, 'text.color': K,
    'axes.spines.top': True, 'axes.spines.right': True, 'axes.grid': False,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5, 'xtick.minor.size': 1.8, 'ytick.minor.size': 1.8,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
    'xtick.minor.visible': True, 'ytick.minor.visible': True, 'xtick.color': K, 'ytick.color': K,
    'legend.frameon': True, 'legend.fancybox': False, 'legend.edgecolor': K, 'legend.framealpha': 1.0,
    'legend.borderpad': 0.4, 'legend.handlelength': 2.0, 'patch.linewidth': 0.5,
    'lines.linewidth': 1.0, 'lines.markersize': 4.0, 'lines.markeredgewidth': 0.8,
    'errorbar.capsize': 2.0, 'hatch.linewidth': 0.5,
    'pdf.fonttype': 42, 'savefig.dpi': 600, 'figure.dpi': 150,
})


def panel(ax, letter, x=0.03, y=0.95, ha='left'):
    ax.text(x, y, f'({letter})', transform=ax.transAxes, ha=ha, va='top', fontsize=9)


def no_minor_x(ax):
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def no_minor_y(ax):
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def save(fig, name):
    """Write a vector PDF, then re-embed fonts with Ghostscript as standard subsets."""
    pdf = OUT / f'{name}.pdf'
    fig.savefig(pdf, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    if shutil.which('gs'):
        tmp = Path(tempfile.mkstemp(suffix='.pdf')[1])
        subprocess.run(['gs', '-q', '-dNOPAUSE', '-dBATCH', '-dSAFER', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.5',
                        '-dEmbedAllFonts=true', '-dSubsetFonts=true', f'-sOutputFile={tmp}', str(pdf)], check=True)
        shutil.move(tmp, pdf)


def bars(ax, x, v, sd=None, colors=None, width=0.6, horizontal=False, log=False):
    colors = colors or [BLUE, ORANGE, GREEN, PURPLE, RED]
    made = []
    for i, (xi, vi) in enumerate(zip(x, v)):
        p = (ax.barh(xi, vi, height=width, edgecolor=K, linewidth=0.5) if horizontal
             else ax.bar(xi, vi, width=width, edgecolor=K, linewidth=0.5))[0]
        made.append((p, colors[i % len(colors)]))
    ax._gradient_bars = made
    if sd is not None:
        if horizontal:
            ax.errorbar(v, x, xerr=sd, fmt='none', ecolor=K, elinewidth=0.6, capthick=0.6)
        else:
            ax.errorbar(x, v, yerr=sd, fmt='none', ecolor=K, elinewidth=0.6, capthick=0.6)


from matplotlib.colors import to_rgb


def _gradient_fill(ax, patch, color, horizontal=False, log=False):
    """Fill a bar patch with a smooth gradient: light tint at the base, full colour at the end."""
    c = np.array(to_rgb(color)); light = 1 - 0.78 * (1 - c)
    t = np.linspace(0, 1, 256)[:, None]
    ramp = light[None, :] * (1 - t) + c[None, :] * t          # 256 x 3, base -> end
    img = ramp[None, :, :] if horizontal else ramp[::-1][:, None, :]
    x0, y0 = patch.get_x(), patch.get_y(); w, h = patch.get_width(), patch.get_height()
    x1, y1 = x0 + w, y0 + h
    if horizontal and log:
        x0 = ax.get_xlim()[0]
    if not horizontal and log:
        y0 = ax.get_ylim()[0]
    im = ax.imshow(img, extent=[x0, x1, y0, y1], aspect='auto', interpolation='bicubic', zorder=patch.get_zorder() - 0.1)
    im.set_clip_path(patch)
    patch.set_facecolor('none')


def gbar(ax, x, v, color, width=0.6, horizontal=False, log=False):
    xl, yl = ax.get_xlim(), ax.get_ylim()
    p = (ax.barh(x, v, height=width, edgecolor=K, linewidth=0.5) if horizontal
         else ax.bar(x, v, width=width, edgecolor=K, linewidth=0.5))[0]
    return p


def finish_gradients(ax, patches, horizontal=False, log=False):
    """Apply gradients after axis limits are final (needed for log axes)."""
    for p, col in patches:
        _gradient_fill(ax, p, col, horizontal=horizontal, log=log)


# ---------------------------------------------------------------- MELD component figures
meld = R('meld_checkpoint_complete_train.json')['metrics']


def meld_conflict():
    test = meld['test']['mapped_four_class_subset']
    fu, cs = test['fusion_and_conflicts']['fusion'], test['fusion_and_conflicts']['conflict_subset']
    names = ['Text only', 'Speech only', 'Prob. average', 'Fixed fusion', 'Fixed + fallback']
    f1 = [test['text']['raw']['macro_f1'], test['speech']['raw']['macro_f1'], fu['naive']['macro_f1'],
          fu['confidence_aware']['macro_f1'], fu['confidence_aware_fallback']['macro_f1']]
    ca = [cs['text_accuracy'], cs['speech_accuracy'], fu['naive']['conflict_accuracy'],
          fu['confidence_aware']['conflict_accuracy'], fu['confidence_aware_fallback']['conflict_accuracy']]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.15))
    for ax, vals, ylabel, letter in ((axes[0], f1, f'Macro-F1 (all, $n$ = {test["sample_count"]:,})', 'a'),
                                     (axes[1], ca, f'Accuracy (conflicts, $n$ = {cs["count"]})', 'b')):
        bars(ax, np.arange(5), vals, colors=[BLUE, SKY, GREEN, ORANGE, RED])
        ax.set_xticks(np.arange(5)); ax.set_xticklabels(names, rotation=30, ha='right', rotation_mode='anchor')
        no_minor_x(ax); ax.tick_params(axis='x', which='both', top=False)
        ax.set_ylim(0, 0.7); ax.set_xlim(-0.6, 4.6); ax.set_ylabel(ylabel); panel(ax, letter, x=0.97, ha='right')
        finish_gradients(ax, ax._gradient_bars)
    fig.tight_layout(w_pad=2.0); save(fig, 'meld_conflict')


def threshold_sensitivity():
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.1))
    for split, ls, mk, col in (('dev', '--', 's', ORANGE), ('test', '-', 'o', RED)):
        v = meld[split]['mapped_four_class_subset']['fallback_threshold_sensitivity']
        ts = sorted(map(float, v)); rows = [v[str(t)] for t in ts]
        lab = 'Development' if split == 'dev' else 'Test'
        axes[0].plot(ts, [r['macro_f1'] for r in rows], ls, marker=mk, color=col, label=lab)
        axes[1].plot(ts, [r['fallback_rate'] for r in rows], ls, marker=mk, color=col, label=lab)
    tx = meld['test']['mapped_four_class_subset']['text']['raw']['macro_f1']
    axes[0].axhline(tx, color=BLUE, lw=1.0, ls='-.', label='Text only (test)')
    axes[0].set_ylim(0, 0.6); axes[0].set_ylabel('Four-class macro-F1')
    axes[1].set_ylim(0, 1.05); axes[1].set_ylabel('Fallback rate')
    for ax, letter in zip(axes, 'ab'):
        ax.set_xlabel(r'Fallback threshold $\tau$'); ax.set_xticks([.4, .5, .6, .7, .8]); ax.set_xlim(0.37, 0.83)
        panel(ax, letter)
    axes[0].legend(loc='lower left', bbox_to_anchor=(0.3, 0.40)); axes[1].legend(loc='lower right')
    fig.tight_layout(w_pad=2); save(fig, 'threshold_sensitivity')


# ---------------------------------------------------------------- substitution condition
def meld_condition():
    d = R('meld_substitution_condition.json')['rows']
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 3.9), sharex=True)
    for col, rule in enumerate(('text-only', 'rescaled')):
        ax, bx = axes[0, col], axes[1, col]
        for split, ls, mk in (('test', '-', 'o'), ('dev', '--', 's')):
            rows = [r for r in d if r['rule'] == rule and r['split'] == split]
            t = [r['threshold'] for r in rows]
            fill = lambda c: c
            ax.plot(t, [r['p_label_correct_low'] for r in rows], ls, marker=mk, color=BLUE, mfc=fill(BLUE))
            ax.plot(t, [r['p_true_neutral_low'] for r in rows], ls, marker='^' if split == 'test' else 'v', color=RED, mfc=fill(RED))
            bx.plot(t, [r['delta_acc'] for r in rows], ls, marker=mk, color=GREEN if split == 'test' else PURPLE)
        ax.set_ylim(0, 0.8); bx.set_ylim(-0.15, 0.01); bx.axhline(0, color=K, lw=0.5, ls=':')
        bx.set_xlabel(r'Confidence threshold $\tau$'); bx.set_xticks([.4, .5, .6, .7, .8]); bx.set_xlim(0.37, 0.83)
        ax.text(0.97, 0.95, 'text-only confidence' if col == 0 else 'rescaled-fusion confidence',
                transform=ax.transAxes, ha='right', va='top', fontsize=8, style='italic')
        panel(ax, 'ab'[col]); panel(bx, 'cd'[col], y=0.12)
    axes[0, 0].set_ylabel(r'Share among inputs with $c<\tau$')
    axes[1, 0].set_ylabel(r'$\Delta$Accuracy (relabel $-$ ungated)')
    h = [Line2D([], [], color=BLUE, marker='o', ls='-'), Line2D([], [], color=RED, marker='^', ls='-'),
         Line2D([], [], color=GREEN, marker='o', ls='-'), Line2D([], [], color=PURPLE, marker='s', ls='--'),
         Line2D([], [], color=K, ls='-'), Line2D([], [], color=K, ls='--')]
    fig.legend(h, [r'$P(\hat{Y}=Y\mid L)$', r'$P(Y=\mathrm{Neutral}\mid L)$', r'$\Delta$Acc, test', r'$\Delta$Acc, development',
                   'Test ($n$ = 2,211)', 'Development ($n$ = 897)'],
               loc='lower center', ncol=3, bbox_to_anchor=(0.53, -0.01), frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1), h_pad=0.6, w_pad=1.2); save(fig, 'meld_substitution_condition')


def accuracy_link():
    syn = R('synthetic_substitution_condition.json')['rows']
    expl = R('controlled_v2_exploratory_contrasts.json')['contrasts']
    fig, ax = plt.subplots(figsize=(FULL * 0.62, 2.55))
    xx = np.linspace(0, 0.19, 10)
    ax.plot(xx, 0.493 * xx, 'k--', lw=0.8, label=r'$y = 0.493\,x$')
    for pref, color, mk, mfc, name in ((0.0, BLUE, 'o', BLUE, 'Preference SD 0'), (0.5, RED, 's', RED, 'Preference SD 0.5')):
        xs, ys, lo, hi = [], [], [], []
        for r in syn:
            c = next(c for c in expl if (c['noise'], c['fusion'], c['preference_sd'], c['left'], c['right'], c['bandit'])
                     == (r['noise'], r['fusion'], pref, 'ungated', 'relabel', False))['metrics']['ndcg_at_10']
            xs.append(-r['delta_acc']); ys.append(c['mean']); lo.append(c['mean'] - c['ci95'][0]); hi.append(c['ci95'][1] - c['mean'])
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk, color=color, mfc=mfc, ms=4.2, elinewidth=0.7, capthick=0.7, label=name)
    ax.axhline(0, color=K, lw=0.5, ls=':')
    ax.set_xlim(0, 0.19); ax.set_ylim(-0.02, 0.1)
    ax.set_xlabel(r'$\mathrm{Acc}(U)-\mathrm{Acc}(R)$')
    ax.set_ylabel(r'$\Delta$NDCG@10 ($U-R$)')
    ax.legend(loc='upper left')
    fig.tight_layout(); save(fig, 'accuracy_ndcg_link')


# ---------------------------------------------------------------- controlled study
def confidence_policies():
    runs = R('controlled_study_v2.json')['runs']
    pol = ('ungated', 'relabel', 'abstain', 'downweight')
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 3.75), sharex=True)
    for i, pref in enumerate((0.0, 0.5)):
        for j, noise in enumerate(('low', 'high', 'asymmetric')):
            ax = axes[i, j]
            for fusion in ('text', 'rescaled'):
                for bandit, ls, mk in ((False, '-', 'o'), (True, '--', 's')):
                    vals = [[r['metrics']['ndcg_at_10'] for r in runs if (r['noise'], r['preference_sd'], r['fusion'], r['bandit'], r['handling'])
                             == (noise, pref, fusion, bandit, h)] for h in pol]
                    m = [np.mean(v) for v in vals]; se = [np.std(v, ddof=1) / np.sqrt(len(v)) for v in vals]
                    off = (-0.09 if fusion == 'text' else 0.09) + (0.0 if not bandit else 0.03)
                    ax.errorbar(np.arange(4) + off, m, yerr=se, fmt=ls, marker=mk, color=RULE[fusion],
                                mfc=RULE[fusion], lw=0.9, ms=3.6, elinewidth=0.6, capsize=1.5, capthick=0.6)
            ax.set_xticks(range(4)); ax.set_xticklabels(['U', 'R', 'A', 'W']); no_minor_x(ax); ax.set_xlim(-0.4, 3.4)
            if i == 0:
                ax.set_title({'low': r'Low noise ($\sigma=0.5$)', 'high': r'High noise ($\sigma=1.5$)',
                              'asymmetric': r'Asymmetric ($\sigma_{\mathrm{tx}}=1.40$, $\sigma_{\mathrm{sp}}=1.90$)'}[noise], fontsize=8)
            if j == 2:
                ax.text(1.04, 0.5, f'Preference SD {pref}', transform=ax.transAxes, rotation=270, va='center', ha='left', fontsize=8)
            panel(ax, 'abcdef'[3 * i + j], x=0.04, y=0.96)
            if j == 0:
                ax.set_ylabel('Mean NDCG@10')
            if i == 1:
                ax.set_xlabel('Policy')
        lo = min(a.get_ylim()[0] for a in axes[i]); hi = max(a.get_ylim()[1] for a in axes[i])
        for a in axes[i]:
            a.set_ylim(lo - 0.03 * (hi - lo), hi + 0.24 * (hi - lo))
        for a in axes[i][1:]:
            a.tick_params(labelleft=False)
    h = [Line2D([], [], color=BLUE, lw=1), Line2D([], [], color=RED, lw=1),
         Line2D([], [], color=K, ls='-', marker='o'), Line2D([], [], color=K, ls='--', marker='s')]
    fig.legend(h, ['Text-only', 'Rescaled fusion', 'Deterministic', 'LinUCB'], loc='lower center', ncol=4,
               bbox_to_anchor=(0.53, -0.005), frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.4, w_pad=0.4); save(fig, 'confidence_policies')


def decomposition():
    expl = R('controlled_v2_exploratory_contrasts.json')['contrasts']
    pre = {}
    with open(ROOT / 'results/controlled_v2_all_contrasts.csv') as f:
        for r in csv.DictReader(f):
            L, Rr = ast.literal_eval(r['left']), ast.literal_eval(r['right'])
            if r['metric'] == 'ndcg_at_10' and L == {'handling': 'downweight', 'bandit': False} and Rr == {'handling': 'relabel', 'bandit': False}:
                pre[(r['noise'], float(r['preference_sd']), r['fusion'])] = (float(r['mean']), ast.literal_eval(r['ci95']))
    get = lambda n, p, f, a, b: next((c['metrics']['ndcg_at_10']['mean'], c['metrics']['ndcg_at_10']['ci95'])
                                      for c in expl if (c['noise'], c['preference_sd'], c['fusion'], c['left'], c['right'], c['bandit']) == (n, p, f, a, b, False))
    cells = [(n, p, f) for n in ('low', 'high', 'asymmetric') for p in (0.0, 0.5) for f in ('text', 'rescaled')]
    series = [(r'$W-R$ (prespecified)', BLUE, 'o', BLUE, -0.25, lambda c: pre[c]),
              (r'$U-R$ (exploratory)', RED, 's', RED, 0.0, lambda c: get(*c, 'ungated', 'relabel')),
              (r'$W-U$ (exploratory)', GREEN, 'D', GREEN, 0.25, lambda c: get(*c, 'downweight', 'ungated'))]
    fig, ax = plt.subplots(figsize=(FULL, 3.9))
    y = np.arange(len(cells))[::-1].astype(float)
    for label, color, mk, mfc, off, fn in series:
        m = np.array([fn(c)[0] for c in cells]); ci = np.array([fn(c)[1] for c in cells])
        ax.errorbar(m, y + off, xerr=[m - ci[:, 0], ci[:, 1] - m], fmt=mk, color=color, mfc=mfc, ms=3.4,
                    elinewidth=0.7, capsize=1.6, capthick=0.6, label=label)
    ax.axvline(0, color=K, lw=0.6, ls=':')
    for b in (3.5, 7.5):
        ax.axhline(b, color=K, lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels([f'{n}, SD {p}, {"text-only" if f == "text" else "rescaled"}' for n, p, f in cells])
    no_minor_y(ax); ax.tick_params(axis='y', which='both', right=False)
    ax.set_ylim(-0.6, 11.6)
    ax.set_xlabel(r'Paired difference in mean NDCG@10 (95% bootstrap CI)')
    ax.legend(loc='lower right')
    fig.tight_layout(); save(fig, 'decomposition_forest')


# ---------------------------------------------------------------- listening log
def nowplaying():
    c = R('nowplaying_transfer.json')['counts']
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.0), gridspec_kw={'width_ratios': [1.55, 1]})
    labels = ['All test events', 'Outside training catalog', 'No unambiguous proxy', 'Eligible']
    vals = [c['test_events'], c['test_unknown_catalog'], c['test_unlabelled_or_conflicting'], c['eligible_events']]
    ax = axes[0]; yy = np.arange(4)[::-1]
    made = [(ax.barh(y, v, height=0.6, edgecolor=K, linewidth=0.5)[0], col) for y, v, col in zip(yy, vals, [BLUE, ORANGE, PURPLE, RED])]
    ax.set_xscale('log'); ax.set_xlim(1e2, 3e5); ax.set_ylim(-0.6, 3.6); finish_gradients(ax, made, horizontal=True, log=True); ax.set_yticks(yy); ax.set_yticklabels(labels)
    no_minor_y(ax); ax.tick_params(axis='y', which='both', right=False, left=False)
    ax.set_xlabel('Number of test events'); panel(ax, 'a', x=0.97, y=0.95, ha='right')
    ax = axes[1]; cls = ['happy', 'sad', 'angry', 'neutral']; v2 = [c['proxy_classes'][k] for k in cls]
    made = [(ax.bar(i, v, width=0.6, edgecolor=K, linewidth=0.5)[0], [ORANGE, BLUE, RED, GREEN][i]) for i, v in enumerate(v2)]
    ax.set_yscale('log'); ax.set_ylim(0.5, 5000); ax.set_xlim(-0.6, 3.6); finish_gradients(ax, made, log=True); ax.set_xticks(range(4)); ax.set_xticklabels(['H', 'S', 'A', 'N'])
    no_minor_x(ax); ax.tick_params(axis='x', which='both', top=False)
    ax.set_ylabel('Eligible events'); ax.set_xlabel('Proxy label'); panel(ax, 'b', x=0.95, ha='right')
    fig.tight_layout(w_pad=1.5); save(fig, 'nowplaying_feasibility')


# ---------------------------------------------------------------- appendix benchmarks
def simple_bars(name, values, sds, names, ylabels):
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.0))
    for ax, v, sd, ylabel, letter in zip(axes, values, sds, ylabels, 'ab'):
        bars(ax, np.arange(3), v, sd=sd, colors=[BLUE, GREEN, ORANGE], width=0.55)
        ax.set_xticks(range(3)); ax.set_xticklabels(names); no_minor_x(ax); ax.tick_params(axis='x', which='both', top=False)
        ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, max(a + s for a, s in zip(v, sd)) * 1.25)
        ax.set_ylabel(ylabel); panel(ax, letter)
        finish_gradients(ax, ax._gradient_bars)
    fig.tight_layout(w_pad=2); save(fig, name)


def simulation():
    d = R('tuned_rerun.json')['held_out_test']
    keys = ['hybrid_linucb_paper_defaults', 'hybrid_linucb_tuned', 'hybrid_no_bandit_same_weights']
    simple_bars('simulation_comparison', [[d[k][m]['mean'] for k in keys] for m in ('ndcg_at_10', 'skip_rate')],
                [[d[k][m]['std'] for k in keys] for m in ('ndcg_at_10', 'skip_rate')],
                ['Fixed\nLinUCB', 'Tuned\nLinUCB', 'Hybrid,\nno bandit'], ['NDCG@10', 'Skip rate'])


def lastfm():
    d = R('lastfm_rerun.json')['aggregate_metrics_macro_averaged_by_user']
    keys = ['popularity', 'tag_content', 'item_item']
    simple_bars('lastfm_comparison', [[d[k][m]['mean'] for k in keys] for m in ('ndcg_at_10', 'recall_at_10')],
                [[d[k][m]['std'] for k in keys] for m in ('ndcg_at_10', 'recall_at_10')],
                ['Popularity', 'Tag content', 'Item-item CF'], ['NDCG@10', 'Recall@10'])


if __name__ == '__main__':
    for fn in (meld_conflict, threshold_sensitivity, meld_condition, accuracy_link, confidence_policies,
               decomposition, nowplaying, simulation, lastfm):
        fn()
    print('publication figures written')
