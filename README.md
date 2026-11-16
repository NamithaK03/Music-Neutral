# Neutral relabeling under uncertain emotion: code and results

This repository contains the code, frozen protocols, saved results and figure and table generators for a study of a common rule in emotion-aware music recommendation. The rule replaces a low-confidence emotion prediction with *Neutral*.

The study asks one question: when does substituting Neutral for an uncertain emotion label help? It has four parts:

- **Analytical condition.** Relabeling changes label accuracy by π_L · [P(Y = Neutral | L) − P(Ŷ = Y | L)], where L is the set of inputs below the confidence threshold and π_L is its share. `experiments/meld_substitution_condition.py` checks this identity numerically for every row.
- **MELD analysis.** The condition is evaluated on released MELD text and audio baseline checkpoints across two splits, three inference rules and nine thresholds.
- **Controlled synthetic study.** It compares four policies (ungated, relabel, abstain, down-weight) using paired seeds, a deterministic genre selector and a LinUCB genre selector.
- **Listening-log feasibility study.** It uses #nowplaying-RS with hashtag mood proxies.

It also includes component checks on DEAM (track affect from audio) and HetRec Last.fm (artist ranking).

## Repository layout

```
sonar/                    Python package: metrics, fusion rules, recommender, LinUCB,
                          controlled-study generator, MELD/DEAM/Last.fm evaluation
experiments/              Study scripts (controlled study v2, MELD and synthetic
                          substitution-condition analyses, listening-log study,
                          logged literature search)
tests/                    Unit tests (28)
results/                  All saved run-level results (JSON/CSV/NPZ)
protocol_archive/         Protocols and sources archived before execution, with
                          SHA-256 manifests and commit timestamps; the Git history of
                          the frozen amendment (commit d53bc1d1) is kept as
                          frozen_amendment_1_git_history.bundle
literature_search/        Raw database-search responses, screening output and status
*_PROTOCOL*.md            Study protocols (hashed; do not edit)
*_RESULTS.md              Readable result reports generated from results/
VALIDATION_*.md, INTERPRETATION_REVISION_*.md
                          Validation notes and post-result scope corrections
make_paper_tables.py      Writes LaTeX table rows to tables/ from results/
make_supplement.py        Writes the supplementary report (Document S1) to supplement/ from results/
make_publication_figures.py
                          Writes vector PDF figures to figures/ from results/
verify_revision_results.py
                          Read-only integrity checks on saved results and archive hashes
```

## Installation

Python 3.10 or later. The controlled study was originally executed with Python 3.10.10 and NumPy 1.26.4.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[meld-study,recommendation-study,figures]"
pip install -r requirements-revision.txt
```

## Quick verification (no external data needed)

```bash
python -m unittest discover -s tests -v
python verify_revision_results.py
```

To inspect the archived commit of the frozen amendment:

```bash
git clone protocol_archive/frozen_amendment_1_git_history.bundle /tmp/frozen_amendment_1
git -C /tmp/frozen_amendment_1 log
```

`verify_revision_results.py` reads the saved outputs only. It checks the following:

- the 96 core conditions and the 1,200 threshold-sweep rows;
- that the original relabel runs match;
- that the frozen protocol hashes and archive chronology are intact;
- that all 80 listening-log runs match the per-user arrays.

## Reproducing analyses, tables and figures from saved results

```bash
python experiments/meld_substitution_condition.py      # MELD substitution condition (54 rows)
python experiments/synthetic_substitution_condition.py # replay of frozen synthetic inference streams
python make_paper_tables.py
python make_publication_figures.py
python make_supplement.py
python report_controlled_v2.py                         # regenerates CONTROLLED_V2_RESULTS.md and CSVs
```

These scripts read only `results/`. Tables and figures are not stored in the repository; the two generator scripts create `tables/` and `figures/` when run. The MELD analysis uses `results/meld_checkpoint_predictions.csv`, which holds temperature-scaled four-class probabilities from the released checkpoints.

## Re-running the experiments

Each archived experiment was run once. `experiments/controlled_study_v2.py` and `experiments/nowplaying_transfer.py` refuse to overwrite an existing execution record. To reproduce a run:

1. Work in a separate clean copy of the repository.
2. Move the existing record aside in that copy.
3. Keep the authoritative results in this repository unchanged.

```bash
python experiments/controlled_study_v2.py   # 96 conditions x 10 seeds + 1,200 sweep runs
```

### External data

No dataset is downloaded automatically. Obtain the data from the original distributors and respect their licences.

| Data | Source | Used by |
|---|---|---|
| MELD dialogues and baseline artifacts | https://github.com/declare-lab/MELD | `sonar evaluate-meld-checkpoints`, `sonar evaluate-meld`, `sonar evaluate-meld-episode-disjoint` |
| DEAM annotations and openSMILE features | MediaEval DEAM distribution | `sonar evaluate-deam --annotations-zip ... --features-zip ...` |
| HetRec 2011 Last.fm 2K | GroupLens HetRec 2011 | `sonar evaluate-lastfm --listening ... --tags ...` |
| #nowplaying-RS (`nowplaying_rs_train_test.zip`, MD5 `32d00b55dbeb2b617fd041c8156d3e86`) | https://zenodo.org/records/3248543 | `experiments/nowplaying_transfer.py` |

For the listening-log study, place the archive at `../recovered_study/data/nowplaying/nowplaying_rs_train_test.zip`, relative to the repository root. Run `sonar <command> --help` for the arguments of each component evaluation.

## Notes on the evidence

- The MELD substitution analysis, the synthetic replay and the ungated/down-weight decomposition were devised after the main results were inspected. They are exploratory.
- The controlled study is fully synthetic. Seeds are replicate simulations, not independent listeners. All intervals are pointwise and unadjusted.
- The deterministic abstention arm is uninformative by construction: every track receives the same emotion-free score.
- The listening-log study is a documented feasibility limit. Only 965 of 90,992 test events qualify, and 99.4% of them carry the happy proxy.
- The literature search is incomplete. Database access errors are recorded in `literature_search/SEARCH_STATUS.md`.

## Data and privacy

The saved MELD predictions contain utterance identifiers, labels and probabilities. They contain no transcripts or audio. Raw listening logs are not redistributed. No new human-subject data were collected.
