# Controlled synthetic follow-up: Amendment 1

**Date written:** 14 November 2026
**Status:** Written before any condition below was run. The original protocol
(`CONTROLLED_PROTOCOL.md`) and its reported results stand unchanged. This
amendment adds conditions; it does not replace or re-analyse earlier ones.
It is not a preregistration with an external registry; commit this file to the
repository (with its hash recorded) before running the extended script.

## 1. Question

When a music recommender is uncertain about the listener's emotion, which
response gives better simulated recommendation outcomes: relabelling to
Neutral, abstaining to an emotion-agnostic recommender, or down-weighting by
confidence? And does the answer change with prediction noise, branch
asymmetry and hidden preferences?

## 2. Unchanged from the original protocol

Catalog generator (80 tracks, 4 genres, Dirichlet affinities with
concentration 4 on own genre and 0.5 elsewhere), uniform true emotions,
logit signal magnitude 2, completion model
0.1 + 0.8·sigmoid(2·(a_{i,e} + p_g − 0.5)), pre-drawn per-track feedback
uniforms, reward +1/−1, 1,000 interactions per seed, scenario seeds
2026–2035, top-10 within-genre lists, graded NDCG@10 with completion
probabilities as relevance, LinUCB with identity initialization,
exploration 0.5 and no decay. No parameter is tuned on outcomes.

## 3. Factors

| Factor | Levels |
|---|---|
| Confidence handling | ungated, relabel (baseline), abstain, down-weight |
| Fusion | rescaled trust, text-only |
| Recommender | deterministic, LinUCB |
| Noise condition | symmetric low (text 0.5, speech 0.5); symmetric high (1.5, 1.5); asymmetric MELD-calibrated (Section 5) |
| Hidden preference SD | 0.0, 0.5 |

Full factorial: 4 × 2 × 2 × 3 × 2 = 96 conditions × 10 seeds = 960 runs.
The fixed (0.7-capped) fusion rule is not carried forward; its behaviour is
already reported in the original study.

## 4. Policy definitions

Let p_tx and p_sp be the branch probability vectors, c = fused trust score,
τ = 0.6, and q the fused probability vector:
- rescaled: q = (m_tx·p_tx + m_sp·p_sp)/(m_tx + m_sp), where m_x = max(p_x);
- text-only: q = p_tx.
The label ê and score c follow the existing rescaled or text-only rule.

- **Ungated (control):** use ê regardless of c.
- **Relabel (baseline):** if c < τ, set ê = neutral; otherwise use ê.
  This is the existing behaviour.
- **Abstain:** if c < τ, ignore emotion. Rank tracks by emotion-agnostic
  affinity (1/4)·Σ_e a_{i,e}. For LinUCB the four emotion indicators are set
  to zero. Otherwise behave as ungated.
- **Down-weight:** no threshold. Rank tracks by Σ_e q_e·a_{i,e}. For LinUCB
  the context is (q_1..q_4, max(q), 1), which keeps dimension 6.

In every policy the deterministic recommender selects the genre with the
highest mean ranking score over its top-10 list. LinUCB contexts are
(4 emotion indicators, c, 1) except where defined above.

## 5. MELD calibration of the asymmetric condition

Before any policy is run, choose text and speech noise SDs separately on a
dedicated calibration seed (9999, 10,000 draws, not reused elsewhere). Search
the grid 0.1 to 3.0 in steps of 0.05, so that branch top-1 accuracy matches
the MELD four-class test accuracies: text 0.685, speech 0.578. Use only
branch accuracy, never recommendation outcomes. Record the selected values in
the results file. Note: the generator's emotions are class-balanced and
MELD's are not, so only accuracy is matched, not macro-F1.

## 6. Metrics

- **Primary:** NDCG@10 (graded).
- **Secondary:** skip rate, expected completion probability of the first
  track, abstention or fallback rate, and emotion accuracy (relabel and
  ungated only).
- **Selective prediction:** for the gated policies, coverage (1 − abstention
  rate) and emotion accuracy on covered inputs. Also report a descriptive
  threshold sweep τ ∈ {0.4, 0.5, 0.6, 0.7, 0.8} for relabel and abstain
  (deterministic recommender only), plus the area under the risk–coverage
  curve of c. The sweep is not used to choose τ, which stays 0.6.

## 7. Comparisons and analysis

Within each fusion × recommender × noise × preference cell, compute per-seed
paired differences on every metric:
1. abstain − relabel
2. down-weight − relabel
3. down-weight − abstain
4. LinUCB − deterministic, for each confidence-handling policy

Report the mean, a 95% percentile bootstrap CI (10,000 resamples of the 10
seed-level differences, RNG seed 0), and the number of seeds with a positive
difference.

**Decision rule:** a policy is described as better or worse in a cell only
if its CI excludes zero. Otherwise the result is reported as "no consistent
difference". With 10 seeds and many cells these are descriptive statements
about this generator; no significance tests or multiplicity-adjusted claims
are made, and every cell is reported whether or not it favours a policy.

**Expectations stated in advance:**
- (a) abstain ≥ relabel, because relabelling recommends Neutral-matched
  music when the emotion is unknown;
- (b) down-weight ≥ relabel;
- (c) no directional expectation for abstain vs down-weight;
- (d) in the asymmetric condition, text-only may match or beat rescaled
  fusion, as on MELD.

If any expectation fails, report it as observed.

## 8. Execution and reporting

- Implement in `experiments/controlled_study_v2.py`, reusing `scenarios()`.
  Extend it only to accept separate text and speech noise SDs. Write all runs
  to `results/controlled_study_v2.json`.
- Run the script once after this file is committed. Any later code fix must
  be logged below with its date and reason, and all affected conditions
  re-run in full.
- Report every cell (main text: NDCG@10 and skip rate for the primary
  comparisons; appendix: all metrics and all cells).

## 9. What this study cannot show

Results describe one hand-specified generator with a known utility function,
equal class frequencies and a single preference vector per run. They are not
evidence about real listeners, and they do not establish a new algorithm. The
deterministic recommender uses the generator's emotion–genre structure, so it
is close to an oracle when preferences do not vary. The follow-up on
#nowplaying-RS with MELD-measured error rates is the intended test on real
listening data.

## Deviation log

### Pre-execution clarification — 14 November 2026 (Asia/Calcutta)

The original protocol and results remain unchanged. No extended-policy run or
asymmetric calibration was executed before these clarifications were added.
This file is a prospective local analysis plan, not an externally registered
protocol. No OSF/Zenodo DOI or registration date may be asserted unless an
actual registration has been completed before execution.

- The core factorial contains 960 runs. The deterministic threshold sweep
  contains 1,200 evaluations; its 240 evaluations at threshold 0.6 reuse core
  outputs. Thus there are 960 additional sweep evaluations, or 1,920 unique
  policy evaluations in total. All are included in the single planned execution.
- All affinity vectors sum to one. Therefore abstention's emotion-agnostic
  score is mathematically 0.25 for every track. Implement this constant exactly,
  avoiding floating-point tie artefacts. Within a genre, ties use ascending
  track index; deterministic genre ties use ascending genre index. Abstention
  is consequently an uninformative control, not a trained non-emotional
  recommender. LinUCB can still learn from reward while its emotion indicators
  are zero. Its trust feature remains c, as specified above.
- Down-weighting means the probability-weighted affinity in Section 4; it
  does not add a further scalar confidence multiplier. The fused label need
  not equal argmax(q), and both definitions are retained as specified.
- For calibration, generate the 10,000 true labels and separate standardized
  text/speech Gaussian arrays once using NumPy default_rng(9999). Reuse them
  for every grid point. Minimize absolute error to the stated rounded target
  accuracy separately for each branch; ties select the smaller noise SD.
  Matching accuracy does not reproduce MELD's error dependence or confusion
  matrix and does not turn the simulation into real-listener evidence.
- Bootstrap entire seed pairs, never individual interactions. Use a common
  10,000-by-10 resampling-index array from default_rng(0) for all contrasts;
  quantiles use NumPy's linear interpolation at 0.025 and 0.975. Report the
  raw positive, negative, and zero seed counts. Lower skip rate is favorable;
  higher NDCG, expected completion, and accuracy are favorable. Coverage and
  fallback rates are descriptive and have no unconditional better direction.
  Intervals are pointwise and unadjusted; exclusion of zero permits only a
  cell-specific descriptive statement, not familywise significance.
- Covered accuracy is null when coverage is zero. Contrasts involving a
  null metric receive null intervals with the valid-seed count; do not silently
  discard seeds or treat undefined accuracy as zero. Emotion accuracy of
  ungated/relabel uses their emitted labels; covered accuracy uses the
  original fused label on c >= threshold for both gated policies.
- Risk--coverage area is computed per seed on original fused-label errors:
  sort by descending c, break ties by ascending interaction index, average
  cumulative error rates at coverage k/1000 for k=1,...,1000. Lower is better.
  It is a score-ranking diagnostic shared by the corresponding gated policies.
- Execution records must include protocol and source SHA-256 hashes, selected
  calibration SDs and achieved accuracies, seeds, library versions, start/end
  times, and all run-level metrics. Any post-execution correction is a dated
  deviation and invalidates a claim of a single unchanged execution.

| Date | Change | Reason | Conditions re-run |
|---|---|---|---|
| | | | |
