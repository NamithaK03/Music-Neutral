# Listening-log transfer analysis plan

Date: 14 November 2026. Written after inspecting the released schema, training
hashtag frequencies and split dates, before computing recommendation outcomes.
This is a local prospective plan, not an external preregistration. It is a
separate observational transfer analysis, not part of Amendment 1's 960 runs.

Use Zenodo 3248543 `nowplaying_rs_train_test.zip`, publisher Context split
(training January--September 2014; test October--December 2014). Verify MD5
32d00b55dbeb2b617fd041c8156d3e86. Deduplicate user/track/timestamp events;
aggregate hashtag labels and exclude conflicting labels. Candidate catalog is
all training tracks, lexicographically ordered. Exclude test targets absent
from this catalog and report exclusions. Allow repeat listening. Evaluate all
eligible test events, with equal weight per user after event averaging.

No listening-event speech, verified four-class listener mood, genre, skips,
or exposure propensities are provided. Use an explicit weak hashtag proxy:
happy={happy,happiness,joy,joyful,excited,enjoy,enjoying,love,loved,lovely};
sad={sad,sadness,unhappy,depressed,crying,heartbroken,lonely};
angry={angry,anger,mad,furious,rage};
neutral={neutral,calm,relaxed,relaxing,chill,chilling}.
Lowercase exact-token matching after stripping a leading #. Calm/chill are
low-arousal proxies, not validated Neutral labels. Other hashtags are missing,
never automatically Neutral. Label assignment uses no track acoustic features.
This mapping is fixed now, not optimized on retrieval outcomes.

Fit training-only track affinity a_ie=(count_ie+1)/(sum_e count_ie+4), using
unambiguous proxy events. Base recommendation probability is
b_ui=0.5*user_track_count/user_event_count + 0.5*global_track_count/global_events.
This explicit non-emotional recommender replaces the synthetic genre selector:
the release has no genre action set, and offline logs cannot train/evaluate the
same online LinUCB counterfactually. Report this architecture adaptation.

For seeds 2026--2035, draw one MELD official-test donor with matching mapped
true emotion per event; reuse its paired text/speech probability vectors across
all policies. This preserves empirical class-conditional errors, confidence,
and branch dependence, unlike independently flipping labels at an aggregate
error rate. Report measured donor-pool and realized branch accuracy/macro-F1;
aggregate error rates may differ because the hashtag-class mixture differs.
MELD errors are injected, not measured on these listeners.

Use text-only and rescaled fusion exactly as Amendment 1. Threshold=0.6.
Ungated ranks b_ui*4*a_i,label. Relabel substitutes Neutral below threshold.
Abstain ranks b_ui below threshold, otherwise ungated. Down-weight ranks
b_ui*4*sum_e q_e*a_ie, with no threshold. Ties use ascending track ID. Also
report the same base recommender without emotion for every event.

Primary: single-held-out-positive NDCG@10 over the full training catalog.
Secondary: Recall@10 and reciprocal rank@10; fallback/abstention/coverage.
Unobserved tracks are unjudged, not proven dislikes. No skip/completion or
causal satisfaction claims. Report event and user counts and all 80 seeded
conditions (2 fusions x 4 policies x 10 seeds), plus the base control.

Contrasts: abstain-relabel, downweight-relabel, downweight-abstain, each policy
minus the base control, in each fusion. Average each user's metric across their
events and then across the ten corruption seeds; paired percentile bootstrap
users, 10,000 resamples, RNG 0, linear 2.5/97.5 percentiles. Report seed-level
contrasts as robustness diagnostics; seeds are not independent listeners.
Intervals are descriptive, pointwise and unadjusted. Retain every condition.
Do not tune thresholds, lexical mapping or model weights after results.

Archive this plan and source hash before one execution. Log any correction.
