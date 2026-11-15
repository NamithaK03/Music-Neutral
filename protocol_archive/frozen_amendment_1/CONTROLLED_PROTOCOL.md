# Controlled synthetic follow-up

This is a new exploratory experiment, specified before its outputs were inspected.
It is not a reproduction of SONAR's original results or a preregistered study.

We compare fixed missing-auxiliary trust (0.7 times primary confidence), rescaled
primary trust, and text-only prediction. Each is crossed with deterministic
emotion matching and disjoint LinUCB. All policies see the same four genre
actions and same catalog, emotion states, branch probability vectors, latent
preferences and per-track feedback uniforms within a seed. Randomness is generated
before policy execution; no action changes the future exogenous sequence.

Fixed design: ten seeds 2026--2035, 1,000 interactions per seed, 80 tracks
(20 per genre), four equally likely emotion states. Text and speech logits are
independent Gaussian perturbations of a one-hot emotion signal of magnitude 2.
Noise SD is 0.5 or 1.5. Genre preferences are a fixed per-run normal draw with
SD 0 or 0.5. Preferences are hidden from policies. Track affinities are Dirichlet
draws with concentration 4 for their genre and 0.5 otherwise.

Synthetic completion probability is 0.1 + 0.8 sigmoid(2*(track affinity to true
emotion + latent genre preference - 0.5)). Feedback is a Bernoulli draw for the
top-ranked track using the shared potential-outcome uniforms. Reward is +1 for
completion and -1 otherwise. Skip rate is one minus completion rate. Within a
genre, tracks are ranked by affinity to the inferred emotion. A low-confidence
prediction becomes neutral (class 3), consistent with the baseline fallback.

LinUCB uses six features: one-hot inferred emotion, fused score, and bias;
identity initialization, exploration 0.5 and no exploration decay. No tuning
is conducted in this follow-up. Deterministic selection takes the genre with
the highest mean affinity score among its top-ten tracks; both policies have
all four actions available.

Report every condition's seed-level outputs and mean/sample SD for NDCG@10
(graded relevance equals completion probability), skip rate, fallback rate,
emotion accuracy, and expected completion probability of the first track.
Also report paired LinUCB-minus-deterministic deltas by seed. These quantify
effects inside this specified generator, not effects on real listeners. The
new six-dimensional architecture must not be conflated with the legacy 32-feature
experiment. Confidence calibration and the synthetic feedback model are assumptions,
not learned from human data. No superiority claim or novel algorithm is presupposed.
