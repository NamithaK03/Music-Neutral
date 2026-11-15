# Database search status — 14 November 2026

**Incomplete; not a systematic-review completion claim.** The frozen search
plan is `../LITERATURE_SEARCH_PROTOCOL.md`. Exact requests, timestamps, raw
responses and failures are retained in this folder.

- OpenAlex: three queries, first 200 records each; 600 retrieved, 595 unique.
  Subsequent cursor pages returned HTTP 401 (authentication required).
- Semantic Scholar: all three bulk-search requests returned HTTP 429.
- Initial automated title/abstract screening flagged 71 records for full-text
  assessment. These are candidates, not 71 confirmed included studies. The
  lexical screen is deliberately broad and includes false positives.
- Full-text screening and author verification are not complete. Access to
  authenticated database search or institutional exports is needed to finish
  retrieval. No accounts were created or emails sent.

Primary-source checks informing the manuscript:

| Work | Verified scope | Implication |
|---|---|---|
| [Jing et al.](https://arxiv.org/abs/2406.14090) | Emotion-distribution and user-preference heterogeneity in Bayesian music recommendation; author abstract checked | Probability/distribution-aware personalization is prior work |
| [Watcharasupat et al.](https://arxiv.org/abs/2501.11570) | Probabilistic losses and inference sampling for uncertainty in perceived music emotion; author abstract checked | Uncertainty estimation in music affect is prior work |
| [Faseela and Supriya](https://doi.org/10.1016/j.array.2026.101158) | Publisher search record describes uncertainty-aware joint emotion/genre recommendation; full text inaccessible | Must examine full text before claiming a distinct uncertainty-aware contribution |
| [Glasserman and Yao](https://doi.org/10.1287/mnsc.38.6.884) | Common random numbers in comparative simulation | Pairing is established methodology |
| [Poddar et al. dataset](https://zenodo.org/records/3248543) | Listening events, content features, hashtag sentiment and standard temporal splits | Supports offline target retrieval, not observed speech or counterfactual skips |

The contribution is an empirical comparison of explicit confidence policies
and their transfer behavior. Do not claim a first uncertainty-aware music
recommender, novel probability weighting, or a novel paired design. The
negative transfer result is part of the contribution, not evidence to omit.
