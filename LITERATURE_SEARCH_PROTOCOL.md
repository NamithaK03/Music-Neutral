# Database search plan

14 November 2026. Question: prior evaluations of uncertainty, abstention,
probability-weighted emotion and music recommendation; distinguish classifier
uncertainty from downstream ranking outcomes. Secondary methodological search:
selective prediction and common random numbers.

Search OpenAlex and Semantic Scholar bibliographic APIs, all years up to
2026-11-14, English search strings, no venue restriction. Save exact requests,
raw responses, dates, totals, errors and pagination. Queries:

1. music AND (recommendation OR recommender) AND (emotion OR mood) AND (uncertainty OR confidence OR abstention OR probabilistic)
2. music AND (recommendation OR recommender) AND (abstention OR "selective prediction")
3. music AND emotion AND ("probability distribution" OR "uncertainty estimation")

Use database-supported Boolean syntax; document translations. Retrieve all
pages up to 2,000 records per query; declare truncation if reached. Deduplicate
by DOI then normalized title. Screen title/abstract for music affect or music
recommendation with uncertainty/probabilistic modelling, retaining reviews and
empirical studies; exclude unrelated music generation, non-music applications
and records without adequate metadata (mark uncertain rather than silently
excluding). One AI-assisted screener; author verification required. Preserve
screening decisions and identify full-text checks separately. Use publisher or
author-hosted primary sources for novelty comparisons. Web discovery is
supplementary and is not a substitute for database retrieval. Do not claim a
complete systematic review if APIs fail or full-text screening is incomplete.
