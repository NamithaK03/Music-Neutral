# Interpretation revision — 14 November 2026

Written after inspecting v2 and listening-log results and receiving the user's
critique. This is a retrospective reporting correction, not Amendment 2,
preregistration, or a new experiment. Frozen protocols, experiment code and
original result JSONs remain unchanged.

1. Removed the title/abstract promise of evaluating informative abstention.
   Averaging normalized track affinities gives a constant 0.25, so the
   deterministic fallback has no informative preference signal. Preserved all
   cells but separated the 12 LinUCB abstention comparisons (one positive,
   one negative, ten intervals containing zero). LinUCB can learn genre
   preference from feedback, although its fallback within-genre ranking still
   uses fixed ties.
2. Reframed the listening-log analysis as a feasibility limit. It retains
   965/90,992 test events (1.06%), 959 happy proxies (99.38%), from 53 users.
   Its -0.0131 difference has a narrowly zero-excluding interval among 14
   unadjusted contrasts; it does not establish failure of four-emotion
   transfer. Balanced validated affect labels or a user study are needed.
3. Added exploratory ungated-minus-relabel and down-weight-minus-ungated
   comparisons from saved v2 runs. All 24 cells, both selectors and all
   metrics are reported, with the existing seed-paired bootstrap convention.
   Deterministic ungated-minus-relabel means are positive in all 12 cells,
   range 0.01321985–0.08550591; eight pointwise intervals exclude zero.
   Down-weight-minus-ungated means range -0.00039827–0.01017292.
   Mean improvements are not claimed to hold in every seed.
4. The connection to MELD's 0.511 text-only versus 0.181 capped-trust
   fusion-plus-relabel macro-F1 is qualitative: fusion and scaling also
   change, so that benchmark difference cannot isolate substitution.

Reporting implementation: `revision_framing.py`, invoked by
`build_revision_report.py`. Experimental policies were not executed again.
The original complete supplement remains unchanged; new exploratory tables
are included in the main manuscript and separate Markdown/JSON report.
