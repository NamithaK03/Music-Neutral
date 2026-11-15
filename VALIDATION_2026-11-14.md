# Validation — 14 November 2026

- Two targeted extension tests passed before the study execution: original
  random-stream preservation and hand-constructed policy/undefined-metric
  behavior. The first unittest invocation used an unavailable package import
  path; discovery resolved that invocation issue without changing source.
- One extended controlled-study execution completed without a policy-code fix
  or rerun: 960 core runs and 960 additional threshold evaluations.
- One separately planned listening-log execution completed: 80 seeded cells.
- `verify_revision_results.py` passed: complete factorial cells, metric bounds,
  all 160 overlapping original relabel runs numerically equal to 1e-12,
  frozen source/protocol hashes, archive-before-execution chronology, and all
  listening summary metrics reconstructed from saved per-user arrays.
- Main manuscript compiled twice after edits: 20 pages. Supplement compiled:
  54 pages. Final LaTeX log has no overfull boxes or undefined references.
- The asymmetric-results table, all-condition figure and listening-results
  pages were rendered and visually checked for readable layout and clipping.
- Original protocol and original experiment source are byte-identical to
  their frozen copies. Original experiment result JSON was not rewritten.

No experimental rerun was performed to improve an outcome. Reporting scripts
were rerun to correct manuscript includes and formatting; they only read
recorded results. No external registration or journal submission took place.
