# Experiment Entry Points

Implemented, question-driven entry points:

- `run_experiment1.py` — core synthetic study for Q1–Q3.
- `run_experiment2.py` — Theorem 5.3 uncertainty-geometry (Appendix A.4).
- `run_experiment3.py` — Q4 hard-regime factorial study.
- `run_experiment4.py` — semi-synthetic benchmark validation (IHDP, ACIC, Twins).
- `run_appendix_clipping.py` — fixed-propensity-clipping ablation (Section 10).
- `aggregate_results.py` — summaries, paired contrasts, and figures.

All open design parameters were approved and finalized on 2026-07-04
and are recorded in `../configs/final.json` and `../DESIGN_CHANGELOG.md`.

Only artifacts under `../outputs/final/` produced from the frozen Git commit and
final configuration are citable. See the "Findings & status" section of the
top-level `../README.md` for what is currently final versus pending.
