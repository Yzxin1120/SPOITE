# SPOITE Experiment

This is the clean, question-driven implementation of the SPOITE empirical
study.

The governing specification is
[`../experiment_design.md`](../experiment_design.md). That document is frozen:
material changes require prior user approval and a changelog entry.

## Reference directories

- `../dccpo/` - Claude's original implementation; read-only reference.
- `../experiment/` - previous section-driven experiments; read-only reference.
- `../SmartPredictThenOptimize-master/` - original SPO reference code.

Code must be re-audited before reuse. Nothing is imported from the old
experiment by absolute path.

## Project status

The shared causal/decision pipeline, synthetic generator, benchmark adapters,
exact allocation/DFI machinery, and Experiment 1--4/appendix runners are
implemented. Smoke outputs are non-citable. Only artifacts under
`outputs/final/` produced from the frozen Git commit and final configuration
are citable.

Final (citable) so far: Experiment 1 (synthetic Q1–Q3), Experiment 2 (geometry),
Experiment 3 (Q4 factorial). **Pending:** Experiment 4 (semi-synthetic) has only
smoke outputs, not final; the clipping appendix has not been run.

## Findings & status

These are honest current findings from the frozen runs, not a claim that the
experiments confirm every hypothesis. They should be read as a work-in-progress
empirical report.

- **Q1 — supported.** CPO+ attains lower decision regret than MSE-DR and MAE-DR
  in both the near-linear and nonlinear regimes (paired, CI excludes zero; see
  `outputs/final/experiment_1_paired_contrasts.csv`).
- **Q3 — supported.** Lower PEHE frequently fails to imply lower regret. The
  paired PEHE/regret ranking-reversal rate is substantial (up to ~0.38; see
  `outputs/final/experiment_1_ranking_reversals.csv`).
- **Theorem 5.3 (Experiment 2) — supported.** Two covariance orientations with
  identical determinant and trace produce sharply different boundary-crossing
  frequency and DFI (`outputs/final/experiment_2_geometry.csv`).
- **Q2 — not supported as stated.** DC-CPO(band) does *not* reduce regret
  relative to CPO+; it is significantly worse on regret in the easier regimes.
  Its one consistent effect is compressing the *tail* (standard deviation) of
  DFI, i.e. worst-case decision fragility, not the median.
- **Q4 — not supported; trend is reversed.** DC-CPO's regret gain over CPO+ does
  *not* grow under weak overlap / long tails / tight budgets. Aggregated over
  `experiment_3`, the per-seed regret win-rate for DC-CPO is ~0.53 at strong
  overlap (`gamma_e=0.5`) and falls to ~0.34 at the near-positivity-violation
  regime (`gamma_e=4`) — the opposite of the predicted direction.

Known caveats affecting the DC-CPO comparison (documented, not yet addressed):

1. The tuned `dc_boundary_penalty_grid` excludes `0`, so DC-CPO cannot fall back
   to CPO+ and is structurally allowed to be slightly worse; adding `0` would let
   validation selection recover CPO+.
2. Hyperparameters are selected on the (noisy) validation pseudo-regret, which is
   least reliable exactly under weak overlap.
3. The implemented boundary surrogate (Eq. 18) penalizes the decision-projected
   distance to the raw pseudo-outcome, so under weak overlap it can pull the
   predictor toward high-variance IPW noise. This is a property of the proposed
   penalty, not an implementation error, and is a gap between the theory
   (Thm. 7.6 targets `d'Σd`) and the finite-sample surrogate.

None of the above changes the correctness of the CPO+ learner or the exact
allocation/DFI oracle, which are unit-tested.

## Layout

```text
configs/          frozen experiment configurations
src/spoite/       shared data, causal, optimization, method, and evaluation code
experiments/      question-driven experiment entry points
tests/            unit and integration tests
outputs/          smoke/provisional/final artifacts
```

## Reproduction

The approved assistant-filled values are in `configs/final.json`; their
rationale and provenance are in `DESIGN_CHANGELOG.md`.

```bash
python -m pytest -q
PYTHONPATH=src python experiments/run_experiment1.py
PYTHONPATH=src python experiments/run_experiment2.py
PYTHONPATH=src python experiments/run_experiment3.py
PYTHONPATH=src python experiments/run_experiment4.py \
  --data-root ../experiment/8.3/semi/data
# Run only after Experiments 1--4 are frozen:
PYTHONPATH=src python experiments/run_appendix_clipping.py
```
