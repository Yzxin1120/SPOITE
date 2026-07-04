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
