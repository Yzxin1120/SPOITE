# Design Changelog

## 2026-07-04 — Assistant-filled execution parameters (approved)

The user authorized the assistant to fill every open implementation parameter
with a recommended value, execute all experiments, and preserve a searchable
record of those choices. This authorization fills parameters; it does not
authorize redefining the frozen framework.

All values below have provenance `assistant-filled`:

- Synthetic DGP: `p=10`, AR(1) covariance with `rho=0.5`,
  `alpha=(1,1,-1,0.5,0,...)/||.||`,
  `m0(x)=x1+0.5*x2+x3^2`, outcome noise SD `1.0`, and no clipping of the
  true assignment propensity.
- Rare-mixture covariates: probability `0.05`, shift `4.0` along `alpha`.
- Synthetic split sizes: train `800`, validation `400`, test `800`.
- Nuisance estimation: 5-fold cross-fitting; ridge outcome regressions and
  logistic propensity regression; numerical estimated-propensity guard
  `1e-6` in the main pipeline. This guard prevents division by zero and is not
  the fixed-clipping appendix intervention.
- Feature class: standardized `[1, X, X^2, X3*X4]`; it represents the near-linear
  CATE and remains misspecified for the nonlinear CATE.
- Decision problem: batches of `m=16`, two groups formed by the within-batch
  median split of `X1`, zero lower bounds, per-group upper bound
  `ceil(0.75*B)`, costs `Uniform(0.3,1.2)`, and budget
  `floor(fraction*m+0.5)`.
- Final inference: 100 full-pipeline bootstrap replicates; paired
  nonparametric bootstrap over seeds with 10,000 resamples for 95% confidence
  intervals.
- Experiment 1 tuning grids: L2 `{1e-4,1e-3,1e-2}`, learning rate
  `{0.05,0.1,0.2}`, 60 epochs; DC boundary penalty `{0.03,0.1,0.3}` and
  active-band quantile `{0.05,0.1,0.2}`.
- Experiment 3 fragility subset: all 50 declared seeds (exceeds the minimum
  of 20).
- IHDP: retain the canonical 672/75 split; use all 100 realizations; obtain
  validation by a deterministic 20% split of the 672 training rows, stratified
  by treatment; never tune on the 75 test rows.
- ACIC: use all 4 available settings and all 5 replicates; deterministic
  stratified 60/20/20 train/validation/test split per outcome file; fit
  median/mode imputation and one-hot encoding on training rows only.
- Twins: fix the standardized assignment direction to the first principal
  component fitted on training covariates, orient it to have positive
  correlation with the first nonconstant feature, use temperatures
  `{0.5,1.0,2.0}`, 20 assignment seeds, a fixed 5,000-row subsample per seed,
  and deterministic stratified 60/20/20 splits. Median imputation and scaling
  are fitted on training rows only.
  Operational clarification: a fixed 60% X-only construction subset fits the
  PCA assignment mechanism first; after treatment is generated, a separate
  treatment-stratified 60/20/20 learner split is formed. Both index sets are
  stored in metadata. No outcome or potential outcome is used by either split
  construction.
- Data storage: preserve raw benchmark data in the legacy reference directory
  and locate it through a configurable data-root argument. No user-specific
  absolute path is embedded in the package.
- Appendix hard cells: nonlinear CATE, rare-mixture covariates,
  `gamma_e=4`, budget fractions `{0.15,0.45}`, 50 seeds, and clipping
  epsilons `{0.01,0.025,0.05}`.

Scientific consequence: these choices operationalize the already-declared
estimands and comparisons. They do not add OCDR, optimized clipping, active
sampling, or any method outside the frozen SPOITE framework.

The governing design is `../experiment_design.md`.

No changes have been approved after baseline v1.0.

Any future entry must include:

- approval date;
- user approval reference;
- proposed and approved change;
- reason;
- affected experiments and outputs.
