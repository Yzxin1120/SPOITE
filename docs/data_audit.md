# Stage 1 Data Audit

Date: 2026-07-04

Status: reference data inspected; adapters not yet approved for scientific
runs.

## Sources inspected

- `../../dccpo/src/dccpo/dgp.py`
- `../../experiment/8.3/semi/loaders/`
- `../../experiment/8.3/semi/data/`

The old implementations were read as evidence. They were not copied into the
new package.

## Synthetic DGP

The old DGP implements the SPOITE near-linear and nonlinear CATE formulas,
correlated Gaussian covariates, logistic treatment assignment, and Gaussian
outcome noise.

It must not be reused unchanged:

- it clips the true assignment propensity to `[0.01, 0.99]`;
- that clipping changes the overlap factor and conflicts with treating
  `gamma_e = 4` as a near-positivity-violation stress test;
- it generates intervention cost inside the data generator, while the new
  design separates causal data from the downstream operational problem;
- it has no rare-mixture covariate mode required by Experiment 3.

Required approval before adapter implementation:

- whether true propensities remain unclipped in the DGP;
- the primary DGP dimension, covariance, baseline-outcome function, and noise;
- rare-mixture probability and shift.

## IHDP

Files:

- train: 672 observations, 25 covariates, 100 realizations;
- test: 75 observations, 25 covariates, 100 realizations.

Available fields include observed treatment, factual outcome, `mu0`, and
`mu1`. Therefore `tau_true = mu1 - mu0` is available for evaluation. True
propensity is not supplied.

Observed treatment rate in realization 0:

- train: approximately 0.183;
- test: approximately 0.213.

The existing loader is structurally usable but does not yet satisfy the new
contract or provenance requirements.

Required approval:

- whether the canonical 672/75 split is retained;
- whether all 100 realizations are final evaluation units;
- how validation is obtained without leaking the 75-observation test split.

## ACIC 2016

Available data:

- 4,802 rows of real covariates;
- 58 raw columns, including 3 categorical columns;
- 20 generated outcome files;
- 4 selected DGP settings crossed with 5 replicates.

Every generated outcome file supplies `z`, `y`, `y0`, `y1`, `mu0`, `mu1`, and
`e`. All inspected numeric fields are finite and row-aligned with the
covariates.

Example overlap diagnostics:

- full/high setting 04 replicate 01: `e` approximately
  `[0.1125, 0.9786]`;
- one-term/high setting 01 replicate 01: `e` approximately
  `[3.0e-10, 0.6679]`.

The one-term regime is a genuine near-overlap-violation stress setting and
must be labeled accordingly.

Required approval:

- which settings and replicates enter primary versus sensitivity results;
- the train/validation/test split rule;
- the categorical encoding and fitted-preprocessing protocol.

## Twins

Available data:

- 71,345 twin pairs;
- 54 raw covariate columns;
- complete `mort_0` and `mort_1` outcome columns;
- 262,836 missing covariate cells.

The old loader must not be reused unchanged:

- treatment and propensity are simulated at load time;
- the propensity direction is a newly sampled random vector;
- the same mutable RNG is also used for assignment and subsampling;
- paired method comparisons can silently receive different constructed data if
  the loader is called repeatedly;
- raw covariate missingness requires fitted preprocessing.

Required approval:

- the treatment-assignment score and whether it is fixed across all runs;
- the assignment-temperature grid;
- the subsampling and split protocol;
- the train-fitted missing-value preprocessing rule.

## Data-storage decision

The 43 MB of reference data currently remains under
`../../experiment/8.3/semi/data/`.

Before adapters are finalized, approval is required for one of:

1. copy immutable raw data into `SPOITE_experiment/data/raw/`;
2. keep one external data root configured by path/environment variable;
3. store only acquisition scripts and require local download.

The new code must not contain user-specific absolute paths.
