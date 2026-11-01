# Pre-registered protocol v1

Committed before any `final` run, so `git log` shows the analysis was fixed before the held-out results
existed. If this file changes after a final run, that change is a deviation and must be reported as one.

## Question

Does a better open-loop trajectory score predict better closed-loop behavior on held-out scenario
families?

## Candidate pool (fixed)

- Four rules: `rule_center`, `rule_offset` (0.4 m bias, built on purpose to cost ADE), `constant_velocity`,
  `stationary`.
- Eight learned models, a full factorial: state or state+history features, clean or noisy expert logs,
  degree 1 or 2. Ridge lambda is chosen per model on dev ADE from {1e-4, 1e-2, 1}.

All candidates output the same plan, run through the same tracker, and face the same paired seeds.

## Splits

Families are disjoint across train, dev and test (`src/eval_lab/scenarios.py`). The manifests in
`experiments/scenarios_*.json` are frozen and their hashes appear in every run manifest. Test families
are read only by `--stage final`.

## Primary endpoint

Safe completion: reached the goal in the time budget with no collision and no off-road. Denominator is
all planned episodes.

## Pre-registered analyses

- **P1, rank agreement.** Spearman correlation between mean ADE@2.5s and mean safe completion across
  policies on test families, with a scenario-bootstrap 95% CI, for the 8 learned models and for all 12.
  Lower ADE means better, so agreement is a negative rho. Disagreement is reported when the learned-pool
  CI includes 0 or is positive. The CI does not cover uncertainty over which policies are in the pool.
- **P2, selection decision.** Among the 8 learned models, pick one by best dev ADE and one by best dev
  safe completion, using dev data only. On test families report the paired difference in safe completion
  (closed-selected minus ADE-selected), resampling scenarios. Reading: IMPROVED, REGRESSED or INCONCLUSIVE
  from whether the 95% CI excludes 0. If both rules pick the same model, the difference is reported as
  not applicable.
- **P3, slices.** Safe completion per test family, for each policy.

## Secondary analyses

- **S1, mechanism control.** The same policies with disturbance 0 and one seed. If a history model that
  fails under disturbance recovers without it, the failure is attributable to leaving the logged state
  distribution. If it still fails, that explanation is not supported and the report must say so.
- **S2, noise.** Variance split between scenarios and seeds, design effect, and the naive
  episode-independent CI half-width next to the scenario-clustered one.
- **S3, flag cross-check.** On MetaDrive, agreement between trace-derived outcomes and simulator flags.
  Disagreement is reported, not reconciled away.

## Failure mining (kinematic fixture)

Baseline: random selection from the test pool at a fixed budget. Intervention: selection using scenario
features plus dev failures only, with a diversity term. Test outcomes are used only to score the selection,
never to make it. Budgets 8, 16, 32 scenarios, 3 seeds, 200 draws per method and budget. Candidate
regressions and a null control are defined in `src/eval_lab/mining.py`. Metrics: regression detection rate,
false-alarm rate on the null, CI width, family coverage, duplicate fraction.

## Stopping and reporting

Null, negative and inconclusive results are reported as they are. Failed and timed-out runs stay in
denominators. No re-running a final stage to get a different number. A required rerun is a new, named run
with a stated reason.
