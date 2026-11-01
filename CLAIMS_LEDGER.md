# Claims ledger

Status words: **tested** by automated checks, **measured** in a retained run, **exploratory** when added after seeing
results, **not established** when the evidence does not support a claim. Numbers come from `reports/pilot.md` and
`reports/study.md`. M2 is the later and stricter test, so where the two differ M2 decides.

## Supported

| Claim | Status | Evidence | Limit |
|---|---|---|---|
| Metrics match hand calculations and refuse bad data | tested | `reports/analytic.md`, `tests/test_metrics.py` | Analytic cases only |
| Selecting by closed-loop results beat selecting by clean-log ADE on held-out families | measured, pre-registered, twice | P2: +0.465, +0.889. M2 C1: +0.459, +0.156 | Toy policies. Resampling families, the M2 MetaDrive CI reaches 0 |
| Clean-log ADE ranked learned models backwards on the pilot families | measured, pre-registered | P1: +0.667, +0.939 | **Not general.** M2: +0.143 fixture, -0.708 MetaDrive, which is the right direction |
| Disturbed-log ADE ranked learned models the right way | measured, pre-registered in M2 | C3: -0.333 fixture, -0.708 MetaDrive; pilot S4 agrees | Eight policies, one log per scenario |
| Dev verdicts on regressions do not fully transfer | measured | 6 of 8 dev-validated held; 3 of 7 dev-silent failed, including `reg_blind_on_curve` at -0.221 | Hand-built regressions on the toy fixture |
| Mined packs beat random packs at the pre-registered test | measured, pre-registered, fragile | M1 at budget 16: +0.141 [+0.005, +0.376]; +0.026 without one regression | Six correlated regressions, toy fixture. Random won on `reg_blind_on_curve` |
| Learned models from these training roads fail on junction maps | measured | Best 0.156 and zero in 6 of 8 families; `rule_center` 1.000 [0.962, 1.000] | One simulator. Depends on the trace off-road rule: 0.205 with simulator flags |
| In the all-candidate pool, the fixture agreed and MetaDrive did not | measured | P1: -0.444, +0.396 | The pool contains extreme baselines |
| Disturbed-log ADE still picks worse than closed-loop results | **mixed** | Pilot S4: +0.192, +0.875. M2 C2: +0.041 inconclusive on the fixture, +0.149 on MetaDrive | Did not replicate on the fixture. MetaDrive CI reaches 0 by family |
| Episode-level intervals are overconfident here | measured | S2: design effect mostly about 3 to 4 | Fixture |
| Reports regenerate byte for byte from retained files | tested | `tests/test_report.py`, `cmp` run | Requires the same run files |
| Tests catch deliberately broken safety rules | tested | `scripts/mutation_check.py`, 45 of 45 after one test was added | Those 45 mutants |
| Real MetaDrive simulator runs were executed | measured | `runs/metadrive-final-v2/` | Headless, one lane, light traffic |

## Not established

| Claim | Why not |
|---|---|
| The copycat effect caused the history-model failures | S5 shows clean history models fail with no outside disturbance (0.000 to 0.017 on the fixture, 0.000 on MetaDrive), but those families are unseen, so unfamiliar roads could explain it too. State models also drift 1 m within seconds |
| Failure mining beats random selection in general | Most of the M1 gap is one regression, five of six share the lead-vehicle mechanism, and mined packs covered fewer families and raised more false flags |
| MetaDrive results generalize to new road types | Eight families, with learned success in only two. Four are near baseline overlap, four are above it, and some single maps reach 0.74 |
| ADE is useless | S4 shows its value depends on the logs it is scored on |
| Any peak-jerk number reflects comfort | Controller steps dominate the finite differences |

## Must never be claimed

- Any statement about real vehicles, fleet safety, or simulation-to-real validity.
- That `rule_center` has a zero failure rate. It has a bound, from 96 M2 scenarios on MetaDrive, and it crashed on some fixture cut-ins.
- That `metadrive-final` is a held-out result. It is superseded because of map-seed leakage.
- That this was independently replicated. One review of code was done by another agent, and the results come from one machine.

## Authorship

Built with Claude Code as an AI pair. Shubhankar Tripathy directed the work and owns the decisions.
