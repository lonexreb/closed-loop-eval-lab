# Pilot report: closed-loop-eval-lab

Generated from retained run files by `src/eval_lab/report.py` (report version 1, metric version 1). Every number below comes from those files. P1 to P3, S1 and S2 were fixed in `experiments/protocol.md` before the first final runs. `experiments/amendment_1.md` lists every later deviation.

**Boundaries.** These are toy linear and quadratic policies on a kinematic fixture and on one open simulator, using CPU only. Nothing here measures a real vehicle or a production planner. A simulation pass is not a safety guarantee. Zero observed events gives an upper bound, not a zero rate.

## Provenance

| Run | Backend | Status | Exit | Code | Split | Scenario hash | Config hash |
|---|---|---|---|---|---|---|---|
| kinematic-dev | kinematic_fixture | SUCCEEDED | 0 | `9c80a7cb823e` | `split-009146a0aef3` | `990aa724c881` | `472e3dc54737` |
| kinematic-final | kinematic_fixture | SUCCEEDED | 0 | `354332d66928` | `split-009146a0aef3` | `990aa724c881` | `472e3dc54737` |
| metadrive-dev-v2 | metadrive-0.4.3 | SUCCEEDED | 0 | `91ca2329a67a` | `split-6a86075bc999` | `a237c55b08f2` | `472e3dc54737` |
| metadrive-final-v2 | metadrive-0.4.3 | SUCCEEDED | 0 | `91ca2329a67a` | `split-6a86075bc999` | `a237c55b08f2` | `472e3dc54737` |

## Result A: kinematic fixture (toy)

**Boundary:** `kinematic_fixture`, a hand-written bicycle model with scripted actors. It is a controlled fixture and not a simulator.

12 policies, 120 held-out scenarios, seeds [0, 1, 2, 3, 4]: 7200 planned episodes.

### Outcomes on held-out families

| Policy (by ADE) | ADE@2.5s m | Safe completion [95% CI] | Coll/ep | Coll/km | Off-road | Timeouts | Failed+unassessable | Planned | Median peak jerk m/s^3 |
|---|---|---|---|---|---|---|---|---|---|
| `rule_center` | 0.57 | 0.937 [0.895, 0.972] | 0.063 | 0.31 | 0.000 | 0 | 0 | 600 | 20.9 |
| `rule_offset` | 0.78 | 0.952 [0.918, 0.980] | 0.048 | 0.23 | 0.000 | 0 | 0 | 600 | 22.5 |
| `ridge_hist_clean_d1` | 0.80 | 0.000 [0.000, 0.030] | 0.002 | 0.05 | 0.998 | 0 | 0 | 600 | 5.2 |
| `ridge_hist_clean_d2` | 1.10 | 0.002 [0.000, 0.005] | 0.032 | 0.39 | 0.685 | 178 | 0 | 600 | 54.9 |
| `ridge_state_noisy_d2` | 1.14 | 0.275 [0.215, 0.337] | 0.260 | 1.75 | 0.295 | 103 | 0 | 600 | 15.3 |
| `ridge_hist_noisy_d2` | 1.19 | 0.350 [0.285, 0.413] | 0.275 | 2.03 | 0.275 | 60 | 0 | 600 | 13.4 |
| `ridge_state_clean_d2` | 1.20 | 0.240 [0.185, 0.298] | 0.243 | 1.84 | 0.467 | 31 | 0 | 600 | 45.8 |
| `ridge_state_clean_d1` | 1.54 | 0.467 [0.393, 0.547] | 0.400 | 2.64 | 0.133 | 0 | 0 | 600 | 5.6 |
| `ridge_state_noisy_d1` | 1.56 | 0.293 [0.225, 0.365] | 0.412 | 2.67 | 0.300 | 0 | 0 | 600 | 7.9 |
| `ridge_hist_noisy_d1` | 1.82 | 0.285 [0.222, 0.350] | 0.403 | 2.60 | 0.315 | 0 | 0 | 600 | 6.4 |
| `constant_velocity` | 1.89 | 0.000 [0.000, 0.030] | 0.045 | 0.69 | 0.963 | 0 | 0 | 600 | 5.3 |
| `stationary` | 16.55 | 0.000 [0.000, 0.030] | 0.000 | 0.00 | 0.002 | 599 | 0 | 600 | 60.8 |

Safe completion CIs resample scenarios. Timeouts, failures and unassessable runs are in every denominator.

### P1: does ADE rank predict closed-loop rank?

| Pool | Policies | Spearman rho [95% CI] | Reading |
|---|---|---|---|
| Learned models | 8 | 0.667 [0.310, 0.874] | inverted: lower ADE went with worse driving |
| All candidates | 12 | -0.444 [-0.549, -0.271] | ADE order agrees with closed-loop order |

### P2: selecting a model by ADE vs by closed-loop results

- Selected by best dev ADE: `ridge_hist_clean_d2` (dev ADE 0.625 m, dev safe completion 0.010).
- Selected by best dev safe completion: `ridge_state_clean_d1` (dev ADE 0.973 m, dev safe completion 0.487).
- On held-out families: safe completion 0.467 (closed-selected) vs 0.002 (ADE-selected).
- Paired difference +0.465, 95% CI [+0.392, +0.545], 120 scenarios. **IMPROVED**.

### Concrete discordant example

First case in scenario order: `curve_lead_brake-000` seed 1. `ridge_hist_clean_d2` has logged-future ADE 0.83 m here and ends OFFROAD. `ridge_state_clean_d1` has ADE 1.35 m and ends COMPLETED. Both faced the same disturbance sequence. Its trace is in `traces.npz` under key `ridge_hist_clean_d2|curve_lead_brake-000|1`.

### P3: slices by held-out family (safe completion)

| Policy | curve_lead_brake | cut_in | narrow_s_curve | tight_curve |
|---|---|---|---|---|
| `rule_center` | 1.000 | 0.747 | 1.000 | 1.000 |
| `rule_offset` | 1.000 | 0.807 | 1.000 | 1.000 |
| `ridge_hist_clean_d1` | 0.000 | 0.000 | 0.000 | 0.000 |
| `ridge_hist_clean_d2` | 0.000 | 0.000 | 0.000 | 0.007 |
| `ridge_state_noisy_d2` | 0.140 | 0.487 | 0.187 | 0.287 |
| `ridge_hist_noisy_d2` | 0.153 | 0.487 | 0.280 | 0.480 |
| `ridge_state_clean_d2` | 0.187 | 0.433 | 0.193 | 0.147 |
| `ridge_state_clean_d1` | 0.087 | 0.307 | 0.680 | 0.793 |
| `ridge_state_noisy_d1` | 0.040 | 0.300 | 0.213 | 0.620 |
| `ridge_hist_noisy_d1` | 0.067 | 0.313 | 0.273 | 0.487 |
| `constant_velocity` | 0.000 | 0.000 | 0.000 | 0.000 |
| `stationary` | 0.000 | 0.000 | 0.000 | 0.000 |

### S1: mechanism control

| Policy | Safe completion, disturbance on (seed 0) | Disturbance off |
|---|---|---|
| `ridge_hist_clean_d1` | 0.000 | 0.000 |
| `ridge_hist_clean_d2` | 0.000 | 0.000 |
| `ridge_hist_noisy_d1` | 0.308 | 0.333 |
| `ridge_hist_noisy_d2` | 0.333 | 0.242 |
| `ridge_state_clean_d1` | 0.467 | 0.550 |
| `ridge_state_clean_d2` | 0.267 | 0.225 |
| `ridge_state_noisy_d1` | 0.317 | 0.400 |
| `ridge_state_noisy_d2` | 0.292 | 0.200 |

**Reading: inconclusive.** History models that stay near zero with the disturbance off: `ridge_hist_clean_d1`, `ridge_hist_clean_d2`. Removing the disturbance did not rescue them, so these runs do not support the idea that external disturbance alone pushes them off their training distribution. This control is also weak: the tracker is not the expert's controller, so even with disturbance off a policy does not reproduce logged states. The cause of these failures is not established here.

### S2: evaluation noise

| Policy | Between-scenario var | Within-scenario (seed) var | ICC | Design effect | Naive CI half-width | Clustered CI half-width |
|---|---|---|---|---|---|---|
| `rule_center` | 0.0430 | 0.0167 | 0.721 | 3.88 | 0.020 | 0.038 |
| `rule_offset` | 0.0271 | 0.0192 | 0.586 | 3.34 | 0.017 | 0.031 |
| `ridge_hist_clean_d2` | 0.0000 | 0.0017 | 0.000 | 1.00 | 0.003 | 0.003 |
| `ridge_state_noisy_d2` | 0.1054 | 0.0950 | 0.526 | 3.10 | 0.036 | 0.061 |
| `ridge_hist_noisy_d2` | 0.1128 | 0.1158 | 0.493 | 2.97 | 0.038 | 0.064 |
| `ridge_state_clean_d2` | 0.0841 | 0.0992 | 0.459 | 2.84 | 0.034 | 0.057 |
| `ridge_state_clean_d1` | 0.1612 | 0.0892 | 0.644 | 3.58 | 0.040 | 0.077 |
| `ridge_state_noisy_d1` | 0.1302 | 0.0783 | 0.624 | 3.50 | 0.036 | 0.070 |
| `ridge_hist_noisy_d1` | 0.1048 | 0.1000 | 0.512 | 3.05 | 0.036 | 0.064 |

3 policies with no variance at all are omitted.

### S4: ADE scored on disturbed logs

Post-hoc (amendment 1). Learned-pool Spearman with disturbed-log ADE: -0.286 [-0.503, -0.095].

Best dev ADE on disturbed logs selects `ridge_state_noisy_d2`. Closed-selected minus this pick on held-out families: +0.192, 95% CI [+0.108, +0.280], **IMPROVED**.

## Result B: MetaDrive (real open simulator)

**Boundary:** MetaDrive 0.4.3, headless, on procedural maps. Models are trained on logs from a rule driver inside MetaDrive. This is simulator evidence only.

12 policies, 24 held-out scenarios, seeds [0, 1, 2]: 864 planned episodes.

### Outcomes on held-out families

| Policy (by ADE) | ADE@2.5s m | Safe completion [95% CI] | Coll/ep | Coll/km | Off-road | Timeouts | Failed+unassessable | Planned | Median peak jerk m/s^3 |
|---|---|---|---|---|---|---|---|---|---|
| `ridge_hist_clean_d2` | 0.20 | 0.000 [0.000, 0.142] | 0.000 | 0.00 | 0.139 | 62 | 0 | 72 | 17.7 |
| `ridge_state_clean_d2` | 0.24 | 0.000 [0.000, 0.142] | 0.000 | 0.00 | 1.000 | 0 | 0 | 72 | 39.2 |
| `ridge_hist_clean_d1` | 0.38 | 0.000 [0.000, 0.142] | 0.000 | 0.00 | 1.000 | 0 | 0 | 72 | 12.0 |
| `ridge_state_clean_d1` | 0.47 | 0.000 [0.000, 0.142] | 0.000 | 0.00 | 0.944 | 4 | 0 | 72 | 10.8 |
| `ridge_hist_noisy_d2` | 0.50 | 0.014 [0.000, 0.042] | 0.000 | 0.00 | 0.486 | 36 | 0 | 72 | 20.3 |
| `ridge_state_noisy_d2` | 0.53 | 0.111 [0.014, 0.236] | 0.014 | 0.04 | 0.597 | 20 | 0 | 72 | 64.2 |
| `ridge_hist_noisy_d1` | 0.64 | 0.778 [0.653, 0.889] | 0.056 | 0.14 | 0.181 | 1 | 0 | 72 | 11.7 |
| `ridge_state_noisy_d1` | 0.68 | 0.889 [0.778, 0.972] | 0.042 | 0.10 | 0.042 | 2 | 0 | 72 | 12.5 |
| `rule_center` | 0.96 | 1.000 [0.858, 1.000] | 0.000 | 0.00 | 0.000 | 0 | 0 | 72 | 35.5 |
| `rule_offset` | 1.16 | 1.000 [0.858, 1.000] | 0.000 | 0.00 | 0.000 | 0 | 0 | 72 | 62.6 |
| `constant_velocity` | 1.96 | 0.000 [0.000, 0.142] | 0.000 | 0.00 | 0.000 | 72 | 0 | 72 | 22.4 |
| `stationary` | 15.10 | 0.000 [0.000, 0.142] | 0.000 | 0.00 | 0.000 | 72 | 0 | 72 | 106.1 |

Safe completion CIs resample scenarios. Timeouts, failures and unassessable runs are in every denominator.

### P1: does ADE rank predict closed-loop rank?

| Pool | Policies | Spearman rho [95% CI] | Reading |
|---|---|---|---|
| Learned models | 8 | 0.939 [0.856, 0.939] | inverted: lower ADE went with worse driving |
| All candidates | 12 | 0.396 [0.370, 0.426] | inverted: lower ADE went with worse driving |

### P2: selecting a model by ADE vs by closed-loop results

- Selected by best dev ADE: `ridge_hist_clean_d2` (dev ADE 0.224 m, dev safe completion 0.000).
- Selected by best dev safe completion: `ridge_state_noisy_d1` (dev ADE 0.622 m, dev safe completion 0.972).
- On held-out families: safe completion 0.889 (closed-selected) vs 0.000 (ADE-selected).
- Paired difference +0.889, 95% CI [+0.778, +0.972], 24 scenarios. **IMPROVED**.

### Concrete discordant example

First case in scenario order: `md_CSCC-400` seed 0. `ridge_hist_clean_d2` has logged-future ADE 0.14 m here and ends TIMEOUT. `ridge_state_noisy_d1` has ADE 0.57 m and ends COMPLETED. Both faced the same disturbance sequence. Its trace is in `traces.npz` under key `ridge_hist_clean_d2|md_CSCC-400|0`.

### P3: slices by held-out family (safe completion)

| Policy | md_CSCC | md_SCCS |
|---|---|---|
| `ridge_hist_clean_d2` | 0.000 | 0.000 |
| `ridge_state_clean_d2` | 0.000 | 0.000 |
| `ridge_hist_clean_d1` | 0.000 | 0.000 |
| `ridge_state_clean_d1` | 0.000 | 0.000 |
| `ridge_hist_noisy_d2` | 0.000 | 0.028 |
| `ridge_state_noisy_d2` | 0.111 | 0.111 |
| `ridge_hist_noisy_d1` | 0.639 | 0.917 |
| `ridge_state_noisy_d1` | 0.833 | 0.944 |
| `rule_center` | 1.000 | 1.000 |
| `rule_offset` | 1.000 | 1.000 |
| `constant_velocity` | 0.000 | 0.000 |
| `stationary` | 0.000 | 0.000 |

### S1: mechanism control

| Policy | Safe completion, disturbance on (seed 0) | Disturbance off |
|---|---|---|
| `ridge_hist_clean_d1` | 0.000 | 0.000 |
| `ridge_hist_clean_d2` | 0.000 | 0.000 |
| `ridge_hist_noisy_d1` | 0.833 | 0.917 |
| `ridge_hist_noisy_d2` | 0.042 | 0.000 |
| `ridge_state_clean_d1` | 0.000 | 0.000 |
| `ridge_state_clean_d2` | 0.000 | 0.583 |
| `ridge_state_noisy_d1` | 0.917 | 0.917 |
| `ridge_state_noisy_d2` | 0.125 | 0.125 |

**Reading: inconclusive.** History models that stay near zero with the disturbance off: `ridge_hist_clean_d1`, `ridge_hist_clean_d2`. Removing the disturbance did not rescue them, so these runs do not support the idea that external disturbance alone pushes them off their training distribution. This control is also weak: the tracker is not the expert's controller, so even with disturbance off a policy does not reproduce logged states. The cause of these failures is not established here.

### S2: evaluation noise

| Policy | Between-scenario var | Within-scenario (seed) var | ICC | Design effect | Naive CI half-width | Clustered CI half-width |
|---|---|---|---|---|---|---|
| `ridge_hist_noisy_d2` | 0.0000 | 0.0139 | 0.000 | 1.00 | 0.027 | 0.021 |
| `ridge_state_noisy_d2` | 0.0745 | 0.0278 | 0.728 | 2.46 | 0.073 | 0.111 |
| `ridge_hist_noisy_d1` | 0.0517 | 0.1250 | 0.293 | 1.59 | 0.097 | 0.118 |
| `ridge_state_noisy_d1` | 0.0459 | 0.0556 | 0.452 | 1.90 | 0.073 | 0.097 |

8 policies with no variance at all are omitted.

### S4: ADE scored on disturbed logs

Post-hoc (amendment 1). Learned-pool Spearman with disturbed-log ADE: -0.685 [-0.702, -0.436].

Best dev ADE on disturbed logs selects `ridge_hist_noisy_d2`. Closed-selected minus this pick on held-out families: +0.875, 95% CI [+0.764, +0.972], **IMPROVED**.

### S3: trace-derived outcomes vs simulator flags

| Event | Both yes | Both no | Trace yes, sim no | Trace no, sim yes |
|---|---|---|---|---|
| collision | 8 | 850 | 0 | 6 |
| off-road | 46 | 548 | 270 | 0 |

Outcomes use the trace column. The simulator's crash-only cases ended early and keep their trace label, usually TIMEOUT, so collision rates above undercount by that amount. None count as safe completion.

Post-hoc check of the off-road disagreement (`scripts/s3_offroad_side.py`). First exit side, by whether the simulator agreed: both left: 4, both right: 42, trace_only left: 254, trace_only right: 16. Left of a one-lane MetaDrive road is the oncoming lane, which MetaDrive still counts as on a lane, so our metric calls these off-road and it does not.

## Superseded run: metadrive-final

`metadrive-final` reused map seeds across families, so test maps retraced training roads, and it cannot be called held-out. It stays on disk and is excluded from Result B. See `experiments/amendment_1.md`. For comparison, its P2 output was:

- Selected by best dev ADE: `ridge_hist_clean_d2` (dev ADE 0.193 m, dev safe completion 0.000).
- Selected by best dev safe completion: `ridge_state_noisy_d1` (dev ADE 0.595 m, dev safe completion 0.972).
- On held-out families: safe completion 0.861 (closed-selected) vs 0.000 (ADE-selected).
- Paired difference +0.861, 95% CI [+0.708, +0.986], 24 scenarios. **IMPROVED**.

After the fix, share of a route within 2 m of its most similar training route: md_CCS (dev_vs_train): mean 0.50, max 0.69; md_CSC (baseline_train_vs_other_train): mean 0.19, max 0.29; md_CSCC (test_vs_train): mean 0.38, max 0.64; md_SCCS (test_vs_train): mean 0.50, max 0.64; md_SCS (baseline_train_vs_other_train): mean 0.25, max 0.40. Same-seed replay is gone, but overlap stays above the unrelated-map baseline because SCCS and CSCC extend the training sequences SCS and CSC. Read Result B as new maps with similar block order, not as novel road types.

## Failure mining (fixture only)

Packs are chosen from scenario definitions and dev failures only. Test outcomes only score the choice. Rates are over 200 draws per method and budget, with 3 seeds per scenario.

Ground truth on the full test pool:

| Candidate | Regressed on full pool | Mean safe-completion change | Worse scenarios |
|---|---|---|---|
| `reg_drifts` | False | +0.061 | 1 |
| `reg_ignores_lead` | True | -0.408 | 50 |
| `reg_steers_late` | False | -0.008 | 14 |
| `reg_too_fast` | False | +0.003 | 10 |
| `null_control` | False | +0.011 | 1 |

| Budget | Method | Detect `reg_ignores_lead` | False flag `reg_drifts` | False flag `reg_steers_late` | False flag `reg_too_fast` | False alarm (A/A null) | Family coverage | Duplicate fraction | Mean CI width |
|---|---|---|---|---|---|---|---|---|---|
| 8 | mined | 1.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.51 | 0.000 | 0.227 |
| 8 | random | 0.65 | 0.00 | 0.01 | 0.01 | 0.00 | 0.92 | 0.009 | 0.356 |
| 16 | mined | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.82 | 0.000 | 0.189 |
| 16 | random | 0.96 | 0.00 | 0.04 | 0.00 | 0.00 | 0.99 | 0.008 | 0.286 |
| 32 | mined | 1.00 | 0.00 | 0.00 | 0.03 | 0.00 | 1.00 | 0.000 | 0.175 |
| 32 | random | 1.00 | 0.00 | 0.04 | 0.04 | 0.00 | 1.00 | 0.008 | 0.213 |

Only 1 of 4 hand-built regressions really regressed on the full pool, so detection rests on that alone. For the others a flag is a false positive. The largest group of development failures was lead-vehicle collisions, the same behavior as the one real regression, so this setup favors the miner and this is not evidence that mining finds unrelated regressions. Family coverage shows the price: mined packs concentrate on fewer families at small budgets.

## Limitations

- Policies are small ridge regressions and do not represent real planners.
- Open-loop scores come from one expert log per scenario. Results for clean-trained models depend on that choice; S4 tests it.
- Confidence intervals cover scenarios inside these families. They do not cover new families, or which policies were in the pool. Eight learned policies is a small sample of policies.
- The kinematic fixture is hand-written, and its actors do not react. MetaDrive results use one simulator, one lane, light traffic and two held-out block families.
- MetaDrive is repeatable to about 0.2 mm, not bit-exact, so reruns can differ in late digits and a marginal outcome could flip.
- Peak jerk comes from differencing speed under an abrupt controller, so it reflects controller steps and is not a usable comfort measure here.
- Rule plans clamp at 40 m, inflating rule ADE on the fixture above about 13.3 m/s. Learned-pool analyses exclude rules.
- `traces.npz` files are retained locally and hashed in run manifests, but are not committed because of size.

## Source files of this report

| File | sha256 |
|---|---|
| `kinematic-dev/episodes.jsonl.gz` | `a9bdde486832f896` |
| `kinematic-dev/episodes_no_disturbance.jsonl.gz` | `22c03b41185a1403` |
| `kinematic-dev/manifest.json` | `47650981a498a34a` |
| `kinematic-dev/model_selection.json` | `aa31f76cf154dd47` |
| `kinematic-dev/open_loop.jsonl.gz` | `5ce6b9dde9e3eb69` |
| `kinematic-final/episodes.jsonl.gz` | `bd898ef36ac71aac` |
| `kinematic-final/episodes_no_disturbance.jsonl.gz` | `00ccfe143a748a8a` |
| `kinematic-final/manifest.json` | `019f31359fa05478` |
| `kinematic-final/model_selection.json` | `aa31f76cf154dd47` |
| `kinematic-final/open_loop.jsonl.gz` | `1aa20c5183923551` |
| `kinematic-sensitivity/manifest.json` | `2d707ef9abc99746` |
| `kinematic-sensitivity/model_selection.json` | `aa31f76cf154dd47` |
| `kinematic-sensitivity/open_loop_disturbed.jsonl.gz` | `95d094aa9391f19e` |
| `metadrive-dev-v2/episodes.jsonl.gz` | `7b53f6a8c7929b22` |
| `metadrive-dev-v2/episodes_no_disturbance.jsonl.gz` | `de86d8af00a876db` |
| `metadrive-dev-v2/manifest.json` | `ceb9479c25e980db` |
| `metadrive-dev-v2/model_selection.json` | `293834fa35d47fe3` |
| `metadrive-dev-v2/open_loop.jsonl.gz` | `3e7f6dbc70f5f7a6` |
| `metadrive-final-v2/episodes.jsonl.gz` | `33886cf39b609840` |
| `metadrive-final-v2/episodes_no_disturbance.jsonl.gz` | `1413dbd614fb7fa2` |
| `metadrive-final-v2/manifest.json` | `e58399316e870d0a` |
| `metadrive-final-v2/model_selection.json` | `293834fa35d47fe3` |
| `metadrive-final-v2/open_loop.jsonl.gz` | `0785cbd4a3a38e5f` |
| `metadrive-sensitivity-v2/manifest.json` | `ca61abddb7a71ccd` |
| `metadrive-sensitivity-v2/model_selection.json` | `293834fa35d47fe3` |
| `metadrive-sensitivity-v2/open_loop_disturbed.jsonl.gz` | `75495b0be6255e74` |
| `mining-final-v2/manifest.json` | `3a7aacd17c48fdf2` |
| `mining-final-v2/mining_results.json` | `29710c9f5d41a842` |
| `mining-final-v2/outcomes.jsonl.gz` | `c8c03bdf65c82f03` |
| `mining-final-v2/packs.json` | `7235d8775387301b` |
| `analysis/map_overlap.json` | `727f394140e40c1b` |
| `analysis/s3_offroad_side.json` | `56af0ba162cff475` |
