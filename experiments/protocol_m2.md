# Pre-registered protocol M2

Committed before any `holdout` run, so `git log` shows the plan came first. A later change to this file is a
deviation and is reported as one. Protocol v1 and amendment 1 still govern the pilot.

## Why a new protocol

The pilot left three gaps. Only one of four hand-built regressions was real. Test families had been seen, so
they could not be reused for anything tuned afterwards. MetaDrive held-out maps sat too close to training
roads. M2 uses new families, regressions validated on dev and families chosen for distance from training.

## What was looked at before this was frozen

- **Regressions, dev only.** 37 variants were screened on dev and 17 regressed. Steering-gain and corner-cutting
  variants had no effect because the tracker corrects them. Variants that fail nearly everything (offset 2.8 m
  and up, stale pose 0.5 s and up) were dropped as uninformative. Two milder stale-pose variants were added.
- **Generators, expert only, spare draws.** With fixture manifest seed 777, `slow_lead_curve` and `close_cut_in`
  were too hard for the expert (60 and 61 percent), so they were softened. No candidate ran.
- **MetaDrive families, geometry and expert only, map seeds 900 and up.** Eight families were picked from 25: the
  lowest route overlap with training among those the expert finished on all six probes. Maps that open with S
  were rejected because they share the training start.

No candidate policy or regression has run on any held-out scenario or held-out family.

## Splits and pool

Train and dev are unchanged. Held-out families live in `experiments/scenarios_{kinematic,metadrive}_m2.json`:

- Fixture: 8 families with 30 scenarios each, 5 seeds. `wide_cruise` and `wide_gentle` are benign on purpose.
- MetaDrive: 8 block-sequence families with 12 maps each, 3 seeds, at map seeds 500 and up.

The candidate pool, model fit, tracker and paired seeds are the same as protocol v1.

## Confirmatory analyses, both worlds

Endpoint: safe completion over all planned episodes. All picks use dev data only, from existing dev and
sensitivity runs. Readings: IMPROVED, REGRESSED or INCONCLUSIVE from whether a 95 percent scenario-resampled CI
excludes 0.

- **C1.** Closed-selected minus clean-log-ADE-selected, paired on held-out families.
- **C2.** Closed-selected minus disturbed-log-ADE-selected. Disturbed ADE is now registered, not post-hoc.
- **C3.** Learned-pool Spearman of held-out safe completion with clean ADE and with disturbed ADE.
- **C4.** Safe completion per family for each policy.

## Secondary

- **S2.** Noise, as in protocol v1.
- **S3.** Trace and simulator flag cross-check on MetaDrive.
- **S5, mechanism replay.** Replaces S1. Disturbance 0, seed 0, and the tracker that made the logs, from the
  logged start. Only the policy's own error can pull a run away from its log. Control: `rule_center` must reach
  0.95 safe completion, or the test is void. Report safe completion and median time to 1 m from the expert log.
  If a clean-trained history model still fails, its own error compounds with no outside push. If it recovers,
  outside disturbance is necessary for that failure. Either way this shows a route to failure, not the only one.

## Regression study (fixture only)

The miner is `select_pack`, unchanged from the pilot. Baseline is random selection at matched budget. Budgets 8,
16 and 32, 3 seeds, 200 draws. The 15 candidates are in `src/eval_lab/m2.py`. `validation.json` is committed here,
so the dev verdicts are fixed before the held-out run.

- **Real regression:** REGRESSED on dev and REGRESSED on the full held-out pool.
- **Primary, M1.** At budget 16, mean of mined minus random detection rate across real regressions, resampling
  regressions. MINED_BETTER or RANDOM_BETTER needs a 95 percent CI excluding 0, otherwise INCONCLUSIVE. Under 3
  real regressions, report it as too few and draw no conclusion.
- **Secondary.** Budgets 8 and 32, false positives on non-regressions, A/A false alarm, CI width, seed variance,
  family coverage, duplicate fraction, distinct packs. Dev-silent candidates that fail held-out are reported as
  dev misses, not counted as real.

Limits known in advance: draws of one miner are not independent, so the 200 draws measure pack noise, not study
replication. Regressions are the unit, and they are hand-built, correlated variants of one baseline. The
delivered regression pack is the mined budget-16 pack of draw 0, with lineage.

## Stopping and reporting

Runs happen once at a clean commit. A rerun is a new, named run with a stated reason. STUDY_COMPLETE needs these
results reported whichever way they fall. Null, negative and inconclusive results stand.
