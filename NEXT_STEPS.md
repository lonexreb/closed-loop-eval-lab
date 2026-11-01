# Next steps

The pilot and the M2 study are complete. The items below would extend it, and none is needed for STUDY_COMPLETE.

## Worth doing next

1. **More families, and family-level CIs registered up front.** Eight is too few when success sits in two of them.
2. **Separate compounding from unfamiliar roads.** Run S5 on train families, where the road is known. If clean history models
   still fail there, their own error is the likelier cause.
3. **A dev set with more interactions.** `reg_blind_on_curve` passed dev because no dev scenario has a lead car on a curve.
4. **Mining on MetaDrive.** Scenario features there are thin, since a map is only a block sequence and seed.
5. **Reduce the sim-only crash gap.** MetaDrive reported 22 crashes in M2 that the trace did not show.
6. **Lane-relative off-road for junctions.** 788 trace off-roads have no simulator flag. Agree a definition before rerunning.

## Small fixes

- Stop rule plans clamping at 40 m by extending the reference, then rerun as a named run.
- Replace peak jerk with a smoothed comfort measure and new definitions in `docs/METRICS.md`.

## Not done on purpose

No upstream PR to MetaDrive. A contribution idea to explore is the reset-determinism drift in `DECISIONS.md`. Search
existing issues and reproduce it on the current commit first.
