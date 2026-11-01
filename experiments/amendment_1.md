# Amendment 1 to protocol v1

Committed before the runs it describes. Written after the first final runs and an independent code
review, so everything here is a post-hoc deviation and is labelled as one in the report. The analysis
plan, endpoint, candidate pool and regressions are unchanged. Only the defects below are repaired.

## Why a rerun is required

`metadrive-final` is **superseded** and stays on disk. Map seeds 0 to 11 were reused across families, and
MetaDrive draws blocks from one random stream per seed, so test maps retraced training roads: 59 to 74
percent of a held-out route lay on one training route, measured by `scripts/map_overlap.py`. Those
results cannot be called held-out. `kinematic-final` has no such defect and is not rerun.

After the seed fix, geometric overlap fell to 38 to 50 percent, against 19 to 25 percent between unrelated
training maps. The rest is expected from the family choice, since SCCS and CSCC extend SCS and CSC. The
families stay as registered, and this is reported as a limitation.

## New named runs

- `metadrive-dev-v2`, `metadrive-final-v2`: the same families and configuration with a separate map-seed
  range per family. The reason is the leak above, not the results.
- `mining-final-v2`: the null control now uses different seeds (a true A/A test). Before, it was identical
  to the baseline and could not raise a false alarm. Regressions are unchanged, including the three that
  were not real regressions. Their flag rates are false positives, not detections.
- `kinematic-sensitivity`, `metadrive-sensitivity-v2`: S4 below.

## Added analysis

**S4, disturbed open-loop sensitivity.** Open-loop scores use undisturbed expert logs while closed-loop runs
use disturbance, so clean-trained models are scored on their own training distribution. With the same
models, also score ADE on expert logs at disturbance 1, then repeat P1 and P2 with that ADE. Reported next
to the originals, without replacing them.

## Corrected reporting

- S1 compares both columns on the same seed. It is reported as inconclusive: the tracker differs from the
  expert, so disturbance 0 does not reproduce the logged state distribution.
- The P1 bootstrap resamples one set of scenarios for both axes. Earlier CIs resampled each axis alone.
- S3 also reports simulator-only crashes and which side cars left the road on.
- Known and unfixed: rule plans clamp at the 40 m reference, inflating rule ADE above about 13.3 m/s on the
  fixture. It cannot affect learned-pool P1 or P2, which exclude rules, or MetaDrive at 10 m/s.
