# Analytic metric check (M0)

Metric version 1. Expected values are hand calculations, shown as comments in `src/eval_lab/analytic.py`. No simulator or model is involved.

| Case | Metric | Expected | Measured | Match |
|---|---|---|---|---|
| zero error | ADE@2.5 | 0.0 | 0.0 | yes |
| zero error | FDE@2.5 | 0.0 | 0.0 | yes |
| 1 m lateral offset | ADE@2.5 | 1.0 | 1.0 | yes |
| 1 m lateral offset | FDE@2.5 | 1.0 | 1.0 | yes |
| 1 m/s speed error | ADE@2.5 | 1.75 | 1.75 | yes |
| 1 m/s speed error | FDE@2.5 | 3.0 | 3.0 | yes |
| masked sample | ADE@2.5 | UNASSESSABLE | UNASSESSABLE | yes |
| shifted timestamp | ADE@2.5 | UNASSESSABLE | UNASSESSABLE | yes |
| parked car | outcome | COLLISION | COLLISION | yes |
| parked car | first collision step | 46 | 46 | yes |
| clear road | outcome | COMPLETED | COMPLETED | yes |
| clear road | distance_m | 100.0 | 100.0 | yes |
| stationary | outcome | TIMEOUT | TIMEOUT | yes |
| stationary | safe_completed | False | False | yes |
