"""The candidate pool, fixed by experiments/protocol.md before any final result exists.

The learned grid is a full factorial, so no candidate is hand-picked for how it scores.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from eval_lab.policies import (
    ConstantVelocity,
    Ridge,
    RuleCenter,
    RuleOffset,
    Stationary,
    fit_ridge,
)
from eval_lab.runner import expert_rollout, log_samples, score_log


def collect(sim, scenarios, seeds, disturbance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sim.disturbance = disturbance
    rows = [log_samples(expert_rollout(sim, sc, seed)) for sc in scenarios for seed in seeds]
    xb, xh, y = zip(*rows)
    return np.concatenate(xb), np.concatenate(xh), np.concatenate(y)


def fit_grid(
    clean: tuple, noisy: tuple, dev_logs: list, lambdas: list[float]
) -> tuple[list[Ridge], list[dict]]:
    """Fit features x data x degree. Lambda is chosen per candidate by development ADE alone."""
    models, selection = [], []
    for use_history, (data_name, data), degree in product(
        (False, True), (("clean", clean), ("noisy", noisy)), (1, 2)
    ):
        xb, xh, y = data
        x = np.concatenate([xb, xh], axis=1) if use_history else xb
        name = f"ridge_{'hist' if use_history else 'state'}_{data_name}_d{degree}"
        trials = []
        for lam in lambdas:
            model = fit_ridge(name, x, y, use_history, degree, lam)
            scores = [score_log(model, log)["ade_m"] for log in dev_logs]
            trials.append((float(np.nanmean([v for v in scores if v is not None])), lam, model))
        best = min(trials, key=lambda v: v[0])
        models.append(best[2])
        selection.append({"name": name, "lambda": best[1], "dev_ade_m": best[0], "rows": len(y)})
    return models, selection


def rule_candidates() -> list:
    return [RuleCenter(), RuleOffset(), ConstantVelocity(), Stationary()]
