"""ADE/FDE against a logged future. See docs/METRICS.md."""

from __future__ import annotations

import numpy as np

from eval_lab.schemas import MetricResult, Trajectory, assessed, unassessable

TIME_TOL_S = 1e-6


def _errors(pred: Trajectory, gt: Trajectory, horizon_s: float) -> np.ndarray | str:
    """L2 error per window sample, or the reason the comparison cannot be assessed."""
    if horizon_s <= 0:
        raise ValueError("horizon must be positive")
    if pred.frame != gt.frame:
        return f"frame mismatch: {pred.frame} vs {gt.frame}"
    if len(gt.t_s) == 0 or len(pred.t_s) == 0:
        return "empty trajectory"
    if not (np.isfinite(gt.t_s).all() and np.isfinite(pred.t_s).all()):
        return "non-finite timestamp"
    if (np.diff(gt.t_s) <= 0).any() or (np.diff(pred.t_s) <= 0).any():
        return "timestamps not strictly increasing"
    end = gt.t_s[0] + horizon_s
    if gt.t_s[-1] < end - TIME_TOL_S:
        return f"ground truth stops at {gt.t_s[-1] - gt.t_s[0]:.3f}s, short of horizon {horizon_s}s"
    window = gt.t_s <= end + TIME_TOL_S
    n = int(window.sum())
    if len(pred.t_s) < n or np.abs(pred.t_s[:n] - gt.t_s[:n]).max() > TIME_TOL_S:
        return "prediction timestamps do not align with ground truth"
    if not (gt.mask[:n].all() and pred.mask[:n].all()):
        return "missing or invalid samples inside the horizon"
    return np.linalg.norm(pred.xy_m[:n] - gt.xy_m[:n], axis=1)


def ade(pred: Trajectory, gt: Trajectory, horizon_s: float) -> MetricResult:
    name = f"ade@{horizon_s:g}s"
    err = _errors(pred, gt, horizon_s)
    if isinstance(err, str):
        return unassessable(name, "m", err)
    return assessed(name, "m", float(err.sum()), len(err))


def fde(pred: Trajectory, gt: Trajectory, horizon_s: float) -> MetricResult:
    name = f"fde@{horizon_s:g}s"
    err = _errors(pred, gt, horizon_s)
    if isinstance(err, str):
        return unassessable(name, "m", err)
    return assessed(name, "m", float(err[-1]), 1)
