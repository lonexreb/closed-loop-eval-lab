"""Uncertainty that respects correlation. The independent unit is the scenario, not the episode."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

N_BOOT = 2000


@dataclass(frozen=True)
class Interval:
    estimate: float
    lo: float
    hi: float
    n_clusters: int
    method: str


def _scenario_means(values: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Average repeats inside each scenario, so scenarios with more repeats do not count more."""
    _, inv = np.unique(clusters, return_inverse=True)
    return np.bincount(inv, weights=values) / np.bincount(inv)


def cluster_ci(values, clusters, seed: int = 0, level: float = 0.95) -> Interval:
    """Percentile bootstrap over scenarios.

    All-zero or all-one results have no bootstrap spread, and [0, 0] would falsely claim certainty,
    so those cases use an exact Clopper-Pearson bound with the scenario count as n.
    """
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    if len(values) == 0:
        raise ValueError("no values")
    m = _scenario_means(values, clusters)
    n = len(m)
    if set(np.unique(values)) <= {0.0, 1.0} and m.min() == m.max() and m[0] in (0.0, 1.0):
        alpha = 1 - level
        if m[0] == 0.0:
            return Interval(
                0.0, 0.0, float(sps.beta.ppf(1 - alpha / 2, 1, n)), n, "clopper-pearson"
            )
        return Interval(1.0, float(sps.beta.ppf(alpha / 2, n, 1)), 1.0, n, "clopper-pearson")
    rng = np.random.default_rng(seed)
    boots = m[rng.integers(0, n, (N_BOOT, n))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(boots, [tail, 100 - tail])
    return Interval(float(m.mean()), float(lo), float(hi), n, "scenario-bootstrap")


def naive_ci_halfwidth(values, level: float = 0.95) -> float:
    """What you get by wrongly treating every episode as independent. Kept to show the gap."""
    v = np.asarray(values, float)
    return float(sps.norm.ppf((1 + level) / 2) * v.std(ddof=1) / np.sqrt(len(v)))


def paired_diff_ci(a, b, clusters, seed: int = 0, level: float = 0.95) -> Interval:
    """Mean of a minus b, resampling scenarios. Inputs must be aligned episode by episode."""
    a, b, clusters = np.asarray(a, float), np.asarray(b, float), np.asarray(clusters)
    if not (len(a) == len(b) == len(clusters)):
        raise ValueError("paired inputs must align")
    d = _scenario_means(a - b, clusters)
    rng = np.random.default_rng(seed)
    n = len(d)
    boots = d[rng.integers(0, n, (N_BOOT, n))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(boots, [tail, 100 - tail])
    return Interval(float(d.mean()), float(lo), float(hi), n, "paired-scenario-bootstrap")


def verdict(diff: Interval, margin: float = 0.0) -> str:
    """Release-style reading of `candidate minus baseline` on a higher-is-better metric."""
    if diff.lo > margin:
        return "IMPROVED"
    if diff.hi < -margin:
        return "REGRESSED"
    return "INCONCLUSIVE"


def variance_components(values, clusters) -> dict:
    """Between-scenario and within-scenario (seed) variance, via one-way random-effects ANOVA."""
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    labels, inv, counts = np.unique(clusters, return_inverse=True, return_counts=True)
    k, n = len(labels), len(values)
    if k < 2 or n == k:
        return {"between": None, "within": None, "icc": None, "design_effect": None}
    means = np.bincount(inv, weights=values) / counts
    within = float(((values - means[inv]) ** 2).sum() / (n - k))
    between_ms = float((counts * (means - values.mean()) ** 2).sum() / (k - 1))
    n0 = (n - (counts**2).sum() / n) / (k - 1)
    between = max(0.0, (between_ms - within) / n0)
    total = between + within
    icc = between / total if total > 0 else 0.0
    return {
        "between": between,
        "within": within,
        "icc": icc,
        "design_effect": 1 + (float(counts.mean()) - 1) * icc,
    }


def rank_correlation_ci(open_by_policy, closed_by_policy, seed: int = 0, level: float = 0.95):
    """Spearman across policies, with one scenario resample shared by both axes.

    Each input is policies x scenarios. This captures scenario noise. It does not capture which
    policies were in the pool, and a dozen policies is a small sample of policies.
    """
    o, c = np.asarray(open_by_policy, float), np.asarray(closed_by_policy, float)
    rho = float(sps.spearmanr(o.mean(1), c.mean(1)).statistic)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, o.shape[1], o.shape[1])  # same scenarios on both axes
        r = sps.spearmanr(o[:, pick].mean(1), c[:, pick].mean(1)).statistic
        if np.isfinite(r):
            boots.append(r)
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(boots, [tail, 100 - tail]) if boots else (np.nan, np.nan)
    return Interval(rho, float(lo), float(hi), o.shape[1], "scenario-bootstrap-spearman")
