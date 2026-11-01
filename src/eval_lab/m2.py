"""M2 regression study: frozen regressions, validated on dev, scored on families no run has touched.

  python -m eval_lab.m2 --stage validate   # dev only. Decides which regressions count as validated.
  python -m eval_lab.m2 --stage holdout    # once. Refuses a dirty tree or an existing run.

The miner is the pilot's `select_pack`, unchanged. A pack is chosen before any held-out outcome
exists in the process, and outcomes only score the choice. See experiments/protocol_m2.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval_lab.adapters.base import WorldState
from eval_lab.experiment import _write_gz, load_holdout, load_manifest
from eval_lab.manifest import code_revision, is_dirty, sha256_file, started_at, write_manifest
from eval_lab.mining import (
    BUDGETS,
    DRAWS,
    NULL_SEEDS,
    SEEDS,
    IgnoresLead,
    NullControl,
    Pack,
    _matrix,
    _outcomes_task,
    detected,
    duplicate_fraction,
    random_pack,
    scenario_features,
    select_pack,
)
from eval_lab.policies import RuleCenter, rule_plan, to_ego_xy
from eval_lab.report import read_gz
from eval_lab.scenarios import DEV, HOLDOUT, Scenario, assert_disjoint, split_id
from eval_lab.statistics import paired_diff_ci, variance_components, verdict

VALIDATION = Path("runs/m2-validation/validation.json")
PACK_BUDGET = 16  # the primary comparison, and the size of the delivered regression pack


@dataclass
class Offset(RuleCenter):
    offset_m: float = 0.0

    def plan(self, st: WorldState) -> np.ndarray:
        return rule_plan(st, self.offset_m)


@dataclass
class Weave(RuleCenter):
    """A slow sideways oscillation in the plan, as a badly damped lateral layer would give."""

    amplitude_m: float = 0.0

    def plan(self, st: WorldState) -> np.ndarray:
        plan = rule_plan(st)
        plan[:, 1] += self.amplitude_m * np.sin(1.2 * st.t_s)
        return plan


@dataclass
class ShortGap(RuleCenter):
    """Keeps only a fraction of the expert's following gap."""

    keep: float = 1.0

    def plan(self, st: WorldState) -> np.ndarray:
        if st.gap_m is None:
            return rule_plan(st)
        extra = (1 - self.keep) * (5.0 + 1.5 * st.speed)
        return rule_plan(dataclasses.replace(st, gap_m=st.gap_m + extra))


@dataclass
class StaleLead(RuleCenter):
    """Sees the lead vehicle as it was `lag` steps ago: late perception, not a wrong plan."""

    lag: int = 0

    def reset(self) -> None:
        self.past: deque = deque(maxlen=self.lag + 1)

    def plan(self, st: WorldState) -> np.ndarray:
        self.past.append(st)
        old = self.past[0]
        return rule_plan(dataclasses.replace(st, gap_m=old.gap_m, lead_speed=old.lead_speed))


@dataclass
class StalePose(RuleCenter):
    """Plans from where the car was `lag` steps ago: late localisation."""

    lag: int = 0

    def reset(self) -> None:
        self.past: deque = deque(maxlen=self.lag + 1)

    def plan(self, st: WorldState) -> np.ndarray:
        self.past.append(st)
        old = self.past[0]
        return rule_plan(
            dataclasses.replace(old, speed=st.speed, gap_m=st.gap_m, lead_speed=st.lead_speed)
        )


@dataclass
class FasterLimit(RuleCenter):
    """Believes the speed limit is higher than it is."""

    factor: float = 1.0

    def plan(self, st: WorldState) -> np.ndarray:
        return rule_plan(dataclasses.replace(st, speed_limit=st.speed_limit * self.factor))


@dataclass
class BlindOnCurve(RuleCenter):
    """Loses the lead vehicle whenever the road ahead bends. Dev has no lead on a curve."""

    name: str = "reg_blind_on_curve"

    def plan(self, st: WorldState) -> np.ndarray:
        bend = abs(float(to_ego_xy(st, st.ref)[-1, 1]))
        if bend > 3.0:
            return rule_plan(dataclasses.replace(st, gap_m=None, lead_speed=None))
        return rule_plan(st)


CANDIDATES: list[RuleCenter] = [
    IgnoresLead(),
    ShortGap(name="reg_short_gap_0.0", keep=0.0),
    ShortGap(name="reg_short_gap_0.1", keep=0.1),
    StaleLead(name="reg_stale_lead_1.5s", lag=15),
    StaleLead(name="reg_stale_lead_2.0s", lag=20),
    Offset(name="reg_offset_2.0", offset_m=2.0),
    Offset(name="reg_offset_2.5", offset_m=2.5),
    Weave(name="reg_weave_2.0", amplitude_m=2.0),
    StalePose(name="reg_stale_pose_0.2s", lag=2),
    StalePose(name="reg_stale_pose_0.3s", lag=3),
    # Expected to pass dev. Whether any of these fails on held-out families is part of the result.
    StaleLead(name="reg_stale_lead_1.2s", lag=12),
    Offset(name="reg_offset_1.5", offset_m=1.5),
    Weave(name="reg_weave_1.0", amplitude_m=1.0),
    FasterLimit(name="reg_faster_limit_2x", factor=2.0),
    BlindOnCurve(),
]


def outcomes(policies, pool: list[Scenario], workers: int) -> list[dict]:
    tasks = [(p, pool, NULL_SEEDS if p.name == "null_control" else SEEDS) for p in policies]
    with ProcessPoolExecutor(workers) as ex:
        return sum(ex.map(_outcomes_task, tasks), [])


def verdict_on(cand: np.ndarray, base: np.ndarray) -> dict:
    ids = np.repeat(np.arange(len(base)), base.shape[1])
    ci = paired_diff_ci(cand.ravel(), base.ravel(), ids)
    return {"diff": ci.estimate, "lo": ci.lo, "hi": ci.hi, "verdict": verdict(ci)}


def validate(run_dir: Path) -> dict:
    cfg = json.loads(Path("experiments/config_v1.json").read_text())["kinematic"]
    scenarios, manifest_path = load_manifest("kinematic", cfg)
    dev = [s for s in scenarios if s.split == DEV]
    rows = outcomes([RuleCenter(), *CANDIDATES], dev, cfg["workers"])
    ids = [s.id for s in dev]
    base = _matrix(rows, "rule_center", ids)
    result = {p.name: verdict_on(_matrix(rows, p.name, ids), base) for p in CANDIDATES}
    (run_dir / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return {
        "data_sha256": sha256_file(manifest_path),
        "split_id": split_id(scenarios),
        "seeds": list(SEEDS),
        "backend": "kinematic_fixture",
    }


def mining_table(packs, rows, policies, ids, family_of, pz) -> list[dict]:
    pos = {sid: i for i, sid in enumerate(ids)}
    base = _matrix(rows, "rule_center", ids)
    n_families = len(set(family_of.values()))
    table = []
    for (method, budget), plist in packs.items():
        picks = [np.array([pos[i] for i in p.ids]) for p in plist]
        entry: dict = {
            "method": method,
            "budget": budget,
            "distinct_packs": len({tuple(sorted(p.ids)) for p in plist}),
            "family_coverage": float(
                np.mean([len({family_of[i] for i in p.ids}) / n_families for p in plist])
            ),
            "duplicate_fraction": float(np.mean([duplicate_fraction(pz[q]) for q in picks])),
        }
        for p in policies:
            seeds = NULL_SEEDS if p.name == "null_control" else SEEDS
            cand = _matrix(rows, p.name, ids, seeds)
            res = [detected(cand, base, q, k) for k, q in enumerate(picks)]
            entry[p.name] = float(np.mean([r[0] for r in res]))
            entry[f"{p.name}_ci_width"] = float(np.mean([r[1] for r in res]))
        table.append(entry)
    return table


def mined_minus_random(table: list[dict], names: list[str], budget: int, seed: int = 0) -> dict:
    """Primary: mean detection gap, resampling regressions, the independent unit here."""
    if not names:
        return {"regressions": 0, "diff": None, "lo": None, "hi": None, "verdict": "NOT_APPLICABLE"}
    by = {(t["method"], t["budget"]): t for t in table}
    gap = np.array([by["mined", budget][n] - by["random", budget][n] for n in names])
    rng = np.random.default_rng(seed)
    boots = gap[rng.integers(0, len(gap), (2000, len(gap)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    reading = "MINED_BETTER" if lo > 0 else "RANDOM_BETTER" if hi < 0 else "INCONCLUSIVE"
    return {
        "regressions": len(gap),
        "diff": float(gap.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "verdict": reading,
    }


def holdout_study(run_dir: Path, dev_episodes: Path) -> dict:
    cfg = json.loads(Path("experiments/config_v1.json").read_text())["kinematic"]
    m2 = json.loads(Path("experiments/config_m2.json").read_text())["kinematic"]
    validation = json.loads(VALIDATION.read_text())
    scenarios, _ = load_manifest("kinematic", cfg)
    pool, pool_path = load_holdout("kinematic", m2)
    assert_disjoint([*scenarios, *pool])
    assert all(s.split == HOLDOUT for s in pool)
    dev = [s for s in scenarios if s.split == DEV]
    dev_rows = read_gz(dev_episodes)

    # Selection comes first and is saved, before one held-out outcome exists in this process.
    packs: dict[tuple[str, int], list[Pack]] = {}
    for budget in BUDGETS:
        packs["mined", budget] = [select_pack(dev_rows, dev, pool, budget, d) for d in range(DRAWS)]
        packs["random", budget] = [random_pack(pool, budget, d) for d in range(DRAWS)]
    delivered = packs["mined", PACK_BUDGET][0]
    (run_dir / "regression_pack.json").write_text(
        json.dumps(
            {
                "scenario_manifest_sha256": sha256_file(pool_path),
                "budget": PACK_BUDGET,
                "seeds": list(SEEDS),
                "ids": list(delivered.ids),
                "lineage": delivered.lineage,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    policies = [*CANDIDATES, NullControl()]
    rows = outcomes([RuleCenter(), *policies], pool, cfg["workers"])
    _write_gz(run_dir / "outcomes.jsonl.gz", rows)

    ids = [s.id for s in pool]
    family_of = {s.id: s.family for s in pool}
    base = _matrix(rows, "rule_center", ids)
    truth = {}
    for p in policies:
        seeds = NULL_SEEDS if p.name == "null_control" else SEEDS
        cand = _matrix(rows, p.name, ids, seeds)
        cluster = np.repeat(np.arange(len(ids)), len(seeds))
        var = variance_components(cand.ravel(), cluster)
        truth[p.name] = {
            **verdict_on(cand, base),
            "dev_verdict": validation.get(p.name, {}).get("verdict", "n/a"),
            "failing_scenarios": int((cand.mean(1) < base.mean(1)).sum()),
            "seed_variance": var["within"],
            "icc": var["icc"],
        }
    px = np.array([scenario_features(s) for s in pool])
    pz = (px - px.mean(0)) / np.where(px.std(0) > 1e-9, px.std(0), 1.0)
    table = mining_table(packs, rows, policies, ids, family_of, pz)
    real = [
        n
        for n, t in truth.items()
        if n != "null_control" and t["verdict"] == "REGRESSED" and t["dev_verdict"] == "REGRESSED"
    ]
    summary = {b: mined_minus_random(table, real, b) for b in BUDGETS}
    (run_dir / "mining_results.json").write_text(
        json.dumps({"truth": truth, "table": table, "real": real, "gap": summary}, indent=2) + "\n"
    )
    return {
        "data_sha256": sha256_file(pool_path),
        "split_id": split_id([*scenarios, *pool]),
        "seeds": list(SEEDS),
        "backend": "kinematic_fixture",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("validate", "holdout"), required=True)
    ap.add_argument("--dev-episodes", default="runs/kinematic-dev/episodes.jsonl.gz")
    ap.add_argument("--run-root", default="runs")
    args = ap.parse_args(argv)
    frozen = args.stage == "holdout"
    run_dir = Path(args.run_root) / ("m2-mining" if frozen else "m2-validation")
    if frozen and is_dirty():
        print("refusing a held-out run on a dirty tree: commit first", file=sys.stderr)
        return 2
    if run_dir.exists():
        print(f"refusing to overwrite {run_dir}", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True)
    started = started_at()
    fields = holdout_study(run_dir, Path(args.dev_episodes)) if frozen else validate(run_dir)
    write_manifest(
        run_dir,
        kind="study",
        status="SUCCEEDED",
        exit_code=0,
        code_revision=code_revision(),
        config_sha256=sha256_file(Path("experiments/config_m2.json")),
        started_at=started,
        limitations=["Regressions are hand-built, on the toy fixture only."],
        eval_split=HOLDOUT if frozen else DEV,
        **fields,
    )
    print(f"wrote {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
