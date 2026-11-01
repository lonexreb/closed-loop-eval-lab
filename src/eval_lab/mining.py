"""Failure mining: pick a small regression pack without ever reading test outcomes.

`select_pack` receives dev rows and scenario definitions only, and refuses anything else, so the
split boundary is a code path and not a promise. Test outcomes appear only in `study`, to score
the selection.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval_lab.adapters.base import WorldState
from eval_lab.adapters.kinematic import KinematicSim
from eval_lab.experiment import load_manifest
from eval_lab.manifest import code_revision, is_dirty, sha256_file, started_at, write_manifest
from eval_lab.policies import PLAN_T, RuleCenter, rule_plan
from eval_lab.runner import row, run_episode
from eval_lab.scenarios import DEV, TEST, Scenario, split_id
from eval_lab.statistics import paired_diff_ci, verdict

EXCLUDED_FROM_MINING = ("stationary", "constant_velocity")  # they fail everywhere
BUDGETS = (8, 16, 32)
DRAWS = 200
SEEDS = (0, 1, 2)
NULL_SEEDS = (10, 11, 12)  # same policy, other seeds: a true A/A test can raise a false alarm
DIVERSITY_WEIGHT = 0.5
DUPLICATE_RADIUS = 0.15


def scenario_features(s: Scenario) -> np.ndarray:
    """Definition of the scenario only. Nothing here depends on any run."""
    kappas = [abs(k) for _, k in s.segments]
    a = s.actors[0] if s.actors else None
    return np.array(
        [
            max(kappas),
            max(kappas) * s.speed_limit_mps**2,  # lateral demand at the limit, m/s^2
            s.half_width_m,
            s.speed_limit_mps,
            0.0 if a is None else 1.0,
            0.0 if a is None or a.brake_at_s is None else a.brake_mps2,
            0.0 if a is None or a.cut_in_at_s is None else 1.0,
        ]
    )


@dataclass(frozen=True)
class Pack:
    ids: tuple[str, ...]
    lineage: dict  # selected scenario id -> the dev scenarios that made it look risky


def select_pack(
    dev_rows: list[dict], dev: list[Scenario], pool: list[Scenario], budget: int, seed: int
) -> Pack:
    """Greedy: high predicted risk, far from what is already chosen. Dev failures set the risk."""
    if any(r["split"] != DEV for r in dev_rows) or any(s.split != DEV for s in dev):
        raise ValueError("mining may read development data only")
    rate: dict[str, list[float]] = {}
    for r in dev_rows:
        if r["policy"] not in EXCLUDED_FROM_MINING:
            rate.setdefault(r["scenario"], []).append(1.0 - float(r["score"]["safe_completed"]))
    fx = np.array([scenario_features(s) for s in dev])
    px = np.array([scenario_features(s) for s in pool])
    both = np.vstack([fx, px])
    mean, scale = both.mean(0), np.where(both.std(0) > 1e-9, both.std(0), 1.0)
    fz, pz = (fx - mean) / scale, (px - mean) / scale
    fail = np.array([np.mean(rate[s.id]) for s in dev])

    d = np.linalg.norm(pz[:, None] - fz[None], axis=-1)
    weight = np.exp(-((d / 2.0) ** 2))
    risk = (weight * fail).sum(1) / np.maximum(weight.sum(1), 1e-9)
    rng = np.random.default_rng(seed)
    risk = risk + rng.gumbel(0.0, 0.02, len(pool))

    chosen: list[int] = []
    for _ in range(budget):
        if chosen:
            spread = np.linalg.norm(pz[:, None] - pz[chosen][None], axis=-1).min(1)
        else:
            spread = np.zeros(len(pool))
        score = risk + DIVERSITY_WEIGHT * np.tanh(spread)
        score[chosen] = -np.inf
        chosen.append(int(score.argmax()))
    lineage = {pool[i].id: [dev[j].id for j in np.argsort(-(weight[i] * fail))[:3]] for i in chosen}
    return Pack(tuple(pool[i].id for i in chosen), lineage)


def random_pack(pool: list[Scenario], budget: int, seed: int) -> Pack:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), budget, replace=False)
    return Pack(tuple(pool[int(i)].id for i in idx), {})


# Candidate regressions: each changes one behavior of the baseline. The null control changes
# nothing, so anything flagged on it is a false alarm.
@dataclass
class TooFast(RuleCenter):
    """Plans 25 percent further along the road, as if the curve speed limit were wrong."""

    name: str = "reg_too_fast"

    def plan(self, st: WorldState) -> np.ndarray:
        return rule_plan(st) * 1.25


@dataclass
class IgnoresLead(RuleCenter):
    """Drops the lead vehicle from its view."""

    name: str = "reg_ignores_lead"

    def plan(self, st: WorldState) -> np.ndarray:
        blind = WorldState(
            st.t_s,
            st.x,
            st.y,
            st.yaw,
            st.speed,
            st.ref,
            st.lateral_m,
            st.heading_error,
            st.speed_limit,
            None,
            None,
        )
        return rule_plan(blind)


@dataclass
class Drifts(RuleCenter):
    """Carries a 1.2 m sideways bias, which only matters on a narrow road."""

    name: str = "reg_drifts"

    def plan(self, st: WorldState) -> np.ndarray:
        return rule_plan(st, 1.2)


@dataclass
class SteersLate(RuleCenter):
    """Looks only at the far end of its plan, so it cuts corners."""

    name: str = "reg_steers_late"

    def plan(self, st: WorldState) -> np.ndarray:
        plan = rule_plan(st)
        plan[:, 1] = plan[:, 1] * (PLAN_T / PLAN_T[-1])
        return plan


@dataclass
class NullControl(RuleCenter):
    name: str = "null_control"


REGRESSIONS = [TooFast(), IgnoresLead(), Drifts(), SteersLate()]


def _outcomes_task(args) -> list[dict]:
    policy, scenarios, seeds = args
    sim = KinematicSim()
    rows = []
    for sc in scenarios:
        for seed in seeds:
            score, _ = run_episode(sim, policy, sc, seed)
            rows.append(row(policy.name, sc, seed, sim.backend, score))
    return rows


def _matrix(rows: list[dict], name: str, ids: list[str], seeds=SEEDS) -> np.ndarray:
    """Scenarios x seeds of safe completion."""
    lookup = {
        (r["scenario"], r["seed"]): r["score"]["safe_completed"]
        for r in rows
        if r["policy"] == name
    }
    return np.array([[float(lookup[(i, s)]) for s in seeds] for i in ids])


def detected(
    cand: np.ndarray, base: np.ndarray, picks: np.ndarray, seed: int
) -> tuple[bool, float]:
    a, b = cand[picks].ravel(), base[picks].ravel()
    clusters = np.repeat(np.arange(len(picks)), len(SEEDS))
    ci = paired_diff_ci(a, b, clusters, seed=seed)
    return verdict(ci) == "REGRESSED", ci.hi - ci.lo


def duplicate_fraction(pz: np.ndarray) -> float:
    if len(pz) < 2:
        return 0.0
    d = np.linalg.norm(pz[:, None] - pz[None], axis=-1)
    iu = np.triu_indices(len(pz), 1)
    return float((d[iu] < DUPLICATE_RADIUS).mean())


def study(run_dir: Path, dev_episodes: Path) -> dict:
    cfg = json.loads(Path("experiments/config_v1.json").read_text())["kinematic"]
    scenarios, manifest_path = load_manifest("kinematic", cfg)
    dev = [s for s in scenarios if s.split == DEV]
    pool = [s for s in scenarios if s.split == TEST]
    with gzip.open(dev_episodes, "rt") as f:
        dev_rows = [json.loads(line) for line in f]

    # Selection happens first and is saved, before a single test outcome exists in this process.
    packs: dict[tuple[str, int], list[Pack]] = {}
    for budget in BUDGETS:
        packs["mined", budget] = [select_pack(dev_rows, dev, pool, budget, d) for d in range(DRAWS)]
        packs["random", budget] = [random_pack(pool, budget, d) for d in range(DRAWS)]
    (run_dir / "packs.json").write_text(
        json.dumps(
            {f"{m}-{b}": [p.__dict__ for p in v[:5]] for (m, b), v in packs.items()}, indent=1
        )
    )

    policies = [RuleCenter(), NullControl(), *REGRESSIONS]
    with ProcessPoolExecutor(cfg["workers"]) as ex:
        tasks = [(p, pool, NULL_SEEDS if p.name == "null_control" else SEEDS) for p in policies]
        rows = sum(ex.map(_outcomes_task, tasks), [])
    with gzip.GzipFile(run_dir / "outcomes.jsonl.gz", "wb", mtime=0) as f:
        f.write(("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n").encode())

    ids = [s.id for s in pool]
    pos = {sid: i for i, sid in enumerate(ids)}
    fam = {s.id: s.family for s in pool}
    base = _matrix(rows, "rule_center", ids)
    px = np.array([scenario_features(s) for s in pool])
    pz = (px - px.mean(0)) / np.where(px.std(0) > 1e-9, px.std(0), 1.0)

    truth = {}
    for p in [*REGRESSIONS, NullControl()]:
        cand = _matrix(rows, p.name, ids, NULL_SEEDS if p.name == "null_control" else SEEDS)
        flag, _ = detected(cand, base, np.arange(len(ids)), 0)
        truth[p.name] = {
            "full_pool_regressed": flag,
            "full_pool_diff": float((cand - base).mean()),
            "failing_scenarios": int((cand.mean(1) < base.mean(1)).sum()),
        }

    table = []
    for (method, budget), plist in packs.items():
        picks = [np.array([pos[i] for i in p.ids]) for p in plist]
        entry: dict = {"method": method, "budget": budget}
        for p in [*REGRESSIONS, NullControl()]:
            cand = _matrix(rows, p.name, ids, NULL_SEEDS if p.name == "null_control" else SEEDS)
            res = [detected(cand, base, q, k) for k, q in enumerate(picks)]
            entry[p.name] = float(np.mean([r[0] for r in res]))
            entry[f"{p.name}_ci_width"] = float(np.mean([r[1] for r in res]))
        entry["family_coverage"] = float(np.mean([len({fam[i] for i in p.ids}) / 4 for p in plist]))
        entry["duplicate_fraction"] = float(np.mean([duplicate_fraction(pz[q]) for q in picks]))
        table.append(entry)
    (run_dir / "mining_results.json").write_text(
        json.dumps({"truth": truth, "table": table}, indent=2, sort_keys=True) + "\n"
    )
    return {
        "data_sha256": sha256_file(manifest_path),
        "split_id": split_id(scenarios),
        "seeds": list(SEEDS),
        "backend": "kinematic_fixture",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-episodes", default="runs/kinematic-dev/episodes.jsonl.gz")
    ap.add_argument("--run-dir", default="runs/mining-final-v2")
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    if is_dirty():
        print("refusing a mining run on a dirty tree: commit first", file=sys.stderr)
        return 2
    if run_dir.exists():
        print(f"refusing to overwrite {run_dir}", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True)
    started = started_at()
    fields = study(run_dir, Path(args.dev_episodes))
    write_manifest(
        run_dir,
        kind="study",
        status="SUCCEEDED",
        exit_code=0,
        code_revision=code_revision(),
        config_sha256=sha256_file(Path("experiments/config_v1.json")),
        started_at=started,
        limitations=["Regressions are hand-built, on the toy fixture only."],
        **fields,
    )
    print(f"wrote {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
