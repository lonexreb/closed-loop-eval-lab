"""Open-loop vs closed-loop comparison. One command, one immutable run directory.

  python -m eval_lab.experiment --world kinematic --stage dev
  python -m eval_lab.experiment --world kinematic --stage final
  python -m eval_lab.experiment --world kinematic --stage holdout

dev touches train and dev families. final is the only code path that reads test families, and
holdout the only one that reads the M2 families. Both refuse a dirty tree or an existing run, so a
final result cannot be re-rolled quietly.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from eval_lab import scenarios as sc_mod
from eval_lab.candidates import collect, fit_grid, rule_candidates
from eval_lab.manifest import code_revision, is_dirty, sha256_file, started_at, write_manifest
from eval_lab.policies import expert_track, track
from eval_lab.runner import expert_rollout, row, run_episode, score_log

CONFIG = Path("experiments/config_v1.json")
M2_CONFIG = Path("experiments/config_m2.json")
TRACKERS = {"fixed": track, "expert": expert_track}
DIVERGED_M = 1.0
_SIM = None


def make_sim(world: str):
    if world == "kinematic":
        from eval_lab.adapters.kinematic import KinematicSim

        return KinematicSim()
    from eval_lab.adapters.metadrive import MetaDriveSim

    return MetaDriveSim()


def load_manifest(world: str, cfg: dict) -> tuple[list[sc_mod.Scenario], Path]:
    """Rebuild and compare with the frozen file, so a generator change cannot pass silently."""
    path = Path(f"experiments/scenarios_{world}.json")
    if world == "kinematic":
        built = sc_mod.build_manifest(cfg["per_family"], cfg["manifest_seed"])
    else:
        built = sc_mod.build_metadrive_manifest(cfg["per_family"])
    sc_mod.assert_disjoint(built)
    if not path.exists():
        sc_mod.write_manifest(built, path)
    frozen = sc_mod.read_manifest(path)
    if sc_mod.manifest_hash(frozen) != sc_mod.manifest_hash(built):
        raise SystemExit(f"{path} no longer matches its generator; refusing to run")
    return frozen, path


def load_holdout(world: str, m2: dict) -> tuple[list[sc_mod.Scenario], Path]:
    """M2 families. They live in their own frozen file so pilot manifests cannot change."""
    path = Path(f"experiments/scenarios_{world}_m2.json")
    if world == "kinematic":
        built = sc_mod.build_m2_manifest(m2["per_family"], m2["manifest_seed"])
    else:
        built = sc_mod.build_metadrive_m2_manifest(m2["per_family"])
    sc_mod.assert_disjoint(built)
    if not path.exists():
        sc_mod.write_manifest(built, path)
    frozen = sc_mod.read_manifest(path)
    if sc_mod.manifest_hash(frozen) != sc_mod.manifest_hash(built):
        raise SystemExit(f"{path} no longer matches its generator; refusing to run")
    return frozen, path


def _closed_loop_task(args) -> tuple[list[dict], dict]:
    world, policy, scenarios, seeds, disturbance, tracker = args
    global _SIM
    if _SIM is None:
        _SIM = make_sim(world)
    _SIM.disturbance = disturbance
    rows, traces = [], {}
    for sc in scenarios:
        for seed in seeds:
            score, trace = run_episode(_SIM, policy, sc, seed, TRACKERS[tracker])
            rows.append(row(policy.name, sc, seed, _SIM.backend, score))
            if trace is not None:
                traces[f"{policy.name}|{sc.id}|{seed}"] = np.column_stack(
                    [trace.ego.t_s, trace.ego.xy_m, trace.ego.yaw_rad, trace.speed_mps]
                ).astype(np.float32)
    return rows, traces


def closed_loop(world, policies, scenarios, seeds, disturbance, workers, tracker="fixed"):
    tasks = [(world, p, scenarios, seeds, disturbance, tracker) for p in policies]
    rows, traces = [], {}
    with ProcessPoolExecutor(workers) as pool:
        for r, t in pool.map(_closed_loop_task, tasks):
            rows += r
            traces.update(t)
    return rows, traces


def _write_gz(path: Path, rows: list[dict]) -> None:
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        f.write(("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n").encode())


def run(world: str, stage: str, run_dir: Path, cfg: dict, log) -> dict:
    scenarios, manifest_path = load_manifest(world, cfg)
    split = sc_mod.DEV if stage == "dev" else sc_mod.TEST
    train = [s for s in scenarios if s.split == sc_mod.TRAIN]
    dev = [s for s in scenarios if s.split == sc_mod.DEV]
    target = [s for s in scenarios if s.split == split]
    assert all(s.split == split for s in target) and target
    log(f"{world} {stage}: train={len(train)} dev={len(dev)} eval={len(target)} split={split}")

    t0 = time.time()
    sim = make_sim(world)
    clean = collect(sim, train, cfg["train_seeds"], 0.0)
    noisy = collect(sim, train, cfg["train_seeds"], cfg["noisy_disturbance"])
    sim.disturbance = 0.0
    dev_logs = [expert_rollout(sim, s, 0) for s in dev]
    models, selection = fit_grid(clean, noisy, dev_logs, cfg["lambdas"])
    log(f"fit {len(models)} models on {len(clean[2])} clean rows in {time.time() - t0:.1f}s")
    (run_dir / "model_selection.json").write_text(json.dumps(selection, indent=2) + "\n")

    t1 = time.time()
    policies = [*rule_candidates(), *models]
    logs = {s.id: expert_rollout(sim, s, 0) for s in target}
    open_rows = []
    for policy in policies:
        for s in target:
            r = score_log(policy, logs[s.id])
            open_rows.append(
                {"policy": policy.name, "scenario": s.id, "family": s.family, "split": s.split, **r}
            )
    _write_gz(run_dir / "open_loop.jsonl.gz", open_rows)
    log(f"open loop: {len(open_rows)} rows in {time.time() - t1:.1f}s")
    if hasattr(sim, "close"):
        sim.close()

    t2 = time.time()
    rows, traces = closed_loop(world, policies, target, cfg["eval_seeds"], 1.0, cfg["workers"])
    _write_gz(run_dir / "episodes.jsonl.gz", rows)
    np.savez_compressed(run_dir / "traces.npz", **traces)
    log(f"closed loop: {len(rows)} episodes in {time.time() - t2:.1f}s")

    # Mechanism control: same models and scenarios with the disturbance switched off.
    t3 = time.time()
    calm, _ = closed_loop(world, policies, target, cfg["eval_seeds"][:1], 0.0, cfg["workers"])
    _write_gz(run_dir / "episodes_no_disturbance.jsonl.gz", calm)
    log(f"mechanism control: {len(calm)} episodes in {time.time() - t3:.1f}s")

    return {
        "data_sha256": sha256_file(manifest_path),
        "split_id": sc_mod.split_id(scenarios),
        "seeds": cfg["eval_seeds"],
        "backend": rows[0]["backend"],
        "eval_split": split,
    }


def sensitivity(world: str, run_dir: Path, cfg: dict, log) -> dict:
    """S4: the same models, with ADE scored on disturbed expert logs. No closed-loop runs here."""
    scenarios, manifest_path = load_manifest(world, cfg)
    train = [s for s in scenarios if s.split == sc_mod.TRAIN]
    dev = [s for s in scenarios if s.split == sc_mod.DEV]
    target = [s for s in scenarios if s.split in (sc_mod.DEV, sc_mod.TEST)]
    sim = make_sim(world)
    clean = collect(sim, train, cfg["train_seeds"], 0.0)
    noisy = collect(sim, train, cfg["train_seeds"], cfg["noisy_disturbance"])
    sim.disturbance = 0.0
    dev_logs = [expert_rollout(sim, s, 0) for s in dev]
    models, selection = fit_grid(clean, noisy, dev_logs, cfg["lambdas"])
    (run_dir / "model_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    sim.disturbance = 1.0
    rows = []
    policies = [*rule_candidates(), *models]
    for s in target:
        log_states = expert_rollout(sim, s, 0)
        for policy in policies:
            r = score_log(policy, log_states)
            rows.append(
                {"policy": policy.name, "scenario": s.id, "family": s.family, "split": s.split, **r}
            )
    _write_gz(run_dir / "open_loop_disturbed.jsonl.gz", rows)
    log(f"disturbed open loop: {len(rows)} rows")
    if hasattr(sim, "close"):
        sim.close()
    return {
        "data_sha256": sha256_file(manifest_path),
        "split_id": sc_mod.split_id(scenarios),
        "seeds": [0],
        "backend": "metadrive-0.4.3" if world == "metadrive" else "kinematic_fixture",
        "eval_split": "dev+test",
    }


def open_rows_for(policies, target, logs) -> list[dict]:
    rows = []
    for policy in policies:
        for s in target:
            r = score_log(policy, logs[s.id])
            rows.append(
                {"policy": policy.name, "scenario": s.id, "family": s.family, "split": s.split, **r}
            )
    return rows


def divergence_s(trace: np.ndarray, log, limit_m: float = DIVERGED_M) -> float | None:
    """First time a closed-loop run is limit_m from its expert log, or None if it never is."""
    n = min(len(trace), len(log))
    expert = np.array([[st.x, st.y] for st in log[:n]])
    far = np.linalg.norm(trace[:n, 1:3] - expert, axis=1) > limit_m
    return float(trace[int(np.argmax(far)), 0]) if far.any() else None


def holdout(world: str, run_dir: Path, cfg: dict, log) -> dict:
    """M2 protocol: same training, dev and models as the pilot, on families no run has touched."""
    m2 = json.loads(M2_CONFIG.read_text())[world]
    scenarios, _ = load_manifest(world, cfg)
    target, target_path = load_holdout(world, m2)
    sc_mod.assert_disjoint([*scenarios, *target])
    train = [s for s in scenarios if s.split == sc_mod.TRAIN]
    dev = [s for s in scenarios if s.split == sc_mod.DEV]
    log(f"{world} holdout: train={len(train)} dev={len(dev)} eval={len(target)}")

    sim = make_sim(world)
    clean = collect(sim, train, cfg["train_seeds"], 0.0)
    noisy = collect(sim, train, cfg["train_seeds"], cfg["noisy_disturbance"])
    sim.disturbance = 0.0
    dev_logs = [expert_rollout(sim, s, 0) for s in dev]
    models, selection = fit_grid(clean, noisy, dev_logs, cfg["lambdas"])
    (run_dir / "model_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    policies = [*rule_candidates(), *models]

    logs = {s.id: expert_rollout(sim, s, 0) for s in target}
    _write_gz(run_dir / "open_loop.jsonl.gz", open_rows_for(policies, target, logs))
    sim.disturbance = 1.0
    noisy_logs = {s.id: expert_rollout(sim, s, 0) for s in target}
    _write_gz(run_dir / "open_loop_disturbed.jsonl.gz", open_rows_for(policies, target, noisy_logs))
    if hasattr(sim, "close"):
        sim.close()

    t0 = time.time()
    seeds, workers = m2["eval_seeds"], cfg["workers"]
    rows, traces = closed_loop(world, policies, target, seeds, 1.0, workers)
    _write_gz(run_dir / "episodes.jsonl.gz", rows)
    np.savez_compressed(run_dir / "traces.npz", **traces)
    log(f"closed loop: {len(rows)} episodes in {time.time() - t0:.1f}s")

    calm, _ = closed_loop(world, policies, target, seeds[:1], 0.0, workers)
    _write_gz(run_dir / "episodes_no_disturbance.jsonl.gz", calm)

    # S5: no disturbance and the logging tracker, from the logged start. The only thing left that
    # can pull a run away from its log is the policy's own prediction error.
    t1 = time.time()
    replay, replay_traces = closed_loop(world, policies, target, seeds[:1], 0.0, workers, "expert")
    _write_gz(run_dir / "episodes_expert_tracker.jsonl.gz", replay)
    diverged = [
        {
            "policy": key.split("|")[0],
            "scenario": key.split("|")[1],
            "t_s": divergence_s(trace, logs[key.split("|")[1]]),
        }
        for key, trace in sorted(replay_traces.items())
    ]
    (run_dir / "s5_divergence.json").write_text(json.dumps(diverged) + "\n")
    log(f"S5: {len(replay)} episodes in {time.time() - t1:.1f}s")
    return {
        "data_sha256": sha256_file(target_path),
        "split_id": sc_mod.split_id([*scenarios, *target]),
        "seeds": seeds,
        "backend": rows[0]["backend"],
        "eval_split": sc_mod.HOLDOUT,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", choices=("kinematic", "metadrive"), required=True)
    ap.add_argument("--stage", choices=("dev", "final", "sensitivity", "holdout"), required=True)
    ap.add_argument("--suffix", default="", help="names a rerun; a stated reason is required")
    ap.add_argument("--run-root", default="runs")
    args = ap.parse_args(argv)
    run_dir = Path(args.run_root) / f"{args.world}-{args.stage}{args.suffix}"
    frozen = args.stage != "dev"
    if frozen and is_dirty():
        print("refusing a final run on a dirty tree: commit first", file=sys.stderr)
        return 2
    if run_dir.exists():
        if frozen:
            print(f"refusing to overwrite final results in {run_dir}", file=sys.stderr)
            return 2
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    all_cfg = json.loads(CONFIG.read_text())
    cfg = all_cfg[args.world]
    started = started_at()
    lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    fields = dict(
        data_sha256="0" * 64, split_id="unknown", seeds=[], backend=args.world, eval_split="unknown"
    )
    exit_code, status = 1, "FAILED"
    try:
        job = {"sensitivity": sensitivity, "holdout": holdout}.get(args.stage)
        fields.update(
            job(args.world, run_dir, cfg, log)
            if job
            else run(args.world, args.stage, run_dir, cfg, log)
        )
        exit_code, status = 0, "SUCCEEDED"
    except Exception as e:  # noqa: BLE001
        log(f"run failed: {type(e).__name__}: {e}")
        raise
    finally:
        (run_dir / "stdout.log").write_text("\n".join(lines) + "\n")
        write_manifest(
            run_dir,
            kind="study" if frozen else "pilot",
            status=status,
            exit_code=exit_code,
            code_revision=code_revision(),
            config_sha256=sha256_file(M2_CONFIG if args.stage == "holdout" else CONFIG),
            started_at=started,
            limitations=[
                "Toy linear and quadratic ridge policies, not a production planner.",
                "A simulator result is not a real-world or fleet safety claim.",
                "kinematic_fixture is a toy fixture; metadrive results are one simulator only.",
            ],
            **fields,
        )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
