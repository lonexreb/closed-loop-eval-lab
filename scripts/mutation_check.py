"""Break each safety rule on purpose and confirm the tests notice. Restores every file afterwards.

A green suite proves little if the tests cannot fail. Exit 1 if any mutant survives.
"""

import subprocess
import sys
from pathlib import Path

M = "src/eval_lab/metrics/closed_loop.py"
OL = "src/eval_lab/metrics/open_loop.py"
ST = "src/eval_lab/statistics.py"
K = "src/eval_lab/adapters/kinematic.py"
R = "src/eval_lab/runner.py"
SC = "src/eval_lab/scenarios.py"
MI = "src/eval_lab/mining.py"
E = "src/eval_lab/experiment.py"
P = "src/eval_lab/policies.py"
S = "src/eval_lab/schemas.py"

M2 = "src/eval_lab/m2.py"
SR = "src/eval_lab/study_report.py"

MUTANTS = [
    ("M2 dirty held-out run allowed", M2, "if frozen and is_dirty():", "if False:"),
    ("M2 gap reading ignores interval", M2, 'reading = "MINED_BETTER" if lo > 0 else', 'reading = "MINED_BETTER" if gap.mean() > 0 else'),
    ("M2 stale lead sees the present", M2, "return rule_plan(dataclasses.replace(st, gap_m=old.gap_m, lead_speed=old.lead_speed))", "return rule_plan(st)"),
    ("M2 short gap inverted", M2, "extra = (1 - self.keep) * (5.0 + 1.5 * st.speed)", "extra = -(1 - self.keep) * (5.0 + 1.5 * st.speed)"),
    ("M2 family seed ranges overlap", SC, "f\"{family}-{MD_M2_BASE_SEED + 40 * n + i:03d}\"", "f\"{family}-{MD_M2_BASE_SEED + i:03d}\""),
    ("M2 divergence off by one test", E, "far = np.linalg.norm(trace[:n, 1:3] - expert, axis=1) > limit_m", "far = np.linalg.norm(trace[:n, 1:3] - expert, axis=1) > 2 * limit_m"),
    ("M2 expert tracker loses speed loop", P, "return Action(float(np.clip(2.0 * (target_speed - st.speed), -6.0, 2.0)), steer)", "return track(plan, st)"),
    ("frame check removed", OL, "if pred.frame != gt.frame:", "if False:"),
    ("mask ignored inside horizon", OL, "if not (gt.mask[:n].all() and pred.mask[:n].all()):", "if False:"),
    ("truncated truth accepted", OL, "if gt.t_s[-1] < end - TIME_TOL_S:", "if False:"),
    ("timestamp alignment ignored", OL, "if len(pred.t_s) < n or np.abs(pred.t_s[:n] - gt.t_s[:n]).max() > TIME_TOL_S:", "if len(pred.t_s) < n:"),
    ("disordered time accepted", OL, "if (np.diff(gt.t_s) <= 0).any() or (np.diff(pred.t_s) <= 0).any():", "if False:"),
    ("FDE uses first sample", OL, "assessed(name, \"m\", float(err[-1]), 1)", "assessed(name, \"m\", float(err[0]), 1)"),
    ("unknown actor treated as clear", M, "collision: bool | None = True if collided else (None if unknown_actor else False)", "collision: bool | None = True if collided else False"),
    ("unknown actor can complete", M, "    elif unknown_actor:\n        outcome = UNASSESSABLE\n", ""),
    ("bad ego telemetry passes", M, "    if problems:\n        return EpisodeScore(", "    if False:\n        return EpisodeScore("),
    ("touching boxes collide", M, "if pa.max() <= pb.min() or pb.max() <= pa.min():", "if pa.max() < pb.min() or pb.max() < pa.min():"),
    ("second box axes skipped", M, "for yaw in (a.yaw, b.yaw):", "for yaw in (a.yaw,):"),
    ("off-road boundary inverted", M, "(np.abs(lat) > trace.half_width_m).any()", "(np.abs(lat) > trace.half_width_m * 2).any()"),
    ("late goal counts", M, "completed = bool((s[in_time] >= trace.goal_s_m).any())", "completed = bool((s >= trace.goal_s_m).any())"),
    ("safe ignores collision", M, "completed and collision is False and not offroad", "completed and not offroad"),
    ("denominator drops failures", M, "    n = len(rows)\n", "    n = max(1, sum(r.outcome != FAILED for r in rows))\n"),
    ("zero km gives zero rate", M, "collisions / km if km > 0 else None", "collisions / km if km > 0 else 0.0"),
    ("worst case ignores unknown", M, "(collisions + unknown) / n", "collisions / n"),
    ("bootstrap ignores clusters", ST, "    m = _scenario_means(values, clusters)\n    n = len(m)\n", "    m = values\n    n = len(m)\n"),
    ("zero events claims zero", ST, "float(sps.beta.ppf(1 - alpha / 2, 1, n))", "0.0"),
    ("paired diff unpaired", ST, "d = _scenario_means(a - b, clusters)", "d = _scenario_means(a, clusters) - np.random.default_rng(seed).permutation(_scenario_means(b, clusters))"),
    ("verdict sign flipped", ST, "if diff.lo > margin:\n        return \"IMPROVED\"", "if diff.hi > margin:\n        return \"IMPROVED\""),
    ("noise differs per reset", K, "rng = np.random.default_rng([zlib.crc32(sc.id.encode()), seed])", "rng = np.random.default_rng()"),
    ("steer rate unlimited", K, "change = np.clip(wanted - self.steer, -STEER_RATE_LIMIT * DT_S, STEER_RATE_LIMIT * DT_S)", "change = wanted - self.steer"),
    ("speed may go negative", K, "self.speed = max(0.0, self.speed + accel * DT_S)", "self.speed = self.speed + accel * DT_S"),
    ("no route runout", K, "route_polyline((*sc.segments, (RUNOUT_M, 0.0)))", "route_polyline(sc.segments)"),
    ("lead never resumes", K, "v = min(spec.v0_mps, v + RESUME_MPS2 * DT_S)", "v = v"),
    ("crash vanishes", R, "return failed_episode(f\"{type(e).__name__}: {e}\"), None", "raise"),
    ("no warm start", R, "warm = len(states) <= HISTORY_STEPS", "warm = False"),
    ("label from wrong frame", R, "ys.append(to_ego_xy(st, future))", "ys.append(future)"),
    ("family in two splits allowed", SC, "if seen.setdefault(s.family, s.split) != s.split:", "if False:"),
    ("mining may read test", MI, "if any(r[\"split\"] != DEV for r in dev_rows) or any(s.split != DEV for s in dev):", "if False:"),
    ("final overwrites", E, "if frozen:\n            print(f\"refusing to overwrite", "if False:\n            print(f\"refusing to overwrite"),
    ("final allowed on dirty tree", E, "if frozen and is_dirty():", "if False:"),
    ("stale manifest accepted", E, "if sc_mod.manifest_hash(frozen) != sc_mod.manifest_hash(built):", "if False:"),
    ("ridge shrinks intercept", P, "reg[-1, -1] = 0.0  # never shrink the intercept", "pass"),
    ("map seeds shared across families", SC, "f\"{family}-{n * 100 + i:03d}\"", "f\"{family}-{i:03d}\""),
    ("rank bootstrap unpaired", ST, "c[:, pick].mean(1)", "c[:, rng.integers(0, c.shape[1], c.shape[1])].mean(1)"),
    ("world/ego rotation wrong", S, "xy = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)", "xy = np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], axis=1)"),
]


def main() -> int:
    survived = []
    only = sys.argv[1:]
    for name, path, old, new in MUTANTS:
        if only and name not in only:
            continue
        p = Path(path)
        original = p.read_text()
        if old not in original:
            print(f"STALE   {name}: pattern not found")
            survived.append(name)
            continue
        try:
            p.write_text(original.replace(old, new, 1))
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "tests", "-q", "-x", "--no-header", "-p", "no:cacheprovider",
                 "--ignore=tests/test_simulator.py"],
                capture_output=True,
                text=True,
                timeout=300,
            )
        finally:
            p.write_text(original)
        killed = r.returncode != 0
        print(f"{'killed ' if killed else 'SURVIVED'} {name}", flush=True)
        if not killed:
            survived.append(name)
    total = len(only) or len(MUTANTS)
    print(f"{total - len(survived)}/{total} mutants killed")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
