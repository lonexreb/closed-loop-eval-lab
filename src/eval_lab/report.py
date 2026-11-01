"""Build reports/pilot.md from retained run files. No clock and no hand-typed result.

  python -m eval_lab.report --out reports/pilot.md

Regenerating from the same runs gives byte-identical output, which a test enforces.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval_lab.manifest import sha256_file
from eval_lab.metrics import EpisodeScore, aggregate
from eval_lab.schemas import METRIC_VERSION
from eval_lab.statistics import (
    cluster_ci,
    naive_ci_halfwidth,
    paired_diff_ci,
    rank_correlation_ci,
    variance_components,
    verdict,
)

REPORT_VERSION = "1"
LEARNED = "ridge_"


def read_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


class Run:
    def __init__(self, path: Path):
        self.path = path
        self.manifest = json.loads((path / "manifest.json").read_text())
        self.rows = read_gz(path / "episodes.jsonl.gz")
        self.open_rows = read_gz(path / "open_loop.jsonl.gz")
        self.calm = read_gz(path / "episodes_no_disturbance.jsonl.gz")
        self.selection = json.loads((path / "model_selection.json").read_text())
        self.policies = sorted({r["policy"] for r in self.rows})
        self.scenarios = sorted({r["scenario"] for r in self.rows})
        self.seeds = sorted({r["seed"] for r in self.rows})

    def scored(self, name: str) -> list[dict]:
        rows = [r for r in self.rows if r["policy"] == name]
        return sorted(rows, key=lambda r: (r["scenario"], r["seed"]))

    def safe(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        rows = self.scored(name)
        return (
            np.array([float(r["score"]["safe_completed"]) for r in rows]),
            np.array([r["scenario"] for r in rows]),
        )

    def safe_by_scenario(self, name: str) -> np.ndarray:
        v, c = self.safe(name)
        return np.array([v[c == s].mean() for s in self.scenarios])

    def ade_by_scenario(self, name: str, ids: list[str]) -> np.ndarray:
        look = {r["scenario"]: r["ade_m"] for r in self.open_rows if r["policy"] == name}
        return np.array([look[i] for i in ids], float)

    def assessable_open_ids(self) -> list[str]:
        """Scenarios every policy has an open-loop score on. Others are listed, not guessed."""
        bad = {r["scenario"] for r in self.open_rows if r["ade_m"] is None}
        return [s for s in self.scenarios if s not in bad]

    def mean_ade(self, name: str) -> float:
        return float(self.ade_by_scenario(name, self.assessable_open_ids()).mean())


def scores(rows: list[dict]) -> list[EpisodeScore]:
    return [EpisodeScore(**r["score"]) for r in rows]


def fmt(v, digits: int = 3) -> str:
    return (
        "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{digits}f}"
    )


def table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def provenance(runs: list[Run]) -> list[str]:
    rows = []
    for r in runs:
        m = r.manifest
        rows.append(
            [
                m["run_id"],
                m["backend"],
                m["status"],
                m["exit_code"],
                f"`{m['code_revision'][:12]}`",
                f"`{m['split_id']}`",
                f"`{m['data_sha256'][:12]}`",
                f"`{m['config_sha256'][:12]}`",
            ]
        )
    head = ["Run", "Backend", "Status", "Exit", "Code", "Split", "Scenario hash", "Config hash"]
    return table(head, rows)


def outcome_table(run: Run) -> list[str]:
    rows = []
    order = sorted(run.policies, key=lambda p: run.mean_ade(p))
    for name in order:
        v, c = run.safe(name)
        a = aggregate(scores(run.scored(name)))
        ci = cluster_ci(v, c)
        o = a["outcomes"]
        jerk = [
            r["score"]["peak_jerk"] for r in run.scored(name) if r["score"]["peak_jerk"] is not None
        ]
        rows.append(
            [
                f"`{name}`",
                fmt(run.mean_ade(name), 2),
                f"{ci.estimate:.3f} [{ci.lo:.3f}, {ci.hi:.3f}]",
                fmt(a["collisions_per_episode"]),
                fmt(a["collisions_per_km"], 2),
                fmt(a["offroad_rate"]),
                o["TIMEOUT"],
                o["FAILED"] + o["UNASSESSABLE"],
                a["planned"],
                fmt(float(np.median(jerk)) if jerk else None, 1),
            ]
        )
    head = [
        "Policy (by ADE)",
        "ADE@2.5s m",
        "Safe completion [95% CI]",
        "Coll/ep",
        "Coll/km",
        "Off-road",
        "Timeouts",
        "Failed+unassessable",
        "Planned",
        "Median peak jerk m/s^3",
    ]
    return table(head, rows)


def p1(run: Run) -> tuple[list[str], dict]:
    ids = run.assessable_open_ids()
    idx = [run.scenarios.index(i) for i in ids]
    out, result = [], {}
    pools = {
        "Learned models": [p for p in run.policies if p.startswith(LEARNED)],
        "All candidates": run.policies,
    }
    rows = []
    for label, names in pools.items():
        o = np.array([run.ade_by_scenario(p, ids) for p in names])
        c = np.array([run.safe_by_scenario(p)[idx] for p in names])
        ci = rank_correlation_ci(o, c)
        reading = (
            "ADE order agrees with closed-loop order"
            if ci.hi < 0
            else "inverted: lower ADE went with worse driving"
            if ci.lo > 0
            else "no reliable agreement (CI includes 0)"
        )
        rows.append([label, len(names), f"{ci.estimate:.3f} [{ci.lo:.3f}, {ci.hi:.3f}]", reading])
        result[label] = ci
    out += table(["Pool", "Policies", "Spearman rho [95% CI]", "Reading"], rows)
    if len(ids) < len(run.scenarios):
        out += [
            "",
            f"Open-loop scores cover {len(ids)} of {len(run.scenarios)} scenarios. The rest had no "
            "expert log long enough, so they are excluded here rather than scored.",
        ]
    return out, result


def p2(dev: Run, final: Run) -> list[str]:
    learned = [p for p in dev.policies if p.startswith(LEARNED)]
    by_ade = min(learned, key=dev.mean_ade)
    by_closed = max(learned, key=lambda p: dev.safe(p)[0].mean())
    lines = [
        f"- Selected by best dev ADE: `{by_ade}` (dev ADE {dev.mean_ade(by_ade):.3f} m, dev safe "
        f"completion {dev.safe(by_ade)[0].mean():.3f}).",
        f"- Selected by best dev safe completion: `{by_closed}` (dev ADE "
        f"{dev.mean_ade(by_closed):.3f} m, dev safe completion {dev.safe(by_closed)[0].mean():.3f}).",
    ]
    chosen = [[(m["name"], m["lambda"]) for m in r.selection] for r in (dev, final)]
    if chosen[0] != chosen[1]:
        lines.append("- WARNING: dev and final runs did not fit identical models.")
    if by_ade == by_closed:
        return [*lines, "- Both rules chose the same model, so the comparison does not apply."]
    a, c = final.safe(by_closed)
    b, _ = final.safe(by_ade)
    ci = paired_diff_ci(a, b, c)
    lines += [
        f"- On held-out families: safe completion {a.mean():.3f} (closed-selected) vs {b.mean():.3f} "
        f"(ADE-selected).",
        f"- Paired difference {ci.estimate:+.3f}, 95% CI [{ci.lo:+.3f}, {ci.hi:+.3f}], {ci.n_clusters} "
        f"scenarios. **{verdict(ci)}**.",
    ]
    return lines


def slices(run: Run) -> list[str]:
    families = sorted({r["family"] for r in run.rows})
    rows = []
    for name in sorted(run.policies, key=run.mean_ade):
        cells = [f"`{name}`"]
        for fam in families:
            sub = [r for r in run.scored(name) if r["family"] == fam]
            cells.append(fmt(float(np.mean([r["score"]["safe_completed"] for r in sub]))))
        rows.append(cells)
    return table(["Policy", *families], rows)


def mechanism(run: Run) -> list[str]:
    seed = min(r["seed"] for r in run.calm)
    rows = []
    for name in sorted(p for p in run.policies if p.startswith(LEARNED)):
        on = [r["score"]["safe_completed"] for r in run.scored(name) if r["seed"] == seed]
        off = [r["score"]["safe_completed"] for r in run.calm if r["policy"] == name]
        rows.append([f"`{name}`", fmt(float(np.mean(on))), fmt(float(np.mean(off)))])
    stuck = [r[0] for r in rows if "hist_clean" in r[0] and float(r[2]) < 0.1]
    return [
        *table(
            ["Policy", f"Safe completion, disturbance on (seed {seed})", "Disturbance off"], rows
        ),
        "",
        f"**Reading: inconclusive.** History models that stay near zero with the disturbance off: {', '.join(stuck) or 'none'}. "
        "Removing the disturbance did not rescue them, so these runs do not support the idea that external "
        "disturbance alone pushes them off their training distribution. This control is also weak: the tracker "
        "is not the expert's controller, so even with disturbance off a policy does not reproduce logged states. "
        "The cause of these failures is not established here.",
    ]


def noise(run: Run) -> list[str]:
    rows = []
    for name in sorted(run.policies, key=run.mean_ade):
        v, c = run.safe(name)
        comp = variance_components(v, c)
        ci = cluster_ci(v, c)
        if comp["icc"] is None or comp["between"] + comp["within"] == 0:
            continue
        rows.append(
            [
                f"`{name}`",
                fmt(comp["between"], 4),
                fmt(comp["within"], 4),
                fmt(comp["icc"]),
                fmt(comp["design_effect"], 2),
                fmt(naive_ci_halfwidth(v), 3),
                fmt((ci.hi - ci.lo) / 2, 3),
            ]
        )
    head = [
        "Policy",
        "Between-scenario var",
        "Within-scenario (seed) var",
        "ICC",
        "Design effect",
        "Naive CI half-width",
        "Clustered CI half-width",
    ]
    dropped = len(run.policies) - len(rows)
    note = f"{dropped} policies with no variance at all are omitted."
    return [*table(head, rows), "", note]


def flags(run: Run) -> list[str]:
    counts: dict[tuple, int] = defaultdict(int)
    for r in run.rows:
        s = r["score"]
        if s["outcome"] in ("FAILED", "UNASSESSABLE"):
            continue
        x = s["extra"]
        counts["collision", bool(s["collision"]), bool(x.get("crash"))] += 1
        counts["off-road", bool(s["offroad"]), bool(x.get("out_of_road"))] += 1
    rows = []
    for kind in ("collision", "off-road"):
        rows.append(
            [
                kind,
                counts[kind, True, True],
                counts[kind, False, False],
                counts[kind, True, False],
                counts[kind, False, True],
            ]
        )
    head = ["Event", "Both yes", "Both no", "Trace yes, sim no", "Trace no, sim yes"]
    return [
        *table(head, rows),
        "",
        "Outcomes use the trace column. The simulator's crash-only cases ended early and keep their "
        "trace label, usually TIMEOUT, so collision rates above undercount by that amount. None count as safe completion.",
    ]


def side_analysis(path: Path) -> list[str]:
    if not path.exists():
        return []
    d = json.loads(path.read_text())
    sides = d["offroad_first_exit_side"]
    cells = ", ".join(f"{k.replace(':', ' ')}: {v}" for k, v in sorted(sides.items()))
    return [
        f"Post-hoc check of the off-road disagreement (`scripts/s3_offroad_side.py`). First exit side, by "
        f"whether the simulator agreed: {cells}. Left of a one-lane MetaDrive road is the oncoming lane, which "
        "MetaDrive still counts as on a lane, so our metric calls these off-road and it does not.",
        "",
    ]


def example(dev: Run, final: Run) -> list[str]:
    learned = [p for p in dev.policies if p.startswith(LEARNED)]
    low = min(learned, key=dev.mean_ade)
    best = max(learned, key=lambda p: dev.safe(p)[0].mean())
    if low == best:
        return ["No concrete example: both selection rules chose the same model."]
    look = {(r["policy"], r["scenario"], r["seed"]): r for r in final.rows}
    for sid in final.scenarios:
        for seed in final.seeds:
            a, b = look[low, sid, seed], look[best, sid, seed]
            if b["score"]["safe_completed"] and not a["score"]["safe_completed"]:
                ids = [sid]
                ade_a, ade_b = (
                    final.ade_by_scenario(low, ids)[0],
                    final.ade_by_scenario(best, ids)[0],
                )
                return [
                    f"First case in scenario order: `{sid}` seed {seed}. `{low}` has logged-future "
                    f"ADE {fmt(float(ade_a), 2)} m here and ends {a['score']['outcome']}. `{best}` has "
                    f"ADE {fmt(float(ade_b), 2)} m and ends {b['score']['outcome']}. Both faced the same "
                    "disturbance sequence. Its trace is in `traces.npz` under key "
                    f"`{low}|{sid}|{seed}`.",
                ]
    return ["No scenario where the ADE-selected model failed and the other succeeded."]


def p2_pair(final: Run, by_closed: str, by_ade: str) -> str:
    a, c = final.safe(by_closed)
    b, _ = final.safe(by_ade)
    ci = paired_diff_ci(a, b, c)
    return f"{ci.estimate:+.3f}, 95% CI [{ci.lo:+.3f}, {ci.hi:+.3f}], **{verdict(ci)}**"


def s4(path: Path, dev: Run, final: Run) -> list[str]:
    """Same models, ADE scored on disturbed expert logs. Post-hoc, from amendment 1."""
    rows = read_gz(path / "open_loop_disturbed.jsonl.gz")
    ade: dict = defaultdict(dict)
    for r in rows:
        ade[r["split"], r["policy"]][r["scenario"]] = r["ade_m"]
    learned = [p for p in dev.policies if p.startswith(LEARNED)]
    score = lambda p: float(np.mean([v for v in ade["dev", p].values() if v is not None]))  # noqa: E731
    by_ade = min(learned, key=score)
    by_closed = max(learned, key=lambda p: dev.safe(p)[0].mean())
    ids = [i for i in final.scenarios if all(ade["test", p][i] is not None for p in learned)]
    idx = [final.scenarios.index(i) for i in ids]
    o = np.array([[ade["test", p][i] for i in ids] for p in learned])
    c = np.array([final.safe_by_scenario(p)[idx] for p in learned])
    ci = rank_correlation_ci(o, c)
    return [
        f"Post-hoc (amendment 1). Learned-pool Spearman with disturbed-log ADE: {ci.estimate:.3f} "
        f"[{ci.lo:.3f}, {ci.hi:.3f}].",
        "",
        f"Best dev ADE on disturbed logs selects `{by_ade}`. Closed-selected minus this pick on held-out "
        + (
            "families: same model, so no comparison."
            if by_ade == by_closed
            else f"families: {p2_pair(final, by_closed, by_ade)}."
        ),
        "",
    ]


def world(
    title: str,
    boundary: str,
    dev: Run,
    final: Run,
    sens: Path | None = None,
    sides: Path | None = None,
) -> list[str]:
    p1_lines, _ = p1(final)
    planned = len(final.rows)
    lines = [
        f"## {title}",
        "",
        boundary,
        "",
        f"{len(final.policies)} policies, {len(final.scenarios)} held-out scenarios, seeds "
        f"{final.seeds}: {planned} planned episodes.",
        "",
        "### Outcomes on held-out families",
        "",
        *outcome_table(final),
        "",
        "Safe completion CIs resample scenarios. Timeouts, failures and unassessable runs are in every "
        "denominator.",
        "",
        "### P1: does ADE rank predict closed-loop rank?",
        "",
        *p1_lines,
        "",
        "### P2: selecting a model by ADE vs by closed-loop results",
        "",
        *p2(dev, final),
        "",
        "### Concrete discordant example",
        "",
        *example(dev, final),
        "",
        "### P3: slices by held-out family (safe completion)",
        "",
        *slices(final),
        "",
        "### S1: mechanism control",
        "",
        *mechanism(final),
        "",
        "### S2: evaluation noise",
        "",
        *noise(final),
        "",
    ]
    if sens is not None and sens.exists():
        lines += ["### S4: ADE scored on disturbed logs", "", *s4(sens, dev, final)]
    if final.rows[0]["backend"].startswith("metadrive"):
        lines += ["### S3: trace-derived outcomes vs simulator flags", "", *flags(final), ""]
        lines += side_analysis(sides) if sides else []
    return lines


def mining(path: Path) -> list[str]:
    data = json.loads((path / "mining_results.json").read_text())
    truth, tab = data["truth"], data["table"]
    names = sorted(k for k in truth if k != "null_control")
    real = [n for n in names if truth[n]["full_pool_regressed"]]
    fake = [n for n in names if not truth[n]["full_pool_regressed"]]
    lines = [
        "## Failure mining (fixture only)",
        "",
        "Packs are chosen from scenario definitions and dev failures only. Test outcomes only score "
        "the choice. Rates are over 200 draws per method and budget, with 3 seeds per scenario.",
        "",
        "Ground truth on the full test pool:",
        "",
    ]
    lines += table(
        ["Candidate", "Regressed on full pool", "Mean safe-completion change", "Worse scenarios"],
        [
            [
                f"`{n}`",
                truth[n]["full_pool_regressed"],
                f"{truth[n]['full_pool_diff']:+.3f}",
                truth[n]["failing_scenarios"],
            ]
            for n in [*names, "null_control"]
        ],
    )
    rows = []
    for e in sorted(tab, key=lambda r: (r["budget"], r["method"])):
        rows.append(
            [
                e["budget"],
                e["method"],
                *[fmt(e[n], 2) for n in real],
                *[fmt(e[n], 2) for n in fake],
                fmt(e["null_control"], 2),
                fmt(e["family_coverage"], 2),
                fmt(e["duplicate_fraction"], 3),
                fmt(float(np.mean([e[f"{n}_ci_width"] for n in names])), 3),
            ]
        )
    head = [
        "Budget",
        "Method",
        *[f"Detect `{n}`" for n in real],
        *[f"False flag `{n}`" for n in fake],
        "False alarm (A/A null)",
        "Family coverage",
        "Duplicate fraction",
        "Mean CI width",
    ]
    verdict_note = (
        f"Only {len(real)} of {len(names)} hand-built regressions really regressed on the full pool, so "
        "detection rests on that alone. For the others a flag is a false positive. The largest group of "
        "development failures was lead-vehicle collisions, the same behavior as the one real regression, so "
        "this setup favors the miner and this is not evidence that mining finds unrelated regressions. "
        "Family coverage shows the price: mined packs concentrate on fewer families at small budgets."
    )
    return [*lines, "", *table(head, rows), "", verdict_note, ""]


LIMITATIONS = [
    "Policies are small ridge regressions and do not represent real planners.",
    "Open-loop scores come from one expert log per scenario. Results for clean-trained models depend on that choice; S4 tests it.",
    "Confidence intervals cover scenarios inside these families. They do not cover new families, or which policies were in the pool. Eight learned policies is a small sample of policies.",
    "The kinematic fixture is hand-written, and its actors do not react. MetaDrive results use one simulator, one lane, light traffic and two held-out block families.",
    "MetaDrive is repeatable to about 0.2 mm, not bit-exact, so reruns can differ in late digits and a marginal outcome could flip.",
    "Peak jerk comes from differencing speed under an abrupt controller, so it reflects controller steps and is not a usable comfort measure here.",
    "Rule plans clamp at 40 m, inflating rule ADE on the fixture above about 13.3 m/s. Learned-pool analyses exclude rules.",
    "`traces.npz` files are retained locally and hashed in run manifests, but are not committed because of size.",
]


def build(root: Path) -> str:
    kd, kf = Run(root / "kinematic-dev"), Run(root / "kinematic-final")
    md, mf = Run(root / "metadrive-dev-v2"), Run(root / "metadrive-final-v2")
    runs = [kd, kf, md, mf]
    old_path = root / "metadrive-final"
    old = [
        "## Superseded run: metadrive-final",
        "",
        "`metadrive-final` reused map seeds across families, so test maps retraced training roads, and it "
        "cannot be called held-out. It stays on disk and is excluded from Result B. See "
        "`experiments/amendment_1.md`. For comparison, its P2 output was:",
        "",
    ]
    if old_path.exists() and (root / "metadrive-dev").exists():
        old += p2(Run(root / "metadrive-dev"), Run(old_path))
    overlap = root / "analysis" / "map_overlap.json"
    if overlap.exists():
        cells = [
            f"{k} ({v['kind']}): mean {v['mean']:.2f}, max {v['max']:.2f}"
            for k, v in sorted(json.loads(overlap.read_text()).items())
        ]
        old += [
            "",
            "After the fix, share of a route within 2 m of its most similar training route: "
            + "; ".join(cells)
            + ". Same-seed replay is gone, but overlap stays above the unrelated-map baseline because "
            "SCCS and CSCC extend the training sequences SCS and CSC. Read Result B as new maps with "
            "similar block order, not as novel road types.",
        ]
    lines = [
        "# Pilot report: closed-loop-eval-lab",
        "",
        f"Generated from retained run files by `src/eval_lab/report.py` (report version "
        f"{REPORT_VERSION}, metric version {METRIC_VERSION}). Every number below comes from those "
        "files. P1 to P3, S1 and S2 were fixed in `experiments/protocol.md` before the first final runs. "
        "`experiments/amendment_1.md` lists every later deviation.",
        "",
        "**Boundaries.** These are toy linear and quadratic policies on a kinematic fixture and on one "
        "open simulator, using CPU only. Nothing here measures a real vehicle or a "
        "production planner. A simulation pass is not a safety guarantee. Zero observed events gives "
        "an upper bound, not a zero rate.",
        "",
        "## Provenance",
        "",
        *provenance(runs),
        "",
        *world(
            "Result A: kinematic fixture (toy)",
            "**Boundary:** `kinematic_fixture`, a hand-written bicycle model with scripted actors. It "
            "is a controlled fixture and not a simulator.",
            kd,
            kf,
            root / "kinematic-sensitivity",
        ),
        *world(
            "Result B: MetaDrive (real open simulator)",
            "**Boundary:** MetaDrive 0.4.3, headless, on procedural maps. Models are trained on logs "
            "from a rule driver inside MetaDrive. This is simulator evidence only.",
            md,
            mf,
            root / "metadrive-sensitivity-v2",
            root / "analysis" / "s3_offroad_side.json",
        ),
        *old,
        "",
    ]
    mined = root / "mining-final-v2"
    if (mined / "mining_results.json").exists():
        lines += mining(mined)
    lines += ["## Limitations", "", *[f"- {x}" for x in LIMITATIONS], ""]
    names = (
        "kinematic-dev",
        "kinematic-final",
        "kinematic-sensitivity",
        "metadrive-dev-v2",
        "metadrive-final-v2",
        "metadrive-sensitivity-v2",
        "mining-final-v2",
        "analysis",
    )
    sources = [
        p
        for r in names
        for p in sorted((root / r).glob("*"))
        if p.is_file()
        and p.name not in ("stdout.log", "traces.npz")
        and not p.name.startswith("m2_")  # later study files are not sources of this report
    ]
    lines += ["## Source files of this report", ""]
    lines += table(
        ["File", "sha256"],
        [[f"`{p.relative_to(root)}`", f"`{sha256_file(p)[:16]}`"] for p in sources],
    )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="reports/pilot.md")
    args = ap.parse_args(argv)
    Path(args.out).write_text(build(Path(args.runs)))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
