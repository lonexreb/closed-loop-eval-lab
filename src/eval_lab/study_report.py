"""Build reports/study.md from retained M2 run files. No clock and no hand-typed result.

python -m eval_lab.study_report --out reports/study.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from eval_lab.m2 import mined_minus_random
from eval_lab.manifest import sha256_file
from eval_lab.report import (
    LEARNED,
    Run,
    flags,
    fmt,
    noise,
    outcome_table,
    provenance,
    read_gz,
    slices,
    table,
)
from eval_lab.statistics import paired_diff_ci, rank_correlation_ci, verdict

BUDGET = 16


def ade_map(path: Path) -> dict:
    out: dict = {}
    for r in read_gz(path / "open_loop_disturbed.jsonl.gz"):
        out.setdefault((r["split"], r["policy"]), {})[r["scenario"]] = r["ade_m"]
    return out


def picks(dev: Run, sensitivity: Path) -> dict:
    """Selections use dev data only. Disturbed ADE comes from the existing sensitivity run."""
    learned = [p for p in dev.policies if p.startswith(LEARNED)]
    disturbed = ade_map(sensitivity)
    score = lambda p: float(  # noqa: E731
        np.mean([v for v in disturbed["dev", p].values() if v is not None])
    )
    return {
        "closed": max(learned, key=lambda p: dev.safe(p)[0].mean()),
        "clean_ade": min(learned, key=dev.mean_ade),
        "disturbed_ade": min(learned, key=score),
    }


def compare(run: Run, a: str, b: str) -> list:
    if a == b:
        return [f"`{a}`", f"`{b}`", "n/a", "n/a", "SAME_MODEL"]
    x, c = run.safe(a)
    y, _ = run.safe(b)
    ci = paired_diff_ci(x, y, c)
    return [
        f"`{a}`",
        f"`{b}`",
        f"{x.mean():.3f} vs {y.mean():.3f}",
        f"{ci.estimate:+.3f} [{ci.lo:+.3f}, {ci.hi:+.3f}]",
        verdict(ci),
    ]


def confirmatory(run: Run, pick: dict) -> list[str]:
    rows = [
        ["C1 clean ADE", *compare(run, pick["closed"], pick["clean_ade"])],
        ["C2 disturbed ADE", *compare(run, pick["closed"], pick["disturbed_ade"])],
    ]
    head = ["Test", "Closed pick", "Other pick", "Safe completion", "Diff [95% CI]", "Reading"]
    return table(head, rows)


def ranks(run: Run) -> list[str]:
    learned = [p for p in run.policies if p.startswith(LEARNED)]
    disturbed = ade_map(run.path)
    rows = []
    for label, source in (
        (
            "clean",
            {
                p: {r["scenario"]: r["ade_m"] for r in run.open_rows if r["policy"] == p}
                for p in learned
            },
        ),
        ("disturbed", {p: disturbed["holdout", p] for p in learned}),
    ):
        ids = [s for s in run.scenarios if all(source[p][s] is not None for p in learned)]
        idx = [run.scenarios.index(s) for s in ids]
        o = np.array([[source[p][s] for s in ids] for p in learned])
        c = np.array([run.safe_by_scenario(p)[idx] for p in learned])
        ci = rank_correlation_ci(o, c)
        reading = "agrees" if ci.hi < 0 else "inverted" if ci.lo > 0 else "no reliable agreement"
        rows.append([label, len(ids), f"{ci.estimate:+.3f} [{ci.lo:+.3f}, {ci.hi:+.3f}]", reading])
    return table(["ADE on", "Scenarios", "Spearman rho [95% CI]", "Reading"], rows)


def family_ci(run: Run, a: str, b: str, seed: int = 0) -> str:
    """Post-hoc. Resamples families, which is the right unit for claims about new families."""
    x, c = run.safe(a)
    y, _ = run.safe(b)
    fam = {r["scenario"]: r["family"] for r in run.rows}
    ci = paired_diff_ci(x, y, np.array([fam[i] for i in c]), seed)
    return (
        f"{ci.estimate:+.3f} [{ci.lo:+.3f}, {ci.hi:+.3f}], {ci.n_clusters} families, {verdict(ci)}"
    )


def sim_flag_safe(run: Run, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Safe completion if MetaDrive's own crash and out-of-road flags replaced the trace test."""
    rows = run.scored(name)
    ok = [
        r["score"]["completed"]
        and not r["score"]["extra"]["crash"]
        # The trace still decides arrival: runs stop 50 m before the simulator's own goal.
        and not r["score"]["extra"]["out_of_road"]
        for r in rows
    ]
    return np.array(ok, float), np.array([r["scenario"] for r in rows])


def sim_flag_compare(run: Run, a: str, b: str) -> str:
    x, c = sim_flag_safe(run, a)
    y, _ = sim_flag_safe(run, b)
    ci = paired_diff_ci(x, y, c)
    return (
        f"{x.mean():.3f} vs {y.mean():.3f}, {ci.estimate:+.3f} [{ci.lo:+.3f}, {ci.hi:+.3f}], "
        f"{verdict(ci)}"
    )


def leave_one_out(path: Path) -> list[str]:
    """Post-hoc. How much of M1 rests on a single regression?"""
    d = json.loads((path / "mining_results.json").read_text())
    rows = []
    for n in d["real"]:
        g = mined_minus_random(d["table"], [x for x in d["real"] if x != n], BUDGET)
        rows.append([f"`{n}`", f"{g['diff']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}]", g["verdict"]])
    return table(["Left out", "Mean gap [95% CI]", "Reading"], rows)


def posthoc(root: Path, k: Run, m: Run, k_pick: dict, m_pick: dict) -> list[str]:
    c, a, d = "closed", "clean_ade", "disturbed_ade"
    return [
        "Added after seeing the results, so these are exploratory. They test how far the "
        "pre-registered readings stretch and do not replace them.",
        "",
        "### Resampling families instead of scenarios",
        "",
        f"- Fixture C1: {family_ci(k, k_pick[c], k_pick[a])}.",
        f"- Fixture C2: {family_ci(k, k_pick[c], k_pick[d])}.",
        f"- MetaDrive C1: {family_ci(m, m_pick[c], m_pick[a])}.",
        f"- MetaDrive C2: {family_ci(m, m_pick[c], m_pick[d])}.",
        "",
        "### MetaDrive with the simulator's own off-road and crash flags",
        "",
        f"- C1: {sim_flag_compare(m, m_pick[c], m_pick[a])}.",
        f"- C2: {sim_flag_compare(m, m_pick[c], m_pick[d])}.",
        "",
        "### M1 at budget 16, leaving out one regression",
        "",
        *leave_one_out(root / "m2-mining"),
    ]


def mechanism(run: Run) -> list[str]:
    replay = read_gz(run.path / "episodes_expert_tracker.jsonl.gz")
    times = json.loads((run.path / "s5_divergence.json").read_text())
    rows = []
    control = 0.0
    for p in sorted({r["policy"] for r in replay}):
        if not (p.startswith(LEARNED) or p == "rule_center"):
            continue
        safe = float(np.mean([r["score"]["safe_completed"] for r in replay if r["policy"] == p]))
        control = safe if p == "rule_center" else control
        ts = [t["t_s"] for t in times if t["policy"] == p]
        hit = [t for t in ts if t is not None]
        rows.append(
            [
                f"`{p}`",
                fmt(safe),
                fmt(float(run.safe(p)[0].mean())),
                fmt(float(np.median(hit)) if hit else None, 1),
                f"{len(hit)}/{len(ts)}",
            ]
        )
    head = ["Policy", "Safe, replay", "Safe, disturbed", "Median s to 1 m", "Runs that reach 1 m"]
    state = "valid" if control >= 0.95 else "VOID"
    return [
        f"Control `rule_center` {control:.3f}, so this test is {state}.",
        "",
        *table(head, rows),
    ]


def named(names: list[str]) -> str:
    return ", ".join(f"`{n}`" for n in names) or "none"


def mean_of(entry: dict, names: list[str]) -> str:
    return fmt(float(np.mean([entry[n] for n in names]))) if names else "n/a"


def mining(path: Path) -> list[str]:
    d = json.loads((path / "mining_results.json").read_text())
    truth, tab, real = d["truth"], d["table"], d["real"]
    rows = [
        [
            f"`{n}`",
            t["dev_verdict"],
            f"{t['diff']:+.3f} [{t['lo']:+.3f}, {t['hi']:+.3f}]",
            t["verdict"],
            t["failing_scenarios"],
            fmt(t["seed_variance"]),
        ]
        for n, t in sorted(truth.items())
    ]
    out = [
        "### Which regressions were real",
        "",
        *table(
            [
                "Candidate",
                "Dev",
                "Held-out diff [95% CI]",
                "Held-out",
                "Worse scenarios",
                "Seed var",
            ],
            rows,
        ),
        "",
        f"Real regressions (dev and held-out): {len(real)}.",
        "",
        "### M1, mined minus random detection",
        "",
    ]
    gap = d["gap"]
    out += table(
        ["Budget", "Regressions", "Mean gap [95% CI]", "Reading"],
        [
            [
                b,
                g["regressions"],
                "n/a"
                if g["diff"] is None
                else f"{g['diff']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}]",
                g["verdict"] if g["regressions"] >= 3 else "TOO_FEW",
            ]
            for b, g in gap.items()
        ],
    )
    fp_names = [
        n
        for n, t in truth.items()
        if n not in real and n != "null_control" and t["verdict"] != "REGRESSED"
    ]
    misses = [
        n
        for n, t in truth.items()
        if t["dev_verdict"] != "REGRESSED" and t["verdict"] == "REGRESSED"
    ]
    lost = [
        n
        for n, t in truth.items()
        if t["dev_verdict"] == "REGRESSED" and t["verdict"] != "REGRESSED"
    ]
    rows = []
    for t in tab:
        width = fmt(float(np.mean([t[f"{n}_ci_width"] for n in real]))) if real else "n/a"
        rows.append(
            [
                t["method"],
                t["budget"],
                mean_of(t, real),
                mean_of(t, misses),
                mean_of(t, fp_names),
                fmt(t["null_control"]),
                width,
                fmt(t["family_coverage"], 2),
                fmt(t["duplicate_fraction"]),
                t["distinct_packs"],
            ]
        )
    out += [
        "",
        "### All budgets",
        "",
        *table(
            [
                "Method",
                "Budget",
                "Detect real",
                "Detect dev-miss",
                "Flag non-regression",
                "A/A alarm",
                "CI width",
                "Family coverage",
                "Duplicates",
                "Distinct packs",
            ],
            rows,
        ),
        "",
        "### Per regression at budget 16",
        "",
    ]
    by = {(t["method"], t["budget"]): t for t in tab}
    out += table(
        ["Candidate", "Mined", "Random"],
        [
            [f"`{n}`", fmt(by["mined", BUDGET][n]), fmt(by["random", BUDGET][n])]
            for n in sorted(truth)
        ],
    )
    out += [
        "",
        f"Dev misses (passed dev, failed held-out): {named(misses)}.",
        f"Dev verdicts that did not hold on held-out families: {named(lost)}.",
    ]
    pack = json.loads((path / "regression_pack.json").read_text())
    fam: dict = {}
    for i in pack["ids"]:
        fam[i.rsplit("-", 1)[0]] = fam.get(i.rsplit("-", 1)[0], 0) + 1
    out += [
        "",
        f"Delivered pack: {len(pack['ids'])} scenarios, {pack['seeds']} seeds. Families: {fam}.",
    ]
    return out


def overlap(root: Path) -> list[str]:
    d = json.loads((root / "analysis/m2_feasibility.json").read_text())
    rows = [
        [f"`{k}`", v["mean"], v["max"], d["metadrive_expert_safe"][k]]
        for k, v in d["metadrive_overlap"].items()
    ]
    return table(["Family", "Mean overlap", "Max", "Expert on spare maps"], rows)


def build(root: Path) -> str:
    k_dev, m_dev = Run(root / "kinematic-dev"), Run(root / "metadrive-dev-v2")
    k, m = Run(root / "kinematic-holdout"), Run(root / "metadrive-holdout")
    k_pick = picks(k_dev, root / "kinematic-sensitivity")
    m_pick = picks(m_dev, root / "metadrive-sensitivity-v2")
    lines = [
        "# M2 study report",
        "",
        "Generated by `python -m eval_lab.study_report`. Plan: `experiments/protocol_m2.md`.",
        "Toy linear and quadratic policies in simulation. The fixture is a toy and MetaDrive is "
        "one simulator. Nothing here is a real-world or fleet result.",
        "",
        "## Confirmatory comparison",
        "",
        "### Fixture (TOY)",
        "",
        *confirmatory(k, k_pick),
        "",
        *ranks(k),
        "",
        "### MetaDrive (REAL SIMULATOR)",
        "",
        *confirmatory(m, m_pick),
        "",
        *ranks(m),
        "",
        "## Outcomes",
        "",
        "### Fixture",
        "",
        *outcome_table(k),
        "",
        "### MetaDrive",
        "",
        *outcome_table(m),
        "",
        "## Per family",
        "",
        "### Fixture",
        "",
        *slices(k),
        "",
        "### MetaDrive",
        "",
        *slices(m),
        "",
        "## Mechanism replay (S5)",
        "",
        "### Fixture",
        "",
        *mechanism(k),
        "",
        "### MetaDrive",
        "",
        *mechanism(m),
        "",
        "## Noise (S2)",
        "",
        "### Fixture",
        "",
        *noise(k),
        "",
        "### MetaDrive",
        "",
        *noise(m),
        "",
        "## MetaDrive flag cross-check (S3)",
        "",
        *flags(m),
        "",
        "## MetaDrive map overlap with training",
        "",
        "Share of a route within 2 m of its closest training route. Unrelated training maps give "
        "0.19 to 0.25. Geometry only.",
        "",
        *overlap(root),
        "",
        "## Regression study (fixture only)",
        "",
        *mining(root / "m2-mining"),
        "",
        "## Post-hoc checks",
        "",
        *posthoc(root, k, m, k_pick, m_pick),
        "",
        "## Provenance",
        "",
        *provenance([k, m]),
        "",
        f"Mining manifest sha256 `{sha256_file(root / 'm2-mining/manifest.json')}`.",
        f"Validation sha256 `{sha256_file(root / 'm2-validation/validation.json')}`.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--out", default="reports/study.md")
    args = ap.parse_args(argv)
    Path(args.out).write_text(build(Path(args.root)))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
