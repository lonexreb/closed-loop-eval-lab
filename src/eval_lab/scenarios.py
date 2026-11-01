"""Frozen scenario manifest. Families, not just scenarios, are disjoint across splits.

Random per-scenario splits would leak: scenarios from one family share a road shape and hazard
type, so a model tuned on some would be tested on near-copies. Whole families are held out instead.
"""

from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

TRAIN, DEV, TEST, HOLDOUT = "train", "dev", "test", "holdout"


@dataclass(frozen=True)
class ActorSpec:
    s0_m: float
    v0_mps: float
    lateral_m: float = 0.0
    brake_at_s: float | None = None  # sim time the actor starts braking
    brake_mps2: float = 0.0
    v_after_mps: float = 0.0
    cut_in_at_s: float | None = None  # sim time the actor starts moving to the ego lane


@dataclass(frozen=True)
class Scenario:
    id: str
    family: str
    split: str
    segments: tuple[tuple[float, float], ...]  # (length m, curvature 1/m)
    half_width_m: float
    speed_limit_mps: float
    actors: tuple[ActorSpec, ...] = ()

    @property
    def length_m(self) -> float:
        return float(sum(seg[0] for seg in self.segments))


def _uniform(r: np.random.Generator, lo: float, hi: float) -> float:
    return round(float(r.uniform(lo, hi)), 3)


def _lead(r: np.random.Generator, brake: bool, **kw) -> ActorSpec:
    return ActorSpec(
        s0_m=_uniform(r, 30, 60),
        v0_mps=_uniform(r, 7, 11),
        brake_at_s=_uniform(r, 4, 9) if brake else None,
        brake_mps2=_uniform(r, 2, 5) if brake else 0.0,
        v_after_mps=_uniform(r, 0, 4) if brake else 0.0,
        **kw,
    )


def straight_cruise(r):
    return ((_uniform(r, 200, 300), 0.0),), 3.5, _uniform(r, 10, 15), ()


def gentle_curve(r):
    k = _uniform(r, 0.003, 0.012) * float(r.choice([-1, 1]))
    return ((40.0, 0.0), (_uniform(r, 80, 140), k), (60.0, 0.0)), 3.5, _uniform(r, 10, 15), ()


def lead_follow(r):
    return ((_uniform(r, 220, 300), 0.0),), 3.5, _uniform(r, 11, 15), (_lead(r, False),)


def s_curve(r):
    k = _uniform(r, 0.012, 0.022)
    n = _uniform(r, 50, 80)
    return ((30.0, 0.0), (n, k), (n, -k), (40.0, 0.0)), 3.0, _uniform(r, 9, 13), ()


def lead_brake(r):
    return ((_uniform(r, 220, 300), 0.0),), 3.5, _uniform(r, 11, 15), (_lead(r, True),)


def tight_curve(r):
    k = _uniform(r, 0.02, 0.04) * float(r.choice([-1, 1]))
    return ((30.0, 0.0), (_uniform(r, 50, 90), k), (60.0, 0.0)), 3.0, _uniform(r, 9, 13), ()


def narrow_s_curve(r):
    k = _uniform(r, 0.015, 0.028)
    n = _uniform(r, 45, 70)
    return ((30.0, 0.0), (n, -k), (n, k), (40.0, 0.0)), 2.2, _uniform(r, 9, 12), ()


def curve_lead_brake(r):
    k = _uniform(r, 0.006, 0.018) * float(r.choice([-1, 1]))
    return ((40.0, 0.0), (160.0, k), (60.0, 0.0)), 3.2, _uniform(r, 10, 14), (_lead(r, True),)


def cut_in(r):
    actor = ActorSpec(
        s0_m=_uniform(r, 25, 45),
        v0_mps=_uniform(r, 6, 10),
        lateral_m=float(r.choice([-1, 1])) * 3.2,
        cut_in_at_s=_uniform(r, 2, 6),
    )
    return ((_uniform(r, 220, 300), 0.0),), 5.0, _uniform(r, 11, 15), (actor,)


FAMILIES = {
    TRAIN: (straight_cruise, gentle_curve, lead_follow),
    DEV: (s_curve, lead_brake),
    TEST: (tight_curve, narrow_s_curve, curve_lead_brake, cut_in),
}


def build_manifest(per_family: int, seed: int) -> list[Scenario]:
    out = []
    for split, families in FAMILIES.items():
        for fn in families:
            for i in range(per_family):
                r = np.random.default_rng([seed, zlib.crc32(fn.__name__.encode()), i])
                segments, half_width, limit, actors = fn(r)
                sid = f"{fn.__name__}-{i:03d}"
                out.append(Scenario(sid, fn.__name__, split, segments, half_width, limit, actors))
    return out


MD_FAMILIES = {TRAIN: ("md_SCS", "md_CSC"), DEV: ("md_CCS",), TEST: ("md_SCCS", "md_CSCC")}


def build_metadrive_manifest(per_family: int, speed_limit_mps: float = 10.0) -> list[Scenario]:
    """A family is a block sequence. The numeric id suffix is the MetaDrive map seed.

    Seed ranges must not repeat across families. MetaDrive draws blocks from one stream per seed,
    so a shared seed makes CSCC replay the roads of CSC, silently breaking the held-out split.
    """
    ordered = [(split, f) for split, families in MD_FAMILIES.items() for f in families]
    return [
        Scenario(f"{family}-{n * 100 + i:03d}", family, split, (), 2.0, speed_limit_mps)
        for n, (split, family) in enumerate(ordered)
        for i in range(per_family)
    ]


def wide_cruise(r):
    return ((_uniform(r, 200, 300), 0.0),), 4.0, _uniform(r, 9, 13), ()


def wide_gentle(r):
    k = _uniform(r, 0.002, 0.006) * float(r.choice([-1, 1]))
    return ((40.0, 0.0), (_uniform(r, 100, 160), k), (60.0, 0.0)), 4.0, _uniform(r, 9, 13), ()


def narrow_lead_brake(r):
    return ((_uniform(r, 220, 300), 0.0),), 2.2, _uniform(r, 10, 13), (_lead(r, True),)


def hairpin(r):
    k = _uniform(r, 0.03, 0.05) * float(r.choice([-1, 1]))
    return ((30.0, 0.0), (_uniform(r, 60, 90), k), (60.0, 0.0)), 2.6, _uniform(r, 8, 11), ()


def chicane(r):
    k = _uniform(r, 0.02, 0.035)
    n = _uniform(r, 25, 40)
    return ((30.0, 0.0), (n, k), (n, -k), (n, k), (n, -k), (40.0, 0.0)), 2.6, _uniform(r, 8, 11), ()


def curve_cut_in(r):
    k = _uniform(r, 0.005, 0.015) * float(r.choice([-1, 1]))
    actor = ActorSpec(
        s0_m=_uniform(r, 25, 45),
        v0_mps=_uniform(r, 6, 10),
        lateral_m=float(r.choice([-1, 1])) * 3.2,
        cut_in_at_s=_uniform(r, 2, 6),
    )
    return ((40.0, 0.0), (160.0, k), (60.0, 0.0)), 5.0, _uniform(r, 10, 13), (actor,)


def close_cut_in(r):
    actor = ActorSpec(
        s0_m=_uniform(r, 24, 36),
        v0_mps=_uniform(r, 6, 9),
        lateral_m=float(r.choice([-1, 1])) * 3.2,
        cut_in_at_s=_uniform(r, 2.5, 5),
    )
    return ((_uniform(r, 220, 300), 0.0),), 5.0, _uniform(r, 10, 13), (actor,)


def slow_lead_curve(r):
    k = _uniform(r, 0.01, 0.025) * float(r.choice([-1, 1]))
    actor = ActorSpec(s0_m=_uniform(r, 30, 60), v0_mps=_uniform(r, 5, 8))
    return ((40.0, 0.0), (160.0, k), (60.0, 0.0)), 3.2, _uniform(r, 10, 13), (actor,)


# M2: families no pilot run ever touched. Two are deliberately benign, because a real pool is mostly
# easy and a miner earns its keep by not spending its budget there.
M2_FAMILIES = (
    wide_cruise,
    wide_gentle,
    narrow_lead_brake,
    hairpin,
    chicane,
    curve_cut_in,
    close_cut_in,
    slow_lead_curve,
)
# Chosen by geometry alone: the eight block sequences with the least overlap with training roads,
# among those the expert finishes on spare seeds. Maps opening with S share the training start.
MD_M2_FAMILIES = (
    "md_OCC",  # O roundabout
    "md_OCO",
    "md_OXC",  # X intersection
    "md_OCS",
    "md_OTC",  # T junction
    "md_XCC",
    "md_RCC",  # R off-ramp
    "md_XTC",
)
MD_M2_BASE_SEED = 500  # above every pilot map seed. Each family owns a 40-seed range.


def build_m2_manifest(per_family: int, seed: int) -> list[Scenario]:
    out = []
    for fn in M2_FAMILIES:
        for i in range(per_family):
            r = np.random.default_rng([seed, zlib.crc32(fn.__name__.encode()), i])
            segments, half_width, limit, actors = fn(r)
            sid = f"{fn.__name__}-{i:03d}"
            out.append(Scenario(sid, fn.__name__, HOLDOUT, segments, half_width, limit, actors))
    return out


def build_metadrive_m2_manifest(per_family: int, speed_limit_mps: float = 10.0) -> list[Scenario]:
    if per_family > 40:
        raise ValueError("seed ranges would overlap")
    return [
        Scenario(
            f"{family}-{MD_M2_BASE_SEED + 40 * n + i:03d}",
            family,
            HOLDOUT,
            (),
            2.0,
            speed_limit_mps,
        )
        for n, family in enumerate(MD_M2_FAMILIES)
        for i in range(per_family)
    ]


def manifest_hash(scenarios: list[Scenario]) -> str:
    return hashlib.sha256(_canonical(scenarios).encode()).hexdigest()


def split_id(scenarios: list[Scenario]) -> str:
    """Identifies the exact split membership, and changes if any scenario moves or changes."""
    return f"split-{manifest_hash(scenarios)[:12]}"


def _canonical(scenarios: list[Scenario]) -> str:
    return json.dumps([asdict(s) for s in scenarios], sort_keys=True, separators=(",", ":"))


def write_manifest(scenarios: list[Scenario], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(scenarios) + "\n")


def read_manifest(path: Path) -> list[Scenario]:
    rows = json.loads(path.read_text())
    return [
        Scenario(
            d["id"],
            d["family"],
            d["split"],
            tuple(tuple(seg) for seg in d["segments"]),
            d["half_width_m"],
            d["speed_limit_mps"],
            tuple(ActorSpec(**a) for a in d["actors"]),
        )
        for d in rows
    ]


def assert_disjoint(scenarios: list[Scenario]) -> None:
    """Fails loudly if a family or scenario id spans two splits."""
    seen: dict[str, str] = {}
    ids = [s.id for s in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate scenario id")
    for s in scenarios:
        if seen.setdefault(s.family, s.split) != s.split:
            raise ValueError(f"family {s.family} is in two splits")
