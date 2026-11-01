"""Paired rollouts, expert logs and open-loop scoring. Every planned episode leaves one record."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from eval_lab.adapters.base import WorldState
from eval_lab.metrics import EpisodeScore, Trace, ade, failed_episode, fde, score_episode
from eval_lab.policies import (
    HISTORY_STEPS,
    PLAN_T,
    Policy,
    base_features,
    expert_action,
    history_features,
    to_ego_xy,
    track,
)
from eval_lab.schemas import Trajectory

STEPS_PER_WAYPOINT = 5  # 0.5 s between waypoints at DT 0.1 s


def rollout(
    sim, policy: Policy, scenario, seed: int, tracker=track
) -> tuple[list[WorldState], Trace]:
    """Closed loop. The first second is driven by the expert for every policy alike.

    Training rows need one second of history, so a history policy has never seen an empty one. Warm
    starting gives all policies a history from the state distribution they were built on.
    """
    st = sim.reset(scenario, seed)
    policy.reset()
    states = [st]
    while not sim.done(st):
        plan = policy.plan(st)  # also fills the policy's history during warm start
        warm = len(states) <= HISTORY_STEPS
        st = sim.step(expert_action(st) if warm else tracker(plan, st))
        states.append(st)
    return states, sim.trace(states)


def expert_rollout(sim, scenario, seed: int) -> list[WorldState]:
    st = sim.reset(scenario, seed)
    states = [st]
    while not sim.done(st):
        st = sim.step(expert_action(st))
        states.append(st)
    return states


def run_episode(
    sim, policy: Policy, scenario, seed: int, tracker=track
) -> tuple[EpisodeScore, Trace | None]:
    """Never lets a crash disappear: it comes back as FAILED and stays in the denominator."""
    try:
        _, trace = rollout(sim, policy, scenario, seed, tracker)
    except Exception as e:  # noqa: BLE001
        return failed_episode(f"{type(e).__name__}: {e}"), None
    score = score_episode(trace)
    score.extra = dict(getattr(sim, "flags", {}))  # the simulator's own verdict, for cross-checking
    return score, trace


def log_samples(states: list[WorldState]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Supervised rows from one log: base features, history features, future waypoints."""
    need = STEPS_PER_WAYPOINT * len(PLAN_T)
    past: deque = deque(maxlen=HISTORY_STEPS + 1)
    xb, xh, ys = [], [], []
    for i, st in enumerate(states):
        past.append(st)
        if i < HISTORY_STEPS or i + need >= len(states):
            continue
        future = np.array(
            [
                [states[i + k * STEPS_PER_WAYPOINT].x, states[i + k * STEPS_PER_WAYPOINT].y]
                for k in range(1, len(PLAN_T) + 1)
            ]
        )
        xb.append(base_features(st))
        xh.append(history_features(past, st))
        ys.append(to_ego_xy(st, future))
    if not ys:
        return np.zeros((0, 16)), np.zeros((0, 6)), np.zeros((0, len(PLAN_T), 2))
    return np.array(xb), np.array(xh), np.array(ys)


def score_log(policy: Policy, states: list[WorldState], stride: int = STEPS_PER_WAYPOINT) -> dict:
    """ADE/FDE@3s of one policy over one expert log.

    The policy replays the log from its start, so a stateful policy sees the logged history it would
    have seen. This is open loop by definition: its plan never affects the next state.
    """
    policy.reset()
    need = STEPS_PER_WAYPOINT * len(PLAN_T)
    ades: list[float] = []
    fdes: list[float] = []
    refused = 0
    for i, st in enumerate(states):
        plan = policy.plan(st)
        if i < HISTORY_STEPS or (i - HISTORY_STEPS) % stride or i + need >= len(states):
            continue
        idx = [i + k * STEPS_PER_WAYPOINT for k in range(1, len(PLAN_T) + 1)]
        truth = to_ego_xy(st, np.array([[states[j].x, states[j].y] for j in idx]))
        frame = f"ego:{i}"
        gt = Trajectory(PLAN_T, truth, np.zeros(len(PLAN_T)), frame)
        pred = Trajectory(PLAN_T, plan, np.zeros(len(PLAN_T)), frame)
        a, f = ade(pred, gt, 2.5), fde(pred, gt, 2.5)
        if a.value is None or f.value is None:
            refused += 1
            continue
        ades.append(a.value)
        fdes.append(f.value)
    return {
        "samples": len(ades),
        "unassessable": refused,
        "ade_m": float(np.mean(ades)) if ades else None,
        "fde_m": float(np.mean(fdes)) if fdes else None,
    }


@dataclass
class EpisodeRow:
    """One line of raw evidence. A table cell can always be traced to these rows."""

    policy: str
    scenario: str
    family: str
    split: str
    seed: int
    backend: str
    score: dict


def row(policy: str, sc, seed: int, backend: str, score: EpisodeScore) -> dict:
    return asdict(EpisodeRow(policy, sc.id, sc.family, sc.split, seed, backend, asdict(score)))
