"""Real MetaDrive runs. Skipped only when the optional simulator is not installed."""

import numpy as np
import pytest

pytest.importorskip("metadrive")

from eval_lab.adapters.metadrive import MetaDriveSim  # noqa: E402
from eval_lab.metrics import COMPLETED, OFFROAD, TIMEOUT, Trace, score_episode  # noqa: E402
from eval_lab.policies import RuleCenter, Stationary  # noqa: E402
from eval_lab.runner import rollout, run_episode  # noqa: E402
from eval_lab.scenarios import build_metadrive_manifest  # noqa: E402

pytestmark = pytest.mark.simulator
SCENARIO = next(s for s in build_metadrive_manifest(3) if s.family == "md_SCS")


@pytest.fixture(scope="module")
def sim():
    s = MetaDriveSim()
    yield s
    s.close()


def test_real_simulator_episode_produces_an_assessable_trace(sim):
    score, trace = run_episode(sim, RuleCenter(), SCENARIO, 0)
    assert isinstance(trace, Trace)
    assert score.outcome == COMPLETED and score.distance_m > 100
    assert sim.backend == "metadrive-0.4.3"


def test_paired_seed_reproduces_within_numerical_noise(sim):
    """Measured drift is about 0.2 mm: float32 physics, not bit-exact. See DECISIONS.md."""
    first, second = (
        rollout(sim, RuleCenter(), SCENARIO, 2)[1],
        rollout(sim, RuleCenter(), SCENARIO, 2)[1],
    )
    assert first.ego.xy_m.shape == second.ego.xy_m.shape
    assert np.abs(first.ego.xy_m - second.ego.xy_m).max() < 0.01
    assert score_episode(first).outcome == score_episode(second).outcome


def test_goal_leaves_road_ahead_so_the_driver_does_not_brake_for_the_map_edge(sim):
    _, trace = rollout(sim, RuleCenter(), SCENARIO, 0)
    assert np.diff(trace.speed_mps).min() / 0.1 > -8.0


def test_stationary_still_cannot_win_in_the_real_simulator(sim):
    score, _ = run_episode(sim, Stationary(), SCENARIO, 0)
    assert score.outcome == TIMEOUT and not score.safe_completed


def test_route_points_are_one_metre_apart(sim):
    sim.reset(SCENARIO, 0)
    gaps = np.linalg.norm(np.diff(sim.route, axis=0), axis=1)
    assert gaps.min() > 0.5 and gaps.max() < 1.5


def test_leaving_the_road_is_off_road_not_a_timeout(sim):
    """MetaDrive's early stop once hid this: runs ended before our metric could see it."""

    class Veers(RuleCenter):
        def plan(self, st):
            plan = super().plan(st)
            plan[:, 1] += 3.0
            return plan

    score, trace = run_episode(sim, Veers(), SCENARIO, 0)
    assert score.outcome == OFFROAD and score.offroad and not score.safe_completed
    assert score.distance_m > 5.0 and trace is not None
