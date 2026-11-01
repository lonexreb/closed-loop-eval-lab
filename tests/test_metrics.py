"""Analytic metric behavior: answers known by hand, and refusal on data that cannot be judged."""

from dataclasses import replace

import numpy as np
import pytest

from eval_lab.analytic import cases, drive_trace, straight
from eval_lab.metrics import (
    COLLISION,
    FAILED,
    OFFROAD,
    TIMEOUT,
    UNASSESSABLE,
    Box,
    ade,
    aggregate,
    boxes_overlap,
    failed_episode,
    fde,
    score_episode,
    station_lateral,
)
from eval_lab.schemas import Pose, Status, Trajectory, to_ego, to_world, wrap_angle


@pytest.mark.parametrize("case,metric,want,got", cases(), ids=lambda v: str(v)[:24])
def test_hand_calculations(case, metric, want, got):
    assert got == pytest.approx(want) if isinstance(want, float) else got == want


# Units and frames


def test_frame_round_trip_and_known_rotation():
    t = np.array([0.0, 1.0])
    world = Trajectory(t, np.array([[1.0, 2.0], [1.0, 3.0]]), np.array([np.pi / 2, np.pi / 2]))
    pose = Pose(1.0, 2.0, np.pi / 2)
    ego = to_ego(world, pose, "a")
    # A point 1 m along world +y is 1 m straight ahead of a vehicle facing +y.
    np.testing.assert_allclose(ego.xy_m, [[0, 0], [1, 0]], atol=1e-12)
    np.testing.assert_allclose(ego.yaw_rad, [0, 0], atol=1e-12)
    back = to_world(ego, pose)
    np.testing.assert_allclose(back.xy_m, world.xy_m)
    np.testing.assert_allclose(back.yaw_rad, world.yaw_rad)


def test_comparing_across_frames_is_refused_even_when_numbers_match():
    gt = straight()
    pred = to_ego(gt, Pose(0, 0, 0), "a")  # identity transform: the numbers are equal
    np.testing.assert_allclose(pred.xy_m, gt.xy_m)
    assert ade(pred, gt, 2.5).status is Status.UNASSESSABLE


def test_wrap_angle_range():
    a = np.linspace(-20, 20, 401)
    w = wrap_angle(a)
    assert (w > -np.pi).all() and (w <= np.pi).all()
    np.testing.assert_allclose(np.cos(w), np.cos(a))
    assert wrap_angle(-np.pi) == pytest.approx(np.pi)


def test_rejects_unknown_frame_and_bad_shapes():
    with pytest.raises(ValueError):
        Trajectory(np.zeros(2), np.zeros((2, 2)), np.zeros(2), frame="map")
    with pytest.raises(ValueError):
        Trajectory(np.zeros(2), np.zeros((3, 2)), np.zeros(2))


# Open loop: refusal


@pytest.mark.parametrize("metric", [ade, fde])
def test_masked_sample_in_prediction_or_truth_is_never_a_score(metric):
    gt, bad = straight(), straight()
    mask = np.ones(6, bool)
    mask[3] = False
    bad = Trajectory(gt.t_s, gt.xy_m, gt.yaw_rad, valid=mask)
    assert metric(bad, gt, 2.5).status is Status.UNASSESSABLE
    assert metric(gt, bad, 2.5).status is Status.UNASSESSABLE


def test_nan_position_and_nan_time_are_unassessable():
    gt = straight()
    xy = gt.xy_m.copy()
    xy[2, 0] = np.nan
    assert ade(Trajectory(gt.t_s, xy, gt.yaw_rad), gt, 2.5).status is Status.UNASSESSABLE
    t = gt.t_s.copy()
    t[1] = np.nan
    r = ade(Trajectory(t, gt.xy_m, gt.yaw_rad), gt, 2.5)
    assert r.status is Status.UNASSESSABLE and r.value is None


def test_duplicate_timestamp_is_unassessable():
    gt = straight()
    t = gt.t_s.copy()
    t[2] = t[1]
    assert ade(Trajectory(t, gt.xy_m, gt.yaw_rad), gt, 2.5).status is Status.UNASSESSABLE


def test_a_clock_shared_by_both_sides_is_still_rejected_when_disordered():
    gt = straight()
    t = gt.t_s.copy()
    t[2], t[3] = t[3], t[2]  # identical on both sides, so an alignment check alone would pass
    bad = Trajectory(t, gt.xy_m, gt.yaw_rad)
    assert ade(bad, bad, 2.5).status is Status.UNASSESSABLE


def test_truncated_truth_cannot_hide_late_error():
    pred, gt = straight(11.0), straight()
    short = Trajectory(gt.t_s[:3], gt.xy_m[:3], gt.yaw_rad[:3])
    assert ade(pred, short, 2.5).status is Status.UNASSESSABLE
    assert ade(pred, short, 1.0).value == pytest.approx(1.0)  # (0.5+1+1.5)/3, an honest window


def test_mask_outside_the_window_does_not_block_it():
    gt = straight()
    mask = np.ones(6, bool)
    mask[5] = False
    masked = Trajectory(gt.t_s, gt.xy_m, gt.yaw_rad, valid=mask)
    assert ade(masked, gt, 2.0).ok
    assert not ade(masked, gt, 2.5).ok


def test_denominator_is_the_sample_count():
    r = ade(straight(lateral=2.0), straight(), 2.5)
    assert (r.numerator, r.denominator, r.unit) == (12.0, 6.0, "m")


# Geometry


def test_boxes_overlap_cases():
    a = Box(0, 0, 0, 4, 2)
    assert boxes_overlap(a, Box(3.9, 0, 0, 4, 2))
    assert not boxes_overlap(a, Box(4.0, 0, 0, 4, 2))  # exactly touching
    assert not boxes_overlap(a, Box(4.1, 0, 0, 4, 2))
    # A 2x2 square rotated 45 deg has half-diagonal sqrt(2) along x; `a` ends at x=2.
    # Overlap needs centre x < 2 + 1.414 = 3.414.
    assert boxes_overlap(a, Box(3.4, 0, np.pi / 4, 2, 2))
    assert not boxes_overlap(a, Box(3.5, 0, np.pi / 4, 2, 2))


def test_bounding_circles_can_intersect_while_boxes_miss():
    # Centre distance 2.687 is below the summed radii 2.828, so a circle check would call this a
    # hit. Along the diagonal: 1.414 (first corner) + 1.0 (rotated flat side) < 2.687.
    assert np.hypot(1.9, 1.9) < 2 * np.sqrt(2)
    assert not boxes_overlap(Box(0, 0, 0, 2, 2), Box(1.9, 1.9, np.pi / 4, 2, 2))


def test_station_and_signed_lateral():
    route = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    s, lat = station_lateral(route, np.array([[5.0, 1.0], [5.0, -2.0], [11.0, 5.0]]))
    np.testing.assert_allclose(s, [5, 5, 15])
    np.testing.assert_allclose(lat, [1, -2, -1])


# Closed loop


def test_offroad_only_past_the_half_width():
    inside, outside = drive_trace(None), drive_trace(None)
    t = inside.ego.t_s
    for lateral, trace, want in ((3.4, inside, False), (3.6, outside, True)):
        xy = np.stack([10 * t, np.full_like(t, lateral)], 1)
        r = score_episode(replace(trace, ego=Trajectory(t, xy, np.zeros_like(t))))
        assert r.offroad is want
    assert r.outcome == OFFROAD and not r.safe_completed


def test_late_goal_is_a_timeout_not_a_completion():
    r = score_episode(replace(drive_trace(None), time_budget_s=5.0))  # 50 m < 90 m goal
    assert r.outcome == TIMEOUT and not r.completed


def test_collision_beats_completion_and_is_never_safe():
    r = score_episode(drive_trace(95.0))
    assert r.outcome == COLLISION and r.completed and not r.safe_completed


def test_missing_ego_telemetry_is_unassessable_not_a_pass():
    trace = drive_trace(None)
    xy = trace.ego.xy_m.copy()
    xy[10] = np.nan
    r = score_episode(replace(trace, ego=Trajectory(trace.ego.t_s, xy, trace.ego.yaw_rad)))
    assert r.outcome == UNASSESSABLE and r.collision is None and not r.safe_completed


def test_unknown_actor_cannot_be_called_collision_free_but_a_seen_hit_counts():
    trace = drive_trace(95.0)
    actors = trace.actors.copy()
    actors[3:] = np.nan
    r = score_episode(replace(trace, actors=actors))
    assert r.outcome == UNASSESSABLE and r.collision is None and not r.safe_completed

    seen = drive_trace(50.0)
    actors = seen.actors.copy()
    actors[3] = np.nan  # blip well before the hit at step 46
    r = score_episode(replace(seen, actors=actors))
    assert r.outcome == COLLISION and r.collision is True


def test_comfort_on_known_motion():
    t = np.arange(50) * 0.1
    v = 0.5 * 2.0 * t**2  # jerk 2 m/s^3 by construction
    s = np.cumsum(v) * 0.1
    trace = replace(
        drive_trace(None, steps=50),
        ego=Trajectory(t, np.stack([s, 0 * t], 1), 0 * t),
        speed_mps=v,
    )
    r = score_episode(trace)
    assert r.peak_jerk == pytest.approx(2.0)
    assert r.peak_long_accel == pytest.approx(9.7, abs=0.05)  # about 2*t at t=4.9
    assert r.peak_lat_accel == pytest.approx(0.0)


def test_comfort_refuses_a_non_uniform_clock():
    trace = drive_trace(None)
    t = trace.ego.t_s.copy()
    t[50:] += 0.05
    r = score_episode(replace(trace, ego=Trajectory(t, trace.ego.xy_m, trace.ego.yaw_rad)))
    assert r.peak_jerk is None


# Denominators


def test_timeouts_failures_and_unassessable_stay_in_denominators():
    ok = score_episode(drive_trace(None))
    hit = score_episode(drive_trace(50.0))
    stuck = score_episode(drive_trace(None, speed=0.0))
    crashed = failed_episode("segfault")
    a = aggregate([ok, hit, stuck, crashed])
    assert a["planned"] == 4
    assert a["safe_completion_rate"] == 0.25
    assert a["collisions_per_episode"] == 0.25
    assert a["worst_case_collisions_per_episode"] == 0.5  # the crashed run is unknown
    assert a["outcomes"][FAILED] == 1 and a["outcomes"][TIMEOUT] == 1


def test_a_policy_that_never_moves_cannot_look_superior():
    still = aggregate([score_episode(drive_trace(None, speed=0.0))] * 5)
    assert still["collisions_per_episode"] == 0.0  # the misleading number
    assert still["collisions_per_km"] is None  # zero exposure is not a zero rate
    assert still["safe_completion_rate"] == 0.0  # and the primary endpoint is not fooled


def test_empty_aggregate_is_unassessable():
    assert aggregate([])["status"] == "UNASSESSABLE"
