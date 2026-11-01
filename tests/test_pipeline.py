"""Physics of the fixture, leakage boundaries, pairing, and the statistics we rely on."""

import json

import numpy as np
import pytest
from jsonschema import validate

from eval_lab import scenarios as S
from eval_lab.adapters.base import Action, WorldState, lead_on_route
from eval_lab.adapters.kinematic import WHEELBASE_M, KinematicSim, route_polyline
from eval_lab.candidates import rule_candidates
from eval_lab.manifest import write_manifest
from eval_lab.metrics import COLLISION, COMPLETED, OFFROAD, TIMEOUT, station_lateral
from eval_lab.mining import IgnoresLead, select_pack
from eval_lab.policies import (
    HISTORY_STEPS,
    ConstantVelocity,
    RuleCenter,
    Stationary,
    expand,
    fit_ridge,
    to_ego_xy,
    track,
)
from eval_lab.runner import expert_rollout, log_samples, rollout, run_episode, score_log
from eval_lab.statistics import (
    cluster_ci,
    naive_ci_halfwidth,
    paired_diff_ci,
    rank_correlation_ci,
    variance_components,
    verdict,
)

MANIFEST = S.build_manifest(4, 11)


def family(name):
    return next(s for s in MANIFEST if s.family == name)


# Geometry and physics


def test_route_of_constant_curvature_is_a_circle():
    k = 0.02
    pts, yaw = route_polyline(((200.0, k),))
    radius = np.hypot(pts[:, 0], pts[:, 1] - 1 / k)
    # A 1 m chord on a 50 m circle has sagitta 1/(8R) = 2.5 mm, the limit of any polyline.
    np.testing.assert_allclose(radius, 1 / k, atol=1 / (8 * (1 / k)))
    assert yaw[-1] == pytest.approx(4.0)


def test_route_points_are_one_metre_apart_so_index_equals_station():
    pts, _ = route_polyline(((50.0, 0.0), (50.0, 0.03), (20.0, -0.03)))
    np.testing.assert_allclose(np.linalg.norm(np.diff(pts, axis=0), axis=1), 1.0, atol=1e-3)


def test_bicycle_turn_radius_matches_the_model():
    sim = KinematicSim(0.0)
    sc = family("straight_cruise")
    sim.reset(sc, 0)
    sim.speed = 5.0
    steer = 0.2
    yaw0 = sim.yaw
    sim.steer = steer
    for _ in range(50):
        sim.step(Action(0.0, steer))
    # 50 steps at 5 m/s: arc 25 m, radius L/tan(delta)
    assert sim.yaw - yaw0 == pytest.approx(25.0 * np.tan(steer) / WHEELBASE_M, rel=1e-6)


def test_steering_rate_limit_is_enforced():
    sim = KinematicSim(0.0)
    sim.reset(family("straight_cruise"), 0)
    sim.step(Action(0.0, 0.6))
    assert sim.steer == pytest.approx(0.06)


def test_speed_is_never_negative_and_accel_is_clipped():
    sim = KinematicSim(0.0)
    sim.reset(family("straight_cruise"), 0)
    start = sim.speed
    sim.step(Action(-100.0, 0.0))
    assert sim.speed == pytest.approx(start - 0.6)
    for _ in range(400):
        sim.step(Action(-6.0, 0.0))
    assert sim.speed == 0.0


def test_projected_lateral_matches_the_metric_implementation():
    sim = KinematicSim(1.0)
    sc = family("gentle_curve")
    st = sim.reset(sc, 3)
    for _ in range(60):
        st = sim.step(track(RuleCenter().plan(st), st))
    _, lat = station_lateral(sim.route, np.array([[st.x, st.y]]))
    assert st.lateral_m == pytest.approx(float(lat[0]), abs=0.02)


def test_lead_detection_uses_route_gap_and_ignores_adjacent_lane():
    route = np.stack([np.arange(200.0), np.zeros(200)], axis=1)
    same = np.array([[60.0, 0.5, 0, 4.5, 1.9, 7.0]])
    beside = np.array([[60.0, 3.5, 0, 4.5, 1.9, 7.0]])
    behind = np.array([[10.0, 0.0, 0, 4.5, 1.9, 7.0]])
    assert lead_on_route(route, 20.0, same, 4.5) == (35.5, 7.0)
    assert lead_on_route(route, 20.0, beside, 4.5) == (None, None)
    assert lead_on_route(route, 20.0, behind, 4.5) == (None, None)


# Fixture behaviour that once went wrong: each of these failed before the fix


@pytest.mark.parametrize("name", ["straight_cruise", "gentle_curve", "lead_follow", "lead_brake"])
def test_careful_rule_driver_finishes_every_nominal_family(name):
    sim = KinematicSim()
    for sc in (s for s in MANIFEST if s.family == name):
        assert run_episode(sim, RuleCenter(), sc, 0)[0].outcome == COMPLETED


def test_actors_keep_moving_past_the_goal_instead_of_parking_at_the_edge():
    sim = KinematicSim()
    sc = family("lead_follow")
    sim.reset(sc, 0)
    last = sim.route[-1]
    # The final actor sample must not be sitting on the route's end while reporting a speed.
    end = sim.actor_arrays[int(sc.length_m / sc.actors[0].v0_mps / 0.1), 0, :2]
    assert np.linalg.norm(end - last) > 1.0


def test_lead_that_brakes_resumes_so_the_lane_is_not_blocked_forever():
    sim = KinematicSim()
    sc = family("lead_brake")
    sim.reset(sc, 0)
    speeds = sim.actor_arrays[:, 0, 5]
    assert speeds.min() <= sc.actors[0].v_after_mps + 1e-6
    assert speeds[-1] == pytest.approx(sc.actors[0].v0_mps)


def test_stationary_plan_times_out_and_is_not_rewarded():
    r, _ = run_episode(KinematicSim(), Stationary(), family("lead_follow"), 0)
    assert r.outcome == TIMEOUT and not r.safe_completed


def test_policy_without_feedback_leaves_the_road():
    r, _ = run_episode(KinematicSim(), ConstantVelocity(), family("straight_cruise"), 0)
    assert r.outcome == OFFROAD


def test_lead_blind_regression_really_collides():
    sim = KinematicSim()
    outcomes = {
        run_episode(sim, IgnoresLead(), s, 0)[0].outcome
        for s in MANIFEST
        if s.family == "lead_brake"
    }
    assert COLLISION in outcomes


def test_a_crash_in_the_policy_is_recorded_not_lost():
    class Bomb(RuleCenter):
        def plan(self, st):
            raise RuntimeError("boom")

    r, trace = run_episode(KinematicSim(), Bomb(), family("straight_cruise"), 0)
    assert r.outcome == "FAILED" and "boom" in r.reason and trace is None


# Determinism and pairing


@pytest.mark.parametrize("name", ["cut_in", "s_curve"])
def test_same_seed_reproduces_exactly(name):
    sc = family(name)
    a = rollout(KinematicSim(), RuleCenter(), sc, 5)[1]
    b = rollout(KinematicSim(), RuleCenter(), sc, 5)[1]
    np.testing.assert_array_equal(a.ego.xy_m, b.ego.xy_m)


def test_different_seed_changes_the_disturbance():
    sc = family("s_curve")
    a = rollout(KinematicSim(), RuleCenter(), sc, 5)[1].ego.xy_m
    b = rollout(KinematicSim(), RuleCenter(), sc, 6)[1].ego.xy_m
    assert a.shape != b.shape or not np.array_equal(a, b)


def test_policies_see_the_same_disturbance_on_a_paired_seed():
    sc = family("gentle_curve")
    s1, s2 = KinematicSim(), KinematicSim()
    s1.reset(sc, 9)
    s2.reset(sc, 9)
    np.testing.assert_array_equal(s1.noise, s2.noise)
    assert (s1.x, s1.y, s1.yaw, s1.speed) == (s2.x, s2.y, s2.yaw, s2.speed)


def test_no_disturbance_means_no_noise():
    sim = KinematicSim(0.0)
    sim.reset(family("gentle_curve"), 1)
    assert not sim.noise.any()


# Learning pipeline


def test_ridge_recovers_a_known_linear_map():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 4))
    w = rng.normal(size=(4, 12))
    y = (x @ w + 3.0).reshape(-1, 6, 2)
    m = fit_ridge("t", x, y, False, 1, 1e-9)
    z = (x[0] - m.mean) / m.scale
    np.testing.assert_allclose((np.append(z, 1.0) @ m.weights).reshape(6, 2), y[0], atol=1e-5)


def test_heavy_regularisation_shrinks_weights_but_not_the_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 4))
    y = (x @ rng.normal(size=(4, 12)) + 7.0).reshape(-1, 6, 2)
    m = fit_ridge("t", x, y, False, 1, 1e9)
    # With features at their mean, the prediction is the training mean of the labels.
    np.testing.assert_allclose(m.weights[-1], y.reshape(len(y), -1).mean(0), atol=1e-6)
    assert np.abs(m.weights[:-1]).max() < 1e-6


def test_quadratic_expansion_counts():
    assert expand(np.ones(4), 2).shape == (4 + 10,)
    assert expand(np.ones(4), 1).shape == (4,)


def test_labels_are_the_logged_future_in_the_ego_frame():
    states = expert_rollout(KinematicSim(0.0), family("gentle_curve"), 0)
    _, _, y = log_samples(states)
    i = HISTORY_STEPS
    want = to_ego_xy(states[i], np.array([[states[i + 5].x, states[i + 5].y]]))
    np.testing.assert_allclose(y[0, 0], want[0])
    assert y[0, 0, 0] > 0  # the future is ahead of the car


def test_history_features_do_not_contain_the_future():
    """Changing a later state must leave earlier features untouched."""
    states = expert_rollout(KinematicSim(0.0), family("gentle_curve"), 0)
    xb, xh, _ = log_samples(states)
    cut = states[: HISTORY_STEPS + 31]
    xb2, xh2, _ = log_samples(cut)
    np.testing.assert_array_equal(xb[: len(xb2)], xb2)
    np.testing.assert_array_equal(xh[: len(xh2)], xh2)


def test_open_loop_scoring_is_truly_open_loop():
    """A plan must not change the log it is scored on, so scoring twice is identical."""
    log = expert_rollout(KinematicSim(0.0), family("s_curve"), 0)
    p = RuleCenter()
    assert score_log(p, log) == score_log(p, log)


def test_all_rule_candidates_produce_the_same_plan_shape():
    st = KinematicSim().reset(family("s_curve"), 0)
    for p in rule_candidates():
        p.reset()
        assert p.plan(st).shape == (6, 2)


def test_first_second_is_warm_started_identically_for_all_policies():
    sc = family("gentle_curve")
    a = rollout(KinematicSim(), Stationary(), sc, 1)[0]
    b = rollout(KinematicSim(), ConstantVelocity(), sc, 1)[0]
    for i in range(HISTORY_STEPS + 1):
        assert (a[i].x, a[i].y) == (b[i].x, b[i].y)


# Splits and leakage


def test_splits_are_disjoint_by_family_and_id():
    S.assert_disjoint(MANIFEST)
    fam = {s: {x.family for x in MANIFEST if x.split == s} for s in (S.TRAIN, S.DEV, S.TEST)}
    assert not (fam[S.TRAIN] & fam[S.DEV] or fam[S.TRAIN] & fam[S.TEST] or fam[S.DEV] & fam[S.TEST])
    md = S.build_metadrive_manifest(2)
    S.assert_disjoint(md)


def test_metadrive_map_seeds_never_repeat_across_families():
    seeds = [s.id.rsplit("-", 1)[1] for s in S.build_metadrive_manifest(12)]
    assert len(seeds) == len(set(seeds))


def test_disjointness_check_catches_a_family_in_two_splits():
    leaky = [*MANIFEST, S.Scenario("x", "cut_in", S.TRAIN, ((10.0, 0.0),), 3.0, 10.0)]
    with pytest.raises(ValueError, match="two splits"):
        S.assert_disjoint(leaky)


def test_manifest_round_trip_keeps_the_hash(tmp_path):
    p = tmp_path / "m.json"
    S.write_manifest(MANIFEST, p)
    assert S.manifest_hash(S.read_manifest(p)) == S.manifest_hash(MANIFEST)


def test_split_id_changes_when_any_scenario_changes():
    changed = list(MANIFEST)
    changed[0] = S.Scenario(
        changed[0].id,
        changed[0].family,
        S.DEV,
        changed[0].segments,
        changed[0].half_width_m,
        changed[0].speed_limit_mps,
    )
    assert S.split_id(changed) != S.split_id(MANIFEST)


def _dev_row(sc, seed=0, safe=False):
    return {
        "policy": "p",
        "scenario": sc.id,
        "split": sc.split,
        "seed": seed,
        "score": {"safe_completed": safe},
    }


def test_mining_refuses_test_outcomes():
    dev = [s for s in MANIFEST if s.split == S.DEV]
    test = [s for s in MANIFEST if s.split == S.TEST]
    rows = [_dev_row(s) for s in dev] + [_dev_row(test[0])]
    with pytest.raises(ValueError, match="development data only"):
        select_pack(rows, dev, test, 4, 0)


def test_mining_has_no_way_to_read_test_outcomes_in_its_signature_and_keeps_lineage():
    dev = [s for s in MANIFEST if s.split == S.DEV]
    test = [s for s in MANIFEST if s.split == S.TEST]
    rows = [_dev_row(s) for s in dev]
    pack = select_pack(rows, dev, test, 4, 0)
    assert len(set(pack.ids)) == 4
    assert all(pack.lineage[i] for i in pack.ids)
    assert all(
        src.startswith(("s_curve", "lead_brake")) for v in pack.lineage.values() for src in v
    )


# Statistics


def test_interval_uses_scenarios_not_episodes():
    # Ten scenarios, each repeated 100 times with identical outcomes: only 10 independent facts.
    values = np.repeat(np.arange(10) % 2, 100).astype(float)
    clusters = np.repeat(np.arange(10), 100)
    ci = cluster_ci(values, clusters)
    assert ci.n_clusters == 10
    assert (ci.hi - ci.lo) / 2 > 5 * naive_ci_halfwidth(values)


def test_repeats_inside_a_scenario_do_not_shrink_the_interval():
    rng = np.random.default_rng(1)
    base = rng.normal(size=20)
    few = cluster_ci(np.repeat(base, 2), np.repeat(np.arange(20), 2), seed=3)
    many = cluster_ci(np.repeat(base, 200), np.repeat(np.arange(20), 200), seed=3)
    assert many.hi - many.lo == pytest.approx(few.hi - few.lo)


def test_zero_observed_failures_is_an_upper_bound_not_zero():
    ci = cluster_ci(np.zeros(60), np.repeat(np.arange(20), 3))
    assert ci.method == "clopper-pearson" and ci.estimate == 0.0
    # Two-sided 95%: 1 - 0.025^(1/20) = 0.1684
    assert ci.hi == pytest.approx(1 - 0.025 ** (1 / 20), rel=1e-6)


def test_all_successes_get_a_lower_bound_below_one():
    ci = cluster_ci(np.ones(30), np.repeat(np.arange(10), 3))
    assert ci.hi == 1.0 and ci.lo == pytest.approx(0.025 ** (1 / 10), rel=1e-6)


def test_bootstrap_interval_covers_a_known_mean_about_95_percent():
    hits = 0
    for trial in range(300):
        rng = np.random.default_rng(trial)
        v = rng.normal(0.3, 1.0, 40)
        ci = cluster_ci(v, np.arange(40), seed=trial)
        hits += ci.lo <= 0.3 <= ci.hi
    assert 0.90 <= hits / 300 <= 0.98


def test_pairing_cancels_shared_scenario_difficulty():
    rng = np.random.default_rng(2)
    hard = rng.normal(0, 5, 50)
    a, b = hard + 1.0 + rng.normal(0, 0.1, 50), hard + rng.normal(0, 0.1, 50)
    ci = paired_diff_ci(a, b, np.arange(50))
    assert ci.lo > 0.9 and ci.hi < 1.1
    unpaired = cluster_ci(a, np.arange(50))
    assert ci.hi - ci.lo < (unpaired.hi - unpaired.lo) / 10


def test_paired_inputs_must_align():
    with pytest.raises(ValueError):
        paired_diff_ci(np.zeros(3), np.zeros(2), np.arange(3))


def test_verdict_reads_the_interval():
    from eval_lab.statistics import Interval

    assert verdict(Interval(0.2, 0.1, 0.3, 9, "x")) == "IMPROVED"
    assert verdict(Interval(-0.2, -0.3, -0.1, 9, "x")) == "REGRESSED"
    assert verdict(Interval(0.1, -0.1, 0.3, 9, "x")) == "INCONCLUSIVE"


def test_variance_components_recover_known_structure():
    rng = np.random.default_rng(3)
    clusters = np.repeat(np.arange(400), 5)
    effect = np.repeat(rng.normal(0, 2.0, 400), 5)
    v = effect + rng.normal(0, 1.0, 2000)
    c = variance_components(v, clusters)
    assert c["between"] == pytest.approx(4.0, rel=0.15)
    assert c["within"] == pytest.approx(1.0, rel=0.1)
    assert c["icc"] == pytest.approx(0.8, abs=0.05)
    assert c["design_effect"] == pytest.approx(1 + 4 * c["icc"])


def test_rank_correlation_resamples_the_same_scenarios_on_both_axes():
    # Per-scenario noise shared by both axes cancels only when the resample is paired.
    rng = np.random.default_rng(5)
    shared = rng.normal(0, 3.0, (6, 25)) + np.arange(6.0)[:, None] * 0.2
    ci = rank_correlation_ci(shared, shared)
    assert ci.estimate == pytest.approx(1.0)
    assert ci.lo == pytest.approx(1.0)


def test_rank_correlation_finds_agreement_and_disagreement():
    rng = np.random.default_rng(4)
    quality = np.arange(8.0)
    o = quality[:, None] + rng.normal(0, 0.1, (8, 30))
    assert rank_correlation_ci(o, o + rng.normal(0, 0.1, (8, 30))).estimate > 0.9
    assert rank_correlation_ci(o, -quality[:, None] + rng.normal(0, 0.1, (8, 30))).estimate < -0.9


# Provenance


def test_manifest_matches_the_schema(tmp_path, monkeypatch):
    repo_lock = __import__("pathlib").Path("uv.lock").resolve()
    schema = json.loads(__import__("pathlib").Path("schemas/run_manifest.schema.json").read_text())
    run_dir = tmp_path / "r1"
    run_dir.mkdir()
    (run_dir / "a.txt").write_text("x")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "uv.lock").write_bytes(repo_lock.read_bytes())
    m = write_manifest(
        run_dir,
        kind="smoke",
        status="SUCCEEDED",
        exit_code=0,
        code_revision="abc",
        config_sha256="0" * 64,
        data_sha256="0" * 64,
        split_id="s",
        seeds=[0],
        backend="b",
        started_at="2026-10-31T00:00:00+00:00",
        limitations=[],
    )
    validate(m, schema)
    assert m["artifacts"][0]["path"] == "a.txt"


def test_world_state_is_what_both_adapters_return():
    st = KinematicSim().reset(family("cut_in"), 0)
    assert isinstance(st, WorldState) and st.ref.shape == (5, 2)
