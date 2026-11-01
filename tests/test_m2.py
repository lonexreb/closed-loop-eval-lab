"""M2 study: the new families really are new, and each regression does what its name says."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from eval_lab import m2
from eval_lab import scenarios as S
from eval_lab.adapters.base import WorldState
from eval_lab.adapters.kinematic import KinematicSim
from eval_lab.experiment import divergence_s
from eval_lab.mining import select_pack
from eval_lab.policies import expert_action, expert_track, rule_plan, track
from eval_lab.report import build as build_pilot
from eval_lab.runner import expert_rollout, run_episode
from eval_lab.study_report import build


def state(**kw) -> WorldState:
    ref = np.column_stack([np.arange(1.0, 41.0, 3.0), np.zeros(14)])
    base = dict(
        t_s=0.0,
        x=0.0,
        y=0.0,
        yaw=0.0,
        speed=10.0,
        ref=ref,
        lateral_m=0.0,
        heading_error=0.0,
        speed_limit=10.0,
        gap_m=None,
        lead_speed=None,
    )
    return WorldState(**{**base, **kw})


def test_holdout_families_share_nothing_with_the_pilot():
    pilot = S.build_manifest(3, 1)
    S.assert_disjoint([*pilot, *S.build_m2_manifest(3, 2)])
    assert not {s.family for s in pilot} & {f.__name__ for f in S.M2_FAMILIES}
    md = S.build_metadrive_manifest(12)
    assert not {s.family for s in md} & set(S.MD_M2_FAMILIES)


def test_metadrive_holdout_map_seeds_are_unused_and_stay_inside_the_map_pool():
    pilot = {int(s.id.rsplit("-", 1)[1]) for s in S.build_metadrive_manifest(12)}
    seeds = [int(s.id.rsplit("-", 1)[1]) for s in S.build_metadrive_m2_manifest(12)]
    assert len(seeds) == len(set(seeds)) and not pilot & set(seeds)
    assert max(seeds) < 900  # 900 and up were spent on the expert feasibility probe
    with pytest.raises(ValueError):
        S.build_metadrive_m2_manifest(41)


def test_frozen_holdout_files_match_their_generators():
    cfg = json.load(open("experiments/config_m2.json"))
    k = S.read_manifest(S.Path("experiments/scenarios_kinematic_m2.json"))
    built = S.build_m2_manifest(cfg["kinematic"]["per_family"], cfg["kinematic"]["manifest_seed"])
    assert S.manifest_hash(k) == S.manifest_hash(built)
    d = S.read_manifest(S.Path("experiments/scenarios_metadrive_m2.json"))
    built = S.build_metadrive_m2_manifest(cfg["metadrive"]["per_family"])
    assert S.manifest_hash(d) == S.manifest_hash(built)


def test_miner_still_refuses_held_out_rows():
    pool = S.build_m2_manifest(1, 1)
    rows = [{"split": S.HOLDOUT, "scenario": pool[0].id, "policy": "x", "score": {}}]
    with pytest.raises(ValueError):
        select_pack(rows, pool, pool, 1, 0)


def test_expert_track_is_exactly_the_logging_controller():
    st = state(speed=7.0)
    assert expert_track(rule_plan(st), st) == expert_action(st)
    assert expert_track(rule_plan(st), st) != track(rule_plan(st), st)


def test_offset_regression_shifts_the_plan_sideways():
    st = state()
    gap = m2.Offset(offset_m=2.0).plan(st) - rule_plan(st)
    assert np.allclose(gap, [0.0, 2.0])


def test_short_gap_at_keep_one_changes_nothing_and_at_zero_drives_closer():
    st = state(gap_m=15.0, lead_speed=5.0)
    assert np.allclose(m2.ShortGap(keep=1.0).plan(st), rule_plan(st))
    assert np.hypot(*m2.ShortGap(keep=0.0).plan(st)[0]) > np.hypot(*rule_plan(st)[0])


def test_short_gap_is_inert_with_no_lead():
    st = state()
    assert np.allclose(m2.ShortGap(keep=0.0).plan(st), rule_plan(st))


def test_stale_lead_sees_the_old_gap_and_the_current_road():
    p = m2.StaleLead(lag=2)
    p.reset()
    near = state(gap_m=6.0, lead_speed=0.0)
    for _ in range(2):
        p.plan(state(gap_m=70.0, lead_speed=10.0))
    assert np.allclose(p.plan(near), rule_plan(replace(near, gap_m=70.0, lead_speed=10.0)))


def test_stale_pose_plans_from_the_old_position():
    p = m2.StalePose(lag=1)
    p.reset()
    p.plan(state(y=3.0))
    now = state(y=0.0)
    assert np.allclose(p.plan(now), rule_plan(replace(now, y=3.0)))


def test_blind_on_curve_acts_only_when_the_road_bends():
    gap = dict(gap_m=7.0, lead_speed=0.0)
    p = m2.BlindOnCurve()
    straight = state(**gap)
    assert np.allclose(p.plan(straight), rule_plan(straight))
    bent = np.column_stack([np.arange(1.0, 41.0, 3.0), np.linspace(0, 8, 14)])
    curved = state(ref=bent, **gap)
    assert np.allclose(p.plan(curved), rule_plan(replace(curved, gap_m=None, lead_speed=None)))


def test_candidate_names_are_unique_so_results_cannot_collide():
    names = [p.name for p in m2.CANDIDATES]
    assert len(names) == len(set(names)) and all(names)


def test_gap_reading_needs_its_interval_to_clear_zero():
    table = [
        {"method": "mined", "budget": 8, "a": 1.0, "b": 0.9, "c": 0.8},
        {"method": "random", "budget": 8, "a": 0.5, "b": 0.5, "c": 0.5},
    ]
    assert m2.mined_minus_random(table, ["a", "b", "c"], 8)["verdict"] == "MINED_BETTER"
    table[0].update(a=0.1, b=0.2, c=0.3)
    table[1].update(a=0.9, b=0.9, c=0.9)
    assert m2.mined_minus_random(table, ["a", "b", "c"], 8)["verdict"] == "RANDOM_BETTER"
    table[0].update(a=1.0, b=0.0, c=0.5)
    table[1].update(a=0.0, b=1.0, c=0.5)
    assert m2.mined_minus_random(table, ["a", "b", "c"], 8)["verdict"] == "INCONCLUSIVE"
    table[0].update(a=1.0, b=0.0, c=0.6)  # positive mean, interval still crosses zero
    assert m2.mined_minus_random(table, ["a", "b", "c"], 8)["verdict"] == "INCONCLUSIVE"
    assert m2.mined_minus_random(table, [], 8)["verdict"] == "NOT_APPLICABLE"


def test_divergence_time_is_the_first_step_past_the_limit():
    log = [state(t_s=i * 0.1, x=float(i)) for i in range(5)]
    trace = np.column_stack([[0.0, 0.1, 0.2, 0.3], [0.0, 1.0, 2.0, 3.0], [0, 0, 0.5, 1.5]])
    assert divergence_s(trace, log) == pytest.approx(0.3)
    assert divergence_s(np.column_stack([[0.0], [0.0], [0.0]]), log) is None


def test_expert_tracker_replay_of_the_rule_policy_finishes_a_new_family():
    sc = S.build_m2_manifest(1, 3)[0]
    sim = KinematicSim(disturbance=0.0)
    score, _ = run_episode(sim, m2.RuleCenter(), sc, 0, expert_track)
    assert score.safe_completed
    assert len(expert_rollout(sim, sc, 0)) > 10


def test_study_report_regenerates_byte_identically_from_retained_runs():
    assert build(Path("runs")) == Path("reports/study.md").read_text()


def test_pilot_report_is_unchanged_by_the_m2_files():
    assert build_pilot(Path("runs")) == Path("reports/pilot.md").read_text()


def test_every_held_out_run_is_a_clean_success_with_the_protocol_committed():
    for name in ("kinematic-holdout", "metadrive-holdout", "m2-mining"):
        m = json.loads(Path(f"runs/{name}/manifest.json").read_text())
        assert (m["status"], m["exit_code"]) == ("SUCCEEDED", 0)
        assert "dirty" not in m["code_revision"]
        assert m["eval_split"] == S.HOLDOUT


def test_holdout_stage_refuses_a_dirty_tree(monkeypatch, tmp_path):
    monkeypatch.setattr(m2, "is_dirty", lambda: True)
    assert m2.main(["--stage", "holdout", "--run-root", str(tmp_path)]) == 2
    assert not (tmp_path / "m2-mining").exists()
