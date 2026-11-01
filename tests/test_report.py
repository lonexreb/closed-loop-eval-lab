"""End to end on a tiny real run: the report is reproducible and traceable to raw files."""

import gzip
import json
import shutil
from pathlib import Path

import pytest

from eval_lab import experiment
from eval_lab.manifest import write_manifest
from eval_lab.report import build

REPO = Path.cwd()
CFG = {
    "per_family": 3,
    "manifest_seed": 5,
    "train_seeds": [0],
    "eval_seeds": [0, 1],
    "noisy_disturbance": 2.5,
    "lambdas": [0.01],
    "workers": 1,
}


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """Real tiny dev and final runs through the production code path."""
    base = tmp_path_factory.mktemp("repo")
    shutil.copy(REPO / "uv.lock", base / "uv.lock")
    (base / "experiments").mkdir()
    runs = base / "runs"
    import os

    os.chdir(base)
    try:
        for stage in ("dev", "final"):
            run_dir = runs / f"kinematic-{stage}"
            run_dir.mkdir(parents=True)
            fields = experiment.run("kinematic", stage, run_dir, CFG, lambda m: None)
            write_manifest(
                run_dir,
                kind="pilot",
                status="SUCCEEDED",
                exit_code=0,
                code_revision="abc123",
                config_sha256="1" * 64,
                started_at="t",
                limitations=[],
                **fields,
            )
            shutil.copytree(run_dir, runs / f"metadrive-{stage}-v2")
    finally:
        os.chdir(REPO)
    return runs


def test_regeneration_is_byte_identical(root):
    assert build(root) == build(root)


def test_split_id_and_data_hash_survive_regeneration(root):
    m = json.loads((root / "kinematic-final" / "manifest.json").read_text())
    text = build(root)
    assert m["split_id"] in text and m["data_sha256"][:12] in text


def test_only_final_runs_read_test_families(root):
    with gzip.open(root / "kinematic-dev" / "episodes.jsonl.gz", "rt") as f:
        dev = [json.loads(x) for x in f]
    with gzip.open(root / "kinematic-final" / "episodes.jsonl.gz", "rt") as f:
        final = [json.loads(x) for x in f]
    assert {r["split"] for r in dev} == {"dev"}
    assert {r["split"] for r in final} == {"test"}


def test_every_planned_episode_is_in_the_raw_file(root):
    with gzip.open(root / "kinematic-final" / "episodes.jsonl.gz", "rt") as f:
        rows = [json.loads(x) for x in f]
    # 12 policies, 4 families x 3 scenarios, 2 seeds
    assert len(rows) == 12 * 12 * 2


def test_changing_a_raw_file_changes_the_report(root, tmp_path):
    copy = tmp_path / "runs"
    shutil.copytree(root, copy)
    path = copy / "kinematic-final" / "episodes.jsonl.gz"
    with gzip.open(path, "rt") as f:
        rows = [json.loads(x) for x in f]
    for r in rows:
        if r["policy"] == "rule_center":
            r["score"]["safe_completed"] = False
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        f.write(("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n").encode())
    assert build(copy) != build(root)


def test_final_stage_refuses_to_overwrite_existing_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs" / "kinematic-final").mkdir(parents=True)
    monkeypatch.setattr(experiment, "is_dirty", lambda: False)
    assert experiment.main(["--world", "kinematic", "--stage", "final"]) == 2


def test_final_stage_refuses_a_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment, "is_dirty", lambda: True)
    assert experiment.main(["--world", "kinematic", "--stage", "final"]) == 2
    assert not (tmp_path / "runs").exists()


def test_a_changed_scenario_generator_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "experiments").mkdir()
    experiment.load_manifest("kinematic", CFG)
    with pytest.raises(SystemExit, match="no longer matches"):
        experiment.load_manifest("kinematic", {**CFG, "manifest_seed": 6})
