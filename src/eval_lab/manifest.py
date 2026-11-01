"""Run provenance matching schemas/run_manifest.schema.json."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def code_revision() -> str:
    """HEAD, plus a hash of the diff when the tree is dirty. A dirty run says so."""
    head = _git("rev-parse", "HEAD").strip()
    if not is_dirty():
        return head
    return f"{head}+dirty.{sha256_text(_git('diff', 'HEAD'))[:12]}"


def is_dirty() -> bool:
    return bool(_git("status", "--porcelain", "--", "src", "experiments", "pyproject.toml").strip())


def hardware() -> dict:
    return {
        "machine": platform.machine(),
        "system": platform.system(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "accelerator": "none used; CPU only",
    }


def versions(names: tuple[str, ...]) -> dict:
    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def started_at() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_manifest(run_dir: Path, **fields) -> dict:
    artifacts = [
        {"path": str(p.relative_to(run_dir)), "sha256": sha256_file(p)}
        for p in sorted(run_dir.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    ]
    manifest = {
        "schema_version": "1.0",
        "run_id": run_dir.name,
        "hardware": hardware(),
        "command": sys.argv,
        "artifacts": artifacts,
        "lockfile_sha256": sha256_file(Path("uv.lock")),
        "versions": versions(("numpy", "scipy", "metadrive-simulator", "panda3d")),
        "finished_at": started_at(),
        **fields,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
