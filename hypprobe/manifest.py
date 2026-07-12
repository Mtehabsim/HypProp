"""Run provenance: every gap-closure stage writes a manifest next to its output.

The re-evaluation found ZERO run provenance in the committed artifacts (gap:
"chat_mode, max_new_tokens, limit, commit hash, model revision are recorded
nowhere"), which made two critical stimulus confounds unfalsifiable after the
fact. This module fixes that: one call per stage records everything needed to
re-interpret the numbers later.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime


def _git_state(repo_dir: str) -> dict:
    def _run(*args):
        try:
            return subprocess.run(["git", *args], cwd=repo_dir, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return "unknown"
    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_run("status", "--porcelain")),
    }


def write_manifest(out_dir: str, stage: str, args: dict | None = None,
                   extra: dict | None = None) -> str:
    """Write ``<out_dir>/manifest_<stage>.json`` and return its path."""
    os.makedirs(out_dir, exist_ok=True)
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    versions = {}
    for mod in ("torch", "numpy", "scipy", "transformers", "hypll"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    manifest = {
        "stage": stage,
        "timestamp": datetime.now().isoformat(),
        "git": _git_state(repo_dir),
        "python": sys.version.split()[0],
        "versions": versions,
        "args": args or {},
        "extra": extra or {},
    }
    path = os.path.join(out_dir, f"manifest_{stage}.json")
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return path
