"""Shared I/O: the activation store format, results dirs, and logging helpers.

An "activation store" is how Phase 0 hands data to later phases. On disk it is
one ``.pt`` file per (model, dataset, sample):

    <activations>/<model>/<dataset>/<sample_id>.pt

Each file is a dict:
    {
      "hidden": Tensor[n_layers, n_tokens, hidden]   # fp32
      "tokens": list[str]                              # per token
      "positions": Tensor[n_tokens]                    # 0-based index in full seq
      "is_thinking": Tensor[n_tokens] (bool)           # matched a reasoning marker
      "is_generated": Tensor[n_tokens] (bool)          # False for prompt, True for CoT
      "label": int
      "label_path": list[int]                          # taxonomy path (for hierarchy)
      "model": str, "dataset": str, "sample_id": str, "text": str
    }

Later phases pool this into (N, hidden) feature matrices per (layer, token-source)
via :func:`pool_features`.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import numpy as np

TOKEN_SOURCES = ("input", "thinking", "last", "all")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def log_line(logfile: str, msg: str) -> None:
    """Append a timestamped line to a log file and echo to stdout."""
    ensure_dir(os.path.dirname(logfile) or ".")
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(logfile, "a") as fh:
        fh.write(line + "\n")


def save_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_csv(path: str, rows: list[dict], columns: list[str] | None = None) -> None:
    """Write a list of dict rows to CSV (no pandas dependency required)."""
    import csv

    ensure_dir(os.path.dirname(path) or ".")
    if not rows:
        open(path, "w").close()
        return
    cols = columns or list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in cols})


def iter_samples(activations_dir: str, model: str | None = None, dataset: str | None = None):
    """Yield loaded sample dicts under an activations directory."""
    import torch

    if not os.path.isdir(activations_dir):
        return
    for m in sorted(os.listdir(activations_dir)):
        if model and m != _safe(model):
            continue
        mdir = os.path.join(activations_dir, m)
        if not os.path.isdir(mdir):
            continue
        for ds in sorted(os.listdir(mdir)):
            if dataset and ds != dataset:
                continue
            dsdir = os.path.join(mdir, ds)
            if not os.path.isdir(dsdir):
                continue
            for fn in sorted(os.listdir(dsdir)):
                if fn.endswith(".pt"):
                    yield torch.load(os.path.join(dsdir, fn), map_location="cpu")


def _safe(name: str) -> str:
    """Filesystem-safe version of a model id (slashes -> underscores)."""
    return name.replace("/", "_")


def sample_path(activations_dir: str, model: str, dataset: str, sample_id: str) -> str:
    d = os.path.join(activations_dir, _safe(model), dataset)
    ensure_dir(d)
    return os.path.join(d, f"{sample_id}.pt")


def pool_features(sample: dict, layer: int, token_source: str) -> np.ndarray | None:
    """Pool one sample's hidden states at ``layer`` for a given token source.

    Returns a single (hidden,) vector or None if no tokens match. ``token_source``
    is one of TOKEN_SOURCES: 'input' (prompt tokens), 'thinking' (reasoning
    markers), 'last' (final token), 'all' (mean over every token).
    """
    hidden = sample["hidden"]  # (n_layers, n_tokens, hidden)
    if hasattr(hidden, "numpy"):
        hidden = hidden.numpy()
    hidden = np.asarray(hidden, dtype=np.float64)
    n_layers = hidden.shape[0]
    layer = min(layer, n_layers - 1)
    h = hidden[layer]  # (n_tokens, hidden)

    is_gen = np.asarray(sample.get("is_generated"))
    is_think = np.asarray(sample.get("is_thinking"))

    if token_source == "input":
        mask = ~is_gen if is_gen is not None and is_gen.size else np.ones(h.shape[0], bool)
    elif token_source == "thinking":
        mask = is_think if is_think is not None and is_think.size else np.zeros(h.shape[0], bool)
    elif token_source == "last":
        mask = np.zeros(h.shape[0], bool)
        mask[-1] = True
    elif token_source == "all":
        mask = np.ones(h.shape[0], bool)
    else:
        raise ValueError(f"unknown token_source {token_source}")

    if mask.sum() == 0:
        return None
    return h[mask].mean(axis=0)


def build_feature_matrix(activations_dir, model, dataset, layer, token_source):
    """Assemble (X, y, label_paths) across all samples for a (layer, source)."""
    xs, ys, paths = [], [], []
    for s in iter_samples(activations_dir, model, dataset):
        vec = pool_features(s, layer, token_source)
        if vec is None:
            continue
        xs.append(vec)
        ys.append(int(s.get("label", 0)))
        paths.append(s.get("label_path", []))
    if not xs:
        return np.empty((0, 0)), np.empty(0, int), []
    return np.stack(xs), np.asarray(ys, int), paths
