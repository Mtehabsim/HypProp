"""Phase 2 CLI: the hyperbolicity-aware adaptive gate.

The gate measures delta_rel of the incoming feature set and picks geometry: use
the hyperbolic probe when the space is tree-like (delta_rel below a threshold),
otherwise fall back to the flat probe. It records its decision and the measured
delta_rel for every setting so we can audit why it chose what it chose.

IMPORTANT (per plan): the gate's INPUT signal is defined by the Phase-1
determinants result. This module ships the delta_rel-driven gate as the default;
once determinants identify the dominant driver (token identity / order /
meaning), pass ``--gate-on {delta,identity,order,meaning}`` to key on it. Until
then the delta gate is the safe, general default.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from ..geometry.delta import delta_hyperbolicity
from ..io import build_feature_matrix, ensure_dir, iter_samples, log_line, save_csv, save_json
from .hmlr import ProbeConfig, fit_probe
from .baselines import prepare_features


def _split(n, seed, frac=0.7):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(frac * n)
    return idx[:cut], idx[cut:]


def _read_determinant_driver(determinants_dir, model, dataset):
    """Read the Phase-1 dominant driver for (model,dataset), gated on std_rel.

    Returns (driver, trustworthy). ``driver`` is the edit with the largest
    |Δδ_rel|; ``trustworthy`` is False if that change is smaller than the
    per-edit noise (std_rel of base or edit), per the project's non-negotiable
    "never trust a gap below std_rel" rule. This is what makes the gate
    'determinants-defined' rather than a hardcoded threshold.
    """
    import csv

    if not determinants_dir:
        return None, False
    path = os.path.join(determinants_dir, "attribution.csv")
    if not os.path.exists(path):
        return None, False
    rows = [r for r in csv.DictReader(open(path))
            if r.get("model") == model and r.get("dataset") == dataset]
    if not rows:
        return None, False
    top = max(rows, key=lambda r: abs(float(r.get("delta_change", 0) or 0)))
    change = abs(float(top.get("delta_change", 0) or 0))
    # std columns may not exist in older CSVs; default to a small floor.
    noise = 0.0
    for key in ("std_rel_base", "std_rel_edit", "std_rel"):
        if key in top and top[key] not in ("", None):
            noise = max(noise, float(top[key]))
    trustworthy = change > max(noise, 1e-6)
    return top.get("edit"), trustworthy


def run(activations_dir, out_dir, determinants_dir=None, seed=0, proj_dim=5,
        delta_threshold=0.25, layer=None, source="last", epochs=300):
    ensure_dir(out_dir)
    adir = ensure_dir(os.path.join(out_dir, "adaptive"))
    logfile = os.path.join(os.path.dirname(out_dir.rstrip("/")) or ".", "logs", "adaptive.log")
    decisions = []
    for model, dataset in sorted({(s["model"], s["dataset"]) for s in iter_samples(activations_dir)}):
        sample = next(iter_samples(activations_dir, model, dataset), None)
        if sample is None:
            continue
        use_layer = (int(np.asarray(sample["hidden"]).shape[0]) - 1) if layer is None else layer
        X, y, _ = build_feature_matrix(activations_dir, model, dataset, use_layer, source)
        if X.shape[0] < 16:
            continue
        n_classes = int(y.max() + 1)
        tr, va = _split(len(y), seed)
        # Shared whitened features (fit on train) -- same space as the comparison.
        xt, xv = prepare_features(X[tr], X[va], whiten=True)

        # Gate signal 1: hyperbolicity of the (whitened) training features.
        dr = delta_hyperbolicity(X[tr], do_whiten=True, seed=seed).delta_rel
        # Gate signal 2 (the Part-1 -> Part-2 link): the Phase-1 dominant driver,
        # if it is trustworthy (change > std_rel). The determinants result now
        # ACTUALLY informs the gate: if hyperbolicity is a token-identity artifact,
        # we do NOT trust it as hierarchy and fall back to flat even when delta is
        # low; a meaning/order driver lets a low-delta setting go hyperbolic.
        driver, driver_ok = _read_determinant_driver(determinants_dir, model, dataset)
        if driver_ok and driver and driver.startswith("token_identity"):
            chosen, reason = "flat", f"driver={driver} (identity artifact -> distrust)"
        elif driver_ok and driver and (driver.startswith("meaning") or driver.startswith("order")):
            chosen = "hyperbolic" if dr < delta_threshold else "flat"
            reason = f"driver={driver} (structural) + delta_rel={dr:.3f}"
        else:
            chosen = "hyperbolic" if dr < delta_threshold else "flat"
            reason = f"delta_rel={dr:.3f} (no trustworthy driver)"

        curvature = 1.0 if chosen == "hyperbolic" else 0.0
        cfg = ProbeConfig(in_dim=xt.shape[1], n_classes=n_classes, proj_dim=proj_dim,
                          curvature=curvature, use_manifold=(chosen == "hyperbolic"),
                          learn_curvature=(chosen == "hyperbolic"),
                          seed=seed, epochs=epochs)
        _, res = fit_probe(xt, y[tr], xv, y[va], cfg)
        decisions.append(dict(model=model, dataset=dataset, layer=use_layer, source=source,
                              seed=seed, delta_rel=round(dr, 4),
                              driver=driver or "", driver_trustworthy=driver_ok,
                              chosen_geometry=chosen,
                              val_acc=round(res.val_acc, 4), macro_f1=round(res.macro_f1, 4)))
        log_line(logfile, f"{model}/{dataset} L{use_layer}: {reason} -> {chosen} "
                          f"(acc={res.val_acc:.3f})")

    save_csv(os.path.join(adir, "gate_decisions.csv"), decisions)
    save_json(os.path.join(adir, f"gate_seed{seed}.json"),
              dict(delta_threshold=delta_threshold, decisions=decisions))
    return decisions


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 2: adaptive hyperbolicity gate.")
    ap.add_argument("--activations", required=True)
    ap.add_argument("--out", default="./results/probes")
    ap.add_argument("--determinants", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proj-dim", type=int, default=5)
    ap.add_argument("--delta-threshold", type=float, default=0.25)
    ap.add_argument("--source", default="last")
    ap.add_argument("--layer", type=int, default=None)
    args = ap.parse_args(argv)
    run(args.activations, args.out, determinants_dir=args.determinants, seed=args.seed,
        proj_dim=args.proj_dim, delta_threshold=args.delta_threshold,
        layer=args.layer, source=args.source)


if __name__ == "__main__":
    main()
