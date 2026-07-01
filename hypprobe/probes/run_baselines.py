"""Phase 2 CLI: train flat baseline probes over the activation store.

Trains euclidean_lr, flat_on_transform, and curvature_zero for the chosen
(layer, token-source) settings and one seed, saving metrics.json per arm. The
matched, cross-arm comparison is assembled later by eval.compare.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from ..io import build_feature_matrix, ensure_dir, iter_samples, log_line, save_json
from .hmlr import ProbeConfig, fit_probe
from .baselines import euclidean_lr


def _split(n, seed, frac=0.7):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(frac * n)
    return idx[:cut], idx[cut:]


def run(activations_dir, out_dir, seed=0, proj_dim=5, layer=None, source="last", epochs=200):
    ensure_dir(out_dir)
    logfile = os.path.join(os.path.dirname(out_dir.rstrip("/")) or ".", "logs", "baselines.log")
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
        _, acc = euclidean_lr(X[tr], y[tr], X[va], y[va], seed=seed)
        arms = {"euclidean_lr": {"val_acc": acc}}
        for name, cfg in [
            ("flat_on_transform", ProbeConfig(in_dim=X.shape[1], n_classes=n_classes,
                                              proj_dim=proj_dim, curvature=1.0,
                                              use_manifold=False, seed=seed, epochs=epochs)),
            ("curvature_zero", ProbeConfig(in_dim=X.shape[1], n_classes=n_classes,
                                           proj_dim=proj_dim, curvature=0.0,
                                           seed=seed, epochs=epochs)),
        ]:
            _, res = fit_probe(X[tr], y[tr], X[va], y[va], cfg)
            arms[name] = {"val_acc": res.val_acc, "macro_f1": res.macro_f1}
        tag = f"{model.replace('/', '_')}_{dataset}_L{use_layer}_{source}_seed{seed}"
        save_json(os.path.join(out_dir, f"baselines_{tag}.json"),
                  dict(model=model, dataset=dataset, layer=use_layer, source=source,
                       seed=seed, proj_dim=proj_dim, arms=arms))
        log_line(logfile, f"{tag}: euclidean_lr={acc:.3f} "
                          f"flat_on_transform={arms['flat_on_transform']['val_acc']:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 2: flat baseline probes.")
    ap.add_argument("--activations", required=True)
    ap.add_argument("--out", default="./results/probes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proj-dim", type=int, default=5)
    ap.add_argument("--source", default="last")
    ap.add_argument("--layer", type=int, default=None)
    args = ap.parse_args(argv)
    run(args.activations, args.out, seed=args.seed, proj_dim=args.proj_dim,
        layer=args.layer, source=args.source)


if __name__ == "__main__":
    main()
