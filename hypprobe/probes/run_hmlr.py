"""Phase 2 CLI: train the hyperbolic probe (H-MLR) over the activation store."""

from __future__ import annotations

import argparse
import os

import numpy as np

from ..io import build_feature_matrix, ensure_dir, iter_samples, log_line, save_json
from .hmlr import ProbeConfig, fit_probe
from .baselines import prepare_features, evaluate_arm


def _split(n, seed, frac=0.7):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(frac * n)
    return idx[:cut], idx[cut:]


def run(activations_dir, out_dir, seed=0, proj_dim=5, curvature=1.0,
        learn_curvature=True, layer=None, source="last", epochs=300):
    ensure_dir(out_dir)
    logfile = os.path.join(os.path.dirname(out_dir.rstrip("/")) or ".", "logs", "hmlr.log")
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
        # Whiten (fit on train, apply to both) so this matches the shared,
        # confound-controlled feature space used by the baselines/comparison.
        xt, xv = prepare_features(X[tr], X[va], whiten=True)
        cfg = ProbeConfig(in_dim=xt.shape[1], n_classes=n_classes, proj_dim=proj_dim,
                          curvature=curvature, learn_curvature=learn_curvature,
                          seed=seed, epochs=epochs)
        acc, f1, sel, mdl, cval = evaluate_arm(xt, y[tr], xv, y[va], cfg)
        tag = f"{model.replace('/', '_')}_{dataset}_L{use_layer}_{source}_seed{seed}"
        save_json(os.path.join(out_dir, f"hmlr_{tag}.json"),
                  dict(model=model, dataset=dataset, layer=use_layer, source=source,
                       seed=seed, proj_dim=proj_dim,
                       arms={"hyperbolic": {"val_acc": acc, "macro_f1": f1,
                                            "selectivity": sel, "mdl_bits": mdl,
                                            "curvature": cval}}))
        log_line(logfile, f"{tag}: hyperbolic acc={acc:.3f} sel={sel:+.3f} "
                          f"mdl={mdl:.0f}b c={cval:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 2: hyperbolic probe (H-MLR).")
    ap.add_argument("--activations", required=True)
    ap.add_argument("--out", default="./results/probes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proj-dim", type=int, default=5)
    ap.add_argument("--curvature", type=float, default=1.0)
    ap.add_argument("--source", default="last")
    ap.add_argument("--layer", type=int, default=None)
    args = ap.parse_args(argv)
    run(args.activations, args.out, seed=args.seed, proj_dim=args.proj_dim,
        curvature=args.curvature, layer=args.layer, source=args.source)


if __name__ == "__main__":
    main()
