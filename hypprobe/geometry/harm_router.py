"""OQ3 — the harm ROUTER: does hyperbolic geometry BLOCK harm better + GENERALIZE?

run6/run7 showed the hazard taxonomy is hyperbolic-DECODABLE (distances fit the
tree better than flat). That is not the same as a useful moderator. This asks the
deployment question directly (the user's design): embed prompts by hazard
category into a small space, assign a NEW prompt to its nearest category
prototype, block if that category is harmful. Compare geometries on two metrics:

  (1) BLOCKING F1 — standard harmful/benign detection via nearest-leaf routing.
      hyperbolic vs matched-euclidean vs a flat logistic baseline. Tests whether
      the tree structure buys accuracy at all.
  (2) ZERO-SHOT to UNSEEN categories — the real payoff a flat classifier can't
      give. Hold out entire harm categories at train time; place their prompts by
      their taxonomic relations; measure harmful-vs-benign accuracy on the held-
      out categories. If hyperbolic routing generalizes to novel hazards better
      than flat, THAT is the deployment case for curvature.

Method: fit the matched hyperbolic / euclidean map (as in tree_probe) so category
prototypes realise the taxonomy tree; route a prompt to argmin distance to a
category prototype; a category is "harmful" if answer==1. Flat baseline = logistic
regression on the raw pooled rep -> safe/unsafe directly (no tree).

Reuses pooled per-prompt reps + label_path (taxonomy) + answer (harm flag) from
the harm_taxonomy / harm_beavertails datasets. No new extraction schema.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from ..io import ensure_dir, iter_samples, log_line, save_csv, save_json
from ..manifest import write_manifest
from .matched_probe import MatchedProbe, _whiten_fit
from .structural_probe import _taxonomy_target


def _pool_last(sample, layer):
    h = np.asarray(sample["hidden"], dtype=np.float64)
    L = min(layer, h.shape[0] - 1)
    return h[L, -1]


def _load(activations_dir, dataset, model, layer, max_n=1200):
    """Pooled reps X, harm flag y (0/1), category id c, supercategory id s."""
    X, y, cat, sup = [], [], [], []
    for smp in iter_samples(activations_dir, model, dataset):
        lp = smp.get("label_path") or [0, 0]
        X.append(_pool_last(smp, layer))
        y.append(int(smp.get("answer", smp.get("label", 0))))
        sup.append(int(lp[0])); cat.append(int(lp[1]) if len(lp) > 1 else 0)
        if len(X) >= max_n:
            break
    return (np.asarray(X), np.asarray(y), np.asarray(cat), np.asarray(sup))


def _fit_map(arm, X, Dtree, proj_dim=5, seed=0, curvature=0.5, epochs=400):
    """Fit matched map so pairwise distances match the taxonomy tree (as tree_probe)."""
    model = MatchedProbe(X.shape[1], proj_dim, arm, seed=seed, curvature=curvature)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    xt = torch.as_tensor(X, dtype=torch.float32)
    dt = torch.as_tensor(Dtree, dtype=torch.float32)
    mask = ~torch.eye(len(xt), dtype=torch.bool)
    tn = (dt[mask] ** 2).sum().clamp_min(1e-9)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((model.dist(xt)[mask] - dt[mask]) ** 2).sum() / tn
        loss.backward(); opt.step()
    return model


def _route_accuracy(model, Xtr, ytr, cat_tr, Xte, yte):
    """Route each test prompt to nearest TRAIN category prototype; predict that
    category's majority harm flag. Returns (accuracy, f1_harmful)."""
    with torch.no_grad():
        Ztr = model.transformed(torch.as_tensor(Xtr, dtype=torch.float32))
        Zte = model.transformed(torch.as_tensor(Xte, dtype=torch.float32))
    # category prototype = mean transformed rep; category harm flag = majority
    cats = sorted(set(cat_tr.tolist()))
    protos, flags = [], []
    for c in cats:
        m = cat_tr == c
        protos.append(Ztr[m].mean(0)); flags.append(int(round(ytr[m].mean())))
    protos = torch.stack(protos); flags = np.asarray(flags)
    c = model.curvature if model.arm == "hyperbolic" else 0.0
    from . import poincare
    n = Zte.shape[0]
    preds = []
    for i in range(n):
        zi = Zte[i:i + 1].expand(len(protos), -1)
        d = poincare.dist(zi, protos, c).numpy()
        preds.append(flags[int(d.argmin())])
    preds = np.asarray(preds)
    acc = float((preds == yte).mean())
    tp = int(((preds == 1) & (yte == 1)).sum()); fp = int(((preds == 1) & (yte == 0)).sum())
    fn = int(((preds == 0) & (yte == 1)).sum())
    f1 = tp / (tp + 0.5 * (fp + fn)) if (tp + fp + fn) else 0.0
    return acc, f1


def _flat_baseline(Xtr, ytr, Xte, yte):
    """Logistic regression safe/unsafe on raw reps — the no-tree control."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000, C=1.0)
    if len(set(ytr.tolist())) < 2:
        return float("nan"), float("nan")
    clf.fit(Xtr, ytr); preds = clf.predict(Xte)
    acc = float((preds == yte).mean())
    tp = int(((preds == 1) & (yte == 1)).sum()); fp = int(((preds == 1) & (yte == 0)).sum())
    fn = int(((preds == 0) & (yte == 1)).sum())
    f1 = tp / (tp + 0.5 * (fp + fn)) if (tp + fp + fn) else 0.0
    return acc, f1


def run(activations_dir, out_dir, dataset="harm_taxonomy", layers=None,
        seeds=(0, 1, 2, 3, 4, 5), proj_dim=5, curvature=0.5):
    ensure_dir(out_dir)
    logfile = os.path.join(os.path.dirname(out_dir.rstrip("/")) or ".", "logs", "harm_router.log")
    rows = []
    models = sorted({s["model"] for s in iter_samples(activations_dir, dataset=dataset)})
    for model in models:
        probe = [s for s in iter_samples(activations_dir, model, dataset)]
        if len(probe) < 40:
            continue
        n_layers = int(np.asarray(probe[0]["hidden"]).shape[0])
        use_layers = layers if layers else [n_layers // 4, n_layers // 2, 3 * n_layers // 4]
        for layer in use_layers:
            X, y, cat, sup = _load(activations_dir, dataset, model, layer)
            whiten = _whiten_fit(X)
            Xw = whiten(X)
            for seed in seeds:
                rng = np.random.default_rng(seed)
                # ---- (1) BLOCKING: standard 70/30 prompt split ----
                perm = rng.permutation(len(Xw)); ntr = int(0.7 * len(Xw))
                tr, te = perm[:ntr], perm[ntr:]
                Dtr = _taxonomy_target([{"label_path": [int(sup[i]), int(cat[i])]} for i in tr])
                for arm in ("hyperbolic", "cond_euclidean"):
                    mdl = _fit_map(arm, Xw[tr], Dtr, proj_dim, seed, curvature)
                    acc, f1 = _route_accuracy(mdl, Xw[tr], y[tr], cat[tr], Xw[te], y[te])
                    rows.append(dict(model=model, layer=layer, seed=seed, eval="blocking",
                                     arm=arm, acc=round(acc, 4), f1=round(f1, 4)))
                fa, ff = _flat_baseline(Xw[tr], y[tr], Xw[te], y[te])
                rows.append(dict(model=model, layer=layer, seed=seed, eval="blocking",
                                 arm="flat_logreg", acc=round(fa, 4), f1=round(ff, 4)))

                # ---- (2) ZERO-SHOT: hold out whole categories ----
                cats = sorted(set(cat.tolist()))
                if len(cats) >= 4:
                    rng.shuffle(cats)
                    held = set(cats[: max(1, len(cats) // 3)])
                    tr = np.array([i for i in range(len(Xw)) if cat[i] not in held])
                    te = np.array([i for i in range(len(Xw)) if cat[i] in held])
                    if len(tr) > 10 and len(te) > 5 and len(set(y[te].tolist())) >= 1:
                        Dtr = _taxonomy_target([{"label_path": [int(sup[i]), int(cat[i])]} for i in tr])
                        for arm in ("hyperbolic", "cond_euclidean"):
                            mdl = _fit_map(arm, Xw[tr], Dtr, proj_dim, seed, curvature)
                            # route held-out prompts to nearest TRAIN category (unseen cats
                            # placed by their supercategory neighbourhood)
                            acc, f1 = _route_accuracy(mdl, Xw[tr], y[tr], cat[tr], Xw[te], y[te])
                            rows.append(dict(model=model, layer=layer, seed=seed, eval="zeroshot",
                                             arm=arm, acc=round(acc, 4), f1=round(f1, 4)))
                        fa, ff = _flat_baseline(Xw[tr], y[tr], Xw[te], y[te])
                        rows.append(dict(model=model, layer=layer, seed=seed, eval="zeroshot",
                                         arm="flat_logreg", acc=round(fa, 4), f1=round(ff, 4)))
            log_line(logfile, f"{model} L{layer}: router blocking+zeroshot, {len(seeds)} seeds")

    save_csv(os.path.join(out_dir, "harm_router.csv"), rows)
    save_json(os.path.join(out_dir, "harm_router_verdict.json"), _verdict(rows))
    write_manifest(out_dir, "harm_router",
                   args=dict(activations=activations_dir, dataset=dataset, seeds=list(seeds)))
    return rows


def _verdict(rows):
    import numpy as np
    out = {}
    for ev in ("blocking", "zeroshot"):
        for arm in ("hyperbolic", "cond_euclidean", "flat_logreg"):
            f1s = [r["f1"] for r in rows if r["eval"] == ev and r["arm"] == arm
                   and isinstance(r["f1"], (int, float)) and np.isfinite(r["f1"])]
            if f1s:
                out[f"{ev}_{arm}_f1"] = round(float(np.mean(f1s)), 4)
    # the two headline gaps
    for ev in ("blocking", "zeroshot"):
        h = out.get(f"{ev}_hyperbolic_f1"); e = out.get(f"{ev}_cond_euclidean_f1")
        fl = out.get(f"{ev}_flat_logreg_f1")
        if h is not None and e is not None:
            out[f"{ev}_hyp_minus_euc"] = round(h - e, 4)
        if h is not None and fl is not None:
            out[f"{ev}_hyp_minus_flat"] = round(h - fl, 4)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Harm router: hyperbolic vs euclidean vs flat, blocking + zero-shot.")
    ap.add_argument("--activations", required=True)
    ap.add_argument("--out", default="./results/harm_router")
    ap.add_argument("--dataset", default="harm_taxonomy")
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--proj-dim", type=int, default=5)
    ap.add_argument("--curvature", type=float, default=0.5)
    args = ap.parse_args(argv)
    run(args.activations, args.out, dataset=args.dataset, layers=args.layers,
        seeds=tuple(args.seeds), proj_dim=args.proj_dim, curvature=args.curvature)


if __name__ == "__main__":
    main()
