"""Phase 3 CLI: obfuscation-cost evaluation (core deliverable).

We reproduce the *idea* of Bailey et al. (2024) at the probe level: an attacker
perturbs a harmful sample's pooled activation to flip a trained probe's decision
(harmful -> benign) while staying close to the original (a proxy for preserving
behaviour). We measure the ATTACKER'S BUDGET -- the perturbation size / steps
needed -- for a flat probe vs the hyperbolic probe, and test TRANSFER (does a
perturbation optimised against one geometry fool the other?).

Honest framing (per plan): the claim is "hyperbolic RAISES the attacker's budget
/ forces more perturbation", never "unbreakable". Because attacks transfer
(Bailey et al.), we explicitly report transfer as well.

This runs on saved activations; it does not need the LLM, so it works on the DGX
or locally on cached features. The probe here is a trained binary
harmful/benign H-MLR (c>0) vs its flat limit (c=0), on the same features.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from ..io import build_feature_matrix, ensure_dir, iter_samples, log_line, save_csv
from ..probes.hmlr import HyperbolicMLR, ProbeConfig


def _train(X, y, curvature, proj_dim, seed, epochs=300, use_manifold=True):
    import torch.nn as nn
    cfg = ProbeConfig(in_dim=X.shape[1], n_classes=int(y.max() + 1), proj_dim=proj_dim,
                      curvature=curvature, use_manifold=use_manifold, seed=seed, epochs=epochs)
    model = HyperbolicMLR(cfg)
    xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); loss = loss_fn(model(xt), yt); loss.backward(); opt.step()
    return model


def _attack_budget(model, x0, target_class, max_steps=200, lr=0.05):
    """PGD-style attack: min perturbation to make model predict target_class.

    Returns (success, perturbation_norm, steps). Optimises the input feature to
    flip the prediction; budget = L2 norm of the perturbation at success.
    """
    x = torch.as_tensor(x0, dtype=torch.float32).clone().requires_grad_(True)
    x_orig = x.detach().clone()
    tgt = torch.tensor([target_class])
    opt = torch.optim.Adam([x], lr=lr)
    import torch.nn.functional as F
    for step in range(max_steps):
        opt.zero_grad()
        logits = model(x.unsqueeze(0))
        loss = F.cross_entropy(logits, tgt)
        loss.backward(); opt.step()
        with torch.no_grad():
            pred = model(x.unsqueeze(0)).argmax(-1).item()
        if pred == target_class:
            return True, float((x.detach() - x_orig).norm()), step + 1
    return False, float((x.detach() - x_orig).norm()), max_steps


def run(probes_dir_unused, activations_dir, out_dir, seed=0, proj_dim=5,
        layer=None, source="last", n_attack=40):
    ensure_dir(out_dir)
    logfile = os.path.join(os.path.dirname(out_dir.rstrip("/")) or ".", "logs", "security.log")
    rows = []
    for model_name, dataset in sorted({(s["model"], s["dataset"]) for s in iter_samples(activations_dir)}):
        sample = next(iter_samples(activations_dir, model_name, dataset), None)
        if sample is None:
            continue
        use_layer = (int(np.asarray(sample["hidden"]).shape[0]) - 1) if layer is None else layer
        X, y, _ = build_feature_matrix(activations_dir, model_name, dataset, use_layer, source)
        if X.shape[0] < 16 or len(set(y.tolist())) < 2:
            log_line(logfile, f"{model_name}/{dataset}: need >=2 classes for attack; skipping")
            continue
        # Binary view: class 0 = benign, >0 = harmful -> collapse to {0,1}.
        yb = (y > 0).astype(int)
        flat = _train(X, yb, 0.0, proj_dim, seed, use_manifold=False)
        hyp = _train(X, yb, 1.0, proj_dim, seed, use_manifold=True)

        harmful_idx = np.where(yb == 1)[0][:n_attack]
        for geom, model in [("flat", flat), ("hyperbolic", hyp)]:
            succ, norms, steps = [], [], []
            for i in harmful_idx:
                s_ok, pnorm, st = _attack_budget(model, X[i], target_class=0)
                succ.append(s_ok); norms.append(pnorm); steps.append(st)
            rows.append(dict(model=model_name, dataset=dataset, geometry=geom,
                             n_attacked=len(harmful_idx),
                             attack_success_rate=round(float(np.mean(succ)), 3),
                             mean_budget_norm=round(float(np.mean(norms)), 4),
                             mean_steps=round(float(np.mean(steps)), 1)))
        # Transfer: perturbation found on flat, applied to hyperbolic and vice versa.
        rows.append(_transfer_row(model_name, dataset, flat, hyp, X, yb, harmful_idx))
        f = rows[-3]; h = rows[-2]
        log_line(logfile, f"{model_name}/{dataset}: flat budget={f['mean_budget_norm']} "
                          f"vs hyperbolic budget={h['mean_budget_norm']} "
                          f"(higher = attacker pays more)")
    save_csv(os.path.join(out_dir, "attack.csv"), rows)
    _maybe_plot(rows, out_dir)
    return rows


def _transfer_row(model_name, dataset, flat, hyp, X, yb, harmful_idx):
    """Attack flat, then check if the same perturbed point fools hyperbolic."""
    transferred = []
    for i in harmful_idx:
        x = torch.as_tensor(X[i], dtype=torch.float32).clone().requires_grad_(True)
        tgt = torch.tensor([0]); opt = torch.optim.Adam([x], lr=0.05)
        import torch.nn.functional as F
        for _ in range(200):
            opt.zero_grad(); loss = F.cross_entropy(flat(x.unsqueeze(0)), tgt)
            loss.backward(); opt.step()
            if flat(x.unsqueeze(0)).argmax(-1).item() == 0:
                break
        with torch.no_grad():
            fooled_hyp = hyp(x.detach().unsqueeze(0)).argmax(-1).item() == 0
        transferred.append(fooled_hyp)
    return dict(model=model_name, dataset=dataset, geometry="transfer_flat_to_hyp",
                n_attacked=len(harmful_idx),
                attack_success_rate=round(float(np.mean(transferred)), 3),
                mean_budget_norm="", mean_steps="")


def _maybe_plot(rows, out_dir):
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    geoms = ["flat", "hyperbolic"]
    vals = []
    for g in geoms:
        gv = [r["mean_budget_norm"] for r in rows
              if r["geometry"] == g and isinstance(r["mean_budget_norm"], (int, float))]
        vals.append(np.mean(gv) if gv else 0.0)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(geoms, vals)
    ax.set_ylabel("mean attacker budget (perturbation L2)")
    ax.set_title("Obfuscation cost: flat vs hyperbolic")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "budget_comparison.png"), dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 3: obfuscation-cost evaluation.")
    ap.add_argument("--probes", default=None, help="(unused; kept for run_all.sh)")
    ap.add_argument("--activations", default="./results/activations")
    ap.add_argument("--out", default="./results/security")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proj-dim", type=int, default=5)
    ap.add_argument("--source", default="last")
    ap.add_argument("--layer", type=int, default=None)
    args = ap.parse_args(argv)
    run(args.probes, args.activations, args.out, seed=args.seed,
        proj_dim=args.proj_dim, layer=args.layer, source=args.source)


if __name__ == "__main__":
    main()
