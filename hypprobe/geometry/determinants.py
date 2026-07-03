"""Phase 1 MAIN SCIENCE: what drives the hyperbolicity of the activations?

We take the activations for the most-hyperbolic setting and apply controlled
edits, re-measuring delta_rel after each. The change (delta of delta_rel) attributes
the hyperbolic structure to one of three drivers:

  - token identity : replace reasoning-marker token vectors with random other
    token vectors from the same sample. If delta_rel rises, the specific tokens
    carried the tree structure.
  - order/position : shuffle the token order before pooling. Because the default
    pooling is mean-over-tokens (order-invariant), we instead build the point
    cloud from per-token vectors and permute which token represents which sample
    position; a rise means sequence order mattered.
  - meaning/semantics: compare against a paraphrase/nonce control provided in the
    activation store (if two variants per sample exist). Absent that, we
    approximate by projecting out the top principal component (a coarse
    "content" direction) and re-measuring.

Each intervention is deliberately simple and label/behaviour-preserving where
possible. The output CSV lets us read off the dominant driver, which then
DEFINES the adaptive probe's gate (Phase 2).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from ..io import (build_feature_matrix, ensure_dir, iter_samples, log_line,
                  pool_features, save_csv)
from .delta import delta_hyperbolicity


def _token_cloud(activations_dir, model, dataset, layer, source, max_samples=400,
                 variant=None):
    """Return (pooled_cloud, raw_tokens). If ``variant`` is set, only samples with
    that variant tag are used (default 'original' when variants exist)."""
    pooled, raw_tokens = [], []
    for s in iter_samples(activations_dir, model, dataset):
        if variant is not None and s.get("variant", "original") != variant:
            continue
        vec = pool_features(s, layer, source)
        if vec is None:
            continue
        pooled.append(vec)
        h = np.asarray(s["hidden"], dtype=np.float64)[min(layer, s["hidden"].shape[0] - 1)]
        raw_tokens.append((h, np.asarray(s.get("is_thinking"))))
        if len(pooled) >= max_samples:
            break
    return np.stack(pooled) if pooled else np.empty((0, 0)), raw_tokens


# --- Controlled edits ---------------------------------------------------------
# CRITICAL fairness rule (from review): every edit MUST use the SAME pooling
# operator and produce the SAME number of points as the base, so that the change
# in delta_rel reflects the named factor and NOT a change of pooling/point-set.
# We fix the pooling operator to POSITION-WEIGHTED mean: pooled = sum_t w_t h_t,
# with weights w_t that encode token position. This single operator is
# order-SENSITIVE (unlike a plain mean), so the order edit is a real test rather
# than a guaranteed null, yet it is identical across all edits.

def _pos_weights(n, rng=None, shuffle=False):
    """Position-encoding pooling weights (linearly increasing, normalised)."""
    w = np.linspace(1.0, 2.0, n)
    if shuffle and rng is not None:
        w = w[rng.permutation(n)]
    return w / w.sum()


def _pooled_base(raw_tokens):
    """Base pooling: position-weighted mean over each sample's tokens."""
    out = []
    for h, _ in raw_tokens:
        if h.shape[0] == 0:
            continue
        w = _pos_weights(h.shape[0])
        out.append((w[:, None] * h).sum(0))
    return np.stack(out) if out else np.empty((0, 0))


def _edit_token_identity(raw_tokens, rng):
    """Same position-weighted pooling, but thinking-token rows swapped for random
    rows first. Isolates TOKEN IDENTITY (pooling/point-set unchanged)."""
    out = []
    for h, is_think in raw_tokens:
        if h.shape[0] == 0:
            continue
        h2 = h.copy()
        think_idx = np.where(is_think)[0] if is_think is not None and is_think.size else []
        for ti in think_idx:
            h2[ti] = h[rng.integers(0, h.shape[0])]
        w = _pos_weights(h.shape[0])
        out.append((w[:, None] * h2).sum(0))
    return np.stack(out) if out else np.empty((0, 0))


def _edit_order_shuffle(raw_tokens, rng):
    """Same position-weighted pooling, but the POSITION WEIGHTS are shuffled, so
    each token is treated as if it sat at a random position. Isolates ORDER while
    keeping the identical pooling operator and point count."""
    out = []
    for h, _ in raw_tokens:
        if h.shape[0] == 0:
            continue
        w = _pos_weights(h.shape[0], rng=rng, shuffle=True)
        out.append((w[:, None] * h).sum(0))
    return np.stack(out) if out else np.empty((0, 0))


def _edit_placebo(raw_tokens, rng):
    """PLACEBO / null-calibration edit: re-pool the IDENTICAL tokens with the
    IDENTICAL operator (only the delta estimator's own resampling differs). Any
    |delta_change| here is pure noise -> a real driver must exceed the placebo."""
    return _pooled_base(raw_tokens)


def _edit_project_out_top_pc(X):
    """Coarse 'meaning' control (fallback when no variant activations exist).

    Removes the leading principal component as a rough proxy for a 'content'
    direction. Used only if the activation store has no nonce/paraphrase
    variants; otherwise we use the real re-run control in :func:`_variant_matrix`.
    """
    if X.shape[0] < 3:
        return X
    Xc = X - X.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    top = vt[0]
    return Xc - np.outer(Xc @ top, top)


def _variant_matrix(activations_dir, model, dataset, layer, source, variant):
    """Pool features for samples whose ``variant`` field matches (real control)."""
    from ..io import pool_features

    xs = []
    for s in iter_samples(activations_dir, model, dataset):
        if s.get("variant", "original") != variant:
            continue
        vec = pool_features(s, layer, source)
        if vec is not None:
            xs.append(vec)
    return np.stack(xs) if xs else np.empty((0, 0))


def _available_variants(activations_dir, model, dataset):
    """Which variant tags are present in this store (e.g. nonce, paraphrase)."""
    vs = set()
    for s in iter_samples(activations_dir, model, dataset):
        vs.add(s.get("variant", "original"))
    return vs


def run(activations_dir, out_dir, whiten=True, seed=0, layer=None, source="thinking"):
    ensure_dir(out_dir)
    logfile = os.path.join(os.path.dirname(out_dir.rstrip("/")) or ".", "logs", "determinants.log")
    rng = np.random.default_rng(seed)
    rows = []
    seen = sorted({(s["model"], s["dataset"]) for s in iter_samples(activations_dir)})
    for model, dataset in seen:
        sample = next(iter_samples(activations_dir, model, dataset), None)
        if sample is None:
            continue
        n_layers = int(np.asarray(sample["hidden"]).shape[0])
        use_layer = (n_layers - 1) if layer is None else layer

        variants = _available_variants(activations_dir, model, dataset)
        has_real_variants = {"nonce", "paraphrase"} & variants

        # Base uses the SHARED position-weighted pooling over the same tokens the
        # edits use, so base and every edit are strictly comparable. When variants
        # exist, base = the 'original' variant only (apples-to-apples vs meaning).
        base_variant = "original" if has_real_variants else None
        _, raw_tokens = _token_cloud(activations_dir, model, dataset, use_layer, source,
                                     variant=base_variant)
        base_X = _pooled_base(raw_tokens)
        if base_X.shape[0] < 8:
            log_line(logfile, f"{model}/{dataset}: too few '{source}' samples; skipping")
            continue
        base_res = delta_hyperbolicity(base_X, do_whiten=whiten, seed=seed)
        base = base_res.delta_rel

        edits = {
            "placebo": _edit_placebo(raw_tokens, rng),       # null calibration
            "token_identity": _edit_token_identity(raw_tokens, rng),
            "order_shuffle": _edit_order_shuffle(raw_tokens, rng),
        }
        # Meaning control: prefer REAL re-run variants. These are re-pooled with
        # the SAME operator so they stay comparable.
        for vname in ("nonce", "paraphrase"):
            if vname in variants:
                _, vtok = _token_cloud(activations_dir, model, dataset, use_layer, source,
                                       variant=vname)
                edits[f"meaning_{vname}"] = _pooled_base(vtok)
        if not has_real_variants:
            edits["meaning_topPC"] = _edit_project_out_top_pc(base_X)

        edit_rows = []
        for edit_name, X_edit in edits.items():
            if X_edit.shape[0] < 8:
                continue
            r = delta_hyperbolicity(X_edit, do_whiten=whiten, seed=seed)
            # Noise floor for this comparison = max of base and edit estimator std.
            noise = max(base_res.std_rel, r.std_rel)
            row = dict(model=model, dataset=dataset, layer=use_layer,
                       token_source=source, edit=edit_name,
                       delta_rel_base=round(base, 4), delta_rel_edit=round(r.delta_rel, 4),
                       delta_change=round(r.delta_rel - base, 4),
                       std_rel=round(noise, 4))
            edit_rows.append(row)
            rows.append(row)

        # Driver selection, gated on BOTH the placebo null and std_rel.
        placebo = next((r for r in edit_rows if r["edit"] == "placebo"), None)
        placebo_mag = abs(placebo["delta_change"]) if placebo else 0.0
        candidates = [r for r in edit_rows if r["edit"] != "placebo"]
        if candidates:
            top = max(candidates, key=lambda r: abs(r["delta_change"]))
            change = abs(top["delta_change"])
            # Must beat BOTH the estimator noise AND the placebo (real no-op) null.
            trustworthy = change > max(top["std_rel"], placebo_mag)
            for r in edit_rows:
                r["is_driver"] = (r is top and trustworthy)
                r["placebo_mag"] = round(placebo_mag, 4)
            verdict = (f"dominant driver = {top['edit']} (delta {top['delta_change']:+.4f}, "
                       f"std {top['std_rel']:.4f}, placebo {placebo_mag:.4f}) -> "
                       + ("TRUSTWORTHY" if trustworthy else "NOT ABOVE NOISE/PLACEBO (no driver)"))
            log_line(logfile, f"{model}/{dataset}: base delta_rel={base:.3f}; {verdict}")

    save_csv(os.path.join(out_dir, "attribution.csv"), rows,
             columns=["model", "dataset", "layer", "token_source", "edit",
                      "delta_rel_base", "delta_rel_edit", "delta_change", "std_rel",
                      "placebo_mag", "is_driver"])
    _maybe_plot(rows, out_dir)
    return rows


def _maybe_plot(rows, out_dir):
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    edits = sorted({r["edit"] for r in rows})
    vals = [np.mean([r["delta_change"] for r in rows if r["edit"] == e]) for e in edits]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(edits, vals)
    ax.set_ylabel("mean change in delta_rel"); ax.axhline(0, color="k", lw=0.7)
    ax.set_title("Which edit destroys the hyperbolic structure?")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "driver_effects.png"), dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 1 main science: determinants.")
    ap.add_argument("--activations", required=True)
    ap.add_argument("--out", default="./results/determinants")
    ap.add_argument("--whiten", dest="whiten", action="store_true", default=True,
                    help="whiten before delta -- ON by default (doctrine)")
    ap.add_argument("--no-whiten", dest="whiten", action="store_false",
                    help="disable whitening (NOT recommended)")
    ap.add_argument("--source", default="thinking")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    run(args.activations, args.out, whiten=args.whiten, seed=args.seed,
        layer=args.layer, source=args.source)


if __name__ == "__main__":
    main()
