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


def _token_cloud(activations_dir, model, dataset, layer, source, max_samples=400):
    """Build a per-sample pooled cloud AND keep raw per-token vectors for edits."""
    pooled, raw_tokens = [], []
    for s in iter_samples(activations_dir, model, dataset):
        vec = pool_features(s, layer, source)
        if vec is None:
            continue
        pooled.append(vec)
        h = np.asarray(s["hidden"], dtype=np.float64)[min(layer, s["hidden"].shape[0] - 1)]
        raw_tokens.append((h, np.asarray(s.get("is_thinking"))))
        if len(pooled) >= max_samples:
            break
    return np.stack(pooled) if pooled else np.empty((0, 0)), raw_tokens


def _edit_token_identity(raw_tokens, rng):
    """Rebuild pooled vectors after swapping thinking-token rows for random rows."""
    out = []
    for h, is_think in raw_tokens:
        if h.shape[0] == 0:
            continue
        h2 = h.copy()
        think_idx = np.where(is_think)[0] if is_think is not None and is_think.size else []
        for ti in think_idx:
            out_idx = rng.integers(0, h.shape[0])
            h2[ti] = h[out_idx]
        out.append(h2.mean(axis=0))
    return np.stack(out) if out else np.empty((0, 0))


def _edit_order_shuffle(raw_tokens, rng):
    """Pool after shuffling token order (tests order via last-token-style pooling)."""
    out = []
    for h, _ in raw_tokens:
        if h.shape[0] == 0:
            continue
        perm = rng.permutation(h.shape[0])
        # Use the (post-shuffle) last token, so order actually matters.
        out.append(h[perm][-1])
    return np.stack(out) if out else np.empty((0, 0))


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

        if has_real_variants:
            # Base = ORIGINAL variant only, so comparisons are apples-to-apples.
            base_X = _variant_matrix(activations_dir, model, dataset, use_layer, source, "original")
        else:
            base_X, _, _ = build_feature_matrix(activations_dir, model, dataset, use_layer, source)
        if base_X.shape[0] < 8:
            log_line(logfile, f"{model}/{dataset}: too few '{source}' samples; skipping")
            continue
        base = delta_hyperbolicity(base_X, do_whiten=whiten, seed=seed).delta_rel

        _, raw_tokens = _token_cloud(activations_dir, model, dataset, use_layer, source)
        edits = {
            "token_identity": _edit_token_identity(raw_tokens, rng),
            "order_shuffle": _edit_order_shuffle(raw_tokens, rng),
        }
        # Meaning control: prefer REAL re-run variants (nonce destroys meaning,
        # keeps structure; paraphrase keeps meaning). Fall back to top-PC proxy.
        if "nonce" in variants:
            edits["meaning_nonce"] = _variant_matrix(
                activations_dir, model, dataset, use_layer, source, "nonce")
        if "paraphrase" in variants:
            edits["meaning_paraphrase"] = _variant_matrix(
                activations_dir, model, dataset, use_layer, source, "paraphrase")
        if not has_real_variants:
            edits["meaning_topPC"] = _edit_project_out_top_pc(base_X)
        for edit_name, X_edit in edits.items():
            if X_edit.shape[0] < 8:
                continue
            d = delta_hyperbolicity(X_edit, do_whiten=whiten, seed=seed).delta_rel
            rows.append(dict(model=model, dataset=dataset, layer=use_layer,
                             token_source=source, edit=edit_name,
                             delta_rel_base=round(base, 4), delta_rel_edit=round(d, 4),
                             delta_change=round(d - base, 4)))
        # Which edit moved delta_rel the most -> dominant driver.
        model_rows = [r for r in rows if r["model"] == model and r["dataset"] == dataset]
        if model_rows:
            driver = max(model_rows, key=lambda r: abs(r["delta_change"]))
            log_line(logfile, f"{model}/{dataset}: base delta_rel={base:.3f}; "
                              f"dominant driver = {driver['edit']} "
                              f"(delta {driver['delta_change']:+.3f})")

    save_csv(os.path.join(out_dir, "attribution.csv"), rows,
             columns=["model", "dataset", "layer", "token_source", "edit",
                      "delta_rel_base", "delta_rel_edit", "delta_change"])
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
    ap.add_argument("--whiten", action="store_true", default=False)
    ap.add_argument("--source", default="thinking")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    run(args.activations, args.out, whiten=args.whiten, seed=args.seed,
        layer=args.layer, source=args.source)


if __name__ == "__main__":
    main()
