"""Phase 1 (token-level): is a single token type hierarchical, and along WHAT?

Motivation (user's idea): the per-token activations are already saved, so we can
ask a finer question than the prompt-level map -- fix a token (e.g. "The") and
ask whether ITS cloud of occurrences is tree-like, and crucially along which
axis:

  * position axis : does delta_rel drop when we look at the token's geometry as a
    function of its POSITION in the sequence? (order/structure signal)
  * context axis  : does it drop across different CONTEXTS at a fixed position
    band? (meaning/context signal)

This is the token-level version of the determinants question (identity / order /
meaning) and directly feeds the robustness bridge: if hierarchy is a property of
token identity/position (cheap to mimic), geometric defense is weak; if it is a
context/meaning property, it is harder to obfuscate.

Two point modes:
  - token       : all tokens pooled into one cloud (representation-manifold shape)
  - token_type  : per frequent-token-type delta_rel, plus the position/context
                  split for each type.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from ..io import (build_token_matrix, ensure_dir, frequent_token_types,
                  iter_samples, log_line, save_csv)
from .delta import delta_hyperbolicity


def _norm_clean(tok: str) -> str:
    return tok.replace("Ġ", "").replace("▁", "").replace("Ċ", "").strip()


def run(activations_dir, out_dir, whiten=True, layer=None, top_k=12,
        min_count=30, seed=0):
    ensure_dir(out_dir)
    logfile = os.path.join(os.path.dirname(out_dir.rstrip("/")) or ".", "logs", "token_geometry.log")
    rows = []
    seen = sorted({(s["model"], s["dataset"]) for s in iter_samples(activations_dir)})
    for model, dataset in seen:
        sample = next(iter_samples(activations_dir, model, dataset), None)
        if sample is None:
            continue
        n_layers = int(np.asarray(sample["hidden"]).shape[0])
        use_layer = (n_layers - 1) if layer is None else layer

        # point_mode = token: whole-cloud manifold shape.
        X_all = build_token_matrix(activations_dir, model, dataset, use_layer)
        if X_all.shape[0] >= 8:
            res = delta_hyperbolicity(X_all, do_whiten=whiten, seed=seed)
            rows.append(dict(model=model, dataset=dataset, layer=use_layer,
                             point_mode="token", token_type="<all>", axis="cloud",
                             delta_rel=round(res.delta_rel, 4),
                             std_rel=round(res.std_rel, 4), n_points=res.n_points))

        # point_mode = token_type: per frequent token, overall + position/context.
        types = frequent_token_types(activations_dir, model, dataset,
                                     top_k=top_k, min_count=min_count)
        for tok in types:
            clean = _norm_clean(tok)

            def filt(t, target=tok):
                return t == target

            X, toks, positions, sids = build_token_matrix(
                activations_dir, model, dataset, use_layer, token_filter=filt,
                with_meta=True)
            if X.shape[0] < 8:
                continue

            overall = delta_hyperbolicity(X, do_whiten=whiten, seed=seed)
            rows.append(dict(model=model, dataset=dataset, layer=use_layer,
                             point_mode="token_type", token_type=clean, axis="overall",
                             delta_rel=round(overall.delta_rel, 4),
                             std_rel=round(overall.std_rel, 4), n_points=X.shape[0]))

            # POSITION axis: sort occurrences by position, does the ordered cloud
            # look more tree-like? We compare delta_rel of the position-sorted set
            # (structure along position) -- reported as its own row.
            pos = np.asarray(positions)
            if len(np.unique(pos)) >= 4:
                # Bin by position, take one representative per bin to expose the
                # position trajectory rather than within-bin context noise.
                order = np.argsort(pos)
                Xp = X[order]
                pos_res = delta_hyperbolicity(Xp, do_whiten=whiten, seed=seed)
                rows.append(dict(model=model, dataset=dataset, layer=use_layer,
                                 point_mode="token_type", token_type=clean,
                                 axis="position", delta_rel=round(pos_res.delta_rel, 4),
                                 std_rel=round(pos_res.std_rel, 4), n_points=Xp.shape[0]))

            # CONTEXT axis: same token, different prompts (contexts), holding
            # position roughly fixed (middle band) to vary meaning not position.
            if len(sids) >= 8:
                med = int(np.median(pos))
                band = np.abs(pos - med) <= max(1, int(0.1 * (pos.max() - pos.min() + 1)))
                Xc = X[band]
                if Xc.shape[0] >= 8:
                    ctx_res = delta_hyperbolicity(Xc, do_whiten=whiten, seed=seed)
                    rows.append(dict(model=model, dataset=dataset, layer=use_layer,
                                     point_mode="token_type", token_type=clean,
                                     axis="context", delta_rel=round(ctx_res.delta_rel, 4),
                                     std_rel=round(ctx_res.std_rel, 4), n_points=Xc.shape[0]))
        log_line(logfile, f"{model}/{dataset}: token geometry for {len(types)} token types")

    save_csv(os.path.join(out_dir, "token_geometry.csv"), rows,
             columns=["model", "dataset", "layer", "point_mode", "token_type",
                      "axis", "delta_rel", "std_rel", "n_points"])
    # Log the per-token position-vs-context contrast (the headline of this module).
    _log_contrasts(rows, logfile)
    return rows


def _log_contrasts(rows, logfile):
    """For each token type, report whether POSITION or CONTEXT is more tree-like."""
    by_tok: dict = {}
    for r in rows:
        if r["point_mode"] != "token_type":
            continue
        by_tok.setdefault((r["model"], r["token_type"]), {})[r["axis"]] = r["delta_rel"]
    for (model, tok), axes in sorted(by_tok.items()):
        pos = axes.get("position")
        ctx = axes.get("context")
        if pos is not None and ctx is not None:
            driver = "POSITION" if pos < ctx else "CONTEXT"
            log_line(logfile, f"{model} token '{tok}': delta_rel position={pos} "
                              f"context={ctx} -> more tree-like along {driver}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 1 (token-level) geometry.")
    ap.add_argument("--activations", required=True)
    ap.add_argument("--out", default="./results/geometry")
    ap.add_argument("--whiten", dest="whiten", action="store_true", default=True)
    ap.add_argument("--no-whiten", dest="whiten", action="store_false")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--min-count", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    run(args.activations, args.out, whiten=args.whiten, layer=args.layer,
        top_k=args.top_k, min_count=args.min_count, seed=args.seed)


if __name__ == "__main__":
    main()
