"""Determinants v2 — the main-science module, rebuilt on the re-evaluation.

Fixes, mapped to confirmed findings:

  SOURCE BUG      — v1 pooled ALL tokens regardless of --source ("determinants
                    ignores its own source argument"). v2 masks tokens to the
                    requested source with the SAME logic as io.pool_features
                    (including sink stripping) before pooling.

  VACUOUS PLACEBO — v1's placebo re-pooled identical tokens with the identical
                    seed -> delta_change EXACTLY 0 always; the driver gate
                    degenerated and produced a demonstrated false positive.
                    v2 uses REAL stochastic nulls:
                      * same-point-set edits (identity/order): the null is the
                        distribution of |delta_change| over R independent RNG
                        redraws of the edit itself, PLUS a split-half null of
                        the base cloud (different-sample noise).
                      * meaning edits (different extracted samples): the null
                        is the split-half base null — the honest
                        different-sample noise scale, 2-4x larger than
                        quadruple noise.
                    A driver must beat its OWN null at the 95th percentile.

  UNDERPOWERED ORDER EDIT — linear weights (max/min = 2, CV 0.21) capped the
                    order effect near the noise floor by construction. v2 uses
                    exponential position weights (CV ~0.87, ~3.7x displacement)
                    — still ONE shared operator across base and all edits, so
                    the fairness rule holds — AND reports the operator's
                    max-attainable effect (power) from a synthetic
                    order-structured control so "no effect" is interpretable.

  WRONG TARGET    — the DGX run located the low-delta signal at the LAST token,
                    where v1's edits are vacuous (a last-token cloud has one
                    token per sample: nothing to swap or shuffle). v2 adds the
                    last-token cluster adjudication: is the last-token cloud's
                    low delta explained by ANSWER-CLASS CLUSTERING (the boring
                    star-tree explanation)? Uses calibration.cluster_null_cloud
                    with the answer/depth labels.

Outputs results/determinants_v2/attribution_v2.csv (+ power + null columns) and
a plain-English log line per (model, dataset).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from ..io import _sink_mask, ensure_dir, iter_samples, log_line, save_csv
from ..manifest import write_manifest
from .calibration import calibrated_delta
from .delta import delta_hyperbolicity


# ---------------------------------------------------------------------------
# Source-respecting token gathering (the v1 source bug fix)
# ---------------------------------------------------------------------------

def _source_mask(sample, h, source):
    """Boolean keep-mask over tokens for a token source, sink-stripped.

    Mirrors io.pool_features' masking exactly (the v1 bug was NOT reusing it).
    """
    is_gen = np.asarray(sample.get("is_generated"))
    is_think = np.asarray(sample.get("is_thinking"))
    n = h.shape[0]
    if source == "input":
        mask = ~is_gen if is_gen is not None and is_gen.size else np.ones(n, bool)
    elif source == "generated":
        mask = is_gen if is_gen is not None and is_gen.size else np.zeros(n, bool)
    elif source == "thinking":
        mask = is_think if is_think is not None and is_think.size else np.zeros(n, bool)
    elif source == "all":
        mask = np.ones(n, bool)
    else:
        raise ValueError(f"determinants_v2 does not support source={source!r} "
                         f"(use 'input'/'generated'/'thinking'/'all'; the LAST-token "
                         f"locus is handled by last_token_adjudication)")
    return mask & _sink_mask(h)


def _gather(activations_dir, model, dataset, layer, source, variant=None,
            max_samples=400):
    """Per-sample (tokens_in_source, is_thinking_in_source, label) triples."""
    out = []
    for s in iter_samples(activations_dir, model, dataset):
        if variant is not None and s.get("variant", "original") != variant:
            continue
        hidden = np.asarray(s["hidden"], dtype=np.float64)
        L = min(layer, hidden.shape[0] - 1)
        h = hidden[L]
        mask = _source_mask(s, h, source)
        if mask.sum() < 2:
            continue
        is_think = np.asarray(s.get("is_thinking"))
        think_sub = (is_think[mask] if is_think is not None and is_think.size
                     else np.zeros(int(mask.sum()), bool))
        out.append((h[mask], think_sub, int(s.get("label", 0))))
        if len(out) >= max_samples:
            break
    return out


# ---------------------------------------------------------------------------
# ONE shared pooling operator (exponential position weights — the power fix)
# ---------------------------------------------------------------------------

def _pos_weights(n, rng=None, shuffle=False, rate=3.0):
    """Exponential position weights w_t ∝ exp(rate * t/(n-1)), normalised.

    CV ~0.87 (vs 0.21 for v1's linear weights): shuffling these moves the
    pooled point ~3.7x further for the same token spread, making the order test
    actually powered. Identical operator across base and every edit.
    """
    t = np.linspace(0.0, 1.0, max(n, 1))
    w = np.exp(rate * t)
    if shuffle and rng is not None:
        w = w[rng.permutation(n)]
    return w / w.sum()


def _pool(gathered, rng=None, edit=None):
    """Pool each sample's tokens with the shared operator, applying an edit.

    edit=None            : base pooling
    edit='identity'      : thinking-token rows swapped for random same-sample rows
    edit='order'         : position weights shuffled
    """
    out = []
    for h, is_think, _ in gathered:
        n = h.shape[0]
        if n == 0:
            continue
        h2 = h
        shuffle = False
        if edit == "identity" and rng is not None:
            h2 = h.copy()
            for ti in np.where(is_think)[0]:
                h2[ti] = h[rng.integers(0, n)]
        elif edit == "order":
            shuffle = True
        w = _pos_weights(n, rng=rng, shuffle=shuffle)
        out.append((w[:, None] * h2).sum(0))
    return np.stack(out) if out else np.empty((0, 0))


# ---------------------------------------------------------------------------
# Real nulls (the vacuous-placebo fix)
# ---------------------------------------------------------------------------

def split_half_null(base_X, metric, seed, n_splits=20, pca_cap=256):
    """Different-sample noise: |delta(half A) - delta(half B)| over random splits.

    This is the null for ANY comparison between clouds made of different
    samples (the meaning edits especially). v1 had nothing like it and used
    quadruple-sampling std, which understates this noise 2-4x.
    """
    rng = np.random.default_rng(seed)
    n = base_X.shape[0]
    diffs = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        ra = delta_hyperbolicity(base_X[a], metric=metric, seed=seed, pca_cap=pca_cap)
        rb = delta_hyperbolicity(base_X[b], metric=metric, seed=seed, pca_cap=pca_cap)
        diffs.append(abs(ra.delta_rel - rb.delta_rel))
    return float(np.percentile(diffs, 95.0))


def edit_redraw_null(gathered, edit, base_delta, metric, seed, n_redraws=20,
                     pca_cap=256):
    """Same-point-set null: spread of delta_change over independent edit redraws.

    For stochastic edits (identity swap / order shuffle) the honest null is the
    edit's OWN randomness: redraw it R times with different RNGs and take the
    95th percentile of |delta_change|. (v1's placebo was deterministic zero.)
    """
    changes = []
    for r in range(n_redraws):
        rng = np.random.default_rng(seed + 1000 + r)
        X = _pool(gathered, rng=rng, edit=edit)
        if X.shape[0] < 8:
            continue
        res = delta_hyperbolicity(X, metric=metric, seed=seed, pca_cap=pca_cap)
        changes.append(res.delta_rel - base_delta)
    if not changes:
        return float("nan"), float("nan")
    return float(np.mean(changes)), float(np.percentile(np.abs(np.asarray(changes)
                 - np.mean(changes)), 95.0))


def order_power(gathered, metric, seed, pca_cap=256):
    """Max attainable order effect under the shared operator (power analysis).

    Build a synthetic control where position CARRIES all the structure: replace
    each sample's tokens with vectors that encode position deterministically,
    then measure how much delta the order shuffle moves. If even this ceiling
    is ~0, the operator cannot detect order and 'no order effect' on real data
    is uninterpretable (this is exactly what v1 could not tell us).
    """
    rng = np.random.default_rng(seed)
    synth = []
    dim = gathered[0][0].shape[1]
    base_dir = rng.standard_normal(dim)
    base_dir /= np.linalg.norm(base_dir)
    depth_dir = rng.standard_normal(dim)
    depth_dir /= np.linalg.norm(depth_dir)
    for h, is_think, label in gathered:
        n = h.shape[0]
        t = np.linspace(0, 1, n)[:, None]
        toks = (t * base_dir[None, :] * 3.0
                + (label + 1) * 0.5 * depth_dir[None, :]
                + 0.1 * rng.standard_normal((n, dim)))
        synth.append((toks, is_think, label))
    Xb = _pool(synth, rng=np.random.default_rng(seed))
    Xo = _pool(synth, rng=np.random.default_rng(seed), edit="order")
    rb = delta_hyperbolicity(Xb, metric=metric, seed=seed, pca_cap=pca_cap)
    ro = delta_hyperbolicity(Xo, metric=metric, seed=seed, pca_cap=pca_cap)
    return abs(ro.delta_rel - rb.delta_rel)


# ---------------------------------------------------------------------------
# Last-token cluster adjudication (pointing at what the DGX actually found)
# ---------------------------------------------------------------------------

def last_token_adjudication(activations_dir, model, dataset, layer, metric,
                            seed, pca_cap=256, max_samples=400):
    """Is the last-token cloud's low delta just answer-class clustering?

    Uses calibrated_delta's cluster null: a GMM with the SAME class means and
    per-class scale as the data. If excess_over_cluster ~ 0, the celebrated
    low-delta 'last' locus is a star of answer clusters — no hierarchy story.
    """
    xs, ys = [], []
    for s in iter_samples(activations_dir, model, dataset):
        if s.get("variant", "original") != "original":
            continue
        hidden = np.asarray(s["hidden"], dtype=np.float64)
        L = min(layer, hidden.shape[0] - 1)
        xs.append(hidden[L, -1])
        ys.append(int(s.get("label", 0)))
        if len(xs) >= max_samples:
            break
    if len(xs) < 16:
        return None
    return calibrated_delta(np.stack(xs), np.asarray(ys), metric=metric,
                            seed=seed, pca_cap=pca_cap)


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(activations_dir, out_dir, source="generated", layer=None, metric="per_cloud",
        seed=0, n_redraws=20, n_splits=20, pca_cap=256):
    ensure_dir(out_dir)
    logfile = os.path.join(os.path.dirname(out_dir.rstrip("/")) or ".",
                           "logs", "determinants_v2.log")
    rows = []
    seen = sorted({(s["model"], s["dataset"]) for s in iter_samples(activations_dir)})
    for model, dataset in seen:
        sample = next(iter_samples(activations_dir, model, dataset), None)
        if sample is None:
            continue
        n_layers = int(np.asarray(sample["hidden"]).shape[0])
        use_layer = (n_layers - 1) if layer is None else layer

        variants = {s.get("variant", "original")
                    for s in iter_samples(activations_dir, model, dataset)}
        base_variant = "original" if ({"nonce", "paraphrase"} & variants) else None

        gathered = _gather(activations_dir, model, dataset, use_layer, source,
                           variant=base_variant)
        if len(gathered) < 16:
            log_line(logfile, f"{model}/{dataset}: <16 samples with '{source}' "
                              f"tokens at L{use_layer}; skipping")
            continue

        base_X = _pool(gathered, rng=np.random.default_rng(seed))
        base = delta_hyperbolicity(base_X, metric=metric, seed=seed, pca_cap=pca_cap)
        null_split = split_half_null(base_X, metric, seed, n_splits=n_splits,
                                     pca_cap=pca_cap)
        pw_order = order_power(gathered, metric, seed, pca_cap=pca_cap)

        # -- same-point-set edits, each judged against its own redraw null --
        edit_results = {}
        for edit in ("identity", "order"):
            mean_change, redraw_null = edit_redraw_null(
                gathered, edit, base.delta_rel, metric, seed,
                n_redraws=n_redraws, pca_cap=pca_cap)
            null95 = max(redraw_null, null_split)
            edit_results[f"token_{edit}"] = (mean_change, null95)

        # -- meaning edits: real re-extracted variants vs split-half null --
        for vname in ("nonce", "paraphrase"):
            if vname not in variants:
                continue
            vg = _gather(activations_dir, model, dataset, use_layer, source,
                         variant=vname)
            if len(vg) < 16:
                continue
            Xv = _pool(vg, rng=np.random.default_rng(seed))
            rv = delta_hyperbolicity(Xv, metric=metric, seed=seed, pca_cap=pca_cap)
            edit_results[f"meaning_{vname}"] = (rv.delta_rel - base.delta_rel,
                                                null_split)

        for edit, (change, null95) in edit_results.items():
            is_driver_candidate = (np.isfinite(change)
                                   and abs(change) > null95)
            rows.append(dict(model=model, dataset=dataset, layer=use_layer,
                             token_source=source, metric=metric, edit=edit,
                             delta_rel_base=round(base.delta_rel, 4),
                             delta_change=round(change, 4) if np.isfinite(change) else "",
                             null95=round(null95, 4) if np.isfinite(null95) else "",
                             split_half_null=round(null_split, 4),
                             order_power_ceiling=round(pw_order, 4),
                             beats_null=is_driver_candidate))

        # -- driver call: largest |change| among those that beat their null --
        cands = [r for r in rows
                 if r["model"] == model and r["dataset"] == dataset
                 and r["beats_null"] and r["delta_change"] != ""]
        if cands:
            top = max(cands, key=lambda r: abs(float(r["delta_change"])))
            for r in rows:
                if r["model"] == model and r["dataset"] == dataset:
                    r["is_driver"] = (r is top)
            verdict = (f"driver = {top['edit']} (delta {top['delta_change']:+}, "
                       f"null95 {top['null95']})")
        else:
            for r in rows:
                if r["model"] == model and r["dataset"] == dataset:
                    r["is_driver"] = False
            verdict = "NO driver beats its null (report as null result)"
        if pw_order < 2 * null_split:
            verdict += (f" [CAUTION: order-power ceiling {pw_order:.4f} < 2x "
                        f"split-half null {null_split:.4f} -> the order test is "
                        f"underpowered here; 'no order effect' is uninterpretable]")
        log_line(logfile, f"{model}/{dataset} [{source}@L{use_layer}, {metric}]: "
                          f"base={base.delta_rel:.3f}; {verdict}")

        # -- last-token adjudication (the DGX H3 locus) --
        cal = last_token_adjudication(activations_dir, model, dataset, use_layer,
                                      metric, seed, pca_cap=pca_cap)
        if cal is not None:
            explained = (np.isfinite(cal.excess_over_cluster)
                         and cal.excess_over_cluster <= max(cal.noise_floor, 1e-6))
            rows.append(dict(model=model, dataset=dataset, layer=use_layer,
                             token_source="last", metric=metric,
                             edit="cluster_adjudication",
                             delta_rel_base=round(cal.delta_data, 4),
                             delta_change=round(cal.excess_over_cluster, 4)
                             if np.isfinite(cal.excess_over_cluster) else "",
                             null95=round(cal.noise_floor, 4),
                             split_half_null="", order_power_ceiling="",
                             beats_null=not explained, is_driver=False))
            log_line(logfile,
                     f"{model}/{dataset} LAST-token adjudication: delta={cal.delta_data:.3f}, "
                     f"cluster_null={cal.delta_cluster:.3f}, excess={cal.excess_over_cluster:+.3f} "
                     + ("-> EXPLAINED BY CLASS CLUSTERING (star tree, no hierarchy)"
                        if explained else "-> exceeds the cluster null (structure beyond classes)"))

    save_csv(os.path.join(out_dir, "attribution_v2.csv"), rows,
             columns=["model", "dataset", "layer", "token_source", "metric", "edit",
                      "delta_rel_base", "delta_change", "null95", "split_half_null",
                      "order_power_ceiling", "beats_null", "is_driver"])
    write_manifest(out_dir, "determinants_v2",
                   args=dict(activations=activations_dir, source=source,
                             layer=layer, metric=metric, seed=seed,
                             n_redraws=n_redraws, n_splits=n_splits))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Determinants v2 (real nulls, "
                                             "source-respecting, powered).")
    ap.add_argument("--activations", required=True)
    ap.add_argument("--out", default="./results/determinants_v2")
    ap.add_argument("--source", default="generated",
                    help="token source for the edits (default 'generated' — the "
                         "axis the Atlas never measured)")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--metric", default="per_cloud")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-redraws", type=int, default=20)
    ap.add_argument("--n-splits", type=int, default=20)
    args = ap.parse_args(argv)
    run(args.activations, args.out, source=args.source, layer=args.layer,
        metric=args.metric, seed=args.seed, n_redraws=args.n_redraws,
        n_splits=args.n_splits)


if __name__ == "__main__":
    main()
