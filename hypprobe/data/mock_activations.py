"""Generate MOCK activation stores for CPU end-to-end testing (no LLM needed).

Produces the exact on-disk format the DGX extractor writes, but with synthetic
hidden states whose classes form a tree (so Phase 1-3 have real structure to
find). Lets us validate the geometry map, determinants, probes, eval, and
security phases locally before spending DGX time.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from ..io import ensure_dir, sample_path


def generate(out_dir, model="mock/tree-7b", dataset="wordnet_control",
             n_layers=8, hidden=64, n_samples=180, n_classes=9,
             n_tokens=12, seed=0, with_variants=True):
    """Write mock .pt samples with hierarchical structure in the late layers.

    If ``with_variants`` we also emit a 'nonce' variant (meaning destroyed ->
    weaker class signal) and a 'paraphrase' variant (meaning kept -> signal
    preserved), so the determinants meaning control has real data to compare.
    """
    rng = np.random.default_rng(seed)
    ensure_dir(out_dir)
    # Class prototypes near the boundary in a low-dim subspace -> tree-like.
    directions = rng.standard_normal((n_classes, hidden))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # variant -> class-signal multiplier. nonce destroys meaning (weak signal),
    # paraphrase preserves it (same as original).
    variant_strength = {"original": 1.0}
    if with_variants:
        variant_strength.update({"nonce": 0.15, "paraphrase": 0.95})

    count = 0
    for i in range(n_samples):
        cls = i % n_classes
        for variant, vmult in variant_strength.items():
            hidden_states = np.zeros((n_layers, n_tokens, hidden), dtype=np.float32)
            for L in range(n_layers):
                strength = (L / (n_layers - 1)) ** 2  # grows toward late layers
                base = strength * 3.0 * vmult * directions[cls]
                hidden_states[L] = (base[None, :]
                                    + rng.standard_normal((n_tokens, hidden)) * 0.5)
            is_think = np.zeros(n_tokens, bool)
            for tp in (n_tokens // 2, n_tokens - 2):
                is_think[tp] = True
                hidden_states[-1, tp] += 4.0 * vmult * directions[cls]
            is_generated = np.zeros(n_tokens, bool)
            is_generated[n_tokens // 3:] = True  # first third = prompt

            # A shared token "The" placed at a VARYING position per sample, and
            # whose late-layer activation encodes its POSITION (not the class).
            # This gives the token_type position-vs-context split real ground
            # truth: "The" should look tree-like along POSITION, flat along
            # CONTEXT. Other tokens keep unique per-slot names.
            the_pos = n_tokens // 3 + (i % (n_tokens // 3))
            tokens = [f"tok{j}" for j in range(n_tokens)]
            tokens[the_pos] = "The"
            pos_frac = the_pos / (n_tokens - 1)
            hidden_states[-1, the_pos] += 3.0 * pos_frac * directions[0]

            sid = f"s{i}" if variant == "original" else f"s{i}__{variant}"
            rec = dict(
                hidden=torch.from_numpy(hidden_states),
                tokens=tokens,
                positions=torch.arange(n_tokens),
                is_generated=torch.from_numpy(is_generated),
                is_thinking=torch.from_numpy(is_think),
                text="mock", model=model, dataset=dataset,
                sample_id=sid, label=cls, label_path=[cls // 3, cls],
                variant=variant, orig_id=f"s{i}",
            )
            torch.save(rec, sample_path(out_dir, model, dataset, sid))
            count += 1
    return count


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate mock activations for CPU tests.")
    ap.add_argument("--out", default="./results/activations")
    ap.add_argument("--n-samples", type=int, default=180)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    n = generate(args.out, n_samples=args.n_samples, seed=args.seed)
    print(f"wrote {n} mock samples to {args.out}")


if __name__ == "__main__":
    main()
