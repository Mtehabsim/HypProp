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

            # Inject an ATTENTION SINK at position 0 in the middle layers, matching
            # the real DGX finding (~250x median norm). This reproduces the delta_rel
            # degeneracy so the sink-stripping fix (io._sink_mask) is exercised
            # end-to-end: without the fix, middle-layer delta collapses; with it,
            # the sink is removed and delta is sane.
            mid = n_layers // 2
            for L in range(max(1, mid - 3), min(n_layers - 1, mid + 4)):
                hidden_states[L, 0] = 250.0 * rng.standard_normal(hidden)

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


def _edge_prototypes(parent, dim, rng, decay=0.6):
    """Node prototypes = sum of shrinking edge vectors along the root->node path.

    Two nodes sharing a longer path-prefix share more edge components, so
    Euclidean prototype distance tracks tree distance (siblings < cousins) and a
    node's NORM grows with depth (the radial signal). Same construction as
    :func:`synthetic.hierarchy_features`, but per arbitrary parent list.
    """
    n = len(parent)
    children = [[] for _ in range(n)]
    roots = []
    for c, p in enumerate(parent):
        (children[p].append(c) if p >= 0 else roots.append(c))
    proto = np.zeros((n, dim))
    order = list(roots)
    head = 0
    edge_vecs = {}
    while head < len(order):
        u = order[head]; head += 1
        for v in children[u]:
            d = rng.standard_normal(dim); d /= (np.linalg.norm(d) + 1e-9)
            depth_v = 0; p = parent[v]
            while p >= 0:
                depth_v += 1; p = parent[p]
            edge_vecs[v] = (decay ** depth_v) * d
            proto[v] = proto[u] + edge_vecs[v]
            order.append(v)
    mx = np.linalg.norm(proto, axis=1).max() + 1e-9
    return 0.9 * proto / mx


def _hyperbolic_cone_layout(parent, depth, dim, seed=12345, radial_rate=0.7):
    """Shared hyperbolic-cone embedding of a tree, lifted to ``dim``.

    Deterministic given the tree SHAPE (parent array), so all prompts sharing a
    shape (all fictional prompts of one branching use the identical
    ``_build_tree`` output) get the IDENTICAL layout — which is what the shared
    decoder in ``tree_probe`` needs to have something to recover, mirroring the
    hypothesis that the model writes tree position into a consistent subspace.

    Nodes are placed in a 2-D Poincare disk: radius grows with depth (so norm
    encodes generality — the radial signal) and each subtree owns an angular
    wedge subdivided among its children (so the tree metric is realised
    hyperbolically). This is the classic construction where a branching tree
    embeds with low distortion in 2-D HYPERBOLIC space but NOT in low-dim
    Euclidean — the crisp positive control the instrument must detect. The 2-D
    coords are lifted into ``dim`` by a shared random rotation, so ambient norm =
    disk radius = a monotone function of depth.
    """
    import math
    rng = np.random.default_rng(seed)
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    roots = []
    for c, p in enumerate(parent):
        (children[p].append(c) if p >= 0 else roots.append(c))
    angle = np.zeros(n)
    r = np.zeros(n)

    def assign(node, a0, a1, d):
        angle[node] = 0.5 * (a0 + a1)
        r[node] = math.tanh(radial_rate * d)
        ch = children[node]
        if ch:
            step = (a1 - a0) / len(ch)
            for k, c in enumerate(ch):
                assign(c, a0 + k * step, a0 + (k + 1) * step, d + 1)

    if len(roots) == 1:
        assign(roots[0], 0.0, 2 * math.pi, 0)
    else:
        step = 2 * math.pi / max(len(roots), 1)
        for k, rt in enumerate(roots):
            assign(rt, k * step, (k + 1) * step, 0)

    coords2 = np.stack([r * np.cos(angle), r * np.sin(angle)], axis=1)  # (n, 2)
    Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    emb = np.zeros((n, dim))
    emb[:, :2] = coords2
    return emb @ Q.T  # (n, dim); ||row|| == disk radius == tanh(radial_rate*depth)


def _bpe_tokens(word, rng, split_prob=0.5):
    """Split a word into byte-level-BPE-style sub-tokens (leading 'Ġ' on first)."""
    if len(word) > 4 and rng.random() < split_prob:
        cut = rng.integers(2, len(word) - 1)
        return ["Ġ" + word[:cut], word[cut:]]
    return ["Ġ" + word]


def generate_prontoqa_tree(out_dir, model="mock/tree-7b", n_prompts=24,
                           n_nodes=15, n_layers=8, hidden=64, signal_layer=5,
                           noise=0.15, seed=0):
    """Mock activation store for the tree probe (CPU positive control).

    Uses the real branching-ontology generator (so ``tree_meta`` is genuine) and
    places each concept token's hidden state at a tree-faithful prototype in one
    'signal' layer (weak elsewhere). Tokens are byte-level-BPE-style (Ġ markers,
    multi-sub-token concepts) so the alignment path is exercised as on real data.
    Concept tree structure is present in ``signal_layer`` -> the instrument SHOULD
    recover it there and NOT in the noise layers (a within-store negative check).
    """
    from .prontoqa_tree import build_prontoqa_tree_all
    ensure_dir(out_dir)
    rng = np.random.default_rng(seed)
    rows = build_prontoqa_tree_all(n_prompts=n_prompts, n_nodes=n_nodes, seed=seed)
    count = 0
    for r in rows:
        tm = r["tree_meta"]
        names, parent = tm["names"], tm["parent"]
        # SHARED hyperbolic-cone layout keyed on the tree SHAPE, scaled up so the
        # signal dominates the additive noise. All prompts of one branching share
        # the identical layout -> the shared decoder has consistent structure to
        # recover, and it is genuinely LOW-DIM HYPERBOLIC (2-D disk lifted), so a
        # hyperbolic decoder beats a flat one at small m (the positive control).
        proto = 3.0 * _hyperbolic_cone_layout(parent, tm["depth"], hidden)

        # Build a token stream from the prompt, emitting concept sub-tokens and a
        # 'Question:' boundary; record, per concept occurrence, its last sub-token.
        prompt = r["prompt"]
        pre, _, ques = prompt.partition("\nQuestion:")
        tokens, concept_at = [], {}   # token_index -> node_id (last sub-token)

        def emit_text(text, is_question):
            for word in text.split():
                stripped = word.strip(".?!,:;")
                nid = names.index(stripped) if stripped in names else None
                subs = (_bpe_tokens(stripped, rng) if stripped else [])
                for k, sub in enumerate(subs):
                    tokens.append(sub)
                    if nid is not None and k == len(subs) - 1:
                        concept_at[len(tokens) - 1] = nid
                # trailing punctuation as its own token
                if word != stripped and word[len(stripped):]:
                    tokens.append(word[len(stripped):])

        emit_text(pre, False)
        prompt_len_marker = len(tokens)
        tokens.append("ĠQuestion")
        tokens.append(":")
        emit_text(ques, True)

        n_tok = len(tokens)
        prompt_len = n_tok            # mock: treat all as prompt (no generation)
        H = np.zeros((n_layers, n_tok, hidden), dtype=np.float32)
        for L in range(n_layers):
            H[L] = rng.standard_normal((n_tok, hidden)) * noise
        # write tree-faithful prototypes at concept tokens in the signal layer
        for ti, nid in concept_at.items():
            H[signal_layer, ti] = proto[nid] + noise * rng.standard_normal(hidden)

        rec = dict(
            hidden=torch.from_numpy(H),
            tokens=tokens,
            positions=torch.arange(n_tok),
            is_generated=torch.zeros(n_tok, dtype=torch.bool),
            is_thinking=torch.zeros(n_tok, dtype=torch.bool),
            text="mock", model=model, dataset="prontoqa_tree",
            sample_id=r["sample_id"], label=r["label"], label_path=r["label_path"],
            variant="original", orig_id=r["sample_id"],
            answer=r.get("answer"), tree_meta=tm, prompt_len=prompt_len,
        )
        torch.save(rec, sample_path(out_dir, model, "prontoqa_tree", r["sample_id"]))
        count += 1
    return count


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate mock activations for CPU tests.")
    ap.add_argument("--out", default="./results/activations")
    ap.add_argument("--n-samples", type=int, default=180)
    ap.add_argument("--tree", action="store_true", help="generate prontoqa_tree mock")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.tree:
        n = generate_prontoqa_tree(args.out, seed=args.seed)
        print(f"wrote {n} mock tree samples to {args.out}")
        return
    n = generate(args.out, n_samples=args.n_samples, seed=args.seed)
    print(f"wrote {n} mock samples to {args.out}")


if __name__ == "__main__":
    main()
