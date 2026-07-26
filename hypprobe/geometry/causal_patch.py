"""Causal test: does the mid-stack TREE SUBSPACE drive the model's answer?

All prior findings are correlational (the tree is decodable). This asks the causal
question: if we ABLATE the decoded tree directions at the peak layer during a
forward pass, does the model's multi-hop answer degrade MORE than if we ablate a
matched random subspace? If yes, the hierarchy is used, not epiphenomenal.

Method (per prompt, at the peak layer L*):
  1. Identify the k-dim TREE SUBSPACE B_tree = top-k right-singular directions of
     the trained tree-probe's projection on this layer's concept reps (the
     directions that carry tree-distance info).
  2. Register a forward hook on layer L* that projects the residual stream onto
     the orthogonal complement of B_tree:  h' = mu + (h-mu) - ((h-mu)B)B^T.
  3. Re-run the model, read the True/False answer logit. Compare answer accuracy
     (and answer-logit margin) under three conditions:
       clean          — no ablation
       ablate_tree    — remove B_tree
       ablate_random  — remove a matched random subspace (top-PC span, variance-
                        matched to B_tree; see composition validation) — the fair
                        control that isolates "this subspace" from "high-variance
                        directions in general".

Decision: tree is CAUSAL iff accuracy_drop(ablate_tree) > accuracy_drop(random)
by a margin, per-seed. Also report drop-per-unit-variance-removed (the fairest
scalar). ProntoQA multi-hop True/False is the task; the answer is decodable from
the final-token logits over the " True"/" False" tokens.

fp32, ulimit-safe. Reuses hidden_state_extractor's model loader conventions.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from ..io import ensure_dir, log_line, save_csv, save_json
from ..manifest import write_manifest


def _tree_subspace(reps, target_d, k, proj_dim=5, seed=0, epochs=300):
    """Return a (d, k) orthonormal basis of the directions the tree-probe uses.

    We fit a light linear map W (d->proj_dim) so ||W(h_i)-W(h_j)|| tracks the
    ground-truth tree distance target_d, then take the top-k right-singular
    vectors of W as the tree subspace. (A full hyperbolic probe is unnecessary
    here — we only need the DIRECTIONS that carry tree structure; the linear map's
    principal axes capture them and are curvature-agnostic, which keeps the causal
    test from smuggling in the geometry we're separately measuring.)
    """
    x = torch.as_tensor(reps, dtype=torch.float32)
    tgt = torch.as_tensor(target_d, dtype=torch.float32)
    W = torch.nn.Linear(x.shape[1], proj_dim, bias=False)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(W.parameters(), lr=1e-2)
    mask = ~torch.eye(len(x), dtype=torch.bool)
    tgt_n = (tgt[mask] ** 2).sum().clamp_min(1e-9)
    for _ in range(epochs):
        opt.zero_grad()
        z = W(x)
        dz = torch.cdist(z, z)
        loss = ((dz[mask] - tgt[mask]) ** 2).sum() / tgt_n
        loss.backward(); opt.step()
    Wm = W.weight.detach().numpy()             # (proj_dim, d)
    U, S, Vt = np.linalg.svd(Wm, full_matrices=False)
    B = Vt[:k].T                                # (d, k) top-k input directions
    # orthonormalize (SVD rows already orthonormal, but be safe)
    Q, _ = np.linalg.qr(B)
    return Q[:, :k]


def _matched_random_subspace(H, B_tree, k, seed=0, n_try=100):
    """Random k-dim subspace from the top-PC span, variance-matched to B_tree.

    Fair control: removing it should delete ~the same total variance as removing
    B_tree, so a larger accuracy drop from B_tree is about tree STRUCTURE, not
    variance magnitude (the pitfall caught in composition validation)."""
    Hc = H - H.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
    top = Vt[: min(4 * k + 10, Vt.shape[0])].T
    def var_removed(B):
        proj = Hc - (Hc @ B) @ B.T
        return float(Hc.var() - proj.var())
    target = var_removed(B_tree)
    rng = np.random.default_rng(seed)
    best = None
    for s in range(n_try):
        M, _ = np.linalg.qr(top @ rng.standard_normal((top.shape[1], k)))
        M = M[:, :k]
        diff = abs(var_removed(M) - target)
        if best is None or diff < best[0]:
            best = (diff, M)
    return best[1]


def _project_out(h, B, mu):
    """h' = mu + (h-mu) - ((h-mu) B) B^T  — remove subspace B (torch, on device)."""
    hc = h - mu
    return mu + hc - (hc @ B) @ B.T


def run(model_name, cache, out_dir, layer=None, k=5, limit=200, seeds=(0, 1, 2, 3, 4, 5),
        dataset="prontoqa_tree", device="cuda", dtype="fp32"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ensure_dir(out_dir)
    logfile = os.path.join(os.path.dirname(out_dir.rstrip("/")) or ".", "logs", "causal_patch.log")

    # --- load prompts (need the ground-truth tree + True/False answer) ---
    path = os.path.join(cache, f"{dataset}.jsonl")
    samples = [json.loads(l) for l in open(path)]
    samples = [s for s in samples if s.get("tree_meta") and "answer" in s][:limit]
    log_line(logfile, f"causal_patch: {len(samples)} prompts from {path}")

    tok = AutoTokenizer.from_pretrained(model_name)
    torch_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}.get(dtype, torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype).to(device).eval()
    n_layers = model.config.num_hidden_layers
    L = layer if layer is not None else n_layers // 2      # default mid-stack
    log_line(logfile, f"{model_name}: {n_layers} layers, patching L{L}, k={k}")

    # token ids for " True"/" False" answers
    def _ans_ids(word):
        ids = tok.encode(word, add_special_tokens=False)
        return ids[0] if ids else None
    true_id, false_id = _ans_ids(" True"), _ans_ids(" False")

    # --- pass 1: collect clean layer-L reps at last token + build tree target ---
    reps, gold, prompts_txt = [], [], []
    for s in samples:
        ids = tok(s["prompt"], return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        h = out.hidden_states[L][0, -1].float().cpu().numpy()   # last-token rep at L
        reps.append(h); gold.append(int(s["answer"])); prompts_txt.append(s["prompt"])
    reps = np.stack(reps)

    # tree target over prompts: |answer_i - answer_j| is too flat; use the queried
    # concept's DEPTH difference as a graded per-prompt target (proxy for "how much
    # tree structure this prompt's rep should carry"). Falls back to answer if absent.
    def _depth(s):
        tm = s["tree_meta"]; t = tm.get("target_node", 0)
        return float(tm.get("depth", [0])[t] if t < len(tm.get("depth", [])) else 0)
    depths = np.array([_depth(s) for s in samples])
    target_d = np.abs(depths[:, None] - depths[None, :])
    if target_d.max() <= 0:
        target_d = np.abs(np.array(gold)[:, None] - np.array(gold)[None, :]).astype(float)

    # --- build tree + matched-random subspaces (per seed for the random one) ---
    B_tree = _tree_subspace(reps, target_d, k)
    B_tree_t = torch.tensor(B_tree, dtype=torch_dtype, device=device)
    mu = torch.tensor(reps.mean(0), dtype=torch_dtype, device=device)

    def var_removed(B):
        Hc = reps - reps.mean(0, keepdims=True)
        return float(Hc.var() - (Hc - (Hc @ B) @ B.T).var())
    vr_tree = var_removed(B_tree)

    # --- hook factory: project out a subspace at layer L ---
    state = {"B": None}
    layer_module = model.model.layers[L] if hasattr(model, "model") else model.transformer.h[L]
    def hook(mod, inp, output):
        if state["B"] is None:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        hs = _project_out(hs, state["B"], mu)
        return (hs,) + output[1:] if isinstance(output, tuple) else hs
    handle = layer_module.register_forward_hook(hook)

    def _accuracy(Bsub):
        state["B"] = Bsub
        correct, margins = 0, []
        for s, g in zip(samples, gold):
            ids = tok(s["prompt"], return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                logits = model(**ids).logits[0, -1]
            lt, lf = float(logits[true_id]), float(logits[false_id])
            pred = 1 if lt > lf else 0
            correct += int(pred == g)
            margins.append((lt - lf) if g == 1 else (lf - lt))   # signed margin toward gold
        state["B"] = None
        return correct / len(samples), float(np.mean(margins))

    rows = []
    acc_clean, marg_clean = _accuracy(None)
    log_line(logfile, f"clean: acc={acc_clean:.3f} margin={marg_clean:+.3f} (vr_tree={vr_tree:.3f})")
    for seed in seeds:
        B_rand = _matched_random_subspace(reps, B_tree, k, seed=seed)
        B_rand_t = torch.tensor(B_rand, dtype=torch_dtype, device=device)
        acc_tree, marg_tree = _accuracy(B_tree_t)
        acc_rand, marg_rand = _accuracy(B_rand_t)
        vr_rand = var_removed(B_rand)
        rows.append(dict(model=model_name, layer=L, k=k, seed=seed,
                         acc_clean=round(acc_clean, 4), acc_ablate_tree=round(acc_tree, 4),
                         acc_ablate_random=round(acc_rand, 4),
                         drop_tree=round(acc_clean - acc_tree, 4),
                         drop_random=round(acc_clean - acc_rand, 4),
                         causal_gap=round((acc_clean - acc_tree) - (acc_clean - acc_rand), 4),
                         var_removed_tree=round(vr_tree, 4), var_removed_random=round(vr_rand, 4),
                         margin_clean=round(marg_clean, 4), margin_tree=round(marg_tree, 4),
                         margin_random=round(marg_rand, 4)))
        log_line(logfile, f"seed {seed}: drop_tree={acc_clean-acc_tree:+.3f} "
                          f"drop_random={acc_clean-acc_rand:+.3f} "
                          f"causal_gap={(acc_clean-acc_tree)-(acc_clean-acc_rand):+.3f}")
    handle.remove()

    save_csv(os.path.join(out_dir, "causal_patch.csv"), rows)
    gaps = [r["causal_gap"] for r in rows]
    verdict = dict(model=model_name, layer=L, k=k, acc_clean=acc_clean,
                   mean_drop_tree=float(np.mean([r["drop_tree"] for r in rows])),
                   mean_drop_random=float(np.mean([r["drop_random"] for r in rows])),
                   mean_causal_gap=float(np.mean(gaps)),
                   causal_gap_all_pos=bool(all(g > 0 for g in gaps)),
                   tree_is_causal=bool(np.mean(gaps) > 0.03 and all(g > 0 for g in gaps)))
    save_json(os.path.join(out_dir, "causal_patch_verdict.json"), verdict)
    log_line(logfile, f"VERDICT: mean causal_gap={verdict['mean_causal_gap']:+.3f} "
                      f"tree_is_causal={verdict['tree_is_causal']}")
    write_manifest(out_dir, "causal_patch", args=dict(model=model_name, layer=L, k=k,
                   dataset=dataset, seeds=list(seeds)), extra=verdict)
    return rows, verdict


def main(argv=None):
    ap = argparse.ArgumentParser(description="Causal ablation of the tree subspace.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", default="./results/data_cache_v3")
    ap.add_argument("--out", default="./results/causal_patch")
    ap.add_argument("--dataset", default="prontoqa_tree")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp32")
    args = ap.parse_args(argv)
    run(args.model, args.cache, args.out, layer=args.layer, k=args.k, limit=args.limit,
        seeds=tuple(args.seeds), dataset=args.dataset, device=args.device, dtype=args.dtype)


if __name__ == "__main__":
    main()
