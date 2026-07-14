"""Phase 0: extract hidden states from an LLM (runs on the DGX).

Design follows the plan's hardened critique:
  - hidden states are captured directly from ``generate(..., output_hidden_states
    =True, return_dict_in_generate=True)`` so prompt and generated-token states
    are aligned with no re-encode/realign step;
  - stored in fp32 (bf16 blows up near the Poincare boundary later);
  - per-token metadata (token string, position, layer, is_thinking, is_generated)
    is saved so alignment is auditable;
  - "thinking" tokens are matched WITHOUT forcing the model to emit them (no
    appended instruction -- that would bias the distribution). Matching is
    tokenizer-aware and multi-token-aware (see reason_markers.py).

``transformers``/``torch.cuda`` are imported lazily so the rest of the package
(geometry, probes, eval) works on a laptop with no transformers installed.
"""

from __future__ import annotations

import argparse
import os

from ..io import ensure_dir, log_line, sample_path
from .reason_markers import ThinkingMatcher


def extract_model_dataset(model_name, dataset, samples, out_dir, logfile,
                          max_new_tokens=256, dtype="fp32", device="cuda",
                          chat_mode="plain"):
    """Extract and save activations for every sample of one (model, dataset).

    ``samples`` is a list of dicts: {"sample_id", "prompt", "label",
    "label_path"}. Returns the number of samples written. ``chat_mode`` controls
    prompt scaffolding across models -- default 'plain' gives EVERY model the
    identical raw prompt, which is required for a fair H1/H2 cross-model contrast
    (a base model with no chat template must not get different scaffolding than a
    chat model).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16,
                   "fp16": torch.float16}[dtype]
    log_line(logfile, f"loading {model_name} ({dtype}) on {device}")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device)
    model.eval()

    # Report the scaffolding decision so cross-model parity is auditable.
    has_tmpl = getattr(tok, "chat_template", None) is not None
    log_line(logfile, f"  {model_name}: chat_template={'yes' if has_tmpl else 'NO'}, "
                      f"chat_mode='{chat_mode}' -> "
                      f"{'chat template' if (chat_mode=='chat' or (chat_mode=='auto' and has_tmpl)) else 'PLAIN prompt'}")

    matcher = ThinkingMatcher(tok)
    written = 0
    for s in samples:
        try:
            rec = _extract_one(model, tok, matcher, s, max_new_tokens, device, chat_mode)
        except Exception as exc:  # keep going; log the skip
            log_line(logfile, f"  skip {s['sample_id']}: {exc}")
            continue
        rec.update(model=model_name, dataset=dataset,
                   sample_id=s["sample_id"], label=s.get("label", 0),
                   label_path=s.get("label_path", []),
                   variant=s.get("variant", "original"),
                   orig_id=s.get("orig_id", s["sample_id"]),
                   answer=s.get("answer"),
                   # ground-truth is-a tree for the tree probe (PREREGISTER3);
                   # None for datasets that don't carry one.
                   tree_meta=s.get("tree_meta"),
                   # provenance (per-sample, so it survives file reshuffles)
                   chat_mode=chat_mode, max_new_tokens=max_new_tokens,
                   dtype=dtype)
        torch.save(rec, sample_path(out_dir, model_name, dataset, s["sample_id"]))
        written += 1
        if written % 25 == 0:
            log_line(logfile, f"  {dataset}: {written}/{len(samples)} "
                              f"(last had {int(rec['is_thinking'].sum())} thinking tokens)")
    log_line(logfile, f"done {model_name}/{dataset}: wrote {written}/{len(samples)}")
    return written


def _format_prompt(tok, prompt, chat_mode):
    """Format a prompt consistently across models to avoid a scaffolding confound.

    A base model (e.g. Qwen2.5-7B) may have NO chat template, while a chat/reasoning
    model does. If we templated one and not the other, the prompt-token geometry
    comparison (H2 especially) would be confounded by scaffolding, not reasoning.
    ``chat_mode`` forces a single policy for the whole run:
      - "plain": raw prompt text, identical for every model (RECOMMENDED for the
        H1/H2 cross-model contrast -- guarantees identical scaffolding).
      - "chat": use each model's chat template (only valid if ALL models have one).
      - "auto": chat template iff the tokenizer actually has one, else plain
        (convenient, but can mix scaffolds across models -- logged, not silent).
    Returns (text, used_chat_bool).
    """
    has_tmpl = getattr(tok, "chat_template", None) is not None
    if chat_mode == "plain" or (chat_mode == "auto" and not has_tmpl):
        return prompt, False
    if chat_mode == "chat" and not has_tmpl:
        raise ValueError("chat_mode='chat' but this tokenizer has no chat template; "
                         "use 'plain' for cross-model parity")
    messages = [{"role": "user", "content": prompt}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), True


def _extract_one(model, tok, matcher, sample, max_new_tokens, device, chat_mode="plain"):
    """Generate, capture aligned hidden states, tag tokens. Returns a record."""
    import torch

    text, used_chat = _format_prompt(tok, sample["prompt"], chat_mode)
    inputs = tok(text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            output_hidden_states=True, return_dict_in_generate=True,
            pad_token_id=tok.pad_token_id,
        )

    seq = gen.sequences[0]                       # (prompt_len + n_gen,)
    n_total = seq.shape[0]
    n_gen = n_total - prompt_len

    # Assemble per-layer hidden states aligned to the full sequence.
    # gen.hidden_states: tuple over generated steps; step 0 covers the prompt
    # (n_layers tensors of shape (1, prompt_len, h)), each later step covers one
    # new token ((1, 1, h)). Concatenate along the token axis per layer.
    step0 = gen.hidden_states[0]
    n_layers = len(step0)
    layer_chunks = [[step0[l][0]] for l in range(n_layers)]  # each (prompt_len, h)
    for step in gen.hidden_states[1:]:
        for l in range(n_layers):
            layer_chunks[l].append(step[l][0])               # (1, h)
    hidden = torch.stack([torch.cat(chunks, dim=0) for chunks in layer_chunks], dim=0)
    hidden = hidden.float().cpu()                            # (n_layers, n_tok, h) fp32

    # Align token count (hidden may cover n_total tokens). NOTE: with KV-cache
    # generation the FINAL sampled token is never re-fed, so hidden has
    # n_total - 1 rows; everything below is sized to hidden's n_tok, and
    # n_generated_with_states records the count that actually has states
    # (v1's n_generated over-counted by exactly 1).
    n_tok = hidden.shape[1]
    ids = seq[:n_tok]
    tokens = tok.convert_ids_to_tokens(ids.tolist())
    positions = torch.arange(n_tok)
    is_generated = positions >= prompt_len
    # Match reasoning markers only among generated tokens, tokenizer-aware.
    is_thinking = matcher.mask(tokens, start=prompt_len)

    # Truncation flag: did generation hit the cap instead of emitting EOS?
    # If True, the 'last' token is a mid-trace cut point, not a conclusion
    # state — downstream 'last'-source analyses must split on this flag.
    hit_cap = bool(n_gen >= max_new_tokens
                   and seq[-1].item() != (tok.eos_token_id or -1))

    return {
        "hidden": hidden,
        "tokens": tokens,
        "positions": positions,
        "is_generated": is_generated,
        "is_thinking": torch.as_tensor(is_thinking, dtype=torch.bool),
        "text": tok.decode(seq[prompt_len:], skip_special_tokens=True),
        "prompt_len": prompt_len,
        "n_generated": int(n_gen),
        "n_generated_with_states": int(is_generated.sum()),
        "truncated": hit_cap,
        "used_chat_template": used_chat,
    }


def _load_samples(dataset, cache_dir):
    """Load prepared samples for a dataset from the data cache (Phase 0.a)."""
    import json

    path = os.path.join(cache_dir, f"{dataset}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run `python -m hypprobe.data.prepare` first")
    out = []
    with open(path) as fh:
        for line in fh:
            out.append(json.loads(line))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Extract LLM hidden states (Phase 0).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out", default="./results/activations")
    ap.add_argument("--cache", default="./results/data_cache")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="cap samples per dataset (0=all)")
    ap.add_argument("--chat-mode", default="plain", choices=["plain", "chat", "auto"],
                    help="prompt scaffolding. 'plain' (default) gives every model the "
                         "identical raw prompt -> fair cross-model H1/H2. 'chat' uses each "
                         "model's chat template (all must have one). 'auto' = chat iff present.")
    args = ap.parse_args(argv)

    ensure_dir(args.out)
    logfile = os.path.join(os.path.dirname(args.out.rstrip("/")) or ".",
                           "logs", "extract.log")
    from ..manifest import write_manifest
    write_manifest(args.out, f"extract_{args.model.replace('/', '_')}",
                   args=vars(args))
    for ds in args.datasets:
        samples = _load_samples(ds, args.cache)
        if args.limit:
            samples = samples[: args.limit]
        log_line(logfile, f"extracting {len(samples)} samples for {args.model}/{ds}")
        extract_model_dataset(args.model, ds, samples, args.out, logfile,
                              max_new_tokens=args.max_new_tokens,
                              dtype=args.dtype, device=args.device,
                              chat_mode=args.chat_mode)


if __name__ == "__main__":
    main()
