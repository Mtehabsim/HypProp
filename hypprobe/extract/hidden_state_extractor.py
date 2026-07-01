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
                          max_new_tokens=256, dtype="fp32", device="cuda"):
    """Extract and save activations for every sample of one (model, dataset).

    ``samples`` is a list of dicts: {"sample_id", "prompt", "label",
    "label_path"}. Returns the number of samples written.
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

    matcher = ThinkingMatcher(tok)
    written = 0
    for s in samples:
        try:
            rec = _extract_one(model, tok, matcher, s, max_new_tokens, device)
        except Exception as exc:  # keep going; log the skip
            log_line(logfile, f"  skip {s['sample_id']}: {exc}")
            continue
        rec.update(model=model_name, dataset=dataset,
                   sample_id=s["sample_id"], label=s.get("label", 0),
                   label_path=s.get("label_path", []))
        torch.save(rec, sample_path(out_dir, model_name, dataset, s["sample_id"]))
        written += 1
        if written % 25 == 0:
            log_line(logfile, f"  {dataset}: {written}/{len(samples)} "
                              f"(last had {int(rec['is_thinking'].sum())} thinking tokens)")
    log_line(logfile, f"done {model_name}/{dataset}: wrote {written}/{len(samples)}")
    return written


def _extract_one(model, tok, matcher, sample, max_new_tokens, device):
    """Generate, capture aligned hidden states, tag tokens. Returns a record."""
    import torch

    messages = [{"role": "user", "content": sample["prompt"]}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

    # Align token count (hidden may cover n_total tokens).
    n_tok = hidden.shape[1]
    ids = seq[:n_tok]
    tokens = tok.convert_ids_to_tokens(ids.tolist())
    positions = torch.arange(n_tok)
    is_generated = positions >= prompt_len
    # Match reasoning markers only among generated tokens, tokenizer-aware.
    is_thinking = matcher.mask(tokens, start=prompt_len)

    return {
        "hidden": hidden,
        "tokens": tokens,
        "positions": positions,
        "is_generated": is_generated,
        "is_thinking": torch.as_tensor(is_thinking, dtype=torch.bool),
        "text": tok.decode(seq[prompt_len:], skip_special_tokens=True),
        "prompt_len": prompt_len,
        "n_generated": int(n_gen),
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
    args = ap.parse_args(argv)

    ensure_dir(args.out)
    logfile = os.path.join(os.path.dirname(args.out.rstrip("/")) or ".",
                           "logs", "extract.log")
    for ds in args.datasets:
        samples = _load_samples(ds, args.cache)
        if args.limit:
            samples = samples[: args.limit]
        log_line(logfile, f"extracting {len(samples)} samples for {args.model}/{ds}")
        extract_model_dataset(args.model, ds, samples, args.out, logfile,
                              max_new_tokens=args.max_new_tokens,
                              dtype=args.dtype, device=args.device)


if __name__ == "__main__":
    main()
