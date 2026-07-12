"""Audit stored generations: did the model actually reason, and were we cut off?

The re-evaluation flagged two unverified stimulus confounds in the DGX run:

  1. CHAT_MODE=plain may have kept DeepSeek-R1-Distill out of its <think>
     reasoning mode entirely — in which case the 'generated'/'thinking' clouds
     never contained reasoning-mode representations and H1/H2 didn't test their
     hypothesis. Nobody ever read the stored text to check.
  2. max_new_tokens=256 may truncate traces, making the 'last' token (H3's
     winning source and the probe default) an arbitrary mid-trace cut point.

This module reads the stored activations' text fields (no GPU needed) and
reports, per (model, dataset, variant, chat regime):
  - fraction of samples with CoT markers in the generated text (step patterns,
    connectives, <think> tags);
  - fraction with an explicit final answer (True/False for prontoqa);
  - fraction truncated (the extractor's 'truncated' flag when present, else a
    length heuristic);
  - generated-length distribution (p10/p50/p90).

Run it right after extraction, BEFORE any geometry: if the reasoning-model arm
shows no reasoning, stop and re-extract with chat mode rather than analyze.
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np

from ..io import ensure_dir, iter_samples, log_line, save_csv
from ..manifest import write_manifest

_COT_PATTERNS = [
    re.compile(r"<think>", re.I),
    re.compile(r"\bstep\s*(?:1|one|by step)\b", re.I),
    re.compile(r"\b(?:first|second|then|therefore|so|thus|hence)\b.*"
               r"\b(?:therefore|so|thus|hence|then)\b", re.I | re.S),
    re.compile(r"\bevery\s+\w+\s+is\s+a\s+\w+", re.I),  # chain restating (prontoqa)
]
_ANSWER_PATTERN = re.compile(r"\b(true|false)\b", re.I)


def _has_cot(text: str) -> bool:
    return any(p.search(text or "") for p in _COT_PATTERNS)


def run(activations_dir, out_dir, min_cot_frac=0.5):
    ensure_dir(out_dir)
    logfile = os.path.join(os.path.dirname(out_dir.rstrip("/")) or ".",
                           "logs", "audit_generations.log")
    groups: dict[tuple, dict] = {}
    for s in iter_samples(activations_dir):
        key = (s["model"], s["dataset"], s.get("variant", "original"),
               "chat" if s.get("used_chat_template") else "plain")
        g = groups.setdefault(key, dict(n=0, cot=0, answered=0, truncated=0,
                                        lengths=[]))
        text = s.get("text", "") or ""
        g["n"] += 1
        g["cot"] += int(_has_cot(text))
        g["answered"] += int(bool(_ANSWER_PATTERN.search(text)))
        n_gen = int(s.get("n_generated_with_states",
                          s.get("n_generated", 0)) or 0)
        g["lengths"].append(n_gen)
        if "truncated" in s:
            g["truncated"] += int(bool(s["truncated"]))
        else:
            # legacy stores (v1) have no flag; use the cap heuristic
            cap = int(s.get("max_new_tokens", 256) or 256)
            g["truncated"] += int(n_gen >= cap - 1)

    rows, warnings = [], []
    for (model, dataset, variant, regime), g in sorted(groups.items()):
        lens = np.asarray(g["lengths"]) if g["lengths"] else np.zeros(1)
        row = dict(model=model, dataset=dataset, variant=variant, chat_regime=regime,
                   n=g["n"],
                   cot_frac=round(g["cot"] / max(g["n"], 1), 3),
                   answered_frac=round(g["answered"] / max(g["n"], 1), 3),
                   truncated_frac=round(g["truncated"] / max(g["n"], 1), 3),
                   gen_len_p10=int(np.percentile(lens, 10)),
                   gen_len_p50=int(np.percentile(lens, 50)),
                   gen_len_p90=int(np.percentile(lens, 90)))
        rows.append(row)
        is_reasoning_model = any(k in model.lower()
                                 for k in ("r1", "distill", "reason"))
        if variant == "original" and is_reasoning_model and row["cot_frac"] < min_cot_frac:
            warnings.append(
                f"{model}/{dataset} [{regime}]: only {row['cot_frac']:.0%} of "
                f"generations look like CoT — the reasoning model may not be in "
                f"reasoning mode under this regime. Do NOT run H1/H2 on this arm; "
                f"re-extract with --chat-mode chat first.")
        if variant == "original" and row["truncated_frac"] > 0.25:
            warnings.append(
                f"{model}/{dataset} [{regime}]: {row['truncated_frac']:.0%} of "
                f"generations truncated at the cap — 'last'-token analyses are "
                f"contaminated; raise --max-new-tokens or split by the flag.")

    save_csv(os.path.join(out_dir, "generation_audit.csv"), rows)
    write_manifest(out_dir, "audit_generations", args=dict(activations=activations_dir))
    for r in rows:
        log_line(logfile, f"{r['model']}/{r['dataset']}/{r['variant']} [{r['chat_regime']}]: "
                          f"n={r['n']} cot={r['cot_frac']} answered={r['answered_frac']} "
                          f"truncated={r['truncated_frac']} len_p50={r['gen_len_p50']}")
    for w in warnings:
        log_line(logfile, "WARNING: " + w)
    if not warnings:
        log_line(logfile, "audit clean: reasoning present, truncation low")
    return rows, warnings


def main(argv=None):
    ap = argparse.ArgumentParser(description="Audit stored generations for CoT "
                                             "presence and truncation.")
    ap.add_argument("--activations", required=True)
    ap.add_argument("--out", default="./results/geometry_v2")
    ap.add_argument("--min-cot-frac", type=float, default=0.5)
    args = ap.parse_args(argv)
    _, warnings = run(args.activations, args.out, min_cot_frac=args.min_cot_frac)
    if warnings:
        raise SystemExit(2)  # non-zero so run_gaps.sh surfaces it


if __name__ == "__main__":
    main()
