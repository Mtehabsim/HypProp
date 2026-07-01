"""Text-level variants for the determinants "meaning" control (Phase 1 science).

To test whether MEANING (not just token identity or order) drives hyperbolicity,
we need to re-run the model on reworded prompts and compare the geometry. This
module builds two variant kinds per prompt:

  - nonce   : replace content words (nouns/verbs/adjectives) with pronounceable
              pseudo-words, keeping function words and sentence STRUCTURE intact.
              Same structure, destroyed meaning -> if delta_rel changes a lot,
              meaning mattered.
  - paraphrase: light, meaning-preserving rewrites (synonym swaps, voice/order
              tweaks) -> same meaning, different surface form -> if delta_rel is
              stable, the structure is carried by meaning rather than surface.

These are deliberately dependency-free (no NLTK/spaCy needed): a small function-
word list marks which tokens to keep; everything else is "content". Good enough
for a controlled contrast; swap in a real paraphraser later if desired.

Output augments the prepared jsonl: each variant becomes its own sample whose
``sample_id`` is ``<orig>__<variant>`` and which carries ``variant`` and
``orig_id`` fields so the determinants step can pair them.
"""

from __future__ import annotations

import argparse
import json
import os

from ..io import ensure_dir, log_line

# Function words we preserve to keep sentence structure recognisable.
FUNCTION_WORDS = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "but",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these",
    "those", "it", "its", "with", "as", "by", "from", "into", "about", "over",
    "i", "you", "he", "she", "they", "we", "his", "her", "their", "my", "your",
    "do", "does", "did", "can", "could", "will", "would", "should", "if", "then",
    "how", "what", "why", "when", "where", "who", "not", "no", "yes",
}

_SYLL = ["ba", "ko", "ti", "lu", "ne", "za", "mi", "ro", "fu", "de", "pa", "so"]

# A few conservative synonym pairs for light paraphrasing.
_SYNONYMS = {
    "describe": "explain", "make": "create", "dangerous": "hazardous",
    "build": "construct", "harmful": "damaging", "big": "large", "small": "tiny",
    "quick": "fast", "begin": "start", "show": "display", "use": "employ",
}


def _pseudoword(token: str, seed_offset: int) -> str:
    """Deterministic pseudo-word roughly matching the length of ``token``."""
    n_syll = max(1, min(4, len(token) // 2))
    out = []
    for i in range(n_syll):
        out.append(_SYLL[(len(token) + i + seed_offset) % len(_SYLL)])
    word = "".join(out)
    return word.capitalize() if token[:1].isupper() else word


def make_nonce(prompt: str) -> str:
    """Replace content words with pseudo-words, keep structure/function words."""
    out_tokens = []
    for i, tok in enumerate(prompt.split(" ")):
        core = tok.strip(".,!?;:\"'")
        suffix = tok[len(tok.rstrip(".,!?;:\"'")):] if tok else ""
        if core.lower() in FUNCTION_WORDS or not core.isalpha():
            out_tokens.append(tok)
        else:
            out_tokens.append(_pseudoword(core, i) + suffix)
    return " ".join(out_tokens)


def make_paraphrase(prompt: str) -> str:
    """Light meaning-preserving rewrite via conservative synonym swaps."""
    out_tokens = []
    for tok in prompt.split(" "):
        core = tok.strip(".,!?;:\"'").lower()
        if core in _SYNONYMS:
            repl = _SYNONYMS[core]
            if tok[:1].isupper():
                repl = repl.capitalize()
            out_tokens.append(repl + tok[len(tok.rstrip(".,!?;:\"'")):])
        else:
            out_tokens.append(tok)
    return " ".join(out_tokens)


def augment_jsonl(in_path: str, out_path: str, kinds=("nonce", "paraphrase")) -> int:
    """Read a prepared jsonl and write it plus variant rows to ``out_path``."""
    rows = []
    with open(in_path) as fh:
        for line in fh:
            obj = json.loads(line)
            obj.setdefault("variant", "original")
            obj.setdefault("orig_id", obj["sample_id"])
            rows.append(obj)
            for kind in kinds:
                text = make_nonce(obj["prompt"]) if kind == "nonce" else make_paraphrase(obj["prompt"])
                rows.append({**obj, "prompt": text, "variant": kind,
                             "orig_id": obj["sample_id"],
                             "sample_id": f"{obj['sample_id']}__{kind}"})
    ensure_dir(os.path.dirname(out_path) or ".")
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Add nonce/paraphrase variants to a prepared dataset.")
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--kinds", nargs="+", default=["nonce", "paraphrase"])
    args = ap.parse_args(argv)
    n = augment_jsonl(args.in_path, args.out_path, tuple(args.kinds))
    logdir = os.path.join(os.path.dirname(os.path.dirname(args.out_path)) or ".", "logs")
    log_line(os.path.join(logdir, "prepare.log"),
             f"variants: {args.in_path} -> {args.out_path} ({n} rows incl. variants)")


if __name__ == "__main__":
    main()
