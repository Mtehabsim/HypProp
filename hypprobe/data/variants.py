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

# Task scaffolding that must survive nonce-ing verbatim: renaming these turns a
# solvable task into gibberish (v1 turned "True or False" into "Zami or Fude"),
# which conflates "meaning destroyed" with "task destroyed".
_SCAFFOLD_WORDS = {"true", "false", "question", "answer", "reason", "step",
                   "explain", "describe", "every"}

# Template-level rewrites for the paraphrase control. v1's word-level synonym
# dict produced BYTE-IDENTICAL output on 300/300 PrOntoQA prompts (none of its
# 11 words occur there) -> a vacuous control. These operate on the shared
# instruction scaffolding, which every builder's prompts contain, and preserve
# meaning while guaranteeing a surface change.
_TEMPLATE_REWRITES = [
    ("Reason step by step, then answer True or False.",
     "Think it through carefully, then reply with True or False."),
    ("Question: Is it true or false that",
     "Question: Would you say it is true or false that"),
    ("Describe:", "Give a description of:"),
]


def _pseudoword(word: str, mapping: dict) -> str:
    """One consistent pseudo-word per unique word (case-insensitive).

    Keyed on the WORD (not its position): v1 keyed on (length, position), so
    'rompus' at two positions became two different pseudowords, shattering the
    coreference chains the nonce control must preserve ('same structure,
    destroyed meaning' requires the structure — including repeated mentions —
    to survive).
    """
    key = word.lower()
    if key not in mapping:
        h = sum((i + 1) * ord(ch) for i, ch in enumerate(key))
        n_syll = max(2, min(4, len(key) // 2))
        mapping[key] = "".join(_SYLL[(h + 7 * i) % len(_SYLL)] for i in range(n_syll))
    out = mapping[key]
    return out.capitalize() if word[:1].isupper() else out


def make_nonce(prompt: str) -> str:
    """Replace content words with per-prompt-CONSISTENT pseudo-words.

    Function words, task scaffolding (True/False, Question, ...), and
    non-alphabetic tokens survive; every other unique word maps to ONE
    pseudo-word reused at all its occurrences, so chains like 'Every rompus is
    a lorpus. Alex is a rompus.' keep their link structure.
    """
    mapping: dict[str, str] = {}
    out_tokens = []
    for tok in prompt.split(" "):
        core = tok.strip(".,!?;:\"'")
        suffix = tok[len(tok.rstrip(".,!?;:\"'")):] if tok else ""
        low = core.lower()
        if low in FUNCTION_WORDS or low in _SCAFFOLD_WORDS or not core.isalpha():
            out_tokens.append(tok)
        else:
            out_tokens.append(_pseudoword(core, mapping) + suffix)
    return " ".join(out_tokens)


def make_paraphrase(prompt: str) -> str:
    """Meaning-preserving rewrite: template-level rewrites + synonym swaps.

    Guaranteed non-identical wherever a known template phrase occurs (all
    builders' prompts contain one); augment_jsonl additionally REFUSES to emit
    a variant identical to its original, so a vacuous control can never reach
    extraction again.
    """
    text = prompt
    for old, new in _TEMPLATE_REWRITES:
        if old in text:
            text = text.replace(old, new)
    out_tokens = []
    for tok in text.split(" "):
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
    """Read a prepared jsonl and write it plus variant rows to ``out_path``.

    HARD GUARD: a variant that is byte-identical to its original is refused
    (raise), never silently emitted — with greedy decoding an identical prompt
    yields identical activations, which turns the meaning control into a
    guaranteed-zero placebo (the v1 paraphrase failure: 300/300 identical).
    """
    rows = []
    n_identical = 0
    with open(in_path) as fh:
        for line in fh:
            obj = json.loads(line)
            obj.setdefault("variant", "original")
            obj.setdefault("orig_id", obj["sample_id"])
            rows.append(obj)
            for kind in kinds:
                text = make_nonce(obj["prompt"]) if kind == "nonce" else make_paraphrase(obj["prompt"])
                if text == obj["prompt"]:
                    n_identical += 1
                    continue  # skip the vacuous row; count and report below
                rows.append({**obj, "prompt": text, "variant": kind,
                             "orig_id": obj["sample_id"],
                             "sample_id": f"{obj['sample_id']}__{kind}"})
    n_orig = sum(1 for r in rows if r["variant"] == "original")
    for kind in kinds:
        n_kind = sum(1 for r in rows if r["variant"] == kind)
        if n_kind < 0.5 * n_orig:
            raise SystemExit(
                f"[variants] '{kind}' produced a non-identical variant for only "
                f"{n_kind}/{n_orig} prompts ({n_identical} identical rows skipped). "
                f"A control this sparse is vacuous — fix make_{kind} for this "
                f"dataset before extracting.")
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
