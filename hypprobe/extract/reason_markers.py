"""Tokenizer-aware, multi-token matching of reasoning ("thinking") markers.

The naive approach (regex on single BPE tokens after stripping a leading 'G-dot')
is broken: words like "therefore"/"analyze"/"because" are usually split into
multiple sub-tokens, and the leading-space marker differs per tokenizer
(byte-level BPE uses 'G-dot', SentencePiece uses an underscore). This module
tokenises each marker phrase with the *actual* tokenizer and matches the
resulting sub-token sequence against the generated token stream, so a marker is
flagged even when it spans several tokens, on any tokenizer.

We deliberately do NOT force the model to emit these markers -- we only detect
markers that appear naturally, keeping the generation distribution unbiased.
"""

from __future__ import annotations

# Reasoning markers (from Qian et al. 2025 / Raj 2026). Neutral logical pivots
# only -- we exclude answer-like words ("harmful"/"violates") that would leak the
# label into a probe of generated tokens.
DEFAULT_MARKERS = (
    "wait", "hmm", "let me", "so", "therefore", "thus", "hence",
    "because", "since", "however", "actually", "analyze",
)


def _clean(token: str) -> str:
    """Normalise a single token to its printable text, stripping BPE space marks."""
    return (
        token.replace("Ġ", " ")   # 'G-dot' byte-level leading space
        .replace("▁", " ")        # SentencePiece underscore
        .replace("Ċ", "\n")       # byte-level newline
        .strip()
        .lower()
    )


class ThinkingMatcher:
    """Match reasoning markers in a token stream for a specific tokenizer."""

    def __init__(self, tokenizer, markers: tuple[str, ...] = DEFAULT_MARKERS):
        self.tokenizer = tokenizer
        self.markers = markers
        # Pre-tokenise each marker phrase (with a leading space, the common case
        # mid-sentence) into its sub-token id sequence, then to cleaned text
        # sequences we can compare against generated tokens.
        self._seqs: list[list[str]] = []
        for m in markers:
            for variant in (" " + m, m):
                ids = tokenizer.encode(variant, add_special_tokens=False)
                toks = tokenizer.convert_ids_to_tokens(ids)
                cleaned = [_clean(t) for t in toks]
                # Drop empties introduced by leading-space-only tokens.
                cleaned = [c for c in cleaned if c != ""]
                if cleaned:
                    self._seqs.append(cleaned)
        # Deduplicate.
        self._seqs = [list(s) for s in {tuple(s) for s in self._seqs}]

    def mask(self, tokens: list[str], start: int = 0) -> list[bool]:
        """Return a per-token boolean mask flagging marker tokens.

        Only positions ``>= start`` (i.e. generated tokens) can be flagged. When
        a marker spans several tokens, every token in the span is flagged.
        """
        cleaned = [_clean(t) for t in tokens]
        mask = [False] * len(tokens)
        # Single-token fast path: any token whose text is itself a marker word.
        marker_words = set(self.markers)
        for i in range(len(tokens)):
            if i >= start and cleaned[i] in marker_words:
                mask[i] = True
        # Multi-token phrases: sliding window match.
        for seq in self._seqs:
            L = len(seq)
            if L == 1:
                continue
            for i in range(max(start, 0), len(tokens) - L + 1):
                window = cleaned[i:i + L]
                # Join to compare, tolerant of the leading-space split.
                if window == seq or "".join(window) == "".join(seq):
                    for j in range(i, i + L):
                        mask[j] = True
        return mask
