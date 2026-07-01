"""Test the tokenizer-aware, multi-token reasoning-marker matcher.

Uses a tiny fake tokenizer so no `transformers` install is needed. The fake
tokenizer splits some markers into multiple sub-tokens (like real BPE does) to
prove multi-token matching works where the naive single-token regex fails.
"""

from hypprobe.extract.reason_markers import ThinkingMatcher, _clean


class FakeTokenizer:
    """Minimal tokenizer: splits 'therefore' into sub-tokens, adds space marks."""

    def encode(self, text, add_special_tokens=False):
        # Map words to fake ids; 'therefore' becomes two pieces.
        pieces = []
        for w in text.strip().split(" "):
            if w == "therefore":
                pieces += ["ther", "efore"]
            elif w:
                pieces.append(w)
        # Represent leading space with 'G-dot' on the first piece.
        out = []
        for i, p in enumerate(pieces):
            out.append("Ġ" + p if (text.startswith(" ") and i == 0) else p)
        return list(range(len(out))) if False else out  # ids == tokens for the fake

    def convert_ids_to_tokens(self, ids):
        return ids  # our "ids" are already the token strings


def test_clean_strips_space_marks():
    assert _clean("Ġwait") == "wait"
    assert _clean("▁So") == "so"
    assert _clean("Therefore") == "therefore"


def test_single_token_marker_matched():
    tok = FakeTokenizer()
    m = ThinkingMatcher(tok)
    tokens = ["The", "Ġanswer", "wait", "Ġhmm"]
    mask = m.mask(tokens, start=0)
    assert mask[2] is True or mask[2]  # 'wait'
    assert mask[3]                     # 'hmm'
    assert not mask[0]


def test_multitoken_marker_matched():
    """'therefore' split as ther+efore must be flagged on BOTH sub-tokens."""
    tok = FakeTokenizer()
    m = ThinkingMatcher(tok)
    tokens = ["Ok", "ther", "efore", "done"]
    mask = m.mask(tokens, start=0)
    assert mask[1] and mask[2], f"multi-token marker not matched: {mask}"


def test_start_offset_excludes_prompt():
    tok = FakeTokenizer()
    m = ThinkingMatcher(tok)
    tokens = ["wait", "here", "wait"]
    mask = m.mask(tokens, start=2)  # only index >=2 can be flagged
    assert not mask[0]
    assert mask[2]
