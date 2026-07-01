"""Tests for the meaning-control variant generators."""

from hypprobe.data.variants import (FUNCTION_WORDS, make_nonce, make_paraphrase)


def test_nonce_preserves_structure_and_function_words():
    prompt = "Describe the loyal dog guarding the house"
    out = make_nonce(prompt)
    toks_in = prompt.split(" ")
    toks_out = out.split(" ")
    # Same number of tokens (structure preserved).
    assert len(toks_in) == len(toks_out)
    # Function words kept verbatim; content words changed.
    for i, (a, b) in enumerate(zip(toks_in, toks_out)):
        if a.lower() in FUNCTION_WORDS:
            assert a == b, f"function word changed: {a} -> {b}"
    # At least one content word actually changed.
    assert any(a != b for a, b in zip(toks_in, toks_out))


def test_paraphrase_preserves_length_and_swaps_synonyms():
    prompt = "Describe a dangerous build"
    out = make_paraphrase(prompt)
    assert len(out.split(" ")) == len(prompt.split(" "))
    # 'dangerous' -> 'hazardous', 'build' -> 'construct', 'Describe' -> 'Explain'
    low = out.lower()
    assert "hazardous" in low and "construct" in low
    assert out.split(" ")[0][0].isupper()  # capitalization preserved


def test_nonce_and_paraphrase_differ():
    prompt = "Describe the quick dog"
    assert make_nonce(prompt) != make_paraphrase(prompt)


def test_augment_jsonl(tmp_path):
    import json
    from hypprobe.data.variants import augment_jsonl

    src = tmp_path / "d.jsonl"
    with open(src, "w") as fh:
        fh.write(json.dumps({"sample_id": "x0", "prompt": "Describe the dog",
                             "label": 0, "label_path": [0, 0]}) + "\n")
    out = tmp_path / "d_aug.jsonl"
    n = augment_jsonl(str(src), str(out))
    assert n == 3  # original + nonce + paraphrase
    rows = [json.loads(l) for l in open(out)]
    variants = {r["variant"] for r in rows}
    assert variants == {"original", "nonce", "paraphrase"}
    for r in rows:
        assert r["orig_id"] == "x0"
