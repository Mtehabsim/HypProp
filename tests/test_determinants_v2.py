"""Tests for determinants_v2: source-respecting edits, real nulls, power."""

import numpy as np
import pytest


def test_source_masking():
    """_source_mask must respect the token source (the v1 source bug fix)."""
    from hypprobe.geometry.determinants_v2 import _source_mask

    sample = {
        "is_generated": np.array([False, False, True, True, True]),
        "is_thinking": np.array([False, False, False, True, False]),
    }
    h = np.ones((5, 8))  # dummy, no sink

    mask_gen = _source_mask(sample, h, "generated")
    assert mask_gen.sum() == 3 and not mask_gen[0] and mask_gen[2]

    mask_think = _source_mask(sample, h, "thinking")
    assert mask_think.sum() == 1 and mask_think[3]

    mask_inp = _source_mask(sample, h, "input")
    assert mask_inp.sum() == 2 and mask_inp[0] and mask_inp[1]


def test_nonce_consistency():
    """make_nonce must give the SAME pseudoword for repeated tokens."""
    from hypprobe.data.variants import make_nonce

    prompt = "Every rompus is a lorpus. Alex is a rompus."
    result = make_nonce(prompt)
    words = result.split()
    # "rompus" appears twice; both must map to the same pseudoword
    rompus_positions = [i for i, w in enumerate(words)
                        if w.rstrip(".,") not in ("Every", "is", "a", "Alex")]
    mapped_rompus = [w.rstrip(".,") for w in words
                     if w.rstrip(".,").lower().startswith(
                         result.split()[1].rstrip(".,").lower()[:3])]
    # More directly: split original and result, check consistent mapping
    orig_words = prompt.split()
    res_words = result.split()
    mapping = {}
    for o, r in zip(orig_words, res_words):
        oc = o.strip(".,").lower()
        rc = r.strip(".,").lower()
        if oc in ("every", "is", "a"):
            continue
        if oc in mapping:
            assert mapping[oc] == rc, (
                f"'{oc}' mapped to both '{mapping[oc]}' and '{rc}' — "
                f"consistency broken (the v1 bug)")
        else:
            mapping[oc] = rc


def test_paraphrase_not_identical():
    """make_paraphrase must produce a non-identical variant on builder prompts."""
    from hypprobe.data.variants import make_paraphrase
    from hypprobe.data.prepare import build_prontoqa

    prompts = build_prontoqa(n_per_depth=10, depths=(1, 2, 3))
    n_changed = sum(1 for p in prompts if make_paraphrase(p["prompt"]) != p["prompt"])
    # The template-level rewrites should hit most prompts (all have the
    # "Reason step by step" instruction).
    assert n_changed >= len(prompts) * 0.9, (
        f"only {n_changed}/{len(prompts)} prompts changed by paraphrase — "
        f"the control is still too sparse")


def test_split_half_null_positive():
    """split_half_null must return a positive number (the honest noise floor)."""
    from hypprobe.geometry.determinants_v2 import split_half_null

    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 32))
    null = split_half_null(X, "raw", seed=0, n_splits=10)
    assert null > 0, f"split-half null should be positive, got {null}"


def test_order_power_ceiling_nonzero():
    """The order power ceiling must be positive on position-structured data."""
    from hypprobe.geometry.determinants_v2 import order_power

    rng = np.random.default_rng(0)
    gathered = []
    for i in range(50):
        n_tok = 12
        h = rng.standard_normal((n_tok, 16))
        is_think = np.zeros(n_tok, bool)
        gathered.append((h, is_think, i % 5))
    pw = order_power(gathered, "raw", seed=0)
    assert pw > 0.01, f"power ceiling should be measurable, got {pw}"
