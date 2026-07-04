"""Tests for cross-model prompt-scaffolding parity (the H2 confound fix)."""

import pytest

from hypprobe.extract.hidden_state_extractor import _format_prompt


class TokWithTemplate:
    chat_template = "{{ x }}"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<CHAT>" + messages[0]["content"] + "</CHAT>"


class TokNoTemplate:
    chat_template = None


def test_plain_is_identical_regardless_of_template():
    """'plain' must give the raw prompt for BOTH a chat model and a base model,
    so the cross-model contrast is not confounded by scaffolding."""
    p = "Every A is a B. Is X a B?"
    a, used_a = _format_prompt(TokWithTemplate(), p, "plain")
    b, used_b = _format_prompt(TokNoTemplate(), p, "plain")
    assert a == p and b == p
    assert used_a is False and used_b is False


def test_chat_mode_uses_template_when_present():
    p = "hello"
    text, used = _format_prompt(TokWithTemplate(), p, "chat")
    assert used is True and text.startswith("<CHAT>")


def test_chat_mode_refuses_base_model():
    """chat_mode='chat' on a template-less base model must raise, not silently
    fall back to a different scaffold."""
    with pytest.raises(ValueError):
        _format_prompt(TokNoTemplate(), "hi", "chat")


def test_auto_mixes_but_is_explicit():
    """'auto' uses template iff present -> can differ across models (logged, not
    silent). We assert the difference is real so callers know to prefer 'plain'."""
    chat_text, used_c = _format_prompt(TokWithTemplate(), "hi", "auto")
    base_text, used_b = _format_prompt(TokNoTemplate(), "hi", "auto")
    assert used_c is True and used_b is False
    assert chat_text != base_text
