import pytest
from unittest.mock import Mock

from flotorch_core.inferencer.guardrails.guardrails_inferencer import GuardRailsInferencer


@pytest.fixture
def base_inferencer():
    """A fake BaseInferencer with predictable return values."""
    bi = Mock()
    # make sure .generate_text returns a (metadata, answer) tuple
    bi.generate_text.return_value = ({"foo": "bar"}, "original answer")
    bi.generate_prompt.return_value = "PROMPT-HERE"
    bi.format_context.return_value = "CONTEXT-STR"
    return bi


@pytest.fixture
def base_guardrail_allow():
    """
    A fake guardrail that does not intervene.
    apply_guardrail should return something where action != 'GUARDRAIL_INTERVENED'
    """
    bg = Mock()
    bg.apply_guardrail.return_value = {
        "action": "ALLOW",
        # outputs / assessments would be ignored in ALLOW case
    }
    return bg


@pytest.fixture
def base_guardrail_block():
    """
    A fake guardrail that *does* intervene.
    """
    bg = Mock()
    bg.apply_guardrail.return_value = {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": ["too_long", "invalid_format"],
        "outputs": [{"text": "blocked answer"}]
    }
    return bg


def test_generate_text_without_intervention(base_inferencer, base_guardrail_allow):
    infer = GuardRailsInferencer(base_inferencer, base_guardrail_allow)
    meta, ans = infer.generate_text("Q?", [{"text": "ctx"}])
    # should simply forward the original metadata+answer
    assert meta == {"foo": "bar"}
    assert ans == "original answer"
    base_inferencer.generate_text.assert_called_once_with("Q?", [{"text": "ctx"}])
    base_guardrail_allow.apply_guardrail.assert_called_once_with("original answer", "OUTPUT")


def test_generate_text_with_intervention(base_inferencer, base_guardrail_block):
    infer = GuardRailsInferencer(base_inferencer, base_guardrail_block)
    meta, ans = infer.generate_text("Q?", [{"text": "ctx"}])
    # on intervention, should return the special blocked metadata + replacement text
    assert meta == {
        "guardrail_output_assessment": ["too_long", "invalid_format"],
        "guardrail_blocked": True
    }
    assert ans == "blocked answer"
    base_inferencer.generate_text.assert_called_once_with("Q?", [{"text": "ctx"}])
    base_guardrail_block.apply_guardrail.assert_called_once_with("original answer", "OUTPUT")


def test_generate_prompt_delegates(base_inferencer, base_guardrail_allow):
    infer = GuardRailsInferencer(base_inferencer, base_guardrail_allow)
    result = infer.generate_prompt("MyQuery", [{"text": "x"}])
    assert result == "PROMPT-HERE"
    base_inferencer.generate_prompt.assert_called_once_with("MyQuery", [{"text": "x"}])


def test_format_context_delegates(base_inferencer, base_guardrail_allow):
    infer = GuardRailsInferencer(base_inferencer, base_guardrail_allow)
    result = infer.format_context([{"text": "y"}])
    assert result == "CONTEXT-STR"
    base_inferencer.format_context.assert_called_once_with([{"text": "y"}])
