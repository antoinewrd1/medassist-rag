"""The safety layer's promises, held to account.

If any test here fails, the build must not ship -- these are the guarantees the
README makes to anyone who types symptoms into this thing.
"""

import pytest

from app.safety import Triage, assess, clamp


@pytest.mark.parametrize("text", [
    "crushing chest pain radiating to my left arm",
    "chest tightness with sweating and nausea",
    "my face is drooping and my speech is slurred",
    "sudden numbness on one side of my body",
    "I can't breathe, my lips are blue",
    "throat is closing up after eating shellfish",
    "bleeding that won't stop",
    "he had a seizure and is unresponsive",
    "fever with stiff neck",
    "vomiting blood",
    "pregnant and bleeding",
    "my 3 week old has a fever",
])
def test_red_flags_escalate_to_emergency(text):
    assert assess(text).level is Triage.EMERGENCY


@pytest.mark.parametrize("text", [
    "I don't want to live anymore",
    "I do not want to live",
    "thinking about killing myself",
    "I want to die",
])
def test_self_harm_returns_crisis_resource_not_ed_referral(text):
    verdict = assess(text)
    assert verdict.level is Triage.EMERGENCY
    assert verdict.crisis_resource is True
    assert "988" in verdict.directive


@pytest.mark.parametrize("text", [
    "runny nose and mild cough",
    "sore throat since yesterday",
    "I killed it at the gym today",
    "my speech therapist appointment is tomorrow",
    "droopy eyelid from allergies for years",
    "mild stomach ache after a big meal",
])
def test_benign_text_is_not_escalated(text):
    assert assess(text).level is not Triage.EMERGENCY


def test_empty_input_is_safe_and_does_not_raise():
    for value in ("", "   ", None):
        assert assess(value).level is Triage.ROUTINE


def test_emergency_blocks_the_llm():
    assert assess("crushing chest pain").blocks_llm is True
    assert assess("mild cough").blocks_llm is False


def test_model_cannot_de_escalate():
    verdict = assess("crushing chest pain radiating to my jaw")
    for suggestion in ("self_care", "routine", "urgent", None, "nonsense"):
        assert clamp(verdict, suggestion) is Triage.EMERGENCY


def test_model_may_escalate():
    verdict = assess("mild cough")           # ROUTINE
    assert clamp(verdict, "urgent") is Triage.URGENT
    assert clamp(verdict, "emergency") is Triage.EMERGENCY


def test_urgent_band_is_distinct_from_emergency():
    assert assess("worst headache of my life").level is Triage.URGENT
    assert assess("can't keep fluids down").level is Triage.URGENT
