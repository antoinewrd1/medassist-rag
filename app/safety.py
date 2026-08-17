"""Deterministic triage safety layer.

This module runs BEFORE the LLM, and its verdict cannot be overridden by
anything the model says afterwards. That ordering is the whole design.

Why it is rule-based and not a model:

  A symptom checker's catastrophic failure mode is not "gave a vague answer".
  It is "told someone with crush-type chest pain and left-arm radiation to rest
  and hydrate". A language model is a probability distribution; for the small
  set of presentations where minutes matter, we want a guarantee, not a high
  likelihood. So red-flag detection is deterministic, unit-tested, and its
  recall is gated in CI at 1.00 -- a single miss fails the build.

  The tradeoff is deliberate over-triage: these patterns will fire on people
  who are not having an emergency. For a non-diagnostic prototype that always
  routes to a human clinician anyway, a false alarm costs a wasted trip. The
  inverse error is unbounded. We accept the false positives.

This is not a medical device and does not diagnose. Every path in this module
terminates in "see a clinician" -- the only variable is how urgently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Triage(str, Enum):
    """Urgency bands. Ordered: EMERGENCY is the most severe."""

    EMERGENCY = "emergency"      # call 911 / go to an ED now
    URGENT = "urgent"            # seek care today
    ROUTINE = "routine"          # schedule with a clinician
    SELF_CARE = "self_care"      # reasonable to monitor at home


# Each rule is (rule_id, human-readable reason, regex).
# Patterns are intentionally broad. Recall beats precision here.
EMERGENCY_RULES: list[tuple[str, str, str]] = [
    (
        "cardiac_chest_pain",
        "Chest pain with features suggesting a cardiac cause",
        r"chest (pain|pressure|tightness|discomfort)|crushing chest|"
        r"pain radiat\w* (to|down|into) (my )?(left )?(arm|jaw|shoulder|back)",
    ),
    (
        "stroke_fast",
        "Signs consistent with stroke (FAST: face, arm, speech, time)",
        r"(face|mouth|facial)[^.!?]{0,20}droop\w*|droop\w*[^.!?]{0,20}(face|mouth)|"
        # Proximity match rather than adjacency: "speech has been slurred since
        # this morning" must fire the same as "slurred speech".
        r"speech[^.!?]{0,25}slurr\w*|slurr\w*[^.!?]{0,25}speech|"
        r"can'?t speak|cannot speak|trouble speaking|words come out wrong|"
        r"sudden (weakness|numbness) (on |in )?(one side|left side|right side)|"
        r"one side of (my |the )?(face|body) (is )?(numb|weak|droop\w*)|"
        r"sudden (confusion|vision loss)",
    ),
    (
        "respiratory_distress",
        "Difficulty breathing or airway compromise",
        r"can'?t breathe|cannot breathe|struggling to breathe|gasping|"
        r"severe shortness of breath|turning blue|lips (are )?blue|"
        r"choking|throat (is )?closing",
    ),
    (
        "anaphylaxis",
        "Signs consistent with a severe allergic reaction",
        r"anaphyla|throat swelling|tongue swelling|swollen (throat|tongue|lips)|"
        r"hives (and|with) (trouble|difficulty) breathing",
    ),
    (
        "severe_bleeding",
        "Uncontrolled bleeding",
        r"bleeding (that )?(won'?t|will not) stop|uncontrolled bleeding|"
        r"spurting blood|losing a lot of blood|hemorrhag",
    ),
    (
        "altered_consciousness",
        "Loss of consciousness or altered mental status",
        r"unconscious|passed out|fainted|unresponsive|won'?t wake up|"
        r"not making sense|seizure|convulsion",
    ),
    (
        "meningitis",
        "Fever with neck stiffness or non-blanching rash",
        r"stiff neck (and|with) fever|fever (and|with) stiff neck|"
        r"rash that doesn'?t fade|non.?blanching rash|light hurts my eyes (and|with) fever",
    ),
    (
        "severe_abdominal",
        "Severe abdominal pain with rigidity or vomiting blood",
        r"vomiting blood|blood in (my )?vomit|coffee.ground|"
        r"rigid abdomen|board.like abdomen|severe abdominal pain",
    ),
    (
        "pregnancy_emergency",
        "Bleeding or severe pain in pregnancy",
        r"pregnan\w+ (and|with) (bleeding|severe pain|no fetal movement)|"
        r"bleeding (and|while) pregnan",
    ),
    (
        "infant_fever",
        "Fever in a very young infant",
        r"(newborn|infant|baby|\b[0-9]\s*(week|month)s? old)\b.{0,40}fever|"
        r"fever.{0,40}\b(newborn|\b[0-2]\s*months? old)\b",
    ),
]

# Handled separately: the response is a crisis resource, not an ED referral.
# NOTE: written to survive contraction variants ("don't" / "do not" / "dont").
# The first draft of this pattern only matched "don't want to live" and missed
# "I do not want to live anymore" -- caught by the eval gate, not by review.
SELF_HARM_PATTERN = (
    r"kill myself|killing myself|suicid|end (my|it) (life|all)|"
    r"do\s*(n'?t| not)\s*want to (live|be alive|wake up)|"
    r"want to (die|be dead)|hurt myself|hurting myself|self.?harm|"
    r"better off dead|no reason to live|nothing to live for"
)

URGENT_RULES: list[tuple[str, str, str]] = [
    (
        "persistent_high_fever",
        "High or persistent fever",
        r"fever (over|above|of) 10[3-9]|10[3-9](\.\d)?\s*(degrees|f\b)|"
        r"fever for (more than |over )?([4-9]|1\d)\s*days",
    ),
    (
        "dehydration",
        "Signs of dehydration or inability to keep fluids down",
        r"can'?t keep (fluids|water|anything) down|not urinating|"
        r"no wet diapers|severely dehydrated",
    ),
    (
        "infection_spreading",
        "Possible spreading infection",
        r"red streaks|spreading redness|wound (is )?(infected|oozing pus)|"
        r"warm (and|,) swollen (and )?painful",
    ),
    (
        "new_severe_headache",
        "Sudden severe headache",
        r"worst headache of my life|thunderclap headache|sudden severe headache",
    ),
]


@dataclass
class TriageVerdict:
    level: Triage
    matched_rules: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    directive: str = ""
    crisis_resource: bool = False

    @property
    def blocks_llm(self) -> bool:
        """Emergency and crisis paths skip generation entirely.

        There is no useful role for a language model between a person
        describing stroke symptoms and being told to call 911.
        """
        return self.level is Triage.EMERGENCY


EMERGENCY_DIRECTIVE = (
    "Your description includes symptoms that can indicate a medical emergency. "
    "Call 911 (or your local emergency number) or go to the nearest emergency "
    "department now. Do not drive yourself if you feel faint or short of breath."
)

CRISIS_DIRECTIVE = (
    "It sounds like you may be going through something very difficult. You do "
    "not have to handle this alone. In the US you can call or text 988 to reach "
    "the Suicide & Crisis Lifeline, 24 hours a day. If you are in immediate "
    "danger, please call 911."
)

URGENT_DIRECTIVE = (
    "These symptoms should be evaluated by a clinician today. Contact your "
    "doctor, an urgent care clinic, or a nurse line for same-day guidance."
)

ROUTINE_DIRECTIVE = (
    "These symptoms are worth discussing with a clinician. Schedule an "
    "appointment, and seek care sooner if anything worsens."
)


def _search(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def assess(symptom_text: str) -> TriageVerdict:
    """Classify urgency from raw symptom text. Never raises, never returns None."""
    text = (symptom_text or "").strip()
    if not text:
        return TriageVerdict(
            level=Triage.ROUTINE,
            directive=ROUTINE_DIRECTIVE,
            reasons=["No symptoms provided"],
        )

    # Order matters: self-harm is checked first so that a crisis resource is
    # never buried under a generic ED referral.
    if _search(SELF_HARM_PATTERN, text):
        return TriageVerdict(
            level=Triage.EMERGENCY,
            matched_rules=["self_harm_risk"],
            reasons=["Expressed thoughts of self-harm or suicide"],
            directive=CRISIS_DIRECTIVE,
            crisis_resource=True,
        )

    hits = [(rid, reason) for rid, reason, pat in EMERGENCY_RULES if _search(pat, text)]
    if hits:
        return TriageVerdict(
            level=Triage.EMERGENCY,
            matched_rules=[r for r, _ in hits],
            reasons=[reason for _, reason in hits],
            directive=EMERGENCY_DIRECTIVE,
        )

    hits = [(rid, reason) for rid, reason, pat in URGENT_RULES if _search(pat, text)]
    if hits:
        return TriageVerdict(
            level=Triage.URGENT,
            matched_rules=[r for r, _ in hits],
            reasons=[reason for _, reason in hits],
            directive=URGENT_DIRECTIVE,
        )

    return TriageVerdict(
        level=Triage.ROUTINE,
        directive=ROUTINE_DIRECTIVE,
        reasons=["No red-flag or urgent patterns detected"],
    )


def clamp(verdict: TriageVerdict, model_suggested: str | None) -> Triage:
    """Let the model raise urgency, never lower it.

    If the rules say EMERGENCY and the model says SELF_CARE, the rules win. If
    the rules say ROUTINE and the model argues for URGENT, we take the model's
    higher reading. Escalation is cheap; de-escalation is the dangerous
    direction, so the model is only trusted in one of them.
    """
    order = [Triage.SELF_CARE, Triage.ROUTINE, Triage.URGENT, Triage.EMERGENCY]
    try:
        suggested = Triage(model_suggested) if model_suggested else verdict.level
    except ValueError:
        return verdict.level
    return max(verdict.level, suggested, key=order.index)
