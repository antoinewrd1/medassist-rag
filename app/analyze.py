"""Request orchestration. The order of operations here is the product.

    1. SAFETY FIRST, unconditionally. Red-flag assessment runs on raw input
       before retrieval or generation. If it returns EMERGENCY, the LLM is
       never called -- there is no useful sentence a model can add between
       "my face is drooping" and "call 911", and every extra generated token
       is another chance to say something harmful.

    2. RETRIEVE WITH A FLOOR. If nothing clears the confidence threshold we
       refuse rather than ground an answer in loosely-related text.

    3. GENERATE INSIDE A FENCE. The model is asked for JSON and gets exactly
       the retrieved passages, nothing else.

    4. VERIFY, DON'T TRUST. Citations are checked against what was actually
       retrieved; unknown ids are dropped. An answer left with no valid
       citation is replaced by a refusal regardless of how confident it reads.

    5. CLAMP UPWARD ONLY. The model may raise urgency, never lower it. The
       rules from step 1 are a floor the model cannot argue below.

Steps 4 and 5 are the ones worth arguing about in review: both assume the model
will eventually do the wrong thing, and both are cheap enough to always run.
"""

from __future__ import annotations

import json
import re

from app.config import get_settings
from app.fhir_out import build_bundle
from app.rag.index import GuidanceIndex, Hit
from app.safety import Triage, TriageVerdict, assess, clamp
from app.schemas import AnalyzeRequest, AnalyzeResponse, Citation

DISCLAIMER = (
    "This is an educational prototype, not a medical device. It does not "
    "diagnose. Always consult a qualified clinician about your symptoms."
)

REFUSAL = "INSUFFICIENT_CONTEXT"

NO_CONTEXT_SUMMARY = (
    "I don't have guidance in my reference set that covers this. I'd rather say "
    "so than guess. Please describe your symptoms to a clinician or a nurse line."
)

SYSTEM_PROMPT = """You are a careful clinical information assistant. You do NOT
diagnose and you do NOT prescribe.

You will receive a patient's symptom description and a set of guidance passages.
Use ONLY those passages.

Return ONE JSON object, no prose, no markdown fences:
{"summary": str, "citations": [str], "suggested_triage": "emergency"|"urgent"|"routine"|"self_care", "next_steps": [str]}

Rules:
- Every factual claim in `summary` must be supported by a passage, and the
  passage id must appear in `citations`.
- If the passages do not address the symptoms, set `summary` to exactly
  INSUFFICIENT_CONTEXT and return empty citations.
- Never name a specific diagnosis as if it were established. Describe what the
  guidance says about symptoms like these.
- `suggested_triage` reflects urgency of seeking human care, not a diagnosis.
- Keep `summary` under 120 words and `next_steps` to at most four items."""

_CITE_RE = re.compile(r"\[(doc-[\w-]+)\]")


def _strip_fences(text: str) -> str:
    return re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip()).strip()


def _emergency_response(
    verdict: TriageVerdict, req: AnalyzeRequest, backend: str
) -> AnalyzeResponse:
    """Short-circuit path. No retrieval, no generation, no ambiguity."""
    return AnalyzeResponse(
        triage=verdict.level,
        directive=verdict.directive,
        summary=verdict.directive,
        next_steps=(
            ["Call or text 988 (US) for the Suicide & Crisis Lifeline",
             "Call 911 if you are in immediate danger",
             "Reach out to someone you trust and let them know how you're feeling"]
            if verdict.crisis_resource
            else ["Call 911 or your local emergency number now",
                  "Do not drive yourself if you feel faint or short of breath",
                  "If someone is with you, ask them to stay"]
        ),
        citations=[],
        matched_safety_rules=verdict.matched_rules,
        safety_reasons=verdict.reasons,
        crisis_resource=verdict.crisis_resource,
        llm_invoked=False,
        grounded=True,          # grounded in a deterministic rule, not a model
        backend=backend,
        fhir=build_bundle(req.symptoms, verdict.level, req.patient_ref),
        disclaimer=DISCLAIMER,
    )


def _parse_model_json(raw: str) -> dict:
    try:
        payload = json.loads(_strip_fences(raw))
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def analyze(req: AnalyzeRequest, index: GuidanceIndex, llm) -> AnalyzeResponse:
    settings = get_settings()
    backend = f"llm={llm.name},embed={settings.embed_backend}"

    # ---- 1. safety gate ---------------------------------------------------
    verdict = assess(req.symptoms)
    if verdict.blocks_llm:
        return _emergency_response(verdict, req, backend)

    # ---- 2. retrieval with a confidence floor -----------------------------
    hits: list[Hit] = index.search_confident(req.symptoms)
    if not hits:
        return AnalyzeResponse(
            triage=verdict.level,
            directive=verdict.directive,
            summary=NO_CONTEXT_SUMMARY,
            next_steps=["Describe these symptoms to a clinician or nurse line"],
            citations=[],
            matched_safety_rules=verdict.matched_rules,
            safety_reasons=verdict.reasons,
            llm_invoked=False,
            grounded=False,
            backend=backend,
            fhir=build_bundle(req.symptoms, verdict.level, req.patient_ref),
            disclaimer=DISCLAIMER,
        )

    # ---- 3. generation, fenced to retrieved context -----------------------
    context = "\n\n".join(f"[{h.doc_id}] {h.text}" for h in hits)
    age_line = f"Patient age: {req.age_years}\n" if req.age_years is not None else ""
    user = f"{age_line}Symptoms: {req.symptoms}\n\nGuidance passages:\n{context}"
    payload = _parse_model_json(llm.complete(SYSTEM_PROMPT, user, json_mode=True).text)

    summary = str(payload.get("summary", "")).strip()
    claimed = payload.get("citations") or []
    next_steps = [str(s) for s in (payload.get("next_steps") or [])][:4]

    # ---- 4. verify citations against what was actually retrieved ----------
    retrieved_ids = {h.doc_id for h in hits}
    cited = {c for c in claimed if c in retrieved_ids}
    cited |= {c for c in _CITE_RE.findall(summary) if c in retrieved_ids}

    ungrounded = summary == REFUSAL or (not cited and summary)
    if ungrounded:
        summary = NO_CONTEXT_SUMMARY
        next_steps = ["Describe these symptoms to a clinician or nurse line"]
        cited = set()

    # ---- 5. clamp upward only ---------------------------------------------
    final_triage: Triage = clamp(verdict, payload.get("suggested_triage"))
    directive = verdict.directive
    if final_triage is not verdict.level:
        from app.safety import URGENT_DIRECTIVE

        directive = URGENT_DIRECTIVE if final_triage is Triage.URGENT else directive

    citations = [
        Citation(**h.to_citation() | {"title": h.title})
        for h in hits
        if h.doc_id in cited
    ]

    return AnalyzeResponse(
        triage=final_triage,
        directive=directive,
        summary=summary,
        next_steps=next_steps,
        citations=citations,
        matched_safety_rules=verdict.matched_rules,
        safety_reasons=verdict.reasons,
        llm_invoked=True,
        grounded=bool(cited),
        backend=backend,
        fhir=build_bundle(req.symptoms, final_triage, req.patient_ref),
        disclaimer=DISCLAIMER,
    )
