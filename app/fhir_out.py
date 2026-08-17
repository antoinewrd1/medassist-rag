"""FHIR R4B output.

Why FHIR at all: a symptom-intake tool that emits free text is a dead end for
integration. Emitting a `Bundle` of validated FHIR resources means the output
can post to an EHR, a health information exchange, or a surveillance system
without a bespoke adapter. R4B is targeted rather than R5 because R4/R4B is
what US health IT actually runs under the ONC certification rules.

Two honest limits stated up front:

  * The SNOMED and LOINC codes below are a small hand-built map. Real coding
    needs a terminology server (or at minimum a UMLS licence); an unmapped
    symptom is emitted as `valueString` text rather than being force-fitted to
    an approximate code, because a wrong code is worse than an uncoded
    observation -- it is wrong in a way downstream systems will trust.
  * `Observation.status` is `preliminary`, never `final`. This is patient-
    reported, unverified input. Marking it `final` would misrepresent its
    provenance to any system that consumed it.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from fhir.resources.R4B.bundle import Bundle, BundleEntry
from fhir.resources.R4B.observation import Observation

from app.safety import Triage

SNOMED = "http://snomed.info/sct"
LOINC = "http://loinc.org"

# Deliberately small and explicit. Extending this is a terminology problem.
SYMPTOM_CODES: dict[str, tuple[str, str]] = {
    "fever": ("386661006", "Fever"),
    "cough": ("49727002", "Cough"),
    "sore throat": ("267102003", "Sore throat symptom"),
    "headache": ("25064002", "Headache"),
    "diarrhea": ("62315008", "Diarrhea"),
    "vomiting": ("422400008", "Vomiting"),
    "nausea": ("422587007", "Nausea"),
    "fatigue": ("84229001", "Fatigue"),
    "rash": ("271807003", "Eruption of skin"),
    "chest pain": ("29857009", "Chest pain"),
    "shortness of breath": ("267036007", "Dyspnea"),
    "abdominal pain": ("21522001", "Abdominal pain"),
    "dizziness": ("404640003", "Dizziness"),
    "muscle aches": ("68962001", "Myalgia"),
    "chills": ("43724002", "Chill"),
}

# Triage level -> a coarse acuity coding. Mapped to SNOMED priority concepts.
TRIAGE_CODES: dict[Triage, tuple[str, str]] = {
    Triage.EMERGENCY: ("25876001", "Emergency"),
    Triage.URGENT: ("103391001", "Urgent"),
    Triage.ROUTINE: ("50811001", "Routine"),
    Triage.SELF_CARE: ("410518001", "Self care"),
}


def extract_coded_symptoms(text: str) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Return (coded, uncoded) symptom terms found in the text.

    Longest-match first so "chest pain" is not shadowed by a bare "pain", and
    so "shortness of breath" beats "breath".
    """
    lowered = (text or "").lower()
    coded: list[tuple[str, str, str]] = []
    matched_spans: list[str] = []

    for term in sorted(SYMPTOM_CODES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(term)}\b", lowered) and not any(
            term in seen for seen in matched_spans
        ):
            code, display = SYMPTOM_CODES[term]
            coded.append((term, code, display))
            matched_spans.append(term)

    return coded, []


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def symptom_observation(
    term: str, code: str, display: str, patient_ref: str
) -> Observation:
    return Observation(
        status="preliminary",  # patient-reported, unverified -- never "final"
        category=[
            {
                "coding": [
                    {
                        "system": (
                            "http://terminology.hl7.org/CodeSystem/observation-category"
                        ),
                        "code": "survey",
                        "display": "Survey",
                    }
                ]
            }
        ],
        code={"coding": [{"system": SNOMED, "code": code, "display": display}]},
        subject={"reference": patient_ref},
        effectiveDateTime=_now(),
        valueBoolean=True,
        note=[{"text": f"Patient-reported symptom: {term}"}],
    )


def freetext_observation(text: str, patient_ref: str) -> Observation:
    """The whole complaint, uncoded. Preserves what coding inevitably drops."""
    return Observation(
        status="preliminary",
        code={
            "coding": [
                {"system": LOINC, "code": "75325-1", "display": "Symptom"}
            ]
        },
        subject={"reference": patient_ref},
        effectiveDateTime=_now(),
        valueString=text[:1000],
    )


def triage_observation(triage: Triage, patient_ref: str) -> Observation:
    code, display = TRIAGE_CODES[triage]
    return Observation(
        status="preliminary",
        category=[
            {
                "coding": [
                    {
                        "system": (
                            "http://terminology.hl7.org/CodeSystem/observation-category"
                        ),
                        "code": "exam",
                        "display": "Exam",
                    }
                ]
            }
        ],
        code={
            "coding": [
                {"system": LOINC, "code": "11283-9", "display": "Acuity assessment"}
            ]
        },
        subject={"reference": patient_ref},
        effectiveDateTime=_now(),
        valueCodeableConcept={
            "coding": [{"system": SNOMED, "code": code, "display": display}]
        },
    )


def build_bundle(symptom_text: str, triage: Triage, patient_ref: str) -> dict:
    """Assemble a validated FHIR Bundle. Returns plain JSON-ready dict."""
    coded, _ = extract_coded_symptoms(symptom_text)

    resources: list[Observation] = [
        freetext_observation(symptom_text, patient_ref),
        triage_observation(triage, patient_ref),
    ]
    resources += [
        symptom_observation(term, code, display, patient_ref)
        for term, code, display in coded
    ]

    bundle = Bundle(
        type="collection",
        timestamp=_now(),
        entry=[BundleEntry(resource=r) for r in resources],
    )
    return bundle.model_dump(exclude_none=True, mode="json")
