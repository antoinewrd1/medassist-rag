from fhir.resources.R4B.bundle import Bundle

from app.fhir_out import build_bundle, extract_coded_symptoms
from app.safety import Triage


def test_bundle_validates_as_fhir_r4b():
    bundle = build_bundle("fever and cough", Triage.ROUTINE, "Patient/example")
    Bundle.model_validate(bundle)          # raises if non-conformant
    assert bundle["resourceType"] == "Bundle"


def test_patient_reported_observations_are_never_final():
    """`final` would misrepresent unverified self-report to a consuming system."""
    bundle = build_bundle("fever", Triage.ROUTINE, "Patient/example")
    statuses = {e["resource"]["status"] for e in bundle["entry"]}
    assert statuses == {"preliminary"}


def test_longest_match_wins_in_coding():
    coded, _ = extract_coded_symptoms("I have chest pain")
    terms = [t for t, _, _ in coded]
    assert "chest pain" in terms


def test_freetext_is_always_preserved():
    """Coding drops nuance; the raw complaint must survive it."""
    text = "weird tingling in my scalp that no code covers"
    bundle = build_bundle(text, Triage.ROUTINE, "Patient/example")
    values = [e["resource"].get("valueString") for e in bundle["entry"]]
    assert any(v and text[:20] in v for v in values)


def test_triage_level_is_coded_in_the_bundle():
    bundle = build_bundle("chest pain", Triage.EMERGENCY, "Patient/example")
    codings = [
        c["code"]
        for e in bundle["entry"]
        for c in e["resource"].get("valueCodeableConcept", {}).get("coding", [])
    ]
    assert "25876001" in codings      # SNOMED: Emergency
