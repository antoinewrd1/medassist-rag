"""Request and response contracts.

Pydantic models rather than loose dicts so that FastAPI generates an accurate
OpenAPI spec and rejects malformed input at the boundary instead of deep inside
the analysis code.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.safety import Triage


class AnalyzeRequest(BaseModel):
    symptoms: str = Field(
        min_length=3,
        max_length=2000,
        description="Free-text symptom description",
        examples=["fever, sore throat and body aches for two days"],
    )
    age_years: int | None = Field(default=None, ge=0, le=120)
    patient_ref: str = Field(
        default="Patient/example",
        description="FHIR reference; this prototype stores no identifiable data",
    )


class Citation(BaseModel):
    doc_id: str
    title: str
    provenance: str
    score: float


class AnalyzeResponse(BaseModel):
    triage: Triage
    directive: str
    summary: str
    next_steps: list[str] = []
    citations: list[Citation] = []
    matched_safety_rules: list[str] = []
    safety_reasons: list[str] = []
    crisis_resource: bool = False
    llm_invoked: bool
    grounded: bool
    backend: str
    fhir: dict | None = None
    disclaimer: str
