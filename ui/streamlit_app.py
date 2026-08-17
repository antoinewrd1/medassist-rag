"""Streamlit frontend.

Two interface decisions carry the safety posture into the UI:

  * The emergency directive renders as a full-width error banner ABOVE the
    summary, never inline. A person reading "call 911" should not have to
    scroll past a paragraph of guidance to find it.
  * Every citation renders with its provenance label attached. The corpus is
    synthetic, and a citation that looks official while being invented is the
    specific failure this project is built to avoid.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("MA_API_URL", "http://localhost:8000")

TRIAGE_STYLE = {
    "emergency": ("🚨", "error"),
    "urgent": ("⚠️", "warning"),
    "routine": ("🩺", "info"),
    "self_care": ("🏠", "success"),
}

st.set_page_config(page_title="MedAssist RAG", page_icon="🩺", layout="centered")

st.title("🩺 MedAssist RAG")
st.caption(
    "Retrieval-grounded symptom triage prototype — **educational only**. "
    "This is not a medical device, it does not diagnose, and it is not a "
    "substitute for a clinician."
)

with st.sidebar:
    st.header("System")
    try:
        health = httpx.get(f"{API_URL}/health", timeout=5).json()
        st.success("API reachable")
        st.json(health)
        if health.get("offline_capable"):
            st.info(
                "Running offline: deterministic stub LLM and local TF-IDF "
                "embeddings. Set MA_LLM_BACKEND=openai for real generation."
            )
    except Exception as exc:  # noqa: BLE001 -- surface any connection problem
        st.error(f"API unreachable at {API_URL}\n\n{exc}")

    st.divider()
    st.caption("Reference corpus")
    try:
        corpus = httpx.get(f"{API_URL}/corpus", timeout=5).json()
        for doc in corpus["documents"]:
            st.caption(f"• {doc['title']}")
        st.warning(corpus["disclaimer"])
    except Exception:  # noqa: BLE001
        pass

symptoms = st.text_area(
    "Describe your symptoms",
    placeholder="e.g. fever, sore throat and body aches for two days",
    height=120,
)
age = st.number_input("Age (optional)", min_value=0, max_value=120, value=0)

if st.button("Analyze", type="primary", use_container_width=True):
    if len(symptoms.strip()) < 3:
        st.warning("Please describe your symptoms in a little more detail.")
    else:
        payload = {"symptoms": symptoms}
        if age > 0:
            payload["age_years"] = int(age)
        try:
            result = httpx.post(f"{API_URL}/analyze", json=payload, timeout=60).json()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Request failed: {exc}")
            st.stop()

        icon, style = TRIAGE_STYLE.get(result["triage"], ("🩺", "info"))
        banner = getattr(st, style)
        # Directive first, always. Never below the fold.
        banner(f"{icon}  **{result['triage'].replace('_', ' ').title()}** — "
               f"{result['directive']}")

        if result.get("crisis_resource"):
            st.info("**988 Suicide & Crisis Lifeline** — call or text 988 (US), 24/7.")

        if result["next_steps"]:
            st.subheader("Next steps")
            for step in result["next_steps"]:
                st.write(f"- {step}")

        if result["llm_invoked"]:
            st.subheader("What the guidance says")
            st.write(result["summary"])
        elif result["triage"] != "emergency":
            st.subheader("Response")
            st.write(result["summary"])

        if result["citations"]:
            st.subheader("Sources")
            for cite in result["citations"]:
                with st.expander(f"{cite['title']}  ·  score {cite['score']}"):
                    st.caption(f"`{cite['doc_id']}`")
                    st.warning(cite["provenance"])

        if result["matched_safety_rules"]:
            st.caption(
                "Triggered safety rules: " + ", ".join(result["matched_safety_rules"])
            )

        with st.expander("FHIR R4B Bundle"):
            st.caption(
                "Structured output for EHR / HIE integration. All observations "
                "are `preliminary` — this is unverified patient-reported input."
            )
            st.json(result["fhir"])

        st.divider()
        st.caption(result["disclaimer"])
