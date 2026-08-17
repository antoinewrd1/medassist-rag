"""FastAPI application.

The index and LLM client are built once at startup rather than per request:
loading FAISS and the fitted embedder on every call would dominate latency.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.analyze import DISCLAIMER, analyze
from app.config import get_settings
from app.db import log_query
from app.llm import get_llm
from app.rag.index import GuidanceIndex
from app.schemas import AnalyzeRequest, AnalyzeResponse

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["index"] = GuidanceIndex.load_or_build()
    _state["llm"] = get_llm()
    yield
    _state.clear()


app = FastAPI(
    title="MedAssist RAG",
    version="1.0.0",
    description=(
        "Retrieval-grounded symptom triage prototype. Educational only; "
        "not a medical device and not a diagnostic tool."
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "llm_backend": s.llm_backend,
        "embed_backend": s.embed_backend,
        "offline_capable": s.offline,
        "documents": len(_state["index"].docs) if "index" in _state else 0,
    }


@app.get("/corpus")
def corpus() -> dict:
    """Expose what the system can actually ground answers in, and its provenance."""
    if "index" not in _state:
        raise HTTPException(status_code=503, detail="index not ready")
    return {
        "documents": [
            {"id": d["id"], "title": d["title"], "provenance": d.get("provenance")}
            for d in _state["index"].docs
        ],
        "disclaimer": DISCLAIMER,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_symptoms(req: AnalyzeRequest) -> AnalyzeResponse:
    if "index" not in _state:
        raise HTTPException(status_code=503, detail="index not ready")
    response = analyze(req, _state["index"], _state["llm"])
    log_query(response, req.symptoms)
    return response
