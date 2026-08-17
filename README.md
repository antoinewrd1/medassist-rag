# MedAssist RAG — retrieval-grounded symptom triage

A FastAPI + Streamlit prototype that takes a free-text symptom description and
returns a **triage urgency level**, a **grounded summary with citations**, and a
**FHIR R4B Bundle** for downstream systems.

The interesting part is not the LLM call. It is the set of guardrails around it:
a deterministic red-flag layer that runs *before* generation and cannot be
overridden, a retrieval confidence floor that refuses rather than guessing, and
an eval harness that fails CI if red-flag recall drops below 1.00.

**This is an educational prototype. It is not a medical device, it does not
diagnose, and it must not be used for real clinical decisions.**

---

## What is verified vs. what is claimed

| Claim | Status |
|---|---|
| Full request path runs offline with no API key | **Verified** — `llm=fake`, `embed=local`; CI runs it on every push |
| Red-flag recall = 1.00 | **Verified on 16 hand-written vignettes** (`evals/run_evals.py`), gated in CI. Sixteen cases is a smoke test, not a clinical validation |
| Over-triage rate = 0.00 on 12 benign vignettes | **Verified** — includes adversarial phrasings ("I killed it at the gym", "droopy eyelid from allergies") |
| Emergency input never reaches the LLM | **Verified** — `test_api.py::test_emergency_short_circuits_before_the_model` asserts `llm_invoked is False` |
| Model cannot de-escalate urgency | **Verified** — `clamp()` is tested against every lower suggestion |
| FHIR R4B Bundle validates | **Verified** — round-tripped through `Bundle.model_validate` |
| RAG citation + refusal accuracy = 1.00 | **Verified with the local TF-IDF backend only.** Numbers with OpenAI embeddings and a real model are **not** claimed |
| Postgres via docker-compose | **Written, not run in CI** — CI uses SQLite through the same SQLAlchemy layer |
| OpenAI backends (`llm=openai`, `embed=openai`) | **Written, not executed** — no key was used in building this |

## Quickstart (no API key, no network)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make index     # build the FAISS index
make test      # 45 tests
make eval      # safety + grounding gates
make api       # http://localhost:8000/docs
make ui        # http://localhost:8501
```

Or the whole stack with Postgres:

```bash
docker compose up --build
```

To use real models, set `MA_LLM_BACKEND=openai`, `MA_EMBED_BACKEND=openai`, and
`OPENAI_API_KEY` (see `.env.example`).

## How a request flows

```
symptom text
   │
   ├─ 1. SAFETY GATE (app/safety.py) ── deterministic regex rules
   │      └─ EMERGENCY? → return directive immediately. LLM never called.
   │
   ├─ 2. RETRIEVE (app/rag/) ── FAISS IndexFlatIP, cosine similarity
   │      └─ nothing above the floor? → refuse, do not guess.
   │
   ├─ 3. GENERATE (app/llm.py) ── JSON mode, fenced to retrieved passages only
   │
   ├─ 4. VERIFY (app/analyze.py) ── citations checked against what was retrieved;
   │      unknown ids dropped; uncited answers replaced with a refusal
   │
   ├─ 5. CLAMP ── model may raise urgency, never lower it
   │
   └─ FHIR R4B Bundle + response
```

## Why the safety layer is rules, not a model

The catastrophic failure of a symptom checker is not a vague answer. It is
telling someone with crush-type chest pain and left-arm radiation to rest and
hydrate.

A language model is a probability distribution. For the small set of
presentations where minutes matter, a guarantee beats a high likelihood — so
red-flag detection is deterministic, unit-tested, and CI-gated at perfect
recall. A single miss fails the build.

The tradeoff is deliberate over-triage: broad patterns will fire on people who
are not having an emergency. For a prototype that routes everyone to a clinician
anyway, a false alarm costs a wasted trip. The inverse error is unbounded. The
eval harness measures the over-triage rate so the tradeoff stays visible rather
than drifting toward "everything is an emergency."

Self-harm language is handled on its own path: it returns the **988 Suicide &
Crisis Lifeline** rather than a generic emergency-department referral.

## Honest limitations

- **The guidance corpus is synthetic.** Every passage was written for this
  project. It paraphrases the *shape* of published clinical guidance but
  carries no clinical authority. Every document has a `provenance` field, the
  API returns it with each citation, and the UI renders it beside the text —
  because a citation that looks official while being invented is worse than no
  citation at all.
- **16 red-flag vignettes is not clinical validation.** Recall of 1.00 means the
  rules catch the cases someone thought to write down. Real validation needs
  clinician-authored vignettes and adversarial phrasing at a scale this
  prototype does not have.
- **The offline stub is extractive, not intelligent.** It returns the first
  sentence of the top passage with a citation. Numbers under
  `llm_backend: "fake"` validate the guardrails and the harness — not a model.
  The eval report records which backend produced it.
- **Local embeddings are TF-IDF + LSA**, which captures term co-occurrence, not
  meaning. It is genuinely weaker at paraphrase than a neural embedding. The
  OpenAI backend exists for that reason and is untested here.
- **Symptom text is PHI.** This prototype logs it because reviewing what the
  system said is the point of the log. Real deployment needs a retention
  policy, encryption at rest, access logging, and a HIPAA-covered hosting
  agreement. `store_text=False` in `app/db.py` disables persistence and is where
  a real deployment would start.
- **SNOMED/LOINC coding is a hand-built map of 15 terms.** Unmapped symptoms are
  emitted as `valueString` rather than force-fitted to an approximate code — a
  wrong code is worse than an uncoded observation, because downstream systems
  trust it. Real coding needs a terminology server.

## Repository layout

```
app/
  safety.py        deterministic triage rules; runs before anything else
  analyze.py       orchestration: safety → retrieve → generate → verify → clamp
  llm.py           OpenAI client + deterministic offline stub
  fhir_out.py      FHIR R4B Observation / Bundle generation
  db.py            SQLAlchemy query logging (SQLite or Postgres)
  main.py          FastAPI: /health, /corpus, /analyze
  schemas.py       Pydantic request/response contracts
  rag/
    corpus.py      guidance documents + provenance labels
    embeddings.py  OpenAI | local TF-IDF+LSA, both L2-normalised
    index.py       FAISS build/save/load/search with confidence floor
ui/                Streamlit frontend
evals/             CI-gated safety + grounding harness
tests/             45 tests
```

## License

MIT. Educational use only — see the disclaimer above.
