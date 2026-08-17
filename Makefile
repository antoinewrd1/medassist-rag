.PHONY: index api ui test eval lint all

index:      ## build the FAISS index from the guidance corpus
	python -m app.rag.index --build

api:        ## run the FastAPI backend on :8000
	uvicorn app.main:app --reload --port 8000

ui:         ## run the Streamlit frontend on :8501
	streamlit run ui/streamlit_app.py

test:
	python -m pytest -q

eval:       ## triage-safety + grounding evals; exits 1 on any gate breach
	python -m evals.run_evals

lint:
	ruff check app tests evals

all: lint test index eval
