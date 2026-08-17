FROM python:3.12-slim

WORKDIR /srv

# Dependencies first: this layer is cached unless requirements.txt changes,
# so day-to-day code edits rebuild in seconds rather than minutes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY ui/ ./ui/
COPY evals/ ./evals/

# Build the FAISS index at image build time so the first request is not slow.
RUN python -m app.rag.index --build

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
