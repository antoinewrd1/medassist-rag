"""Two embedding backends behind one interface.

  openai -- text-embedding-3-small. What you would actually deploy.
  local  -- TF-IDF reduced to a dense vector with TruncatedSVD (classical LSA).

The local backend exists so the test suite and CI run with no API key, no
network, and no cost, and so the eval harness can be regression-tested
independently of a vendor. It is genuinely weaker at paraphrase than a neural
embedding -- LSA captures term co-occurrence, not meaning -- and the README
says so rather than implying the two are equivalent.

Both produce L2-normalised vectors, so FAISS inner-product search is cosine
similarity in both cases and the score threshold means the same thing.
"""

from __future__ import annotations

import numpy as np

from app.config import get_settings


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0          # a zero vector stays zero rather than NaN
    return (mat / norms).astype("float32")


class LocalEmbedder:
    """TF-IDF + LSA. Deterministic, offline, no model download."""

    name = "local-tfidf-lsa"

    def __init__(self, n_components: int = 64) -> None:
        self.n_components = n_components
        self._vectorizer = None
        self._svd = None

    def fit(self, texts: list[str]) -> "LocalEmbedder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),      # bigrams catch "chest pain", "sore throat"
            sublinear_tf=True,
        )
        tfidf = self._vectorizer.fit_transform(texts)
        # SVD cannot produce more components than the corpus supports.
        k = min(self.n_components, tfidf.shape[1] - 1, max(len(texts) - 1, 1))
        self._svd = TruncatedSVD(n_components=k, random_state=0)
        self._svd.fit(tfidf)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._vectorizer is None or self._svd is None:
            raise RuntimeError("LocalEmbedder.fit() must be called before encode()")
        return _l2_normalize(self._svd.transform(self._vectorizer.transform(texts)))

    @property
    def dim(self) -> int:
        return int(self._svd.n_components) if self._svd else self.n_components


class OpenAIEmbedder:
    """Hosted embeddings. Stateless -- `fit` is a no-op by design."""

    name = "openai"

    def __init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model = get_settings().embed_model
        self._dim = 1536

    def fit(self, texts: list[str]) -> "OpenAIEmbedder":
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype="float32")
        self._dim = vecs.shape[1]
        return _l2_normalize(vecs)

    @property
    def dim(self) -> int:
        return self._dim


def get_embedder():
    backend = get_settings().embed_backend
    if backend == "openai":
        return OpenAIEmbedder()
    if backend == "local":
        return LocalEmbedder()
    raise ValueError(f"unknown MA_EMBED_BACKEND={backend!r}")
