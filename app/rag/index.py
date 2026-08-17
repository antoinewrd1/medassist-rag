"""FAISS vector index over the guidance corpus.

`IndexFlatIP` is exact inner-product search. With L2-normalised vectors that is
cosine similarity, and with a corpus of a dozen documents an approximate index
(IVF, HNSW) would be slower and less accurate than brute force. Choosing the
simple index here is the correct engineering answer, not a shortcut -- ANN
structures earn their complexity somewhere north of 100k vectors.

The index and the fitted embedder are persisted together. That pairing is not
incidental: the local embedder learns its vocabulary from the corpus, so an
index built with one embedder and queried with a freshly-fitted one would
return vectors from a different space and quietly produce nonsense scores.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss

from app.config import get_settings
from app.rag.corpus import DOCUMENTS


@dataclass
class Hit:
    doc_id: str
    title: str
    text: str
    provenance: str
    score: float

    def to_citation(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "provenance": self.provenance,
            "score": round(self.score, 4),
        }


class GuidanceIndex:
    def __init__(self, embedder, index, docs: list[dict]) -> None:
        self.embedder = embedder
        self.index = index
        self.docs = docs

    # ---- construction ----------------------------------------------------
    @classmethod
    def build(cls, docs: list[dict] | None = None, embedder=None) -> "GuidanceIndex":
        from app.rag.embeddings import get_embedder

        docs = docs or DOCUMENTS
        embedder = embedder or get_embedder()
        corpus = [f"{d['title']}. {d['text']}" for d in docs]
        embedder.fit(corpus)
        vectors = embedder.encode(corpus)

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(embedder, index, docs)

    # ---- persistence -----------------------------------------------------
    def save(self, directory: Path | None = None) -> Path:
        directory = Path(directory or get_settings().index_dir)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "guidance.faiss"))
        (directory / "docs.json").write_text(json.dumps(self.docs, indent=2))
        # The embedder ships with the index; see the module docstring.
        with (directory / "embedder.pkl").open("wb") as fh:
            pickle.dump(self.embedder, fh)
        return directory

    @classmethod
    def load(cls, directory: Path | None = None) -> "GuidanceIndex":
        directory = Path(directory or get_settings().index_dir)
        index = faiss.read_index(str(directory / "guidance.faiss"))
        docs = json.loads((directory / "docs.json").read_text())
        with (directory / "embedder.pkl").open("rb") as fh:
            embedder = pickle.load(fh)
        return cls(embedder, index, docs)

    @classmethod
    def load_or_build(cls, directory: Path | None = None) -> "GuidanceIndex":
        directory = Path(directory or get_settings().index_dir)
        if (directory / "guidance.faiss").exists():
            try:
                return cls.load(directory)
            except Exception:  # corrupt, stale or version-mismatched: rebuild
                pass
        return cls.build()

    # ---- query -----------------------------------------------------------
    def search(self, query: str, k: int | None = None) -> list[Hit]:
        settings = get_settings()
        k = k or settings.top_k
        vec = self.embedder.encode([query])
        scores, idxs = self.index.search(vec, min(k, len(self.docs)))

        hits: list[Hit] = []
        for score, idx in zip(scores[0], idxs[0], strict=True):
            if idx < 0:
                continue
            doc = self.docs[int(idx)]
            hits.append(
                Hit(
                    doc_id=doc["id"],
                    title=doc["title"],
                    text=doc["text"],
                    provenance=doc.get("provenance", "unknown"),
                    score=float(score),
                )
            )
        return hits

    def search_confident(self, query: str, k: int | None = None) -> list[Hit]:
        """Search, then discard everything below the confidence floor.

        Returning an empty list is a valid and useful outcome: the caller
        refuses rather than grounding an answer in loosely-related text. Weak
        retrieval is what produces confident, wrong, cited-looking answers.
        """
        floor = get_settings().min_score
        return [h for h in self.search(query, k) if h.score >= floor]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--query", default=None)
    args = ap.parse_args()

    if args.build:
        idx = GuidanceIndex.build()
        out = idx.save()
        s = get_settings()
        print(
            f"indexed {len(idx.docs)} documents "
            f"(embedder={idx.embedder.name}, dim={idx.embedder.dim}) -> {out}\n"
            f"  confidence floor: {s.min_score}"
        )
    if args.query:
        idx = GuidanceIndex.load_or_build()
        for h in idx.search(args.query):
            print(f"  {h.score:.4f}  {h.doc_id:24s} {h.title}")
    if not args.build and not args.query:
        ap.print_help()


if __name__ == "__main__":
    main()
