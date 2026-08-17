from app.rag.corpus import DOCUMENTS
from app.rag.index import GuidanceIndex


def test_every_document_declares_provenance():
    """An uncited-looking citation is worse than none. Provenance is mandatory."""
    for doc in DOCUMENTS:
        assert doc.get("provenance"), f"{doc['id']} has no provenance"
        assert "Synthetic" in doc["provenance"]


def test_on_topic_query_retrieves_expected_document(index):
    hits = index.search("burning when I urinate and urgency")
    assert hits[0].doc_id == "doc-uti"
    assert hits[0].score > 0.5


def test_out_of_domain_query_falls_below_confidence_floor(index):
    assert index.search_confident("how do I fix my car transmission") == []


def test_embeddings_are_normalised(index):
    import numpy as np
    vec = index.embedder.encode(["fever and cough"])
    assert np.isclose(np.linalg.norm(vec[0]), 1.0, atol=1e-5)


def test_index_roundtrips_through_disk(tmp_path, index):
    index.save(tmp_path)
    reloaded = GuidanceIndex.load(tmp_path)
    # The embedder must travel with the index or scores land in a different space.
    before = index.search("diarrhea and vomiting")[0]
    after = reloaded.search("diarrhea and vomiting")[0]
    assert before.doc_id == after.doc_id
    assert abs(before.score - after.score) < 1e-5
