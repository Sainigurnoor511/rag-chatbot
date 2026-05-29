from app.retrieval.chunking import compute_doc_id, chunk_text


def test_doc_id_is_deterministic_and_stable():
    assert compute_doc_id("report.pdf") == compute_doc_id("report.pdf")
    assert compute_doc_id("report.pdf") != compute_doc_id("other.pdf")
    assert len(compute_doc_id("report.pdf")) == 16


def test_chunk_text_attaches_metadata():
    text = "para one.\n\n" + ("word " * 500) + "\n\npara three."
    docs = chunk_text(text, channel_id="chan-1", filename="report.pdf",
                      chunk_size=200, chunk_overlap=20)
    assert len(docs) > 1
    doc_id = compute_doc_id("report.pdf")
    for i, d in enumerate(docs):
        assert d.metadata["channel_id"] == "chan-1"
        assert d.metadata["source"] == "report.pdf"
        assert d.metadata["doc_id"] == doc_id
        assert d.metadata["chunk_id"] == f"{doc_id}-{i}"
        assert d.page_content.strip() != ""


def test_chunk_text_empty_returns_empty():
    assert chunk_text("", channel_id="c", filename="f.pdf") == []
