import app.utilities.rag_utilities as util_mod


def test_load_embeddings_uses_channel_collection(monkeypatch, tmp_path):
    captured = {}

    class _FakeChroma:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    # channel dir must exist and be non-empty to pass the guard
    channel_dir = tmp_path / "chan-1"
    channel_dir.mkdir()
    (channel_dir / "chroma.sqlite3").write_text("x")

    monkeypatch.setattr(util_mod, "EMBEDDING_DIR", str(tmp_path))
    monkeypatch.setattr(util_mod, "Chroma", _FakeChroma)
    monkeypatch.setattr(util_mod.RAGUtilities, "__init__", lambda self: None)

    inst = util_mod.RAGUtilities()
    inst.embedding_model = object()
    inst.load_embeddings("chan-1")

    assert captured["collection_name"] == "rag_channel"
    assert captured["persist_directory"].endswith("chan-1")
