from app.repository import channel_repository as repo


def test_register_and_list_documents(fake_redis):
    repo.register_document("chan-1", "doc-a", "alpha.pdf")
    repo.register_document("chan-1", "doc-b", "beta.docx")

    docs = repo.list_documents("chan-1")
    by_id = {d["doc_id"]: d["filename"] for d in docs}
    assert by_id == {"doc-a": "alpha.pdf", "doc-b": "beta.docx"}


def test_register_sets_ttl(fake_redis):
    repo.register_document("chan-ttl", "doc-a", "alpha.pdf")
    ttl = fake_redis.ttl("channel:chan-ttl:docs")
    assert 0 < ttl <= 1800


def test_list_documents_unknown_channel_is_empty(fake_redis):
    assert repo.list_documents("nope") == []


def test_remove_channel_clears_manifest(fake_redis):
    repo.register_document("chan-1", "doc-a", "alpha.pdf")
    repo.remove_channel("chan-1")
    assert repo.list_documents("chan-1") == []
