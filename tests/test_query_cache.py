import app.repository.query_cache as qc


def test_cache_key_is_stable_and_scoped():
    k1 = qc.make_key("chan-1", "Hello There", "a.pdf")
    k2 = qc.make_key("chan-1", "hello there", "a.pdf")
    k3 = qc.make_key("chan-2", "Hello There", "a.pdf")
    assert k1 == k2
    assert k1 != k3
    assert k1.startswith("qcache:")


def test_get_set_roundtrip(fake_redis):
    qc.set_cached("chan-1", "q", None, "the answer", ttl=300)
    assert qc.get_cached("chan-1", "q", None) == "the answer"


def test_get_missing_returns_none(fake_redis):
    assert qc.get_cached("chan-1", "absent", None) is None
