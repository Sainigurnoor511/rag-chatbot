from app.config.settings import settings


def test_phase2_retrieval_settings_defaults():
    assert settings.DENSE_TOP_K == 20
    assert settings.BM25_TOP_K == 20
    assert settings.RRF_K == 60
    assert settings.RERANK_TOP_N == 5
    assert settings.RERANKER_MODEL == "BAAI/bge-reranker-base"
