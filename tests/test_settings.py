from app.config.settings import settings


def test_phase1_settings_defaults():
    assert settings.CHUNK_SIZE == 1000
    assert settings.CHUNK_OVERLAP == 150
    assert settings.CHROMA_COLLECTION_NAME == "rag_channel"
    assert settings.CHANNEL_TTL_SECONDS == 1800
