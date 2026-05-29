import importlib


def test_inmemory_cleanup_module_removed():
    """The per-process cleanup task is replaced by Redis TTL on the manifest."""
    try:
        importlib.import_module("app.utilities.file_embeddings_handler")
        raised = False
    except ModuleNotFoundError:
        raised = True
    assert raised, "file_embeddings_handler should be removed in favor of Redis TTL"


def test_main_imports_without_inmemory_handler():
    import main
    assert hasattr(main, "app")
