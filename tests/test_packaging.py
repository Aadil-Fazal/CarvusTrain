"""Packaging and public API import verification."""

import importlib


def test_import_carvustrain():
    import carvustrain

    assert carvustrain.__version__
    assert hasattr(carvustrain, "Model")


def test_public_api_exports():
    from carvustrain import (
        AgentModel,
        ChatModel,
        KnowledgeBase,
        LearningValidator,
        Model,
        activation,
        callbacks,
        layers,
        losses,
        metrics,
    )

    assert Model is not None
    assert ChatModel is not None
    assert AgentModel is not None
    assert KnowledgeBase is not None
    assert LearningValidator is not None
    assert activation is not None
    assert callbacks is not None
    assert layers is not None
    assert losses is not None
    assert metrics is not None


def test_submodule_imports():
    for name in (
        "carvustrain.model",
        "carvustrain.memory",
        "carvustrain.knowledge",
        "carvustrain.websearch",
        "carvustrain.rag",
        "carvustrain.cli",
        "carvustrain.layers",
        "carvustrain.losses",
        "carvustrain.metrics",
        "carvustrain.activation",
        "carvustrain.callbacks",
    ):
        mod = importlib.import_module(name)
        assert mod is not None


def test_cli_entrypoint_resolvable():
    from carvustrain.cli import main

    assert callable(main)
