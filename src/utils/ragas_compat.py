"""Compatibility helpers for optional Vertex AI imports in Ragas.

Ragas 0.4.x imports the legacy ``langchain_community`` Vertex AI modules
unconditionally. Modern LangChain releases moved those integrations to
``langchain-google-vertexai``. The lab supports Gemini through
``langchain-google-genai`` and does not need Vertex AI for evaluation, so a
small optional-module shim keeps non-Vertex providers usable as well.
"""

import importlib
import sys
import types


def install_ragas_vertex_compat() -> None:
    """Make Ragas' legacy optional Vertex imports safe to resolve.

    If the legacy modules are available, nothing is changed. Otherwise the
    modern classes are exposed at the old import locations. When the optional
    Vertex package is not installed, harmless placeholder classes are used;
    they are only referenced by Ragas' type-dispatch list and never selected
    for the lab's OpenAI/Gemini/Anthropic/Ollama providers.
    """

    try:
        importlib.import_module("langchain_community.chat_models.vertexai")
        importlib.import_module("langchain_community.llms")
        return
    except (ImportError, ModuleNotFoundError):
        pass

    try:
        from langchain_google_vertexai.chat_models import ChatVertexAI
        from langchain_google_vertexai.llms import VertexAI
    except (ImportError, ModuleNotFoundError):
        class ChatVertexAI:  # pragma: no cover - only a compatibility type
            pass

        class VertexAI:  # pragma: no cover - only a compatibility type
            pass

    legacy_chat_module = types.ModuleType(
        "langchain_community.chat_models.vertexai"
    )
    legacy_chat_module.ChatVertexAI = ChatVertexAI
    sys.modules[legacy_chat_module.__name__] = legacy_chat_module

    community_llms = importlib.import_module("langchain_community.llms")
    community_llms.VertexAI = VertexAI
