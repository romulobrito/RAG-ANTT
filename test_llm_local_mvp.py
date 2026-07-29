#!/usr/bin/env python3
"""
Testes sandbox do provedor Ollama (MVP aditivo, sem API).

Executar:
    cd RAG-ANTT
    source venv/bin/activate
    python test_llm_local_mvp.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEFAULT_LLM_PROVIDER,
    LLM_PROVIDERS,
    cloud_fallback_enabled,
    get_allowed_llm_providers,
)
from llm_providers import get_available_providers


def test_provedores_cloud_inalterados() -> None:
    """Garante que OpenAI e DeepSeek permanecem registrados."""
    assert "openai" in LLM_PROVIDERS
    assert "deepseek" in LLM_PROVIDERS
    assert LLM_PROVIDERS["openai"].get("requires_api_key") is True
    assert LLM_PROVIDERS["deepseek"].get("requires_api_key") is True


def test_default_provider_ainda_deepseek() -> None:
    """Padrao de geracao continua DeepSeek ate decisao explicita."""
    assert DEFAULT_LLM_PROVIDER == "deepseek"


def test_config_ollama_registrado() -> None:
    """Ollama entra como terceiro provedor aditivo."""
    assert "ollama" in LLM_PROVIDERS
    cfg = LLM_PROVIDERS["ollama"]
    assert cfg.get("requires_api_key") is False
    assert "llama3.2:3b" in cfg["models"]
    assert callable(cfg["get_api_key"])
    assert callable(cfg["get_base_url"])


def test_filtro_antt_prod_so_ollama() -> None:
    """RAG_LLM_ALLOWED_PROVIDERS=ollama restringe a sidebar."""
    anterior = os.environ.get("RAG_LLM_ALLOWED_PROVIDERS")
    try:
        os.environ["RAG_LLM_ALLOWED_PROVIDERS"] = "ollama"
        assert get_allowed_llm_providers() == ["ollama"]
        disponiveis = get_available_providers()
        assert list(disponiveis.keys()) == ["ollama"]
    finally:
        if anterior is None:
            os.environ.pop("RAG_LLM_ALLOWED_PROVIDERS", None)
        else:
            os.environ["RAG_LLM_ALLOWED_PROVIDERS"] = anterior


def test_cloud_fallback_desligado_em_antt_prod() -> None:
    """Perfil antt_prod desliga fallback cloud por padrao."""
    ant_profile = os.environ.get("RAG_DEPLOY_PROFILE")
    ant_fb = os.environ.get("RAG_LLM_CLOUD_FALLBACK")
    try:
        os.environ.pop("RAG_LLM_CLOUD_FALLBACK", None)
        os.environ["RAG_DEPLOY_PROFILE"] = "antt_prod"
        assert cloud_fallback_enabled() is False
        os.environ["RAG_DEPLOY_PROFILE"] = "dev"
        assert cloud_fallback_enabled() is True
        os.environ["RAG_LLM_CLOUD_FALLBACK"] = "false"
        assert cloud_fallback_enabled() is False
    finally:
        if ant_profile is None:
            os.environ.pop("RAG_DEPLOY_PROFILE", None)
        else:
            os.environ["RAG_DEPLOY_PROFILE"] = ant_profile
        if ant_fb is None:
            os.environ.pop("RAG_LLM_CLOUD_FALLBACK", None)
        else:
            os.environ["RAG_LLM_CLOUD_FALLBACK"] = ant_fb


def test_limite_contexto_ollama() -> None:
    """Contexto Ollama usa limite menor que cloud."""
    from antt_rag_unified import (
        _MAX_CONTEXT_CHARS,
        _LIMITE_CONTEXTO_OLLAMA,
        _limite_contexto_chars,
    )

    assert _limite_contexto_chars("deepseek") == _MAX_CONTEXT_CHARS
    assert _limite_contexto_chars("ollama") == _LIMITE_CONTEXTO_OLLAMA
    assert _limite_contexto_chars("llama3.2:3b") == _LIMITE_CONTEXTO_OLLAMA


def main() -> int:
    """Executa todos os testes e reporta falhas."""
    testes = [
        test_provedores_cloud_inalterados,
        test_default_provider_ainda_deepseek,
        test_config_ollama_registrado,
        test_filtro_antt_prod_so_ollama,
        test_cloud_fallback_desligado_em_antt_prod,
        test_limite_contexto_ollama,
    ]
    falhas = 0
    for teste in testes:
        try:
            teste()
            print(f"OK  {teste.__name__}")
        except Exception as exc:  # noqa: BLE001
            falhas += 1
            print(f"FAIL {teste.__name__}: {exc}")
    print(f"\n{len(testes) - falhas}/{len(testes)} passou")
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
