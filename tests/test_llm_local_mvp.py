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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def test_historico_conversa_padrao() -> None:
    """Sem env, a continuidade permanece a definida em config.py."""
    from config import (
        get_historico_chars_resposta,
        get_historico_turnos_guardados,
        get_historico_turnos_reescrita,
    )

    nomes = (
        "RAG_HISTORICO_TURNOS",
        "RAG_HISTORICO_REESCRITA",
        "RAG_HISTORICO_CHARS_RESPOSTA",
    )
    anteriores = {nome: os.environ.get(nome) for nome in nomes}
    try:
        for nome in nomes:
            os.environ.pop(nome, None)
        assert get_historico_turnos_guardados() == 10
        assert get_historico_turnos_reescrita() == 8
        assert get_historico_chars_resposta() == 1000
        os.environ["RAG_HISTORICO_REESCRITA"] = "4"
        assert get_historico_turnos_reescrita() == 4
    finally:
        for nome, anterior in anteriores.items():
            if anterior is None:
                os.environ.pop(nome, None)
            else:
                os.environ[nome] = anterior


def test_vagas_geracao_local_padrao_e_teto() -> None:
    """Padrao e uma vaga; valor invalido cai para 1; teto interno e 8."""
    from config import get_vagas_geracao_local

    anterior = os.environ.get("RAG_VAGAS_GERACAO_LOCAL")
    try:
        os.environ.pop("RAG_VAGAS_GERACAO_LOCAL", None)
        assert get_vagas_geracao_local() == 1
        os.environ["RAG_VAGAS_GERACAO_LOCAL"] = "0"
        assert get_vagas_geracao_local() == 1
        os.environ["RAG_VAGAS_GERACAO_LOCAL"] = "abc"
        assert get_vagas_geracao_local() == 1
        os.environ["RAG_VAGAS_GERACAO_LOCAL"] = "3"
        assert get_vagas_geracao_local() == 3
        os.environ["RAG_VAGAS_GERACAO_LOCAL"] = "40"
        assert get_vagas_geracao_local() == 8
    finally:
        if anterior is None:
            os.environ.pop("RAG_VAGAS_GERACAO_LOCAL", None)
        else:
            os.environ["RAG_VAGAS_GERACAO_LOCAL"] = anterior


def test_ollama_uma_vaga_por_vez() -> None:
    """Com limite 1, a segunda geracao local espera a primeira terminar."""
    import threading
    import time

    from llm_providers import vaga_geracao

    anterior = os.environ.get("RAG_VAGAS_GERACAO_LOCAL")
    os.environ["RAG_VAGAS_GERACAO_LOCAL"] = "1"
    ordem: list = []
    try:
        def _trabalho(nome: str) -> None:
            with vaga_geracao("ollama"):
                ordem.append(nome + "-ini")
                time.sleep(0.15)
                ordem.append(nome + "-fim")

        primeira = threading.Thread(target=_trabalho, args=("a",))
        segunda = threading.Thread(target=_trabalho, args=("b",))
        primeira.start()
        time.sleep(0.05)
        segunda.start()
        primeira.join(timeout=2)
        segunda.join(timeout=2)
        assert ordem == ["a-ini", "a-fim", "b-ini", "b-fim"]
    finally:
        if anterior is None:
            os.environ.pop("RAG_VAGAS_GERACAO_LOCAL", None)
        else:
            os.environ["RAG_VAGAS_GERACAO_LOCAL"] = anterior


def test_provedor_externo_nao_espera_vaga() -> None:
    """OpenAI/DeepSeek nao entram na fila local."""
    import threading

    from llm_providers import vaga_geracao

    barreira = threading.Barrier(2)

    def _trabalho() -> None:
        with vaga_geracao("deepseek"):
            barreira.wait(timeout=2)

    primeira = threading.Thread(target=_trabalho)
    segunda = threading.Thread(target=_trabalho)
    primeira.start()
    segunda.start()
    primeira.join(timeout=3)
    segunda.join(timeout=3)
    assert not primeira.is_alive()
    assert not segunda.is_alive()


def test_vaga_aninhada_no_mesmo_thread() -> None:
    """Dois passos no mesmo thread nao pedem uma segunda vaga."""
    from llm_providers import vaga_geracao

    anterior = os.environ.get("RAG_VAGAS_GERACAO_LOCAL")
    os.environ["RAG_VAGAS_GERACAO_LOCAL"] = "1"
    try:
        with vaga_geracao("qwen2.5:7b"):
            with vaga_geracao("ollama"):
                assert True
    finally:
        if anterior is None:
            os.environ.pop("RAG_VAGAS_GERACAO_LOCAL", None)
        else:
            os.environ["RAG_VAGAS_GERACAO_LOCAL"] = anterior


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
        test_historico_conversa_padrao,
        test_vagas_geracao_local_padrao_e_teto,
        test_ollama_uma_vaga_por_vez,
        test_provedor_externo_nao_espera_vaga,
        test_vaga_aninhada_no_mesmo_thread,
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
