"""
Consulta, ready e reindex da API. Ollama e indice ficam mockados.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, List, Optional

import pytest
from fastapi.testclient import TestClient

import rag_service
from api.app import app
from rag_service import (
    DocumentHit,
    QueryResult,
    RagGenerationError,
    RagNotReadyError,
    StatusServico,
)

RAIZ = Path(__file__).resolve().parent.parent

_PEDIDO = {
    "pergunta": (
        "Qual o IRI maximo da pista principal na manutencao "
        "segundo a INM 34/2024?"
    ),
    "filtros": {"tipo_documento": "INM", "ano": 2024, "numero": "34"},
    "correlation_id": "sigesc-ticket-8891",
}


@pytest.fixture
def cliente(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Cliente com key e sem aquecer o indice de verdade."""
    monkeypatch.setenv("RAG_API_KEY", "chave-teste")
    monkeypatch.setattr(rag_service, "aquecer_indice", lambda embedding_provider=None: False)
    with TestClient(app) as cliente_http:
        yield cliente_http


def _cabecalhos() -> dict:
    """Header de servico dos testes."""
    return {"X-API-Key": "chave-teste"}


def _resultado() -> QueryResult:
    """Fontes minimas do exemplo INM 34."""
    return QueryResult(
        resposta="Segundo a INM 34/2024, o IRI maximo e 2,7 m/km.",
        modelo_usado="qwen2.5:7b",
        provider="ollama",
        documentos_consultados=[
            DocumentHit(
                tipo="INM",
                numero="34",
                ano="2024",
                trecho="IRI | Principal | 2,7 m/km",
                caminho="dados_antt/INM/2024/INM-00000034-2024.md",
                relevancia=0.91,
            )
        ],
        total_documentos_encontrados=30,
        embedding_provider="local",
        vectorstore_utilizado="vectorstore_local",
    )


def test_post_inm_34(
    cliente: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O POST ilustrativo devolve request_id, fontes e o correlation_id."""
    monkeypatch.setattr(rag_service, "consultar", lambda *args, **kwargs: _resultado())
    resposta = cliente.post("/api/query", headers=_cabecalhos(), json=_PEDIDO)
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["correlation_id"] == "sigesc-ticket-8891"
    assert corpo["request_id"]
    assert corpo["documentos_consultados"][0]["tipo"] == "INM"
    assert corpo["documentos_consultados"][0]["numero"] == "34"


def test_pergunta_vazia(cliente: TestClient) -> None:
    """Pergunta vazia responde 400."""
    resposta = cliente.post(
        "/api/query",
        headers=_cabecalhos(),
        json={"pergunta": ""},
    )
    assert resposta.status_code == 400
    assert resposta.json()["detail"] == "pedido invalido"


def test_user_id_rejeitado(cliente: TestClient) -> None:
    """Campo extra user_id nao entra no contrato."""
    pedido = dict(_PEDIDO)
    pedido["user_id"] = "fiscal"
    resposta = cliente.post("/api/query", headers=_cabecalhos(), json=pedido)
    assert resposta.status_code == 400


def test_indice_indisponivel(
    cliente: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RagNotReadyError vira 503."""
    def _falha(*args: object, **kwargs: object) -> QueryResult:
        raise RagNotReadyError("indice ausente")

    monkeypatch.setattr(rag_service, "consultar", _falha)
    resposta = cliente.post("/api/query", headers=_cabecalhos(), json=_PEDIDO)
    assert resposta.status_code == 503
    assert "traceback" not in resposta.text.lower()


def test_consulta_longa(
    cliente: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Causa consulta_longa vira 504."""
    def _longa(*args: object, **kwargs: object) -> QueryResult:
        raise RagGenerationError("consulta_longa: estourou o tempo")

    monkeypatch.setattr(rag_service, "consultar", _longa)
    resposta = cliente.post("/api/query", headers=_cabecalhos(), json=_PEDIDO)
    assert resposta.status_code == 504


def test_resposta_vazia_do_provedor(
    cliente: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Falha de conteudo vazio do provedor vira 503."""
    def _vazia(*args: object, **kwargs: object) -> QueryResult:
        raise RagGenerationError("resposta_vazia_do_provedor")

    monkeypatch.setattr(rag_service, "consultar", _vazia)
    resposta = cliente.post("/api/query", headers=_cabecalhos(), json=_PEDIDO)
    assert resposta.status_code == 503
    assert resposta.json()["detail"] == "resposta_vazia_do_provedor"


def test_ready_ollama_fora(
    cliente: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ready 503 quando o Ollama esta fora, sem API Key."""
    monkeypatch.setattr(
        rag_service,
        "obter_status",
        lambda: StatusServico(
            vectorstore_ok=True,
            vectorstore_path="vectorstore_local",
            ollama_ok=False,
            ollama_mensagem="fora",
            n_docs=1,
            provedores_liberados=["ollama"],
            embeddings_liberados=["local"],
            embedding_provider="local",
        ),
    )
    resposta = cliente.get("/api/ready")
    assert resposta.status_code == 503
    assert resposta.json()["ollama"] is False


def test_reindex_aceito(
    cliente: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sem lock, a API responde 202 e nao reindexa na hora do assert do corpo."""
    chamadas: List[Optional[str]] = []

    def _falso(embedding_provider: Optional[str] = None) -> SimpleNamespace:
        chamadas.append(embedding_provider)
        return SimpleNamespace(sucesso=True, mensagem="ok", job_id="interno")

    monkeypatch.setattr(rag_service, "reindexacao_em_andamento", lambda: False)
    monkeypatch.setattr(rag_service, "disparar_reindexacao", _falso)
    resposta = cliente.post("/api/reindex", headers=_cabecalhos())
    assert resposta.status_code == 202
    assert resposta.json()["status"] == "accepted"
    assert resposta.json()["job_id"]
    assert chamadas == ["local"]


def test_reindex_ocupado(
    cliente: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lock ativo responde 409 e nao dispara outro job."""
    monkeypatch.setattr(rag_service, "reindexacao_em_andamento", lambda: True)

    def _proibido(embedding_provider: Optional[str] = None) -> SimpleNamespace:
        raise AssertionError("reindex nao deveria rodar")

    monkeypatch.setattr(rag_service, "disparar_reindexacao", _proibido)
    resposta = cliente.post("/api/reindex", headers=_cabecalhos())
    assert resposta.status_code == 409
    assert resposta.json()["detail"] == "reindex_em_andamento"


def test_import_app_nao_carrega_streamlit() -> None:
    """Carregar a API nao puxa a tela."""
    codigo = (
        "import sys\n"
        "import api.app\n"
        "assert 'streamlit' not in sys.modules, sorted(sys.modules)\n"
    )
    concluido = subprocess.run(
        [sys.executable, "-c", codigo],
        cwd=str(RAIZ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert concluido.returncode == 0, concluido.stderr
