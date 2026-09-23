"""
Auth da API HTTP. Nao chama Ollama nem reindexa a base.
"""

from __future__ import annotations

from typing import Iterator

import pytest
from fastapi.testclient import TestClient

import rag_service
from api.app import app
from rag_service import DocumentHit, QueryResult


@pytest.fixture
def cliente(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Cliente com a key de teste e sem carregar o indice real."""
    monkeypatch.setenv("RAG_API_KEY", "chave-teste")
    monkeypatch.setattr(rag_service, "aquecer_indice", lambda embedding_provider=None: False)
    with TestClient(app) as cliente_http:
        yield cliente_http


def _resultado() -> QueryResult:
    """Resposta falsa, no formato do facade."""
    return QueryResult(
        resposta="resposta de teste",
        modelo_usado="qwen2.5:7b",
        provider="ollama",
        documentos_consultados=[
            DocumentHit(
                tipo="INM",
                numero="34",
                ano="2024",
                trecho="trecho",
                caminho="dados_antt/INM/2024/INM-00000034-2024.md",
                relevancia=0.9,
            )
        ],
        total_documentos_encontrados=1,
        embedding_provider="local",
        vectorstore_utilizado="vectorstore_local",
    )


def test_swagger_declara_cadeado(cliente: TestClient) -> None:
    """OpenAPI declara X-API-Key nas rotas de negocio e nao no health."""
    resposta = cliente.get("/openapi.json")
    assert resposta.status_code == 200
    corpo = resposta.json()
    esquemas = corpo["components"]["securitySchemes"]
    nomes = [item.get("name") for item in esquemas.values()]
    assert "X-API-Key" in nomes
    assert corpo["paths"]["/api/status"]["get"].get("security")
    assert "security" not in corpo["paths"]["/api/health"]["get"]


def test_health_sem_key(cliente: TestClient) -> None:
    """Health responde 200 sem header."""
    resposta = cliente.get("/api/health")
    assert resposta.status_code == 200
    assert resposta.json()["status"] == "ok"


def test_query_sem_key(cliente: TestClient) -> None:
    """Consulta sem header responde 401."""
    resposta = cliente.post("/api/query", json={"pergunta": "IRI?"})
    assert resposta.status_code == 401
    assert resposta.json()["detail"] == "api_key_invalid"


def test_query_key_errada(cliente: TestClient) -> None:
    """Key diferente da env responde 401."""
    resposta = cliente.post(
        "/api/query",
        headers={"X-API-Key": "outra"},
        json={"pergunta": "IRI?"},
    )
    assert resposta.status_code == 401


def test_query_env_vazia(
    cliente: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pod sem Secret responde 503, nao 401."""
    monkeypatch.setenv("RAG_API_KEY", "")
    resposta = cliente.post(
        "/api/query",
        headers={"X-API-Key": "chave-teste"},
        json={"pergunta": "IRI?"},
    )
    assert resposta.status_code == 503
    assert resposta.json()["detail"] == "api_key_not_configured"


def test_query_com_key(
    cliente: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Key certa chega no consultar e devolve 200."""
    monkeypatch.setattr(rag_service, "consultar", lambda *args, **kwargs: _resultado())
    resposta = cliente.post(
        "/api/query",
        headers={"X-API-Key": "chave-teste"},
        json={"pergunta": "IRI?"},
    )
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["resposta"] == "resposta de teste"
    assert corpo["documentos_consultados"][0]["tipo"] == "INM"
