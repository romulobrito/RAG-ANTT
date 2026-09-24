"""Contrato HTTP do upload incremental e polling."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

import config
import ingestao.jobs
import rag_service
from api.app import app


@pytest.fixture
def cliente(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Sobe app sem ativar worker real no lifespan."""
    monkeypatch.setenv("RAG_API_KEY", "teste")
    monkeypatch.setenv("RAG_INCREMENTAL_UPLOAD_ENABLED", "false")
    monkeypatch.setattr(
        rag_service,
        "aquecer_indice",
        lambda embedding_provider=None: False,
    )
    with TestClient(app) as cliente_http:
        yield cliente_http


def _headers() -> dict:
    """Cabecalho de servico."""
    return {"X-API-Key": "teste"}


def test_upload_incremental_retorna_202(
    cliente: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upload habilitado devolve job sem bloquear conversao."""
    monkeypatch.setattr(config, "incremental_upload_enabled", lambda: True)
    monkeypatch.setattr(
        ingestao.jobs,
        "submeter_upload",
        lambda conteudo, nome: SimpleNamespace(
            job_id="abc-123",
            nome=nome,
            formato="pdf",
        ),
    )
    resposta = cliente.post(
        "/api/documents",
        headers=_headers(),
        files={"arquivo": ("novo.pdf", b"%PDF-1.4\nconteudo")},
    )
    assert resposta.status_code == 202
    assert resposta.json() == {
        "job_id": "abc-123",
        "status": "queued",
        "nome": "novo.pdf",
        "formato": "pdf",
    }


def test_polling_job(
    cliente: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Job concluido devolve geracao, chunks e avisos."""
    monkeypatch.setattr(
        ingestao.jobs,
        "obter_job",
        lambda job_id: {
            "job_id": job_id,
            "status": "succeeded",
            "nome": "novo.pdf",
            "formato": "pdf",
            "origem": "api",
            "mensagem": "Documento disponivel para consulta.",
            "avisos": [],
            "geracao": "g2",
            "chunks": 3,
            "caminho": "dados_antt/entrada/novo.pdf",
            "created_at": "2026-09-24T10:00:00Z",
            "updated_at": "2026-09-24T10:01:00Z",
        },
    )
    resposta = cliente.get("/api/jobs/abc-123", headers=_headers())
    assert resposta.status_code == 200
    assert resposta.json()["geracao"] == "g2"
    assert resposta.json()["chunks"] == 3
