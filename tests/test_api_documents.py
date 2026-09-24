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


def test_upload_duplicado_retorna_409(
    cliente: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nome ou hash repetido e conflito explicito."""
    from ingestao.indice_incremental import DocumentoDuplicadoError

    def _duplicado(conteudo: bytes, nome: str) -> object:
        del conteudo
        del nome
        raise DocumentoDuplicadoError("nome_arquivo_duplicado")

    monkeypatch.setattr(config, "incremental_upload_enabled", lambda: True)
    monkeypatch.setattr(ingestao.jobs, "submeter_upload", _duplicado)
    resposta = cliente.post(
        "/api/documents",
        headers=_headers(),
        files={"arquivo": ("repetido.pdf", b"%PDF-1.4\nconteudo")},
    )
    assert resposta.status_code == 409
    assert resposta.json()["detail"] == "nome_arquivo_duplicado"


def test_upload_grande_retorna_413(
    cliente: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Falha de tamanho usa o status HTTP apropriado."""
    from ingestao.formatos import DocumentoInvalidoError

    def _grande(conteudo: bytes, nome: str) -> object:
        del conteudo
        del nome
        raise DocumentoInvalidoError("tamanho excedido")

    monkeypatch.setattr(config, "incremental_upload_enabled", lambda: True)
    monkeypatch.setattr(ingestao.jobs, "submeter_upload", _grande)
    resposta = cliente.post(
        "/api/documents",
        headers=_headers(),
        files={"arquivo": ("grande.pdf", b"%PDF-1.4\nconteudo")},
    )
    assert resposta.status_code == 413
    assert resposta.json()["detail"] == "tamanho excedido"


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


def test_polling_job_inexistente_retorna_404(
    cliente: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Job desconhecido usa o envelope de erro comum."""
    from ingestao.jobs import JobNaoEncontradoError

    def _inexistente(job_id: str) -> dict:
        del job_id
        raise JobNaoEncontradoError("ausente")

    monkeypatch.setattr(ingestao.jobs, "obter_job", _inexistente)
    resposta = cliente.get("/api/jobs/ausente", headers=_headers())
    assert resposta.status_code == 404
    assert resposta.json()["detail"] == "job_nao_encontrado"


def test_openapi_documenta_status_reais(cliente: TestClient) -> None:
    """Swagger mostra os status efetivamente devolvidos pela API."""
    esquema = cliente.get("/openapi.json").json()
    query = esquema["paths"]["/api/query"]["post"]["responses"]
    upload = esquema["paths"]["/api/documents"]["post"]["responses"]
    job = esquema["paths"]["/api/jobs/{job_id}"]["get"]["responses"]

    assert {"200", "400", "503", "504"}.issubset(query)
    assert {"201", "202", "400", "409", "413"}.issubset(upload)
    assert {"200", "404"}.issubset(job)
