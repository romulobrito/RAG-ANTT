"""Persistencia e estados finais dos jobs de ingestao."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ingestao import jobs


def test_job_conclui_com_avisos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker persiste geracao, chunks e avisos sem perder o job_id."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(jobs, "PASTA_JOBS", str(tmp_path / "jobs"))
    monkeypatch.setattr(jobs, "PASTA_STAGING", str(tmp_path / "staging"))
    monkeypatch.setattr(
        jobs,
        "validar_duplicidade_ativa",
        lambda documento: None,
    )
    criado = jobs.submeter_upload(
        b"%PDF-1.4\nconteudo",
        "novo.pdf",
    )
    monkeypatch.setattr(
        jobs,
        "indexar_documento",
        lambda documento, job_id, staging: SimpleNamespace(
            avisos=["Imagem secundaria sem OCR."],
            geracao="g2",
            chunks=4,
            caminho_original="dados_antt/entrada/novo.pdf",
        ),
    )
    jobs._processar(criado.job_id)
    estado = jobs.obter_job(criado.job_id)
    assert estado["status"] == "succeeded_with_warnings"
    assert estado["geracao"] == "g2"
    assert estado["chunks"] == 4


def test_job_falho_preserva_diagnostico(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Erro do indice vira estado failed em vez de sumir com o job."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(jobs, "PASTA_JOBS", str(tmp_path / "jobs"))
    monkeypatch.setattr(jobs, "PASTA_STAGING", str(tmp_path / "staging"))
    monkeypatch.setattr(
        jobs,
        "validar_duplicidade_ativa",
        lambda documento: None,
    )
    criado = jobs.submeter_upload(
        b"%PDF-1.4\nconteudo",
        "falha.pdf",
    )

    def _falhar(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("falha controlada")

    monkeypatch.setattr(jobs, "indexar_documento", _falhar)
    jobs._processar(criado.job_id)
    estado = jobs.obter_job(criado.job_id)
    assert estado["status"] == "failed"
    assert "falha controlada" in str(estado["mensagem"])
