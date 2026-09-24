"""Scanner aceita apenas PDF completo e nao duplica o manifest."""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestao.geracoes import (
    ativar_geracao,
    criar_diretorio_geracao,
    escrever_catalogo,
    escrever_manifest,
)
from ingestao import scanner


def _geracao_vazia(tmp_path: Path) -> None:
    """Cria geracao valida para o scanner consultar."""
    raiz = tmp_path / "vectorstore_local"
    geracao = criar_diretorio_geracao(str(raiz))
    (Path(geracao.caminho) / "index.faiss").write_bytes(b"faiss")
    (Path(geracao.caminho) / "index.pkl").write_bytes(b"pickle")
    escrever_catalogo(geracao, [])
    escrever_manifest(geracao, [])
    ativar_geracao(geracao, str(raiz))


def test_pdf_precisa_ficar_estavel_duas_varreduras(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Primeiro ciclo observa e o segundo reclama por rename."""
    monkeypatch.chdir(tmp_path)
    _geracao_vazia(tmp_path)
    inbox = tmp_path / "dados_antt" / "entrada"
    staging = tmp_path / "dados_antt" / ".staging"
    inbox.mkdir(parents=True)
    monkeypatch.setattr(scanner, "PASTA_INBOX", str(inbox))
    monkeypatch.setattr(scanner, "PASTA_STAGING", str(staging))
    scanner._ESTABILIDADE.clear()
    chamados = []
    monkeypatch.setattr(
        scanner,
        "submeter_arquivo_reclamado",
        lambda caminho, nome: chamados.append((caminho, nome)),
    )
    (inbox / "novo.pdf").write_bytes(b"%PDF-1.4\nconteudo")

    assert scanner.varrer_uma_vez() == 0
    assert scanner.varrer_uma_vez() == 1
    assert chamados[0][1] == "novo.pdf"
    assert not (inbox / "novo.pdf").exists()


def test_part_docx_e_xlsx_sao_ignorados(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caixa compartilhada e exclusiva para PDF final."""
    monkeypatch.chdir(tmp_path)
    _geracao_vazia(tmp_path)
    inbox = tmp_path / "dados_antt" / "entrada"
    staging = tmp_path / "dados_antt" / ".staging"
    inbox.mkdir(parents=True)
    monkeypatch.setattr(scanner, "PASTA_INBOX", str(inbox))
    monkeypatch.setattr(scanner, "PASTA_STAGING", str(staging))
    scanner._ESTABILIDADE.clear()
    (inbox / "a.pdf.part").write_bytes(b"%PDF")
    (inbox / "b.docx").write_bytes(b"zip")
    (inbox / "c.xlsx").write_bytes(b"zip")
    assert scanner.varrer_uma_vez() == 0
    assert len(list(inbox.iterdir())) == 3
