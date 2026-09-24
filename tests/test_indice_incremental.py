"""Inclusao incremental preserva documentos antigos e faz commit por geracao."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import antt_rag_unified
from ingestao import indice_incremental
from ingestao.conversores import ResultadoConversao
from ingestao.formatos import validar_documento
from ingestao.geracoes import ler_catalogo_ativo, resolver_geracao_ativa


class _FaissFalso:
    """FAISS em memoria com persistencia minima para testar orquestracao."""

    def __init__(self, total: int = 1) -> None:
        self.total = total
        self.index_to_docstore_id = {
            indice: str(indice) for indice in range(total)
        }

    @classmethod
    def load_local(
        cls,
        caminho: str,
        embeddings: object,
        allow_dangerous_deserialization: bool,
    ) -> "_FaissFalso":
        """Abre contador salvo junto do indice falso."""
        del embeddings, allow_dangerous_deserialization
        contador = Path(caminho) / "count.txt"
        total = int(contador.read_text()) if contador.exists() else 1
        return cls(total)

    def add_documents(self, documentos: list) -> None:
        """Acrescenta somente os chunks recebidos."""
        self.total += len(documentos)
        self.index_to_docstore_id = {
            indice: str(indice) for indice in range(self.total)
        }

    def save_local(self, caminho: str) -> None:
        """Cria artefatos esperados da geracao."""
        pasta = Path(caminho)
        pasta.mkdir(parents=True, exist_ok=True)
        (pasta / "index.faiss").write_bytes(b"faiss")
        (pasta / "index.pkl").write_bytes(b"pickle")
        (pasta / "count.txt").write_text(str(self.total), encoding="ascii")


def test_incremental_preserva_antigo_e_adiciona_novo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Geracao nova contem contador antigo mais chunks novos."""
    monkeypatch.chdir(tmp_path)
    raiz = tmp_path / "vectorstore_local"
    raiz.mkdir()
    (raiz / "index.faiss").write_bytes(b"faiss")
    (raiz / "index.pkl").write_bytes(b"pickle")
    (raiz / "count.txt").write_text("1", encoding="ascii")
    (tmp_path / "relatorio_documentos.json").write_text("[]", encoding="utf-8")

    monkeypatch.setattr(
        indice_incremental,
        "converter_documento",
        lambda documento, staging: ResultadoConversao(
            markdown="# INM 99/2024\n\nTabela\n\n| Item | Valor |\n| --- | --- |\n| IRI | 2,7 |",
        ),
    )
    monkeypatch.setattr(
        antt_rag_unified,
        "_criar_embeddings_local",
        lambda: object(),
    )
    monkeypatch.setattr(
        antt_rag_unified,
        "criar_chunks_documento",
        lambda meta: [SimpleNamespace(metadata=meta), SimpleNamespace(metadata=meta)],
    )
    import langchain_community.vectorstores

    monkeypatch.setattr(
        langchain_community.vectorstores,
        "FAISS",
        _FaissFalso,
    )
    documento = validar_documento(
        b"%PDF-1.4\nconteudo",
        "INM-00000099-2024.pdf",
    )
    resultado = indice_incremental.indexar_documento(
        documento,
        "job-1",
        str(tmp_path / "staging"),
        raiz_indice=str(raiz),
    )
    ativa = resolver_geracao_ativa(str(raiz))
    assert ativa.id == resultado.geracao
    assert (Path(ativa.caminho) / "count.txt").read_text() == "3"
    catalogo = ler_catalogo_ativo(str(raiz))
    assert catalogo is not None
    assert catalogo[-1]["numero"] == "99"
