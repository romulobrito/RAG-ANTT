"""Migracao e troca atomica das geracoes do indice."""

from __future__ import annotations

import json
from pathlib import Path

from ingestao.geracoes import (
    ativar_geracao,
    bootstrap_indice_legado,
    criar_diretorio_geracao,
    escrever_catalogo,
    escrever_manifest,
    ler_catalogo_ativo,
    ler_id_atual,
    resolver_geracao_ativa,
)


def _indice_falso(pasta: Path, marcador: str) -> None:
    """Cria os dois arquivos obrigatorios sem abrir FAISS."""
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "index.faiss").write_text(marcador, encoding="ascii")
    (pasta / "index.pkl").write_text(marcador, encoding="ascii")


def test_bootstrap_nao_recalcula_indice(tmp_path: Path) -> None:
    """Arquivos legados sao copiados sem mudar bytes."""
    raiz = tmp_path / "vectorstore_local"
    _indice_falso(raiz, "indice atual")
    relatorio = tmp_path / "relatorio.json"
    relatorio.write_text(
        json.dumps([{"arquivo_md": "dados_antt/a.md", "tipo": "DOC"}]),
        encoding="utf-8",
    )
    geracao = bootstrap_indice_legado(str(raiz), str(relatorio))
    assert (Path(geracao.caminho) / "index.faiss").read_text(
        encoding="ascii"
    ) == "indice atual"
    assert ler_id_atual(str(raiz)) == geracao.id
    assert resolver_geracao_ativa(str(raiz)).id == geracao.id


def test_troca_de_ponteiro_preserva_geracao_anterior(tmp_path: Path) -> None:
    """Commit muda current.json e mantem rollback no disco."""
    raiz = tmp_path / "vectorstore_local"
    primeira = criar_diretorio_geracao(str(raiz))
    _indice_falso(Path(primeira.caminho), "primeiro")
    escrever_catalogo(primeira, [{"arquivo_md": "a.md"}])
    escrever_manifest(primeira, [])
    ativar_geracao(primeira, str(raiz))

    segunda = criar_diretorio_geracao(str(raiz))
    _indice_falso(Path(segunda.caminho), "segundo")
    escrever_catalogo(segunda, [{"arquivo_md": "b.md"}])
    escrever_manifest(segunda, [])
    ativar_geracao(segunda, str(raiz))

    assert ler_id_atual(str(raiz)) == segunda.id
    assert Path(primeira.caminho).is_dir()
    assert ler_catalogo_ativo(str(raiz)) == [{"arquivo_md": "b.md"}]
