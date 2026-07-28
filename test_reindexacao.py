"""
Testes da substituicao do indice durante a atualizacao da base.

O pipeline anterior apagava o indice em uso antes de construir o novo. Isso
deixava a base sem nenhuma versao consultavel durante toda a reindexacao (mais
de uma hora no acervo atual) e, em caso de falha no meio do processo, exigia
reindexar do zero para voltar a responder perguntas.

Estes testes fixam o comportamento atual: o indice novo e construido em
diretorio separado e a troca so acontece no fim, com restauracao do anterior se
a substituicao falhar.
"""

import os

import pytest

from antt_rag_unified import _substituir_vectorstore


def _criar_indice(diretorio, marcador):
    """Cria um diretorio de indice simulado com um arquivo identificavel."""
    os.makedirs(diretorio, exist_ok=True)
    with open(os.path.join(diretorio, "index.faiss"), "w", encoding="utf-8") as f:
        f.write(marcador)


def _ler_indice(diretorio):
    """Le o marcador do indice simulado."""
    with open(os.path.join(diretorio, "index.faiss"), encoding="utf-8") as f:
        return f.read()


def test_substituicao_coloca_indice_novo_em_producao(tmp_path):
    """Apos a troca, o diretorio em uso deve conter o indice recem-criado."""
    destino = str(tmp_path / "vectorstore_local")
    origem = str(tmp_path / "vectorstore_local.novo")
    _criar_indice(destino, "indice antigo")
    _criar_indice(origem, "indice novo")

    _substituir_vectorstore(origem, destino)

    assert _ler_indice(destino) == "indice novo"
    assert not os.path.exists(origem)


def test_substituicao_remove_backup_apos_sucesso(tmp_path):
    """O indice anterior nao deve permanecer em disco depois da troca."""
    destino = str(tmp_path / "vectorstore_local")
    origem = str(tmp_path / "vectorstore_local.novo")
    _criar_indice(destino, "indice antigo")
    _criar_indice(origem, "indice novo")

    _substituir_vectorstore(origem, destino)

    assert not os.path.exists(f"{destino}.anterior")


def test_substituicao_sem_indice_anterior(tmp_path):
    """Primeira indexacao da base nao tem indice a preservar."""
    destino = str(tmp_path / "vectorstore_local")
    origem = str(tmp_path / "vectorstore_local.novo")
    _criar_indice(origem, "indice novo")

    _substituir_vectorstore(origem, destino)

    assert _ler_indice(destino) == "indice novo"


def test_substituicao_descarta_backup_orfao(tmp_path):
    """
    Backup de uma troca interrompida nao deve impedir a proxima atualizacao.

    Sem essa limpeza, a renomeacao falharia porque o destino do backup ja
    existiria.
    """
    destino = str(tmp_path / "vectorstore_local")
    origem = str(tmp_path / "vectorstore_local.novo")
    _criar_indice(destino, "indice antigo")
    _criar_indice(origem, "indice novo")
    _criar_indice(f"{destino}.anterior", "backup orfao")

    _substituir_vectorstore(origem, destino)

    assert _ler_indice(destino) == "indice novo"
    assert not os.path.exists(f"{destino}.anterior")


def test_substituicao_restaura_indice_anterior_em_falha(tmp_path, monkeypatch):
    """
    Falha na troca nao pode deixar a base sem indice consultavel.

    O cenario simulado e o da renomeacao do indice novo falhando depois de o
    anterior ja ter sido movido para o backup.
    """
    destino = str(tmp_path / "vectorstore_local")
    origem = str(tmp_path / "vectorstore_local.novo")
    _criar_indice(destino, "indice antigo")
    _criar_indice(origem, "indice novo")

    rename_original = os.rename
    chamadas = {"total": 0}

    def rename_que_falha_na_segunda(caminho_origem, caminho_destino):
        """Permite mover o indice antigo, mas falha ao promover o novo."""
        chamadas["total"] += 1
        if chamadas["total"] == 2:
            raise OSError("falha simulada ao promover o indice novo")
        return rename_original(caminho_origem, caminho_destino)

    monkeypatch.setattr(os, "rename", rename_que_falha_na_segunda)

    with pytest.raises(OSError):
        _substituir_vectorstore(origem, destino)

    monkeypatch.setattr(os, "rename", rename_original)

    assert os.path.isdir(destino)
    assert _ler_indice(destino) == "indice antigo"
