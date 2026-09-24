"""Geracoes imutaveis do indice e ponteiro atomico de ativacao."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

from config import (
    DEFAULT_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    get_embedding_provider,
)


RAIZ_INDICE = "vectorstore_local"
PASTA_GERACOES = "generations"
ARQUIVO_ATUAL = "current.json"
ARQUIVO_MANIFEST = "manifest.json"
ARQUIVO_CATALOGO = "catalog.json"


class GeracaoIndiceError(RuntimeError):
    """Falha ao resolver, criar ou ativar uma geracao."""


@dataclass(frozen=True)
class GeracaoAtiva:
    """Identifica a geracao consultavel."""

    id: str
    caminho: str


def _escrever_json_atomico(caminho: str, conteudo: object) -> None:
    """Escreve JSON no mesmo filesystem e troca por rename atomico."""
    diretorio = os.path.dirname(caminho) or "."
    os.makedirs(diretorio, exist_ok=True)
    descritor, temporario = tempfile.mkstemp(
        prefix=".tmp-",
        suffix=".json",
        dir=diretorio,
    )
    try:
        with os.fdopen(descritor, "w", encoding="utf-8") as arquivo:
            json.dump(conteudo, arquivo, ensure_ascii=True, indent=2)
            arquivo.flush()
            os.fsync(arquivo.fileno())
        os.replace(temporario, caminho)
        try:
            fd_diretorio = os.open(diretorio, os.O_DIRECTORY)
            try:
                os.fsync(fd_diretorio)
            finally:
                os.close(fd_diretorio)
        except OSError:
            pass
    finally:
        if os.path.exists(temporario):
            os.remove(temporario)


def _ler_json(caminho: str) -> object:
    """Le JSON ou propaga erro com caminho identificavel."""
    try:
        with open(caminho, "r", encoding="utf-8") as arquivo:
            return json.load(arquivo)
    except (OSError, ValueError) as exc:
        raise GeracaoIndiceError(
            "JSON de geracao ilegivel em {0}: {1}".format(caminho, exc)
        ) from exc


def _hash_arquivo(caminho: str) -> str:
    """Calcula SHA-256 em fluxo."""
    digest = hashlib.sha256()
    with open(caminho, "rb") as arquivo:
        while True:
            bloco = arquivo.read(1024 * 1024)
            if not bloco:
                break
            digest.update(bloco)
    return digest.hexdigest()


def _documentos_do_catalogo(catalogo: object) -> List[Dict[str, object]]:
    """Transforma entradas antigas em registros de manifest."""
    if not isinstance(catalogo, list):
        return []
    documentos: List[Dict[str, object]] = []
    for entrada in catalogo:
        if not isinstance(entrada, dict):
            continue
        caminho = str(entrada.get("arquivo_md") or entrada.get("caminho") or "")
        sha256 = ""
        if caminho and os.path.isfile(caminho):
            try:
                sha256 = _hash_arquivo(caminho)
            except OSError:
                sha256 = ""
        documentos.append(
            {
                "nome": os.path.basename(caminho),
                "caminho": caminho,
                "sha256": sha256,
                "chunks": 0,
            }
        )
    return documentos


def caminho_geracoes(raiz: str = RAIZ_INDICE) -> str:
    """Retorna diretorio que contem as geracoes."""
    return os.path.join(raiz, PASTA_GERACOES)


def caminho_current(raiz: str = RAIZ_INDICE) -> str:
    """Retorna caminho do ponteiro atomico."""
    return os.path.join(raiz, ARQUIVO_ATUAL)


def ler_id_atual(raiz: str = RAIZ_INDICE) -> Optional[str]:
    """Retorna id ativo, ou None para um indice legado."""
    ponteiro = caminho_current(raiz)
    if not os.path.isfile(ponteiro):
        return None
    bruto = _ler_json(ponteiro)
    if not isinstance(bruto, dict):
        raise GeracaoIndiceError("current.json precisa ser um objeto.")
    geracao = bruto.get("generation")
    if not isinstance(geracao, str) or not geracao:
        raise GeracaoIndiceError("current.json nao informa generation.")
    return geracao


def resolver_geracao_ativa(
    raiz: str = RAIZ_INDICE,
    permitir_legado: bool = True,
) -> GeracaoAtiva:
    """Resolve a pasta carregada pelo FAISS."""
    geracao = ler_id_atual(raiz)
    if geracao is None:
        if permitir_legado and all(
            os.path.isfile(os.path.join(raiz, nome))
            for nome in ("index.faiss", "index.pkl")
        ):
            return GeracaoAtiva(id="legacy", caminho=raiz)
        raise GeracaoIndiceError("Nenhuma geracao de indice esta ativa.")
    caminho = os.path.join(caminho_geracoes(raiz), geracao)
    if not all(
        os.path.isfile(os.path.join(caminho, nome))
        for nome in ("index.faiss", "index.pkl", ARQUIVO_MANIFEST, ARQUIVO_CATALOGO)
    ):
        raise GeracaoIndiceError(
            "Geracao ativa incompleta: {0}.".format(geracao)
        )
    return GeracaoAtiva(id=geracao, caminho=caminho)


def _novo_id() -> str:
    """Gera id ordenavel e sem caracteres especiais."""
    return "{0}-{1}".format(
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        uuid.uuid4().hex[:12],
    )


def criar_diretorio_geracao(raiz: str = RAIZ_INDICE) -> GeracaoAtiva:
    """Reserva um diretorio imutavel ainda nao ativo."""
    geracao = _novo_id()
    caminho = os.path.join(caminho_geracoes(raiz), geracao)
    os.makedirs(caminho, exist_ok=False)
    return GeracaoAtiva(id=geracao, caminho=caminho)


def escrever_manifest(
    geracao: GeracaoAtiva,
    documentos: List[Mapping[str, object]],
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> None:
    """Grava manifest completo da geracao candidata."""
    provedor = embedding_provider or get_embedding_provider()
    modelo = embedding_model or (
        LOCAL_EMBEDDING_MODEL
        if provedor == "local"
        else DEFAULT_EMBEDDING_MODEL
    )
    _escrever_json_atomico(
        os.path.join(geracao.caminho, ARQUIVO_MANIFEST),
        {
            "schema_version": 1,
            "generation": geracao.id,
            "created_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
            "embedding_provider": provedor,
            "embedding_model": modelo,
            "documents": list(documentos),
        },
    )


def escrever_catalogo(geracao: GeracaoAtiva, catalogo: object) -> None:
    """Grava catalogo pertencente a geracao."""
    _escrever_json_atomico(
        os.path.join(geracao.caminho, ARQUIVO_CATALOGO),
        catalogo,
    )


def ativar_geracao(geracao: GeracaoAtiva, raiz: str = RAIZ_INDICE) -> None:
    """Faz o commit atomico da geracao validada."""
    esperado = os.path.join(caminho_geracoes(raiz), geracao.id)
    if os.path.abspath(esperado) != os.path.abspath(geracao.caminho):
        raise GeracaoIndiceError("Geracao candidata fora da raiz.")
    _escrever_json_atomico(
        caminho_current(raiz),
        {
            "generation": geracao.id,
            "activated_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
        },
    )


def ler_manifest(geracao: GeracaoAtiva) -> Dict[str, object]:
    """Le e valida minimamente o manifest."""
    bruto = _ler_json(os.path.join(geracao.caminho, ARQUIVO_MANIFEST))
    if not isinstance(bruto, dict):
        raise GeracaoIndiceError("Manifest precisa ser um objeto.")
    return bruto


def ler_catalogo_ativo(raiz: str = RAIZ_INDICE) -> Optional[List[object]]:
    """Le catalogo da geracao ativa ou retorna None no legado."""
    geracao = resolver_geracao_ativa(raiz)
    if geracao.id == "legacy":
        return None
    bruto = _ler_json(os.path.join(geracao.caminho, ARQUIVO_CATALOGO))
    if not isinstance(bruto, list):
        raise GeracaoIndiceError("Catalogo da geracao precisa ser lista.")
    return bruto


def bootstrap_indice_legado(
    raiz: str = RAIZ_INDICE,
    relatorio: Optional[str] = None,
) -> GeracaoAtiva:
    """Migra arquivos legados para uma geracao sem recalcular embeddings."""
    caminho_relatorio = relatorio or os.environ.get(
        "RAG_CATALOG_PATH",
        "relatorio_documentos.json",
    )
    existente = ler_id_atual(raiz)
    if existente is not None:
        return resolver_geracao_ativa(raiz, permitir_legado=False)
    for nome in ("index.faiss", "index.pkl"):
        if not os.path.isfile(os.path.join(raiz, nome)):
            raise GeracaoIndiceError(
                "Indice legado incompleto; faltando {0}.".format(nome)
            )
    catalogo: object = []
    if os.path.isfile(caminho_relatorio):
        catalogo = _ler_json(caminho_relatorio)
    if not isinstance(catalogo, list):
        raise GeracaoIndiceError("Relatorio legado precisa ser lista.")

    candidata = criar_diretorio_geracao(raiz)
    try:
        for nome in ("index.faiss", "index.pkl"):
            shutil.copy2(
                os.path.join(raiz, nome),
                os.path.join(candidata.caminho, nome),
            )
        escrever_catalogo(candidata, catalogo)
        escrever_manifest(candidata, _documentos_do_catalogo(catalogo))
        ativar_geracao(candidata, raiz)
    except Exception:
        shutil.rmtree(candidata.caminho, ignore_errors=True)
        raise
    return candidata


def espelhar_catalogo(
    catalogo: object,
    destino: Optional[str] = None,
) -> None:
    """Mantem o relatorio antigo para ferramentas externas."""
    caminho_destino = destino or os.environ.get(
        "RAG_CATALOG_PATH",
        "relatorio_documentos.json",
    )
    try:
        _escrever_json_atomico(caminho_destino, catalogo)
    except OSError:
        # Um bind mount de arquivo permite sobrescrever o conteudo, mas nao
        # permite trocar o inode com os.replace. O catalogo ativo continua
        # sendo o arquivo imutavel dentro da geracao.
        with open(caminho_destino, "w", encoding="utf-8") as arquivo:
            json.dump(catalogo, arquivo, ensure_ascii=True, indent=2)
            arquivo.flush()
            os.fsync(arquivo.fileno())


def remover_geracoes_antigas(
    raiz: str = RAIZ_INDICE,
    manter: int = 2,
) -> None:
    """Remove geracoes inativas antigas, preservando rollback recente."""
    atual = ler_id_atual(raiz)
    pasta = caminho_geracoes(raiz)
    if not os.path.isdir(pasta):
        return
    entradas = [
        nome
        for nome in sorted(os.listdir(pasta), reverse=True)
        if os.path.isdir(os.path.join(pasta, nome))
    ]
    preservadas = set(entradas[: max(2, manter)])
    if atual:
        preservadas.add(atual)
    for nome in entradas:
        if nome not in preservadas:
            shutil.rmtree(os.path.join(pasta, nome), ignore_errors=True)
