"""Transacao incremental do indice FAISS compartilhado."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

from config import LOCAL_EMBEDDING_MODEL, get_embedding_provider
from ingestao.conversores import ResultadoConversao, converter_documento
from ingestao.formatos import DocumentoValidado
from ingestao.geracoes import (
    ARQUIVO_CATALOGO,
    GeracaoAtiva,
    GeracaoIndiceError,
    ativar_geracao,
    bootstrap_indice_legado,
    criar_diretorio_geracao,
    espelhar_catalogo,
    escrever_catalogo,
    escrever_manifest,
    ler_catalogo_ativo,
    ler_manifest,
    remover_geracoes_antigas,
    resolver_geracao_ativa,
)


PASTA_ENTRADA = os.path.join("dados_antt", "entrada")
PASTA_LOCKS = os.path.join("dados_antt", ".locks")
PASTA_TRANSACOES = os.path.join("dados_antt", ".transactions")
LOCK_INDICE = os.path.join(PASTA_LOCKS, "indexacao.lock")


class IndiceOcupadoError(RuntimeError):
    """Outro escritor possui o lock compartilhado."""


class DocumentoDuplicadoError(RuntimeError):
    """Nome ou hash ja faz parte da geracao ativa."""


class EmbeddingIncompativelError(RuntimeError):
    """O indice usa embedding diferente do ambiente ativo."""


@dataclass(frozen=True)
class ResultadoIncremental:
    """Resultado do commit de uma nova geracao."""

    geracao: str
    caminho_original: str
    caminho_markdown: str
    chunks: int
    avisos: List[str]


class LockIndice:
    """Lock de arquivo atomico no volume compartilhado."""

    def __init__(self, caminho: str = LOCK_INDICE, timeout: int = 3600) -> None:
        self.caminho = caminho
        self.timeout = timeout
        self.adquirido = False

    def __enter__(self) -> "LockIndice":
        os.makedirs(os.path.dirname(self.caminho), exist_ok=True)
        if os.path.exists(self.caminho):
            try:
                idade = time.time() - os.path.getmtime(self.caminho)
                if idade > self.timeout:
                    os.remove(self.caminho)
            except OSError:
                pass
        try:
            descritor = os.open(
                self.caminho,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as exc:
            raise IndiceOcupadoError("indexacao_em_andamento") from exc
        with os.fdopen(descritor, "w", encoding="ascii") as arquivo:
            arquivo.write(str(os.getpid()))
        self.adquirido = True
        return self

    def __exit__(self, tipo: object, valor: object, traceback: object) -> None:
        del tipo, valor, traceback
        if self.adquirido:
            try:
                os.remove(self.caminho)
            except OSError:
                pass
        self.adquirido = False


def _escrever_journal(job_id: str, etapa: str, dados: Mapping[str, object]) -> None:
    """Persiste etapa de recuperacao por replace atomico."""
    os.makedirs(PASTA_TRANSACOES, exist_ok=True)
    destino = os.path.join(PASTA_TRANSACOES, "{0}.json".format(job_id))
    descritor, temporario = tempfile.mkstemp(
        prefix=".tmp-",
        suffix=".json",
        dir=PASTA_TRANSACOES,
    )
    try:
        with os.fdopen(descritor, "w", encoding="utf-8") as arquivo:
            json.dump(
                {"job_id": job_id, "stage": etapa, **dict(dados)},
                arquivo,
                ensure_ascii=True,
                indent=2,
            )
            arquivo.flush()
            os.fsync(arquivo.fileno())
        os.replace(temporario, destino)
    finally:
        if os.path.exists(temporario):
            os.remove(temporario)


def _catalogar(caminho_md: str, documento: DocumentoValidado, resultado: ResultadoConversao) -> Dict[str, object]:
    """Gera a entrada de catalogo compativel com o relatorio atual."""
    from scripts.gerar_relatorio import _extrair_metadados_do_nome

    meta = _extrair_metadados_do_nome(
        os.path.basename(caminho_md),
        caminho_md,
        sugerir_sigla=lambda trecho, siglas: "OUTROS",
    )
    meta["arquivo_md"] = caminho_md
    meta["formato_origem"] = documento.formato.value
    meta["metodo_extracao"] = "nativo+ocr"
    meta["avisos_extracao"] = list(resultado.avisos)
    meta["origem_tabela"] = (
        "nativa_ou_ocr" if resultado.tabelas else "sem_tabela"
    )
    return meta


def _validar_duplicidade(
    manifest: Mapping[str, object],
    documento: DocumentoValidado,
) -> None:
    """Rejeita mesmo nome ou hash sob o lock."""
    bruto = manifest.get("documents", [])
    if not isinstance(bruto, list):
        return
    for item in bruto:
        if not isinstance(item, dict):
            continue
        nome = str(item.get("source_name") or item.get("nome") or "")
        sha256 = str(item.get("source_sha256") or item.get("sha256") or "")
        nome_base = os.path.splitext(nome)[0].casefold()
        documento_base = os.path.splitext(documento.nome)[0].casefold()
        if nome and (
            nome.casefold() == documento.nome.casefold()
            or nome_base == documento_base
        ):
            raise DocumentoDuplicadoError("nome_arquivo_duplicado")
        if sha256 and sha256 == documento.sha256:
            raise DocumentoDuplicadoError("conteudo_duplicado")


def validar_duplicidade_ativa(documento: DocumentoValidado) -> None:
    """Faz pre-check rapido; a transacao repete a verificacao sob lock."""
    ativa = resolver_geracao_ativa()
    manifest = ler_manifest(ativa)
    _validar_duplicidade(manifest, documento)
    nome_base = os.path.splitext(documento.nome)[0]
    if (
        os.path.exists(os.path.join(PASTA_ENTRADA, documento.nome))
        or os.path.exists(os.path.join(PASTA_ENTRADA, nome_base + ".md"))
    ):
        raise DocumentoDuplicadoError("nome_arquivo_duplicado")


def _preparar_embeddings(manifest: Mapping[str, object]) -> object:
    """Confere o modelo e cria embeddings locais sem chave de chat."""
    provedor = get_embedding_provider()
    provedor_indice = str(manifest.get("embedding_provider") or "local")
    modelo_indice = str(
        manifest.get("embedding_model") or LOCAL_EMBEDDING_MODEL
    )
    if provedor != provedor_indice:
        raise EmbeddingIncompativelError(
            "embedding_provider_divergente_requer_rebuild"
        )
    if provedor == "local" and modelo_indice != LOCAL_EMBEDDING_MODEL:
        raise EmbeddingIncompativelError(
            "embedding_model_divergente_requer_rebuild"
        )
    if provedor != "local":
        raise EmbeddingIncompativelError(
            "incremental_externo_ainda_nao_homologado"
        )
    from antt_rag_unified import _criar_embeddings_local

    embeddings = _criar_embeddings_local()
    if embeddings is None:
        raise GeracaoIndiceError("Falha ao carregar embeddings locais.")
    return embeddings


def indexar_documento(
    documento: DocumentoValidado,
    job_id: str,
    staging_dir: str,
    raiz_indice: str = "vectorstore_local",
) -> ResultadoIncremental:
    """Converte e adiciona um documento em uma nova geracao atomica."""
    from langchain_community.vectorstores import FAISS
    from antt_rag_unified import criar_chunks_documento

    os.makedirs(staging_dir, exist_ok=True)
    conversao = converter_documento(documento, staging_dir)
    nome_base = os.path.splitext(documento.nome)[0]
    caminho_original = os.path.join(PASTA_ENTRADA, documento.nome)
    caminho_md = os.path.join(PASTA_ENTRADA, nome_base + ".md")
    markdown_staging = os.path.join(staging_dir, nome_base + ".md")
    with open(markdown_staging, "w", encoding="utf-8") as arquivo:
        arquivo.write(conversao.markdown)

    candidata: Optional[GeracaoAtiva] = None
    promovidos: List[str] = []
    geracao_anterior = ""
    with LockIndice():
        try:
            ativa = bootstrap_indice_legado(raiz_indice)
            geracao_anterior = ativa.id
            manifest = ler_manifest(ativa)
            _validar_duplicidade(manifest, documento)
            if os.path.exists(caminho_original) or os.path.exists(caminho_md):
                raise DocumentoDuplicadoError("nome_arquivo_duplicado")
            embeddings = _preparar_embeddings(manifest)

            os.makedirs(PASTA_ENTRADA, exist_ok=True)
            with open(caminho_original, "wb") as arquivo:
                arquivo.write(documento.conteudo)
            promovidos.append(caminho_original)
            shutil.copy2(markdown_staging, caminho_md)
            promovidos.append(caminho_md)
            _escrever_journal(
                job_id,
                "files_promoted",
                {
                    "previous_generation": geracao_anterior,
                    "files": promovidos,
                },
            )

            entrada_catalogo = _catalogar(caminho_md, documento, conversao)
            chunks = criar_chunks_documento(entrada_catalogo)
            if not chunks:
                raise GeracaoIndiceError("Documento nao gerou chunks.")

            catalogo = ler_catalogo_ativo(raiz_indice) or []
            novo_catalogo = list(catalogo) + [entrada_catalogo]
            documentos_manifest = manifest.get("documents", [])
            if not isinstance(documentos_manifest, list):
                documentos_manifest = []
            novo_manifest = list(documentos_manifest) + [
                {
                    "nome": os.path.basename(caminho_md),
                    "caminho": caminho_md,
                    "sha256": documento.sha256,
                    "source_name": documento.nome,
                    "source_sha256": documento.sha256,
                    "chunks": len(chunks),
                }
            ]

            candidata = criar_diretorio_geracao(raiz_indice)
            vectorstore = FAISS.load_local(
                ativa.caminho,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            vectorstore.add_documents(chunks)
            vectorstore.save_local(candidata.caminho)
            escrever_catalogo(candidata, novo_catalogo)
            escrever_manifest(candidata, novo_manifest)

            validado = FAISS.load_local(
                candidata.caminho,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            docstore = getattr(validado, "index_to_docstore_id", {})
            if len(docstore) < len(chunks):
                raise GeracaoIndiceError("Geracao candidata sem os novos chunks.")
            _escrever_journal(
                job_id,
                "candidate_validated",
                {
                    "previous_generation": geracao_anterior,
                    "candidate_generation": candidata.id,
                    "files": promovidos,
                },
            )
            ativar_geracao(candidata, raiz_indice)
            espelhar_catalogo(novo_catalogo)
            _escrever_journal(
                job_id,
                "committed",
                {
                    "previous_generation": geracao_anterior,
                    "candidate_generation": candidata.id,
                    "files": promovidos,
                },
            )
            remover_geracoes_antigas(raiz_indice, manter=2)
            return ResultadoIncremental(
                geracao=candidata.id,
                caminho_original=caminho_original,
                caminho_markdown=caminho_md,
                chunks=len(chunks),
                avisos=list(conversao.avisos),
            )
        except Exception:
            if candidata is not None:
                shutil.rmtree(candidata.caminho, ignore_errors=True)
            for caminho in reversed(promovidos):
                try:
                    os.remove(caminho)
                except OSError:
                    pass
            _escrever_journal(
                job_id,
                "rolled_back",
                {
                    "previous_generation": geracao_anterior,
                    "files": promovidos,
                },
            )
            raise
