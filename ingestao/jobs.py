"""Fila persistida de um unico escritor para ingestao incremental."""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

from config import logger
from ingestao.formatos import DocumentoValidado, validar_documento
from ingestao.indice_incremental import (
    indexar_documento,
    validar_duplicidade_ativa,
)


PASTA_JOBS = os.path.join("dados_antt", ".jobs")
PASTA_STAGING = os.path.join("dados_antt", ".staging")


class JobNaoEncontradoError(KeyError):
    """Job solicitado nao existe."""


@dataclass(frozen=True)
class JobCriado:
    """Identidade devolvida ao chamador apos persistencia."""

    job_id: str
    nome: str
    formato: str


def _caminho_job(job_id: str) -> str:
    """Monta caminho sem aceitar path traversal."""
    if not job_id or any(c not in "0123456789abcdef-" for c in job_id):
        raise JobNaoEncontradoError(job_id)
    return os.path.join(PASTA_JOBS, "{0}.json".format(job_id))


def _escrever_json(caminho: str, dados: Dict[str, object]) -> None:
    """Persiste estado por replace atomico."""
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    descritor, temporario = tempfile.mkstemp(
        prefix=".tmp-",
        suffix=".json",
        dir=os.path.dirname(caminho),
    )
    try:
        with os.fdopen(descritor, "w", encoding="utf-8") as arquivo:
            json.dump(dados, arquivo, ensure_ascii=True, indent=2)
            arquivo.flush()
            os.fsync(arquivo.fileno())
        os.replace(temporario, caminho)
    finally:
        if os.path.exists(temporario):
            os.remove(temporario)


def obter_job(job_id: str) -> Dict[str, object]:
    """Le o estado persistido de um job."""
    caminho = _caminho_job(job_id)
    try:
        with open(caminho, "r", encoding="utf-8") as arquivo:
            bruto = json.load(arquivo)
    except FileNotFoundError as exc:
        raise JobNaoEncontradoError(job_id) from exc
    if not isinstance(bruto, dict):
        raise JobNaoEncontradoError(job_id)
    return bruto


def _atualizar_job(job_id: str, **campos: object) -> Dict[str, object]:
    """Atualiza job sem perder campos anteriores."""
    dados = obter_job(job_id)
    dados.update(campos)
    dados["updated_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(),
    )
    _escrever_json(_caminho_job(job_id), dados)
    return dados


def _criar_registro(
    documento: DocumentoValidado,
    origem: str,
    staging_dir: str,
) -> JobCriado:
    """Cria estado queued depois que a fonte esta no staging."""
    job_id = os.path.basename(staging_dir)
    agora = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dados: Dict[str, object] = {
        "job_id": job_id,
        "status": "queued",
        "nome": documento.nome,
        "formato": documento.formato.value,
        "sha256": documento.sha256,
        "origem": origem,
        "staging_dir": staging_dir,
        "source_path": os.path.join(staging_dir, documento.nome),
        "avisos": [],
        "mensagem": "Documento aguardando indexacao.",
        "created_at": agora,
        "updated_at": agora,
    }
    _escrever_json(_caminho_job(job_id), dados)
    return JobCriado(
        job_id=job_id,
        nome=documento.nome,
        formato=documento.formato.value,
    )


_CONTEXTO = multiprocessing.get_context("spawn")
_PARAR: Optional[object] = None
_PROCESSO: Optional[multiprocessing.Process] = None
_GUARDA_THREAD = threading.Lock()


def submeter_upload(conteudo: bytes, nome: str) -> JobCriado:
    """Valida, persiste em staging e enfileira um upload."""
    documento = validar_documento(conteudo, nome)
    validar_duplicidade_ativa(documento)
    job_id = str(uuid.uuid4())
    staging = os.path.join(PASTA_STAGING, job_id)
    os.makedirs(staging, exist_ok=False)
    source = os.path.join(staging, documento.nome)
    try:
        with open(source, "xb") as arquivo:
            arquivo.write(documento.conteudo)
        criado = _criar_registro(documento, "api", staging)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return criado


def submeter_arquivo_reclamado(
    caminho: str,
    nome_original: str,
) -> JobCriado:
    """Registra PDF que o scanner ja moveu atomicamente para staging."""
    with open(caminho, "rb") as arquivo:
        documento = validar_documento(
            arquivo.read(),
            nome_original,
            formatos=["pdf"],
        )
    staging = os.path.dirname(caminho)
    destino = os.path.join(staging, documento.nome)
    if caminho != destino:
        os.replace(caminho, destino)
    criado = _criar_registro(documento, "inbox", staging)
    return criado


def _processar(job_id: str) -> None:
    """Executa um job persistido e captura erro sem derrubar o worker."""
    dados = _atualizar_job(
        job_id,
        status="running",
        mensagem="Convertendo e atualizando o indice.",
    )
    staging = str(dados["staging_dir"])
    source = str(dados["source_path"])
    origem = str(dados.get("origem") or "api")
    try:
        with open(source, "rb") as arquivo:
            documento = validar_documento(
                arquivo.read(),
                str(dados["nome"]),
                formatos=[str(dados["formato"])],
            )
        resultado = indexar_documento(documento, job_id, staging)
        status = (
            "succeeded_with_warnings"
            if resultado.avisos
            else "succeeded"
        )
        _atualizar_job(
            job_id,
            status=status,
            mensagem="Documento disponivel para consulta.",
            avisos=resultado.avisos,
            geracao=resultado.geracao,
            chunks=resultado.chunks,
            caminho=resultado.caminho_original,
        )
        shutil.rmtree(staging, ignore_errors=True)
    except Exception as exc:
        logger.exception("Falha no job de ingestao %s", job_id)
        if origem == "inbox" and os.path.isfile(source):
            destino = os.path.join("dados_antt", "entrada", str(dados["nome"]))
            if not os.path.exists(destino):
                os.makedirs(os.path.dirname(destino), exist_ok=True)
                os.replace(source, destino)
        _atualizar_job(
            job_id,
            status="failed",
            mensagem=str(exc),
        )


def _loop_worker(evento_parar: object) -> None:
    """Busca jobs queued no disco sem bloquear o processo HTTP."""
    esperar = getattr(evento_parar, "wait")
    while not esperar(0.5):
        candidatos: List[tuple[str, str]] = []
        try:
            nomes = os.listdir(PASTA_JOBS)
        except OSError:
            continue
        for nome in nomes:
            if not nome.endswith(".json"):
                continue
            job_id = nome[:-5]
            try:
                dados = obter_job(job_id)
            except JobNaoEncontradoError:
                continue
            if dados.get("status") == "queued":
                candidatos.append(
                    (str(dados.get("created_at") or ""), job_id)
                )
        if candidatos:
            _processar(sorted(candidatos)[0][1])


def iniciar_worker() -> None:
    """Inicia o unico worker deste processo e recupera queued."""
    global _PARAR, _PROCESSO
    with _GUARDA_THREAD:
        if _PROCESSO is not None and _PROCESSO.is_alive():
            return
        os.makedirs(PASTA_JOBS, exist_ok=True)
        os.makedirs(PASTA_STAGING, exist_ok=True)
        for nome in sorted(os.listdir(PASTA_JOBS)):
            if not nome.endswith(".json"):
                continue
            job_id = nome[:-5]
            try:
                dados = obter_job(job_id)
            except JobNaoEncontradoError:
                continue
            status = str(dados.get("status") or "")
            if status == "running":
                _atualizar_job(
                    job_id,
                    status="failed",
                    mensagem="Job interrompido pelo reinicio do servico.",
                )
        _PARAR = _CONTEXTO.Event()
        _PROCESSO = _CONTEXTO.Process(
            target=_loop_worker,
            args=(_PARAR,),
            name="rag-ingestao-worker",
            daemon=True,
        )
        _PROCESSO.start()


def parar_worker() -> None:
    """Solicita parada sem interromper gravacao em curso."""
    global _PARAR, _PROCESSO
    evento = _PARAR
    if evento is not None:
        getattr(evento, "set")()
    processo = _PROCESSO
    if processo is not None:
        processo.join(timeout=5.0)
        if processo.is_alive():
            processo.terminate()
            processo.join(timeout=2.0)
    _PROCESSO = None
    _PARAR = None


def enfileirar_existente(job_id: str) -> None:
    """Valida que um job persistido esta pronto para o worker."""
    dados = obter_job(job_id)
    if dados.get("status") != "queued":
        raise ValueError("Somente job queued pode ser enfileirado.")
