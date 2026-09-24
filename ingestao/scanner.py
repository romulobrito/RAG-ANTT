"""Scanner conservador de PDFs recebidos pelo volume compartilhado."""

from __future__ import annotations

import hashlib
import os
import threading
import time
import uuid
from typing import Dict, Optional, Tuple

from config import get_inbox_scan_seconds, logger
from ingestao.geracoes import ler_manifest, resolver_geracao_ativa
from ingestao.jobs import PASTA_STAGING, submeter_arquivo_reclamado


PASTA_INBOX = os.path.join("dados_antt", "entrada")
_ESTABILIDADE: Dict[str, Tuple[int, float]] = {}
_PARAR = threading.Event()
_THREAD: Optional[threading.Thread] = None
_GUARDA = threading.Lock()


def _hash(caminho: str) -> str:
    """Calcula SHA-256 do candidato estavel."""
    digest = hashlib.sha256()
    with open(caminho, "rb") as arquivo:
        while True:
            bloco = arquivo.read(1024 * 1024)
            if not bloco:
                break
            digest.update(bloco)
    return digest.hexdigest()


def _ja_indexado(nome: str, sha256: str) -> bool:
    """Consulta manifest sem modificar indice."""
    try:
        manifest = ler_manifest(resolver_geracao_ativa())
    except Exception:
        return True
    documentos = manifest.get("documents", [])
    if not isinstance(documentos, list):
        return False
    md_esperado = os.path.splitext(nome)[0] + ".md"
    for item in documentos:
        if not isinstance(item, dict):
            continue
        nome_item = str(item.get("source_name") or item.get("nome") or "")
        hash_item = str(
            item.get("source_sha256") or item.get("sha256") or ""
        )
        if nome_item.casefold() in (nome.casefold(), md_esperado.casefold()):
            return True
        if hash_item and hash_item == sha256:
            return True
    return False


def varrer_uma_vez() -> int:
    """Enfileira PDFs estaveis e retorna a quantidade reclamada."""
    os.makedirs(PASTA_INBOX, exist_ok=True)
    os.makedirs(PASTA_STAGING, exist_ok=True)
    vistos = set()
    reclamados = 0
    for nome in sorted(os.listdir(PASTA_INBOX)):
        caminho = os.path.join(PASTA_INBOX, nome)
        if (
            nome.startswith(".")
            or nome.endswith(".part")
            or not nome.lower().endswith(".pdf")
            or not os.path.isfile(caminho)
        ):
            continue
        vistos.add(caminho)
        try:
            estado = (os.path.getsize(caminho), os.path.getmtime(caminho))
        except OSError:
            continue
        anterior = _ESTABILIDADE.get(caminho)
        _ESTABILIDADE[caminho] = estado
        if anterior != estado:
            continue
        try:
            sha256 = _hash(caminho)
        except OSError:
            continue
        if _ja_indexado(nome, sha256):
            continue
        job_id = str(uuid.uuid4())
        staging = os.path.join(PASTA_STAGING, job_id)
        os.makedirs(staging, exist_ok=False)
        reclamado = os.path.join(staging, nome)
        try:
            os.replace(caminho, reclamado)
            submeter_arquivo_reclamado(reclamado, nome)
            reclamados += 1
            _ESTABILIDADE.pop(caminho, None)
        except Exception:
            if os.path.isfile(reclamado) and not os.path.exists(caminho):
                os.replace(reclamado, caminho)
            try:
                os.rmdir(staging)
            except OSError:
                pass
            logger.exception("Falha ao reclamar PDF da inbox: %s", nome)
    for caminho in list(_ESTABILIDADE):
        if caminho not in vistos:
            _ESTABILIDADE.pop(caminho, None)
    return reclamados


def _loop() -> None:
    """Executa varredura periodica sem sobrepor ciclos."""
    intervalo = get_inbox_scan_seconds()
    while not _PARAR.wait(intervalo):
        try:
            varrer_uma_vez()
        except Exception:
            logger.exception("Falha na varredura da inbox")


def iniciar_scanner() -> None:
    """Inicia scanner depois que o bootstrap do indice terminou."""
    global _THREAD
    with _GUARDA:
        if _THREAD is not None and _THREAD.is_alive():
            return
        _PARAR.clear()
        # Primeira passagem apenas registra tamanho e mtime.
        varrer_uma_vez()
        _THREAD = threading.Thread(
            target=_loop,
            name="rag-inbox-scanner",
            daemon=True,
        )
        _THREAD.start()


def parar_scanner() -> None:
    """Para o scanner durante shutdown."""
    global _THREAD
    _PARAR.set()
    thread = _THREAD
    if thread is not None:
        thread.join(timeout=2.0)
    _THREAD = None
