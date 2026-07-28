"""
Montagem de contexto RAG compartimentado por documento-fonte.

Agrupa chunks do mesmo documento e delimita cada fonte com cercas
FONTE INICIO / FONTE FIM para reduzir fusao indevida pelo LLM.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Sequence, Tuple


def id_fonte_documento(metadados: Dict[str, object]) -> str:
    """
    Monta identificador legivel da fonte a partir dos metadados do chunk.

    Args:
        metadados: Metadata do Document LangChain.

    Returns:
        String no formato "Tipo numero/ano" (ou fallbacks seguros).
    """
    if not isinstance(metadados, dict):
        return "Documento N/A/N/A"
    tipo = (
        metadados.get("nome_tipo")
        or metadados.get("tipo_documento")
        or "Documento"
    )
    numero = metadados.get("numero", "N/A")
    ano = metadados.get("ano", "N/A")
    return f"{tipo} {numero}/{ano}"


def montar_contexto_compartimentado(
    documentos: Sequence[object],
) -> Tuple[str, set, Dict[str, int]]:
    """
    Agrupa chunks por documento e delimita cada fonte com cercas explicitas.

    A ordem das fontes segue a primeira aparicao no ranking de retrieval.
    Dentro de cada fonte, os trechos mantem a ordem relativa original.

    Args:
        documentos: Lista de Document recuperados.

    Returns:
        (texto_contexto, conjunto_de_tipos, contagem_por_tipo).
    """
    grupos: OrderedDict[str, List[object]] = OrderedDict()
    tipos_documentos: set = set()
    contagem_por_tipo: Dict[str, int] = {}

    for doc in documentos:
        metadados = getattr(doc, "metadata", None) or {}
        if not isinstance(metadados, dict):
            metadados = {}
        tipo = str(
            metadados.get("nome_tipo")
            or metadados.get("tipo_documento")
            or "Documento"
        )
        tipos_documentos.add(tipo)
        contagem_por_tipo[tipo] = contagem_por_tipo.get(tipo, 0) + 1
        doc_id = id_fonte_documento(metadados)
        if doc_id not in grupos:
            grupos[doc_id] = []
        grupos[doc_id].append(doc)

    blocos: List[str] = []
    for doc_id, chunks in grupos.items():
        linhas: List[str] = [
            f"===== FONTE INICIO: {doc_id} =====",
            f"ID: {doc_id}",
            (
                "Escopo: as regras, valores e formalizacoes abaixo "
                "aplicam-se SOMENTE a esta fonte."
            ),
            (
                "Nao use o conteudo deste bloco para inferir "
                "obrigacoes de outras fontes."
            ),
        ]
        meta0 = getattr(chunks[0], "metadata", None) or {}
        if isinstance(meta0, dict):
            caminho = meta0.get("caminho")
            if caminho:
                linhas.append(f"Caminho: {caminho}")
        total = len(chunks)
        for indice, doc in enumerate(chunks, start=1):
            meta = getattr(doc, "metadata", None) or {}
            if not isinstance(meta, dict):
                meta = {}
            chunk_n = meta.get("chunk", indice)
            total_n = meta.get("total_chunks", total)
            conteudo = getattr(doc, "page_content", "") or ""
            linhas.append(
                f"--- Trecho {indice}/{total} (chunk {chunk_n}/{total_n}) ---"
            )
            linhas.append(conteudo)
        linhas.append(f"===== FONTE FIM: {doc_id} =====")
        blocos.append("\n".join(linhas))

    return "\n\n".join(blocos), tipos_documentos, contagem_por_tipo
