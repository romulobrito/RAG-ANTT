"""
Packing estrutural de chunks para orcamento de contexto do LLM.

Seleciona quais chunks entram no limite de caracteres com base em sinais
estruturais (tabela, fonte estruturada, densidade numerica, ranking), sem
listas de topicos ou siglas de dominio.

Invariante de identidade: se o contexto completo cabe no orcamento, a lista
de entrada e devolvida intacta (mesma ordem, mesmos chunks).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document

from contexto_compartimentado import (
    id_fonte_documento,
    montar_contexto_compartimentado,
)
from retrieval_hibrido import (
    chunk_tem_tabela,
    classificar_fonte_qualidade,
)

# Fracao maxima do orcamento de chars atribuivel a uma unica fonte quando
# o packing esta ativo (orcamento estourado).
FRACAO_TETO_POR_FONTE = 0.40


def densidade_numerica(texto: str) -> float:
    """
    Calcula a proporcao de digitos no texto, entre 0.0 e 1.0.

    Args:
        texto: Conteudo do chunk.

    Returns:
        Fracao de caracteres que sao digitos.
    """
    if not texto:
        return 0.0
    digitos = sum(1 for c in texto if c.isdigit())
    return digitos / len(texto)


def score_estrutural(documento: Document, rank_index: int) -> float:
    """
    Pontua um chunk para packing sob orcamento apertado.

    Sinais sao estruturais/qualidade de fonte e posicao no ranking.
    Nao usa palavras de dominio.

    Args:
        documento: Chunk recuperado.
        rank_index: Posicao 0-based na lista de retrieval (menor = melhor).

    Returns:
        Score numerico (maior = mais prioritario no packing).
    """
    metadados = documento.metadata if isinstance(documento.metadata, dict) else {}
    texto = documento.page_content or ""
    fonte = classificar_fonte_qualidade(texto, metadados)

    score = 0.0
    # Ranking: primeiros chunks do retrieval ainda importam.
    score += max(0.0, 20.0 - float(rank_index))

    if metadados.get("prioritario"):
        score += 40.0
    if metadados.get("injetado_auxiliar"):
        score += 25.0
    if fonte == "estruturada":
        score += 30.0
    elif fonte == "ocr":
        score -= 5.0

    if chunk_tem_tabela(documento):
        score += 35.0

    dens = densidade_numerica(texto)
    score += min(20.0, dens * 100.0)

    return score


def cortar_preservando_tabela(texto: str, max_chars: int) -> str:
    """
    Reduz um texto ao orcamento preferindo linhas de tabela markdown.

    Args:
        texto: Conteudo original.
        max_chars: Limite de caracteres.

    Returns:
        Texto cortado (ou original se couber).
    """
    if max_chars <= 0:
        return ""
    if len(texto) <= max_chars:
        return texto

    linhas = texto.split("\n")
    linhas_tabela = [ln for ln in linhas if ln.strip().startswith("|")]
    if linhas_tabela:
        selecionadas: List[str] = []
        tamanho = 0
        for ln in linhas_tabela:
            custo = len(ln) + (1 if selecionadas else 0)
            if tamanho + custo > max_chars:
                break
            selecionadas.append(ln)
            tamanho += custo
        if selecionadas:
            return "\n".join(selecionadas)

    return texto[:max_chars]


def _estimar_tamanho_contexto(
    documentos: Sequence[Document],
    overhead_instrucoes: int,
) -> int:
    """
    Estima chars do prompt de contexto (instrucoes + corpo montado).

    Args:
        documentos: Chunks candidatos.
        overhead_instrucoes: Tamanho das instrucoes prefixadas.

    Returns:
        Tamanho total estimado em caracteres.
    """
    if not documentos:
        return max(0, int(overhead_instrucoes))
    corpo, _, _ = montar_contexto_compartimentado(documentos)
    # "\n\n" entre instrucoes e corpo, como em _preparar_contexto_resposta.
    separador = 2 if overhead_instrucoes > 0 else 0
    return int(overhead_instrucoes) + separador + len(corpo)


def _documento_com_conteudo(documento: Document, novo_conteudo: str) -> Document:
    """
    Cria copia do Document com page_content substituido.

    Args:
        documento: Documento original.
        novo_conteudo: Novo texto do chunk.

    Returns:
        Novo Document (metadata copiado).
    """
    meta = documento.metadata if isinstance(documento.metadata, dict) else {}
    return Document(page_content=novo_conteudo, metadata=dict(meta))


def selecionar_chunks_para_orcamento(
    documentos: Sequence[Document],
    limite_chars: int,
    overhead_instrucoes: int = 0,
    fracao_teto_fonte: float = FRACAO_TETO_POR_FONTE,
) -> List[Document]:
    """
    Seleciona chunks que cabem no orcamento de caracteres.

    Invariante: se o contexto com TODOS os chunks cabe em limite_chars,
    devolve a lista intacta (mesma ordem). Caso contrario, seleciona de
    forma gulosa por score estrutural, com teto por fonte, e devolve os
    escolhidos na ordem original de retrieval.

    Args:
        documentos: Chunks ja recuperados (ordem de ranking).
        limite_chars: Orcamento total do contexto (inclui instrucoes).
        overhead_instrucoes: Chars das instrucoes prefixadas ao corpo.
        fracao_teto_fonte: Fracao maxima do orcamento por id de fonte
            quando o packing esta ativo.

    Returns:
        Lista de Document a montar no contexto.
    """
    docs: List[Document] = list(documentos)
    if not docs:
        return []

    limite = max(1, int(limite_chars))
    overhead = max(0, int(overhead_instrucoes))
    fracao = fracao_teto_fonte
    if fracao <= 0.0 or fracao > 1.0:
        fracao = FRACAO_TETO_POR_FONTE
    teto_fonte = max(1, int(limite * fracao))

    tamanho_full = _estimar_tamanho_contexto(docs, overhead)
    if tamanho_full <= limite:
        return docs

    ranqueados = [
        (score_estrutural(doc, indice), indice, doc)
        for indice, doc in enumerate(docs)
    ]
    ranqueados.sort(key=lambda item: (-item[0], item[1]))

    selecionados_idx: List[int] = []
    chars_por_fonte: Dict[str, int] = {}

    for _score, indice, doc in ranqueados:
        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        fonte = id_fonte_documento(meta)
        conteudo_len = len(doc.page_content or "")
        uso_atual = chars_por_fonte.get(fonte, 0)

        # Teto por fonte: permite o primeiro chunk da fonte mesmo se grande;
        # chunks seguintes da mesma fonte respeitam o teto.
        if uso_atual > 0 and (uso_atual + conteudo_len) > teto_fonte:
            continue

        trial_idx = selecionados_idx + [indice]
        trial_docs = [docs[j] for j in sorted(trial_idx)]
        tamanho_trial = _estimar_tamanho_contexto(trial_docs, overhead)
        if tamanho_trial > limite:
            continue

        selecionados_idx.append(indice)
        chars_por_fonte[fonte] = uso_atual + conteudo_len

    if selecionados_idx:
        return [docs[i] for i in sorted(selecionados_idx)]

    # Nenhum chunk coube inteiro: fica o de maior score, cortado se preciso.
    _melhor_score, melhor_idx, melhor_doc = ranqueados[0]
    orcamento_corpo = max(1, limite - overhead - 2)
    # Reserva aproximada para cercas FONTE (cabecalho + rodape).
    reserva_cerca = 220
    max_conteudo = max(1, orcamento_corpo - reserva_cerca)
    texto = melhor_doc.page_content or ""
    if len(texto) <= max_conteudo:
        return [melhor_doc]
    texto_cortado = cortar_preservando_tabela(texto, max_conteudo)
    return [_documento_com_conteudo(melhor_doc, texto_cortado)]


def eh_identificador_ollama(modelo_usado: Optional[str]) -> bool:
    """
    Indica se o identificador de modelo corresponde a inferencia Ollama.

    Args:
        modelo_usado: String do provedor/modelo (ex.: ollama, qwen2.5:7b).

    Returns:
        True se for rota local Ollama.
    """
    texto = (modelo_usado or "").lower()
    return any(
        chave in texto
        for chave in ("ollama", "llama3", "qwen2.5", "phi3")
    )
