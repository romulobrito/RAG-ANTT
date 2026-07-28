"""
Harness de avaliacao multi-dominio do retrieval e da geracao.

Metricas deterministicas (sem LLM juiz):
  - cobertura dos valores esperados nos trechos recuperados
  - cobertura dos valores esperados na resposta gerada (completude)
  - acerto do documento-alvo no top-k
  - fracao de chunks estruturados vs OCR no contexto
  - latencia de retrieval e de geracao

Metricas RAGAS (opcionais, com --com-ragas; usam LLM juiz):
  - faithfulness
  - answer_relevancy
  - context_precision
  - context_recall

Uso:
    python avaliar_retrieval.py
    python avaliar_retrieval.py --k 16 --casos pavimento_generico,iri_principal
    python avaliar_retrieval.py --saida relatorios_avaliacao
    python avaliar_retrieval.py --com-geracao --casos iri_principal
    python avaliar_retrieval.py --com-ragas --casos iri_principal,dadm_vdm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CasoAvaliacao:
    """
    Caso de teste do retrieval com gabarito multi-dominio.

    Attributes:
        identificador: Chave curta.
        dominio: Agrupamento (pavimento, prazos, eef, etc.).
        pergunta: Texto enviado ao pesquisar_documentos.
        valores_esperados: Trechos que devem aparecer no contexto recuperado.
        doc_esperado: (tipo, numero, ano) do documento-alvo, ou None.
        min_estruturados: Minimo de chunks estruturados desejavel (0 = ignorar).
    """

    identificador: str
    dominio: str
    pergunta: str
    valores_esperados: Tuple[str, ...]
    doc_esperado: Optional[Tuple[str, str, str]]
    min_estruturados: int = 0


# Gabaritos derivados das tabelas auxiliares e normas indexadas. Mantidos em
# ASCII para bater com a normalizacao do harness.
CASOS_AVALIACAO: Tuple[CasoAvaliacao, ...] = (
    CasoAvaliacao(
        identificador="pavimento_generico",
        dominio="pavimento",
        pergunta="Quais sao os parametros tecnicos para pavimentos rodoviarios?",
        valores_esperados=(
            "3,5 m/km",
            "2,7 m/km",
            "3,0 m/km",
            "Dadm",
            "IFI",
            "12 mm",
            "20%",
            "5 cm",
            "pavimento rigido",
            "faixas paralelas",
        ),
        doc_esperado=("INM", "34", "2024"),
        min_estruturados=3,
    ),
    CasoAvaliacao(
        identificador="iri_principal",
        dominio="pavimento",
        pergunta=(
            "Quais sao os limites de IRI da pista principal na "
            "Instrucao Normativa 34 de 2024?"
        ),
        valores_esperados=("3,5 m/km", "2,7 m/km", "60%", "40%"),
        doc_esperado=("INM", "34", "2024"),
        min_estruturados=1,
    ),
    CasoAvaliacao(
        identificador="dadm_vdm",
        dominio="pavimento",
        pergunta=(
            "Qual a deflexao admissivel Dadm para VDM entre 1000 e 2500 "
            "na INM 34/2024?"
        ),
        valores_esperados=("50",),
        doc_esperado=("INM", "34", "2024"),
        min_estruturados=1,
    ),
    CasoAvaliacao(
        identificador="ifi_manutencao",
        dominio="pavimento",
        pergunta=(
            "Qual o valor de IFI na fase de manutencao segundo a "
            "Instrucao Normativa 34/2024?"
        ),
        valores_esperados=("0,2",),
        doc_esperado=("INM", "34", "2024"),
        min_estruturados=1,
    ),
    CasoAvaliacao(
        identificador="flechas_principal",
        dominio="pavimento",
        pergunta=(
            "Quais os limites de flechas nas trilhas de roda da pista "
            "principal na INM 34/2024?"
        ),
        valores_esperados=("12 mm", "7 mm"),
        doc_esperado=("INM", "34", "2024"),
        min_estruturados=1,
    ),
    CasoAvaliacao(
        identificador="cronograma_revisao",
        dominio="prazos",
        pergunta=(
            "Quais sao os prazos do cronograma de revisao ordinaria em "
            "relacao a data-base, conforme o Anexo I da Instrucao Normativa "
            "18 de 2023?"
        ),
        valores_esperados=("140", "90", "75", "35"),
        doc_esperado=("INM", "18", "2023"),
        min_estruturados=0,
    ),
    CasoAvaliacao(
        identificador="prazos_eef",
        dominio="eef",
        pergunta=(
            "Quais os prazos maximos para analise de admissibilidade e para "
            "a decisao da Diretoria no pedido de recomposicao do equilibrio "
            "economico-financeiro, conforme a Instrucao Normativa 33 de 2024?"
        ),
        valores_esperados=("60 dias", "180 dias"),
        doc_esperado=("INM", "33", "2024"),
        min_estruturados=0,
    ),
    CasoAvaliacao(
        identificador="recurso_admissibilidade",
        dominio="eef",
        pergunta=(
            "Qual o prazo para recurso contra o indeferimento da "
            "admissibilidade na Instrucao Normativa 33 de 2024?"
        ),
        valores_esperados=("15 dias",),
        doc_esperado=("INM", "33", "2024"),
        min_estruturados=0,
    ),
)

# Nomes das metricas RAGAS gravadas no relatorio.
METRICAS_RAGAS: Tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def normalizar_texto(valor: str) -> str:
    """Normaliza texto para comparacao de gabarito (sem acento, minusculas)."""
    if not isinstance(valor, str):
        return ""
    decomposto = unicodedata.normalize("NFKD", valor)
    sem_acento = "".join(c for c in decomposto if not unicodedata.combining(c))
    return sem_acento.lower()


def cobertura_valores(contexto: str, esperados: Sequence[str]) -> Tuple[float, List[str], List[str]]:
    """
    Calcula a fracao de valores esperados presentes no contexto.

    Args:
        contexto: Texto concatenado dos chunks recuperados ou da resposta.
        esperados: Gabarito.

    Returns:
        (cobertura 0-1, encontrados, ausentes).
    """
    contexto_n = normalizar_texto(contexto)
    encontrados: List[str] = []
    ausentes: List[str] = []
    for esperado in esperados:
        alvo = normalizar_texto(esperado)
        if alvo and alvo in contexto_n:
            encontrados.append(esperado)
        else:
            ausentes.append(esperado)
    total = len(esperados) or 1
    return len(encontrados) / total, encontrados, ausentes


def documento_bate(doc_meta: Dict[str, object], esperado: Tuple[str, str, str]) -> bool:
    """Verifica se o metadado do chunk corresponde ao documento esperado."""
    tipo_e, numero_e, ano_e = esperado
    tipo = str(doc_meta.get("tipo_documento") or doc_meta.get("nome_tipo") or "")
    numero = str(doc_meta.get("numero") or "").lstrip("0") or "0"
    ano = str(doc_meta.get("ano") or "")
    return (
        normalizar_texto(tipo).startswith(normalizar_texto(tipo_e)[:3])
        and numero == (numero_e.lstrip("0") or "0")
        and ano == ano_e
    )


def montar_referencia(caso: CasoAvaliacao) -> str:
    """
    Monta texto de referencia para metricas RAGAS que exigem ground_truth.

    Args:
        caso: Caso com gabarito de valores e documento.

    Returns:
        Frase ASCII com os elementos minimos da resposta correta.
    """
    partes: List[str] = [f"A resposta deve incluir: {', '.join(caso.valores_esperados)}."]
    if caso.doc_esperado is not None:
        tipo, numero, ano = caso.doc_esperado
        partes.append(f"Documento de referencia: {tipo} {numero}/{ano}.")
    return " ".join(partes)


def extrair_textos_documentos(documentos: Sequence[object]) -> List[str]:
    """Extrai page_content de cada Document recuperado."""
    textos: List[str] = []
    for doc in documentos:
        conteudo = getattr(doc, "page_content", "") or ""
        if isinstance(conteudo, str) and conteudo.strip():
            textos.append(conteudo)
        else:
            textos.append("")
    return textos


def avaliar_caso(caso: CasoAvaliacao, documentos: Sequence[object]) -> Dict[str, object]:
    """
    Avalia um unico caso a partir da lista de Document recuperada.

    Args:
        caso: Definicao do gabarito.
        documentos: Resultado de pesquisar_documentos.

    Returns:
        Dicionario com metricas do caso.
    """
    from retrieval_hibrido import classificar_fonte_qualidade

    contextos = extrair_textos_documentos(documentos)
    hit_doc = False
    n_estruturada = 0
    n_ocr = 0

    for indice, doc in enumerate(documentos):
        meta = getattr(doc, "metadata", {}) or {}
        fonte = classificar_fonte_qualidade(contextos[indice], meta)
        if fonte == "estruturada":
            n_estruturada += 1
        elif fonte == "ocr":
            n_ocr += 1
        if caso.doc_esperado and documento_bate(meta, caso.doc_esperado):
            hit_doc = True

    contexto = "\n".join(contextos)
    cobertura, encontrados, ausentes = cobertura_valores(
        contexto, caso.valores_esperados
    )

    return {
        "identificador": caso.identificador,
        "dominio": caso.dominio,
        "n_docs": len(documentos),
        "cobertura": cobertura,
        "encontrados": encontrados,
        "ausentes": ausentes,
        "hit_documento": hit_doc if caso.doc_esperado else None,
        "n_estruturada": n_estruturada,
        "n_ocr": n_ocr,
        "ok_estruturados": n_estruturada >= caso.min_estruturados,
        "contextos": contextos,
        "referencia": montar_referencia(caso),
    }


def avaliar_completude_resposta(
    caso: CasoAvaliacao,
    resposta: str,
) -> Dict[str, object]:
    """
    Mede completude factual da resposta contra o gabarito normativo.

    Args:
        caso: Caso com valores esperados.
        resposta: Texto gerado pelo LLM.

    Returns:
        Metricas de cobertura da resposta.
    """
    cobertura, encontrados, ausentes = cobertura_valores(
        resposta, caso.valores_esperados
    )
    return {
        "completude_resposta": cobertura,
        "encontrados_resposta": encontrados,
        "ausentes_resposta": ausentes,
    }


def criar_llm_geracao(
    provider: str,
    model: Optional[str],
    max_tokens: int,
) -> Tuple[object, str]:
    """
    Instancia o LLM de geracao via llm_providers.

    Args:
        provider: deepseek ou openai.
        model: Chave do modelo em LLM_PROVIDERS, ou None para o padrao.
        max_tokens: Teto de tokens da resposta.

    Returns:
        (llm, nome_modelo_exibido).
    """
    from llm_providers import create_llm_manager

    manager = create_llm_manager(
        provider=provider,
        model=model,
        embedding_provider="local",
    )
    llm = manager.get_llm(temperature=0.1, max_tokens=max_tokens)
    nome = manager.config["models"][manager.model]
    return llm, str(nome)


def criar_wrappers_ragas(
    llm: object,
    embedding_provider: str = "local",
) -> Tuple[object, object]:
    """
    Empacota LLM e embeddings no formato exigido pelo RAGAS 0.1.x.

    Args:
        llm: ChatOpenAI (ou compativel) ja configurado.
        embedding_provider: local ou openai (para answer_relevancy).

    Returns:
        (llm_wrapper, embeddings_wrapper).
    """
    from langchain_core.embeddings import Embeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from llm_providers import LocalEmbeddings, create_llm_manager

    class _AdaptadorEmbeddings(Embeddings):
        """Adapta LocalEmbeddings/OpenAIEmbeddings a interface LangChain."""

        def __init__(self, interno: object) -> None:
            self._interno = interno

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            resultado = self._interno.embed_documents(texts)
            return [[float(v) for v in vetor] for vetor in resultado]

        def embed_query(self, text: str) -> List[float]:
            vetor = self._interno.embed_query(text)
            return [float(v) for v in vetor]

    if embedding_provider == "openai":
        manager = create_llm_manager(
            provider="openai",
            embedding_provider="openai",
        )
        interno = manager.get_embeddings()
    else:
        interno = LocalEmbeddings()

    return (
        LangchainLLMWrapper(llm),
        LangchainEmbeddingsWrapper(_AdaptadorEmbeddings(interno)),
    )


def _float_ou_nulo(valor: object) -> Optional[float]:
    """Converte score RAGAS para float, tratando NaN como None."""
    if valor is None:
        return None
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return None
    if numero != numero:  # NaN
        return None
    return numero


def avaliar_com_ragas(
    amostras: Sequence[Mapping[str, object]],
    llm_wrapper: object,
    embeddings_wrapper: object,
) -> List[Dict[str, Optional[float]]]:
    """
    Executa faithfulness, relevancy, precision e recall via RAGAS.

    Args:
        amostras: Lista com question, answer, contexts, ground_truth.
        llm_wrapper: LangchainLLMWrapper do juiz.
        embeddings_wrapper: LangchainEmbeddingsWrapper.

    Returns:
        Lista de dicionarios com as quatro metricas por amostra.
    """
    if not amostras:
        return []

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    perguntas: List[str] = []
    respostas: List[str] = []
    contextos: List[List[str]] = []
    referencias: List[str] = []

    for item in amostras:
        pergunta = item.get("question")
        resposta = item.get("answer")
        ctx = item.get("contexts")
        ref = item.get("ground_truth")
        if not isinstance(pergunta, str) or not isinstance(resposta, str):
            raise ValueError("Amostra RAGAS exige question e answer como str.")
        if not isinstance(ctx, list) or not all(isinstance(c, str) for c in ctx):
            raise ValueError("Amostra RAGAS exige contexts como list[str].")
        if not isinstance(ref, str):
            raise ValueError("Amostra RAGAS exige ground_truth como str.")
        perguntas.append(pergunta)
        respostas.append(resposta)
        contextos.append(ctx if ctx else [""])
        referencias.append(ref)

    dataset = Dataset.from_dict(
        {
            "question": perguntas,
            "answer": respostas,
            "contexts": contextos,
            "ground_truth": referencias,
        }
    )

    resultado = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm_wrapper,
        embeddings=embeddings_wrapper,
        raise_exceptions=False,
    )

    tabela = resultado.to_pandas()
    saidas: List[Dict[str, Optional[float]]] = []
    for indice in range(len(amostras)):
        linha: Dict[str, Optional[float]] = {}
        for metrica in METRICAS_RAGAS:
            if metrica in tabela.columns:
                linha[metrica] = _float_ou_nulo(tabela.iloc[indice][metrica])
            else:
                linha[metrica] = None
        saidas.append(linha)
    return saidas


def _media_opcional(valores: Sequence[Optional[float]]) -> Optional[float]:
    """Media ignorando None; None se a lista estiver vazia."""
    validos = [v for v in valores if v is not None]
    if not validos:
        return None
    return sum(validos) / len(validos)


def gerar_relatorio_markdown(
    resultados: Sequence[Dict[str, object]],
    metadados_execucao: Dict[str, object],
) -> str:
    """Monta o relatorio markdown da execucao."""
    com_ragas = bool(metadados_execucao.get("com_ragas"))
    com_geracao = bool(metadados_execucao.get("com_geracao"))

    linhas = [
        "# Avaliacao de retrieval multi-dominio",
        "",
        f"- Data: {metadados_execucao.get('data', '')}",
        f"- k: {metadados_execucao.get('k', '')}",
        f"- Embeddings: {metadados_execucao.get('embeddings', '')}",
        f"- Geracao: {'sim' if com_geracao else 'nao'}",
        f"- RAGAS: {'sim' if com_ragas else 'nao'}",
        f"- Casos: {len(resultados)}",
        "",
        "## Resumo",
        "",
    ]

    cabecalho = (
        "| Caso | Dominio | Cobertura | Completude resp. | Hit doc | "
        "Estruturados | OCR | Retr (s) | Ger (s) |"
    )
    separador = "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    if com_ragas:
        cabecalho = (
            f"{cabecalho} Faith | Relev | CtxPrec | CtxRec |"
        )
        separador = f"{separador} --- | --- | --- | --- |"
    linhas.extend([cabecalho, separador])

    coberturas: List[float] = []
    completudes: List[float] = []
    for item in resultados:
        cob = float(item["cobertura"])
        coberturas.append(cob)
        hit = item["hit_documento"]
        hit_txt = "sim" if hit else ("n/a" if hit is None else "nao")
        comp = item.get("completude_resposta")
        if isinstance(comp, (int, float)):
            completudes.append(float(comp))
            comp_txt = f"{float(comp):.0%}"
        else:
            comp_txt = "n/a"
        linha = (
            f"| {item['identificador']} | {item['dominio']} | {cob:.0%} | "
            f"{comp_txt} | {hit_txt} | {item['n_estruturada']} | {item['n_ocr']} | "
            f"{float(item.get('tempo_retrieval_s', item.get('tempo_s', 0))):.2f} | "
            f"{float(item.get('tempo_geracao_s', 0) or 0):.2f} |"
        )
        if com_ragas:
            def _fmt(chave: str) -> str:
                valor = item.get(chave)
                if isinstance(valor, (int, float)):
                    return f"{float(valor):.2f}"
                return "n/a"

            linha = (
                f"{linha} {_fmt('faithfulness')} | {_fmt('answer_relevancy')} | "
                f"{_fmt('context_precision')} | {_fmt('context_recall')} |"
            )
        linhas.append(linha)

    media = sum(coberturas) / len(coberturas) if coberturas else 0.0
    hits = [r for r in resultados if r["hit_documento"] is True]
    elegiveis = [r for r in resultados if r["hit_documento"] is not None]
    taxa_hit = (len(hits) / len(elegiveis)) if elegiveis else 0.0
    media_comp = (
        sum(completudes) / len(completudes) if completudes else None
    )

    linhas.extend(
        [
            "",
            f"**Cobertura media (retrieval):** {media:.0%}",
            f"**Taxa de hit do documento-alvo:** {taxa_hit:.0%} "
            f"({len(hits)}/{len(elegiveis)})",
        ]
    )
    if media_comp is not None:
        linhas.append(f"**Completude media (resposta):** {media_comp:.0%}")

    if com_ragas:
        for metrica in METRICAS_RAGAS:
            media_m = _media_opcional(
                [
                    float(r[metrica]) if isinstance(r.get(metrica), (int, float)) else None
                    for r in resultados
                ]
            )
            if media_m is None:
                linhas.append(f"**Media {metrica}:** n/a")
            else:
                linhas.append(f"**Media {metrica}:** {media_m:.3f}")

    linhas.extend(["", "## Detalhes", ""])

    for item in resultados:
        linhas.extend(
            [
                f"### {item['identificador']}",
                "",
                f"- Dominio: {item['dominio']}",
                f"- Cobertura retrieval: {float(item['cobertura']):.0%}",
                f"- Encontrados: {', '.join(item['encontrados']) or '-'}",
                f"- Ausentes: {', '.join(item['ausentes']) or '-'}",
                f"- Chunks estruturados: {item['n_estruturada']} "
                f"(meta minima ok: {item['ok_estruturados']})",
            ]
        )
        if "completude_resposta" in item:
            linhas.append(
                f"- Completude resposta: "
                f"{float(item['completude_resposta']):.0%}"
            )
            ausentes_r = item.get("ausentes_resposta") or []
            if isinstance(ausentes_r, list) and ausentes_r:
                linhas.append(
                    f"- Ausentes na resposta: {', '.join(str(a) for a in ausentes_r)}"
                )
        if com_ragas:
            for metrica in METRICAS_RAGAS:
                valor = item.get(metrica)
                if isinstance(valor, (int, float)):
                    linhas.append(f"- {metrica}: {float(valor):.3f}")
                else:
                    linhas.append(f"- {metrica}: n/a")
        if item.get("resposta_preview"):
            linhas.append(f"- Preview resposta: {item['resposta_preview']}")
        linhas.append("")

    return "\n".join(linhas)


def _selecionar_casos(filtro: str) -> Tuple[CasoAvaliacao, ...]:
    """Filtra CASOS_AVALIACAO pelos identificadores informados."""
    if not filtro.strip():
        return CASOS_AVALIACAO
    selecionados = {c.strip() for c in filtro.split(",") if c.strip()}
    casos = tuple(c for c in CASOS_AVALIACAO if c.identificador in selecionados)
    return casos


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Ponto de entrada CLI do harness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=16, help="Trechos por pergunta.")
    parser.add_argument(
        "--embeddings",
        default="local",
        help="Provedor de embeddings (local/free/openai).",
    )
    parser.add_argument(
        "--casos",
        default="",
        help="Identificadores separados por virgula (padrao: todos).",
    )
    parser.add_argument(
        "--saida",
        default="relatorios_avaliacao",
        help="Diretorio de saida dos relatorios.",
    )
    parser.add_argument(
        "--com-geracao",
        action="store_true",
        help="Gera resposta com o LLM e mede completude/latencia de geracao.",
    )
    parser.add_argument(
        "--com-ragas",
        action="store_true",
        help=(
            "Alem da geracao, calcula faithfulness, answer_relevancy, "
            "context_precision e context_recall via RAGAS."
        ),
    )
    parser.add_argument(
        "--llm-provider",
        default="deepseek",
        help="Provedor do LLM de geracao/juiz (deepseek/openai).",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Modelo do provedor (padrao: primeiro da config).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="max_tokens da geracao.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    com_ragas = bool(args.com_ragas)
    com_geracao = bool(args.com_geracao) or com_ragas

    casos = _selecionar_casos(args.casos)
    if not casos:
        print("Nenhum caso corresponde ao filtro.", file=sys.stderr)
        return 2

    from antt_rag_unified import (
        carregar_vectorstore_com_provider,
        gerar_resposta,
        pesquisar_documentos,
    )
    from retrieval_hibrido import limpar_cache_indice_lexical

    limpar_cache_indice_lexical()
    print(f"Carregando indice (embeddings={args.embeddings})...")
    try:
        vectorstore = carregar_vectorstore_com_provider(args.embeddings)
    except Exception as exc:  # noqa: BLE001
        print(f"Falha ao carregar indice: {exc}", file=sys.stderr)
        return 1

    llm = None
    nome_modelo = ""
    if com_geracao:
        try:
            llm, nome_modelo = criar_llm_geracao(
                args.llm_provider,
                args.llm_model,
                args.max_tokens,
            )
            print(f"LLM geracao: {nome_modelo}")
        except Exception as exc:  # noqa: BLE001
            print(f"Falha ao criar LLM: {exc}", file=sys.stderr)
            return 1

    resultados: List[Dict[str, object]] = []
    amostras_ragas: List[Dict[str, object]] = []

    for indice, caso in enumerate(casos, start=1):
        print(f"[{indice}/{len(casos)}] {caso.identificador}")
        inicio_ret = time.time()
        docs = pesquisar_documentos(
            caso.pergunta,
            vectorstore,
            k=args.k,
            embedding_provider=args.embeddings,
        )
        tempo_ret = time.time() - inicio_ret
        metrica = avaliar_caso(caso, docs)
        metrica["tempo_retrieval_s"] = round(tempo_ret, 3)
        metrica["tempo_s"] = metrica["tempo_retrieval_s"]

        if com_geracao and llm is not None:
            inicio_ger = time.time()
            try:
                resposta, modelo_usado = gerar_resposta(
                    caso.pergunta,
                    docs,
                    llm,
                    modelo_usado=nome_modelo,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"    ERRO geracao: {exc}", file=sys.stderr)
                resposta = ""
                modelo_usado = nome_modelo
            tempo_ger = time.time() - inicio_ger
            metrica["tempo_geracao_s"] = round(tempo_ger, 3)
            metrica["modelo"] = modelo_usado
            metrica["resposta"] = resposta
            metrica["resposta_preview"] = (
                resposta[:240].replace("\n", " ") if resposta else ""
            )
            metrica.update(avaliar_completude_resposta(caso, resposta))
            amostras_ragas.append(
                {
                    "question": caso.pergunta,
                    "answer": resposta,
                    "contexts": list(metrica["contextos"]),
                    "ground_truth": str(metrica["referencia"]),
                }
            )
            print(
                f"    cobertura={metrica['cobertura']:.0%} "
                f"completude_resp={float(metrica['completude_resposta']):.0%} "
                f"estrut={metrica['n_estruturada']} "
                f"retr={tempo_ret:.2f}s ger={tempo_ger:.2f}s"
            )
        else:
            print(
                f"    cobertura={metrica['cobertura']:.0%} "
                f"estrut={metrica['n_estruturada']} "
                f"ausentes={metrica['ausentes']} "
                f"tempo={tempo_ret:.2f}s"
            )

        # Contextos e referencia ficam no JSON completo; o preview do MD e curto.
        resultados.append(metrica)

    if com_ragas:
        print("Executando metricas RAGAS (LLM juiz)...")
        try:
            if llm is None:
                raise RuntimeError("LLM juiz nao inicializado.")
            llm_w, emb_w = criar_wrappers_ragas(llm, embedding_provider="local")
            inicio_ragas = time.time()
            scores = avaliar_com_ragas(amostras_ragas, llm_w, emb_w)
            tempo_ragas = time.time() - inicio_ragas
            for metrica, score in zip(resultados, scores):
                metrica.update(score)
            print(f"RAGAS concluido em {tempo_ragas:.1f}s")
            for metrica in resultados:
                partes = []
                for nome in METRICAS_RAGAS:
                    valor = metrica.get(nome)
                    if isinstance(valor, (int, float)):
                        partes.append(f"{nome}={float(valor):.2f}")
                if partes:
                    print(f"    {metrica['identificador']}: " + " ".join(partes))
        except Exception as exc:  # noqa: BLE001
            print(f"Falha no RAGAS: {exc}", file=sys.stderr)
            return 1

    os.makedirs(args.saida, exist_ok=True)
    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta: Dict[str, object] = {
        "data": datetime.now().isoformat(timespec="seconds"),
        "k": args.k,
        "embeddings": args.embeddings,
        "com_geracao": com_geracao,
        "com_ragas": com_ragas,
        "llm_provider": args.llm_provider if com_geracao else None,
        "llm_model": nome_modelo if com_geracao else None,
        "max_tokens": args.max_tokens if com_geracao else None,
    }
    caminho_md = os.path.join(args.saida, f"retrieval_{carimbo}.md")
    caminho_json = os.path.join(args.saida, f"retrieval_{carimbo}.json")

    with open(caminho_md, "w", encoding="utf-8") as f:
        f.write(gerar_relatorio_markdown(resultados, meta))
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(
            {"meta": meta, "resultados": resultados},
            f,
            ensure_ascii=True,
            indent=2,
        )

    print(f"\nRelatorio: {caminho_md}")
    print(f"JSON: {caminho_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
