"""
Harness de avaliacao multi-dominio do retrieval.

Mede, sem chamar o LLM de geracao:
  - cobertura dos valores esperados nos trechos recuperados (recall de gabarito)
  - acerto do documento-alvo no top-k
  - fracao de chunks estruturados vs OCR no contexto
  - latencia da consulta

Uso:
    python avaliar_retrieval.py
    python avaliar_retrieval.py --k 16 --casos pavimento_generico,iri_principal
    python avaliar_retrieval.py --saida relatorios_avaliacao
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
from typing import Dict, List, Optional, Sequence, Tuple


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
        contexto: Texto concatenado dos chunks recuperados.
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

    contextos = []
    hit_doc = False
    n_estruturada = 0
    n_ocr = 0

    for doc in documentos:
        contextos.append(getattr(doc, "page_content", "") or "")
        meta = getattr(doc, "metadata", {}) or {}
        fonte = classificar_fonte_qualidade(contextos[-1], meta)
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
    }


def gerar_relatorio_markdown(
    resultados: Sequence[Dict[str, object]],
    metadados_execucao: Dict[str, object],
) -> str:
    """Monta o relatorio markdown da execucao."""
    linhas = [
        "# Avaliacao de retrieval multi-dominio",
        "",
        f"- Data: {metadados_execucao.get('data', '')}",
        f"- k: {metadados_execucao.get('k', '')}",
        f"- Embeddings: {metadados_execucao.get('embeddings', '')}",
        f"- Casos: {len(resultados)}",
        "",
        "## Resumo",
        "",
        "| Caso | Dominio | Cobertura | Hit doc | Estruturados | OCR | Tempo (s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    coberturas = []
    for item in resultados:
        cob = float(item["cobertura"])
        coberturas.append(cob)
        hit = item["hit_documento"]
        hit_txt = "sim" if hit else ("n/a" if hit is None else "nao")
        linhas.append(
            f"| {item['identificador']} | {item['dominio']} | {cob:.0%} | "
            f"{hit_txt} | {item['n_estruturada']} | {item['n_ocr']} | "
            f"{item.get('tempo_s', 0):.2f} |"
        )

    media = sum(coberturas) / len(coberturas) if coberturas else 0.0
    hits = [r for r in resultados if r["hit_documento"] is True]
    elegiveis = [r for r in resultados if r["hit_documento"] is not None]
    taxa_hit = (len(hits) / len(elegiveis)) if elegiveis else 0.0

    linhas.extend(
        [
            "",
            f"**Cobertura media:** {media:.0%}",
            f"**Taxa de hit do documento-alvo:** {taxa_hit:.0%} "
            f"({len(hits)}/{len(elegiveis)})",
            "",
            "## Detalhes",
            "",
        ]
    )

    for item in resultados:
        linhas.extend(
            [
                f"### {item['identificador']}",
                "",
                f"- Dominio: {item['dominio']}",
                f"- Cobertura: {float(item['cobertura']):.0%}",
                f"- Encontrados: {', '.join(item['encontrados']) or '-'}",
                f"- Ausentes: {', '.join(item['ausentes']) or '-'}",
                f"- Chunks estruturados: {item['n_estruturada']} "
                f"(meta minima ok: {item['ok_estruturados']})",
                "",
            ]
        )

    return "\n".join(linhas)


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
    args = parser.parse_args(list(argv) if argv is not None else None)

    casos = CASOS_AVALIACAO
    if args.casos.strip():
        selecionados = {c.strip() for c in args.casos.split(",") if c.strip()}
        casos = tuple(c for c in CASOS_AVALIACAO if c.identificador in selecionados)
        if not casos:
            print("Nenhum caso corresponde ao filtro.", file=sys.stderr)
            return 2

    from antt_rag_unified import (
        carregar_vectorstore_com_provider,
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

    resultados: List[Dict[str, object]] = []
    for indice, caso in enumerate(casos, start=1):
        print(f"[{indice}/{len(casos)}] {caso.identificador}")
        inicio = time.time()
        docs = pesquisar_documentos(
            caso.pergunta,
            vectorstore,
            k=args.k,
            embedding_provider=args.embeddings,
        )
        tempo = time.time() - inicio
        metrica = avaliar_caso(caso, docs)
        metrica["tempo_s"] = round(tempo, 3)
        resultados.append(metrica)
        print(
            f"    cobertura={metrica['cobertura']:.0%} "
            f"estrut={metrica['n_estruturada']} "
            f"ausentes={metrica['ausentes']} "
            f"tempo={tempo:.2f}s"
        )

    os.makedirs(args.saida, exist_ok=True)
    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {
        "data": datetime.now().isoformat(timespec="seconds"),
        "k": args.k,
        "embeddings": args.embeddings,
    }
    caminho_md = os.path.join(args.saida, f"retrieval_{carimbo}.md")
    caminho_json = os.path.join(args.saida, f"retrieval_{carimbo}.json")

    with open(caminho_md, "w", encoding="utf-8") as f:
        f.write(gerar_relatorio_markdown(resultados, meta))
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "resultados": resultados}, f, ensure_ascii=True, indent=2)

    print(f"\nRelatorio: {caminho_md}")
    print(f"JSON: {caminho_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
