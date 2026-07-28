"""
Comparacao A/B de modelos de embedding open source em CPU.

Constroi indices FAISS temporarios sobre o mesmo corpus (documentos
criticos + distratores), executa as perguntas de referencia (IRI, prazos)
e mede:
  - cobertura dos valores esperados nos trechos recuperados
  - acerto do documento-alvo no top-k
  - tempo de carga, indexacao e consulta

A unica variavel e o modelo de embedding. O LLM nao entra no teste.

Uso:
    python comparar_embeddings.py
    python comparar_embeddings.py --modelos minilm,e5-small
    python comparar_embeddings.py --k 10 --distratores 40
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from antt_rag_unified import (
    _dividir_por_estrutura,
    _mesclar_tabelas_auxiliares,
)
from comparar_modelos import CASOS_REFERENCIA, CasoReferencia


# ---------------------------------------------------------------------------
# Modelos sob teste
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeloEmbedding:
    """
    Configuracao de um modelo de embedding para o experimento.

    Attributes:
        chave: Identificador curto na linha de comando.
        repo_id: Identificador HuggingFace / sentence-transformers.
        query_prefix: Prefixo aplicado a consultas (ex.: modelos E5).
        doc_prefix: Prefixo aplicado a documentos.
        trust_remote_code: Necessario para alguns repositorios.
        descricao: Texto curto para o relatorio.
    """

    chave: str
    repo_id: str
    query_prefix: str
    doc_prefix: str
    trust_remote_code: bool
    descricao: str


MODELOS: Dict[str, ModeloEmbedding] = {
    "minilm": ModeloEmbedding(
        chave="minilm",
        repo_id="paraphrase-multilingual-MiniLM-L12-v2",
        query_prefix="",
        doc_prefix="",
        trust_remote_code=False,
        descricao="Baseline atual do sistema (leve, 384 dims)",
    ),
    "e5-small": ModeloEmbedding(
        chave="e5-small",
        repo_id="intfloat/multilingual-e5-small",
        query_prefix="query: ",
        doc_prefix="passage: ",
        trust_remote_code=False,
        descricao="E5 multilingual small (bom custo/beneficio em CPU)",
    ),
    "bge-m3": ModeloEmbedding(
        chave="bge-m3",
        repo_id="BAAI/bge-m3",
        query_prefix="",
        doc_prefix="",
        trust_remote_code=True,
        descricao="BGE-M3 (forte em retrieval multilingue; ~2.2 GB)",
    ),
    "e5-base": ModeloEmbedding(
        chave="e5-base",
        repo_id="intfloat/multilingual-e5-base",
        query_prefix="query: ",
        doc_prefix="passage: ",
        trust_remote_code=False,
        descricao="E5 multilingual base (meio-termo qualidade/peso)",
    ),
}

# BGE-M3 fica opcional: o download exige ~2.2 GB livres. O padrao cobre o
# baseline atual e o melhor candidato leve para CPU.
MODELOS_PADRAO: Tuple[str, ...] = ("minilm", "e5-small", "e5-base")

# Documentos que contem os gabaritos das perguntas de referencia.
DOCS_CRITICOS: Tuple[Tuple[str, str, str, str], ...] = (
    ("INM", "Instrucao Normativa", "34", "2024"),
    ("INM", "Instrucao Normativa", "18", "2023"),
    ("INM", "Instrucao Normativa", "33", "2024"),
    ("RES", "Resolucao", "6000", "2022"),
)

# Documento esperado por caso de referencia (tipo, numero, ano).
DOC_ESPERADO_POR_CASO: Dict[str, Tuple[str, str, str]] = {
    "iri_principal": ("INM", "34", "2024"),
    "iri_marginal": ("INM", "34", "2024"),
    "dadm_fwd": ("INM", "34", "2024"),
    "ifi_manutencao": ("INM", "34", "2024"),
    "cronograma_revisao": ("INM", "18", "2023"),
    "prazos_eef": ("INM", "33", "2024"),
    "recurso_admissibilidade": ("INM", "33", "2024"),
}


# ---------------------------------------------------------------------------
# Embeddings LangChain sobre SentenceTransformer
# ---------------------------------------------------------------------------


class SentenceTransformerEmbeddings(Embeddings):
    """
    Adaptador LangChain para sentence-transformers com prefixos opcionais.

    Normaliza os vetores (L2) para que a distancia do FAISS se aproxime da
    similaridade de cosseno, padrao recomendado para E5 e BGE.
    """

    def __init__(
        self,
        repo_id: str,
        query_prefix: str = "",
        doc_prefix: str = "",
        trust_remote_code: bool = False,
    ) -> None:
        """
        Args:
            repo_id: Identificador do modelo no HuggingFace.
            query_prefix: Prefixo das consultas.
            doc_prefix: Prefixo dos documentos.
            trust_remote_code: Permite codigo remoto do repositorio.

        Raises:
            RuntimeError: Se o modelo nao puder ser carregado.
        """
        from sentence_transformers import SentenceTransformer

        self.repo_id = repo_id
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        inicio = time.perf_counter()
        try:
            self.model = SentenceTransformer(
                repo_id,
                device="cpu",
                trust_remote_code=trust_remote_code,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Falha ao carregar embedding '{repo_id}': {exc}"
            ) from exc
        self.tempo_carga_s = time.perf_counter() - inicio

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Gera embeddings para documentos (com prefixo de passagem, se houver)."""
        if not texts:
            return []
        preparados = [
            f"{self.doc_prefix}{texto}" if self.doc_prefix else texto
            for texto in texts
        ]
        vetores = self.model.encode(
            preparados,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [vetor.tolist() for vetor in vetores]

    def embed_query(self, text: str) -> List[float]:
        """Gera embedding para uma consulta (com prefixo de query, se houver)."""
        preparado = f"{self.query_prefix}{text}" if self.query_prefix else text
        vetor = self.model.encode(
            [preparado],
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0]
        return vetor.tolist()


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def _normalizar_numero(numero: str) -> str:
    """Remove zeros a esquerda do numero documental."""
    texto = str(numero).strip()
    if texto.isdigit():
        return str(int(texto))
    return texto


def carregar_catalogo(caminho: str = "relatorio_documentos.json") -> List[dict]:
    """
    Le o catalogo de documentos.

    Args:
        caminho: Caminho do JSON gerado por gerar_relatorio.

    Returns:
        Lista de entradas do catalogo.
    """
    with open(caminho, "r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)
    if not isinstance(dados, list):
        raise ValueError(f"Catalogo invalido em {caminho}")
    return dados


def carregar_documento_leve(
    caminho: str,
    tipo: str,
    nome_tipo: str,
    numero: str,
    ano: str,
) -> List[Document]:
    """
    Carrega um markdown em chunks sem OCR de imagens externas.

    O OCR tornaria o experimento inviavel (minutos por distrator). As tabelas
    auxiliares locais continuam sendo mescladas, o que cobre IRI/prazos.

    Args:
        caminho: Arquivo .md do documento.
        tipo: Sigla documental.
        nome_tipo: Nome por extenso.
        numero: Numero do documento.
        ano: Ano do documento.

    Returns:
        Lista de chunks Document.
    """
    try:
        with open(caminho, "r", encoding="utf-8") as arquivo:
            conteudo = arquivo.read()
    except OSError:
        return []

    if not conteudo.strip():
        return []

    conteudo = _mesclar_tabelas_auxiliares(conteudo, tipo, numero, ano)
    partes = _dividir_por_estrutura(conteudo)
    documentos: List[Document] = []
    for indice, texto in enumerate(partes):
        documentos.append(
            Document(
                page_content=texto,
                metadata={
                    "tipo_documento": tipo,
                    "nome_tipo": nome_tipo,
                    "numero": str(numero).zfill(8),
                    "ano": str(ano),
                    "caminho": caminho,
                    "chunk": indice + 1,
                    "total_chunks": len(partes),
                },
            )
        )
    return documentos


def montar_corpus(
    num_distratores: int,
    catalogo: Sequence[dict],
) -> List[Document]:
    """
    Monta o corpus do experimento: docs criticos + distratores do catalogo.

    Args:
        num_distratores: Quantidade maxima de documentos adicionais.
        catalogo: Entradas de relatorio_documentos.json.

    Returns:
        Lista de chunks Document prontos para indexacao.

    Raises:
        FileNotFoundError: Se algum documento critico estiver ausente.
    """
    chunks: List[Document] = []
    criticos_ids = {
        (tipo, _normalizar_numero(numero), ano)
        for tipo, _nome, numero, ano in DOCS_CRITICOS
    }

    for tipo, nome_tipo, numero, ano in DOCS_CRITICOS:
        caminho = _localizar_arquivo(catalogo, tipo, numero, ano)
        if caminho is None or not os.path.isfile(caminho):
            raise FileNotFoundError(
                f"Documento critico ausente: {tipo} {numero}/{ano}"
            )
        docs = carregar_documento_leve(caminho, tipo, nome_tipo, numero, ano)
        chunks.extend(docs)

    distratores_carregados = 0
    for entrada in catalogo:
        if distratores_carregados >= num_distratores:
            break
        tipo = str(entrada.get("tipo", ""))
        numero = _normalizar_numero(str(entrada.get("numero", "")))
        ano = str(entrada.get("ano", ""))
        if (tipo, numero, ano) in criticos_ids:
            continue
        caminho = entrada.get("arquivo_md") or entrada.get("caminho")
        if not caminho or not os.path.isfile(caminho):
            continue
        nome_tipo = str(entrada.get("nome_tipo", tipo))
        docs = carregar_documento_leve(caminho, tipo, nome_tipo, numero, ano)
        if not docs:
            continue
        chunks.extend(docs)
        distratores_carregados += 1

    return chunks


def _localizar_arquivo(
    catalogo: Sequence[dict],
    tipo: str,
    numero: str,
    ano: str,
) -> Optional[str]:
    """Localiza o caminho .md de um documento no catalogo."""
    alvo_num = _normalizar_numero(numero)
    for entrada in catalogo:
        if str(entrada.get("tipo", "")) != tipo:
            continue
        if str(entrada.get("ano", "")) != str(ano):
            continue
        if _normalizar_numero(str(entrada.get("numero", ""))) != alvo_num:
            continue
        caminho = entrada.get("arquivo_md") or entrada.get("caminho")
        if isinstance(caminho, str):
            return caminho
    return None


# ---------------------------------------------------------------------------
# Avaliacao
# ---------------------------------------------------------------------------


def normalizar_texto(texto: str) -> str:
    """Normaliza texto para comparacao de cobertura."""
    decomposto = unicodedata.normalize("NFKD", texto)
    sem_acento = "".join(c for c in decomposto if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", sem_acento).strip().lower()


def cobertura_esperados(texto: str, esperados: Sequence[str]) -> Tuple[float, List[str]]:
    """
    Calcula a fracao dos valores esperados presentes no texto.

    Returns:
        Tupla (cobertura 0-1, lista dos valores encontrados).
    """
    if not esperados:
        return 1.0, []
    base = normalizar_texto(texto)
    achados = [item for item in esperados if normalizar_texto(item) in base]
    return len(achados) / len(esperados), achados


def documento_no_top(
    documentos: Sequence[Document],
    tipo: str,
    numero: str,
    ano: str,
) -> bool:
    """Indica se o documento-alvo aparece entre os trechos recuperados."""
    alvo_num = _normalizar_numero(numero)
    padrao_nome = re.compile(
        rf"{re.escape(tipo.upper())}-0*{re.escape(alvo_num)}-{re.escape(str(ano))}"
    )
    for doc in documentos:
        meta = doc.metadata
        tipo_doc = str(
            meta.get("tipo_documento", meta.get("tipo", ""))
        ).upper()
        num_ok = _normalizar_numero(str(meta.get("numero", ""))) == alvo_num
        ano_ok = str(meta.get("ano", "")) == str(ano)
        tipo_ok = tipo_doc == tipo.upper()
        if num_ok and ano_ok and tipo_ok:
            return True
        caminho = str(meta.get("caminho", ""))
        if padrao_nome.search(os.path.basename(caminho).upper()):
            return True
    return False


@dataclass
class ResultadoCaso:
    """Resultado de um caso para um modelo."""

    identificador: str
    cobertura: float
    achados: List[str]
    hit_documento: bool
    tempo_consulta_s: float
    trechos: int


@dataclass
class ResultadoModelo:
    """Agregado de um modelo no experimento."""

    chave: str
    repo_id: str
    descricao: str
    tempo_carga_s: float
    tempo_indexacao_s: float
    num_chunks: int
    casos: List[ResultadoCaso]
    erro: Optional[str] = None

    @property
    def cobertura_media(self) -> Optional[float]:
        """Media da cobertura dos valores esperados, ou None se falhou."""
        if self.erro or not self.casos:
            return None
        return sum(caso.cobertura for caso in self.casos) / len(self.casos)

    @property
    def hit_documento_media(self) -> Optional[float]:
        """Taxa de acerto do documento-alvo no top-k."""
        if self.erro or not self.casos:
            return None
        return sum(1 for caso in self.casos if caso.hit_documento) / len(self.casos)

    @property
    def tempo_consulta_medio_s(self) -> Optional[float]:
        """Tempo medio de consulta em segundos."""
        if self.erro or not self.casos:
            return None
        return sum(caso.tempo_consulta_s for caso in self.casos) / len(self.casos)


def avaliar_modelo(
    modelo: ModeloEmbedding,
    corpus: Sequence[Document],
    casos: Sequence[CasoReferencia],
    k: int,
) -> ResultadoModelo:
    """
    Indexa o corpus com o modelo e avalia as perguntas de referencia.

    Args:
        modelo: Configuracao do embedding.
        corpus: Chunks a indexar.
        casos: Perguntas de referencia.
        k: Numero de trechos recuperados por pergunta.

    Returns:
        Resultado agregado do modelo.
    """
    try:
        embeddings = SentenceTransformerEmbeddings(
            repo_id=modelo.repo_id,
            query_prefix=modelo.query_prefix,
            doc_prefix=modelo.doc_prefix,
            trust_remote_code=modelo.trust_remote_code,
        )
    except Exception as exc:
        return ResultadoModelo(
            chave=modelo.chave,
            repo_id=modelo.repo_id,
            descricao=modelo.descricao,
            tempo_carga_s=0.0,
            tempo_indexacao_s=0.0,
            num_chunks=len(corpus),
            casos=[],
            erro=str(exc),
        )

    diretorio = tempfile.mkdtemp(prefix=f"emb_{modelo.chave}_")
    try:
        inicio_idx = time.perf_counter()
        vectorstore = FAISS.from_documents(list(corpus), embeddings)
        tempo_idx = time.perf_counter() - inicio_idx
        vectorstore.save_local(diretorio)

        resultados_casos: List[ResultadoCaso] = []
        for caso in casos:
            inicio_q = time.perf_counter()
            docs = vectorstore.similarity_search(caso.pergunta, k=k)
            tempo_q = time.perf_counter() - inicio_q
            texto = "\n".join(doc.page_content for doc in docs)
            cobertura, achados = cobertura_esperados(texto, caso.esperados)
            alvo = DOC_ESPERADO_POR_CASO.get(caso.identificador)
            hit = False
            if alvo is not None:
                hit = documento_no_top(docs, alvo[0], alvo[1], alvo[2])
            resultados_casos.append(
                ResultadoCaso(
                    identificador=caso.identificador,
                    cobertura=cobertura,
                    achados=achados,
                    hit_documento=hit,
                    tempo_consulta_s=tempo_q,
                    trechos=len(docs),
                )
            )

        return ResultadoModelo(
            chave=modelo.chave,
            repo_id=modelo.repo_id,
            descricao=modelo.descricao,
            tempo_carga_s=embeddings.tempo_carga_s,
            tempo_indexacao_s=tempo_idx,
            num_chunks=len(corpus),
            casos=resultados_casos,
        )
    except Exception as exc:
        return ResultadoModelo(
            chave=modelo.chave,
            repo_id=modelo.repo_id,
            descricao=modelo.descricao,
            tempo_carga_s=getattr(embeddings, "tempo_carga_s", 0.0),
            tempo_indexacao_s=0.0,
            num_chunks=len(corpus),
            casos=[],
            erro=str(exc),
        )
    finally:
        shutil.rmtree(diretorio, ignore_errors=True)


# ---------------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------------


def _fmt_pct(valor: Optional[float]) -> str:
    """Formata fracao como percentual."""
    if valor is None:
        return "n/d"
    return f"{100.0 * valor:.1f}%"


def _fmt_s(valor: Optional[float]) -> str:
    """Formata segundos."""
    if valor is None:
        return "n/d"
    return f"{valor:.2f}s"


def montar_relatorio(
    resultados: Sequence[ResultadoModelo],
    casos: Sequence[CasoReferencia],
    k: int,
    num_distratores: int,
) -> str:
    """
    Monta o relatorio Markdown do experimento.

    Args:
        resultados: Resultados por modelo.
        casos: Casos avaliados.
        k: Top-k usado.
        num_distratores: Distratores incluidos no corpus.

    Returns:
        Texto Markdown do relatorio.
    """
    linhas: List[str] = [
        "# Comparacao de modelos de embedding (CPU)",
        "",
        f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Parametros",
        "",
        f"- Top-k: {k}",
        f"- Distratores no corpus: {num_distratores}",
        f"- Documentos criticos: {', '.join(f'{t} {n}/{a}' for t, _, n, a in DOCS_CRITICOS)}",
        f"- Casos: {', '.join(caso.identificador for caso in casos)}",
        "",
        "O escore de cobertura verifica se os valores de referencia aparecem",
        "nos trechos recuperados (nao envolve LLM). Hit de documento indica",
        "se o arquivo correto entrou no top-k.",
        "",
        "## Resumo",
        "",
        "| Modelo | Cobertura media | Hit documento | Carga | Indexacao | Consulta media | Chunks |",
        "|--------|-----------------|---------------|-------|-----------|----------------|--------|",
    ]

    for item in resultados:
        if item.erro:
            linhas.append(
                f"| {item.chave} | erro | erro | {_fmt_s(item.tempo_carga_s)} | "
                f"- | - | {item.num_chunks} |"
            )
        else:
            linhas.append(
                f"| {item.chave} | {_fmt_pct(item.cobertura_media)} | "
                f"{_fmt_pct(item.hit_documento_media)} | "
                f"{_fmt_s(item.tempo_carga_s)} | "
                f"{_fmt_s(item.tempo_indexacao_s)} | "
                f"{_fmt_s(item.tempo_consulta_medio_s)} | "
                f"{item.num_chunks} |"
            )

    linhas.extend(["", "## Detalhe por caso", ""])
    for caso in casos:
        linhas.append(f"### {caso.identificador}")
        linhas.append("")
        linhas.append(f"- Pergunta: {caso.pergunta}")
        linhas.append(f"- Origem: {caso.origem}")
        linhas.append(f"- Esperados: {', '.join(caso.esperados)}")
        linhas.append("")
        linhas.append("| Modelo | Cobertura | Achados | Hit doc | Tempo |")
        linhas.append("|--------|-----------|---------|---------|-------|")
        for item in resultados:
            if item.erro:
                linhas.append(f"| {item.chave} | erro | - | - | - |")
                continue
            mapa = {c.identificador: c for c in item.casos}
            r = mapa.get(caso.identificador)
            if r is None:
                linhas.append(f"| {item.chave} | n/d | - | - | - |")
                continue
            linhas.append(
                f"| {item.chave} | {_fmt_pct(r.cobertura)} | "
                f"{', '.join(r.achados) if r.achados else '-'} | "
                f"{'sim' if r.hit_documento else 'nao'} | "
                f"{_fmt_s(r.tempo_consulta_s)} |"
            )
        linhas.append("")

    linhas.extend(
        [
            "## Erros de carga/indexacao",
            "",
        ]
    )
    houve_erro = False
    for item in resultados:
        if item.erro:
            houve_erro = True
            linhas.append(f"- **{item.chave}** (`{item.repo_id}`): {item.erro}")
    if not houve_erro:
        linhas.append("- Nenhum.")

    linhas.extend(
        [
            "",
            "## Observacoes",
            "",
            "- Trocar o embedding em producao exige reindexacao completa.",
            "- E5 usa prefixos `query:` / `passage:`; o adaptador ja aplica.",
            "- BGE-M3 e mais pesado em CPU; avalie latencia antes de adotar.",
            "- Este teste isola a recuperacao; a qualidade final da resposta",
            "  ainda depende do LLM e do prompt.",
            "",
        ]
    )
    return "\n".join(linhas)


def recomendar(resultados: Sequence[ResultadoModelo]) -> str:
    """
    Escolhe o melhor modelo pelo criterio cobertura, depois hit, depois tempo.

    Returns:
        Frase de recomendacao.
    """
    validos = [item for item in resultados if item.erro is None and item.casos]
    if not validos:
        return "Nenhum modelo concluiu o teste; verifique a carga dos pesos."

    def chave_ord(item: ResultadoModelo) -> Tuple[float, float, float]:
        cobertura = item.cobertura_media or 0.0
        hit = item.hit_documento_media or 0.0
        # Menor tempo de consulta e melhor (inverter sinal)
        tempo = -(item.tempo_consulta_medio_s or 999.0)
        return (cobertura, hit, tempo)

    melhor = max(validos, key=chave_ord)
    baseline = next((item for item in validos if item.chave == "minilm"), None)
    extra = ""
    if baseline and melhor.chave != "minilm":
        ganho = (melhor.cobertura_media or 0) - (baseline.cobertura_media or 0)
        extra = f" Ganho de cobertura vs MiniLM: {100 * ganho:+.1f} p.p."
    return (
        f"Recomendacao deste experimento: **{melhor.chave}** "
        f"(`{melhor.repo_id}`) - cobertura {_fmt_pct(melhor.cobertura_media)}, "
        f"hit documento {_fmt_pct(melhor.hit_documento_media)}, "
        f"consulta media {_fmt_s(melhor.tempo_consulta_medio_s)}."
        f"{extra}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Interpreta a linha de comando."""
    parser = argparse.ArgumentParser(
        description="Compara modelos de embedding open source em CPU."
    )
    parser.add_argument(
        "--modelos",
        default=",".join(MODELOS_PADRAO),
        help=f"Lista separada por virgula. Opcoes: {', '.join(MODELOS)}",
    )
    parser.add_argument(
        "--perguntas",
        default="",
        help="Identificadores de caso separados por virgula (padrao: todos).",
    )
    parser.add_argument("--k", type=int, default=10, help="Trechos por consulta.")
    parser.add_argument(
        "--distratores",
        type=int,
        default=40,
        help="Documentos adicionais no corpus alem dos criticos.",
    )
    parser.add_argument(
        "--saida",
        default="",
        help="Caminho do relatorio Markdown (padrao: relatorios_comparacao/).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Ponto de entrada do experimento.

    Returns:
        Codigo de saida do processo (0 sucesso, 1 erro).
    """
    args = parse_args(argv)
    chaves = [item.strip() for item in args.modelos.split(",") if item.strip()]
    desconhecidos = [chave for chave in chaves if chave not in MODELOS]
    if desconhecidos:
        print(f"Modelos desconhecidos: {', '.join(desconhecidos)}", file=sys.stderr)
        print(f"Opcoes: {', '.join(MODELOS)}", file=sys.stderr)
        return 1

    if args.perguntas.strip():
        ids = {item.strip() for item in args.perguntas.split(",") if item.strip()}
        casos = [caso for caso in CASOS_REFERENCIA if caso.identificador in ids]
        if not casos:
            print("Nenhuma pergunta valida selecionada.", file=sys.stderr)
            return 1
    else:
        casos = list(CASOS_REFERENCIA)

    print("Carregando catalogo e corpus...")
    catalogo = carregar_catalogo()
    corpus = montar_corpus(args.distratores, catalogo)
    print(
        f"Corpus: {len(corpus)} chunks "
        f"(criticos + ate {args.distratores} distratores)"
    )

    resultados: List[ResultadoModelo] = []
    for chave in chaves:
        modelo = MODELOS[chave]
        print(f"\n=== Avaliando {chave} ({modelo.repo_id}) ===")
        resultado = avaliar_modelo(modelo, corpus, casos, args.k)
        resultados.append(resultado)
        if resultado.erro:
            print(f"ERRO: {resultado.erro}")
        else:
            print(
                f"Cobertura={_fmt_pct(resultado.cobertura_media)} "
                f"Hit={_fmt_pct(resultado.hit_documento_media)} "
                f"Carga={_fmt_s(resultado.tempo_carga_s)} "
                f"Index={_fmt_s(resultado.tempo_indexacao_s)} "
                f"Consulta={_fmt_s(resultado.tempo_consulta_medio_s)}"
            )

    relatorio = montar_relatorio(resultados, casos, args.k, args.distratores)
    recomendacao = recomendar(resultados)
    relatorio = relatorio + "\n## Conclusao\n\n" + recomendacao + "\n"

    if args.saida:
        caminho_saida = args.saida
    else:
        os.makedirs("relatorios_comparacao", exist_ok=True)
        carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
        caminho_saida = os.path.join(
            "relatorios_comparacao", f"embeddings_{carimbo}.md"
        )

    with open(caminho_saida, "w", encoding="utf-8") as arquivo:
        arquivo.write(relatorio)

    print("\n" + recomendacao)
    print(f"Relatorio: {caminho_saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
