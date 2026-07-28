"""
Recuperacao hibrida e contextualizacao de chunks para o RAG-ANTT.

Este modulo concentra tecnicas de recuperacao independentes de dominio, isto e,
sem listas fixas de parametros, siglas ou assuntos regulatorios. Todas as
decisoes derivam da estrutura do proprio documento (titulos, anexos, tabelas) e
de estatisticas do corpus.

Componentes:

1. Contextualizacao de chunk (`contextualizar_chunks`)
   Prefixa cada chunk com uma trilha estrutural ("breadcrumb") derivada do
   markdown do documento: identificador da norma, ementa, hierarquia de
   titulos/anexos e cabecalho da tabela vigente. Resolve a assimetria de
   vocabulario dos chunks tabulares, que contem apenas rotulos curtos e numeros
   e por isso ficam semanticamente distantes de perguntas em linguagem natural.

2. Indice lexical BM25 (`IndiceLexical`)
   Busca esparsa sobre os mesmos chunks do indice vetorial. Recupera de forma
   confiavel siglas, numeros, unidades e referencias normativas, onde a busca
   densa e fraca.

3. Fusao de ranqueamentos (`fundir_rrf`)
   Reciprocal Rank Fusion entre as listas densa e lexical, sem necessidade de
   calibrar escalas de score heterogeneas.

4. Expansao por documento-pai (`expandir_com_irmaos_tabulares`)
   Quando um documento e considerado relevante, seus chunks irmaos tabulares
   (tipicamente anexos com limites numericos) sao anexados ao contexto, mesmo
   que isoladamente nao tenham similaridade com a pergunta.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi

    BM25_DISPONIVEL = True
except ImportError:  # pragma: no cover - ambiente sem a dependencia opcional
    BM25Okapi = None
    BM25_DISPONIVEL = False
    logger.warning(
        "rank_bm25 nao instalado: busca lexical desabilitada "
        "(instale com: pip install rank_bm25)"
    )

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - fallback para versoes antigas
    from langchain.schema import Document


# ---------------------------------------------------------------------------
# Normalizacao e tokenizacao
# ---------------------------------------------------------------------------

# Stopwords do portugues sem acento. Lista curta e proposital: em corpus
# normativo, remover palavras funcionais melhora o BM25 sem descartar termos
# tecnicos.
_STOPWORDS: Set[str] = {
    "a", "ao", "aos", "as", "ate", "com", "como", "da", "das", "de", "dela",
    "delas", "dele", "deles", "do", "dos", "e", "ela", "elas", "ele", "eles",
    "em", "entre", "era", "eram", "essa", "essas", "esse", "esses", "esta",
    "estas", "este", "estes", "eu", "foi", "for", "foram", "ha", "isso",
    "isto", "ja", "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu", "meus",
    "muito", "na", "nao", "nas", "nem", "no", "nos", "num", "numa", "o", "os",
    "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "qual", "quais",
    "quando", "que", "quem", "se", "seja", "sejam", "sem", "ser", "seu",
    "seus", "so", "sao", "sua", "suas", "tambem", "tem", "temos", "ter",
    "uma", "um", "voce", "vos",
}

# Token = sequencia alfanumerica, admitindo separador decimal interno para que
# valores como "3,5" ou "2.7" sobrevivam a tokenizacao.
_PADRAO_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")

# Separador decimal normalizado para ponto, de modo que "3,5" na norma e "3.5"
# na pergunta gerem o mesmo token.
_PADRAO_DECIMAL_VIRGULA = re.compile(r"(?<=\d),(?=\d)")


def normalizar_para_busca(texto: str) -> str:
    """
    Normaliza texto para comparacao lexical: remove acentos e aplica minusculas.

    Args:
        texto: Texto de entrada, possivelmente com acentos e maiusculas.

    Returns:
        Texto sem diacriticos, em minusculas. String vazia se a entrada nao for
        um texto valido.
    """
    if not isinstance(texto, str) or not texto:
        return ""
    decomposto = unicodedata.normalize("NFKD", texto)
    sem_acento = "".join(c for c in decomposto if not unicodedata.combining(c))
    return sem_acento.lower()


def tokenizar(texto: str) -> List[str]:
    """
    Converte texto em tokens para indexacao/consulta BM25.

    Preserva numeros decimais (normalizando virgula para ponto) e descarta
    stopwords e tokens alfabeticos de um unico caractere. Digitos isolados sao
    mantidos porque carregam informacao em tabelas normativas.

    Args:
        texto: Texto bruto do chunk ou da pergunta.

    Returns:
        Lista de tokens normalizados.
    """
    normalizado = normalizar_para_busca(texto)
    if not normalizado:
        return []
    normalizado = _PADRAO_DECIMAL_VIRGULA.sub(".", normalizado)

    tokens: List[str] = []
    for token in _PADRAO_TOKEN.findall(normalizado):
        if token in _STOPWORDS:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        tokens.append(token)
    return tokens


# ---------------------------------------------------------------------------
# Deteccao de estrutura e contextualizacao de chunks
# ---------------------------------------------------------------------------

_PADRAO_TITULO_MD = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")

# O padrao estrutural e aplicado sobre a linha normalizada (sem acentos), de
# modo que a expressao permanece em ASCII e ainda reconhece "CAPITULO",
# "SECAO", "APENDICE" grafados com acento no documento.
_PADRAO_ESTRUTURA = re.compile(
    r"^\s*("
    r"anexo\b"
    r"|apendice\b"
    r"|capitulo\s+[ivxlcdm0-9]"
    r"|secao\s+[ivxlcdm0-9]"
    r"|subsecao\s+[ivxlcdm0-9]"
    r"|titulo\s+[ivxlcdm0-9]"
    r"|tabela\s+[0-9ivxlcdm]"
    r"|quadro\s+[0-9ivxlcdm]"
    r")"
)
_PADRAO_SEPARADOR_TABELA = re.compile(r"^\s*\|?[\s:|-]*-{3,}[\s:|-]*\|?\s*$")
_PADRAO_DECIMAL = re.compile(r"\d+[.,]\d+")

# Limites de tamanho do prefixo, para que o breadcrumb ajude a recuperacao sem
# inflar o chunk nem consumir orcamento de contexto do LLM.
_MAX_CHARS_EMENTA = 220
_MAX_CHARS_TRILHA = 240
_MAX_CHARS_CABECALHO = 220
_MAX_CHARS_MARCADOR = 90
_MAX_NIVEIS_TRILHA = 4

# Densidade minima de digitos para classificar o chunk como portador de valores
# numericos, usada quando nao ha decimais explicitos.
_DENSIDADE_NUMERICA_MINIMA = 0.03


def _detectar_marcador_estrutural(linha: str) -> str:
    """
    Detecta se a linha abre um bloco estrutural (anexo, capitulo, tabela).

    A deteccao ocorre sobre a linha normalizada, mas o texto retornado preserva
    a grafia original do documento.

    Args:
        linha: Linha do markdown, com ou sem marcacao de titulo.

    Returns:
        Texto do marcador estrutural, ou string vazia se a linha nao abrir um
        bloco.
    """
    if _PADRAO_ESTRUTURA.match(normalizar_para_busca(linha)):
        return _truncar(linha.strip("# ").strip(), _MAX_CHARS_MARCADOR)
    return ""


def _linha_e_tabela(linha: str) -> bool:
    """Indica se a linha pertence a uma tabela markdown (delimitada por |)."""
    despida = linha.strip()
    return despida.startswith("|") and despida.count("|") >= 2


def chunk_tem_tabela(documento: Document) -> bool:
    """
    Indica se um chunk contem tabela, por metadado ou por inspecao do conteudo.

    A inspecao do conteudo garante funcionamento tambem em indices antigos,
    criados antes da gravacao do metadado `contem_tabelas`.

    Args:
        documento: Chunk recuperado.

    Returns:
        True se o chunk contiver estrutura tabular markdown.
    """
    try:
        if documento.metadata.get("contem_tabelas") == "Sim":
            return True
        texto = documento.page_content or ""
    except AttributeError:
        return False

    linhas = texto.split("\n")
    if any(_PADRAO_SEPARADOR_TABELA.match(ln) for ln in linhas):
        return True
    # Duas linhas delimitadas por | bastam: chunks de continuacao de tabela
    # ficam sem a linha separadora depois da divisao do documento.
    return sum(1 for ln in linhas if _linha_e_tabela(ln)) >= 2


def _densidade_numerica(texto: str) -> float:
    """Calcula a proporcao de digitos no texto, entre 0.0 e 1.0."""
    if not texto:
        return 0.0
    digitos = sum(1 for c in texto if c.isdigit())
    return digitos / len(texto)


def _tem_valores_numericos(texto: str) -> bool:
    """
    Indica se o chunk carrega valores numericos relevantes.

    Usa presenca de decimais ou densidade de digitos, sem depender de unidades
    ou parametros de um dominio especifico.

    Args:
        texto: Conteudo do chunk.

    Returns:
        True se o chunk aparentar conter dados quantitativos.
    """
    if not texto:
        return False
    if _PADRAO_DECIMAL.search(texto):
        return True
    return _densidade_numerica(texto) >= _DENSIDADE_NUMERICA_MINIMA


def _identificador_documento(metadados: Dict[str, object]) -> str:
    """
    Monta o identificador legivel da norma a partir dos metadados do documento.

    Args:
        metadados: Metadados do documento (tipo, numero, ano).

    Returns:
        Identificador como "INM 34/2024", ou string vazia se faltarem dados.
    """
    tipo = str(metadados.get("nome_tipo") or metadados.get("tipo_documento") or "")
    numero = str(metadados.get("numero") or "")
    ano = str(metadados.get("ano") or "")

    # Numeros sao gravados com zeros a esquerda no indice; a forma sem zeros e
    # a que aparece nas perguntas dos usuarios.
    numero_limpo = numero.lstrip("0") or numero

    partes = [p for p in (tipo.strip(), numero_limpo.strip()) if p]
    identificador = " ".join(partes)
    if identificador and ano:
        identificador = f"{identificador}/{ano}"
    return identificador.strip()


def _truncar(texto: str, limite: int) -> str:
    """Corta o texto no limite informado, sem cortar palavra pela metade."""
    texto = " ".join((texto or "").split())
    if len(texto) <= limite:
        return texto
    corte = texto[:limite]
    ultimo_espaco = corte.rfind(" ")
    if ultimo_espaco > limite // 2:
        corte = corte[:ultimo_espaco]
    return corte.rstrip(" ,;:.-")


def _trilha_de_hierarquia(hierarquia: Dict[int, str], estrutura: str) -> str:
    """
    Serializa a hierarquia de titulos ativa em uma trilha legivel.

    Args:
        hierarquia: Mapa de nivel de titulo markdown para texto do titulo.
        estrutura: Marcador estrutural vigente (ANEXO, CAPITULO, TABELA...).

    Returns:
        Trilha como "ANEXO I > Parametros de desempenho".
    """
    partes: List[str] = []
    for nivel in sorted(hierarquia):
        titulo = hierarquia[nivel].strip()
        if titulo and titulo not in partes:
            partes.append(titulo)
    if estrutura and estrutura not in partes:
        partes.append(estrutura.strip())
    partes = partes[-_MAX_NIVEIS_TRILHA:]
    return _truncar(" > ".join(partes), _MAX_CHARS_TRILHA)


def _analisar_chunk(
    texto: str,
    hierarquia: Dict[int, str],
    estrutura_atual: str,
    cabecalho_tabela: str,
) -> Tuple[Dict[int, str], str, str, bool]:
    """
    Percorre o chunk atualizando o estado estrutural do documento.

    A hierarquia retornada considera todos os titulos presentes no chunk. Como
    a divisao do documento corta imediatamente antes de artigos, secoes e
    anexos, os titulos aparecem no inicio do chunk e descrevem seu conteudo.

    Args:
        texto: Conteudo do chunk.
        hierarquia: Hierarquia de titulos herdada do chunk anterior.
        estrutura_atual: Marcador estrutural herdado.
        cabecalho_tabela: Cabecalho da ultima tabela vista.

    Returns:
        Tupla com (hierarquia atualizada, marcador estrutural atualizado,
        cabecalho de tabela atualizado, indicador de continuacao de tabela).
    """
    hierarquia = dict(hierarquia)
    linhas = texto.split("\n")
    tabela_abre_no_chunk = False
    primeira_linha_util = ""

    for indice, linha in enumerate(linhas):
        if not linha.strip():
            continue
        if not primeira_linha_util:
            primeira_linha_util = linha.strip()

        titulo_md = _PADRAO_TITULO_MD.match(linha)
        if titulo_md:
            nivel = len(titulo_md.group(1))
            texto_titulo = titulo_md.group(2).strip()
            # Um titulo encerra os niveis mais profundos abertos antes dele.
            for nivel_aberto in [n for n in hierarquia if n >= nivel]:
                del hierarquia[nivel_aberto]
            hierarquia[nivel] = texto_titulo
            marcador = _detectar_marcador_estrutural(texto_titulo)
            if marcador:
                estrutura_atual = marcador
            continue

        marcador = _detectar_marcador_estrutural(linha)
        if marcador:
            estrutura_atual = marcador
            continue

        # Cabecalho de tabela: linha com | imediatamente antes do separador.
        if _PADRAO_SEPARADOR_TABELA.match(linha) and indice > 0:
            for anterior in range(indice - 1, -1, -1):
                candidato = linhas[anterior].strip()
                if candidato:
                    if _linha_e_tabela(candidato):
                        cabecalho_tabela = _truncar(
                            candidato.strip("|").replace("|", "; "),
                            _MAX_CHARS_CABECALHO,
                        )
                        tabela_abre_no_chunk = True
                    break

    # Chunk que comeca com linhas de tabela sem abrir cabecalho proprio e
    # continuacao de uma tabela cortada pelo splitter.
    continuacao_tabela = (
        bool(primeira_linha_util)
        and _linha_e_tabela(primeira_linha_util)
        and not tabela_abre_no_chunk
    )

    return hierarquia, estrutura_atual, cabecalho_tabela, continuacao_tabela


def contextualizar_chunks(
    chunks: Sequence[str],
    metadados_documento: Dict[str, object],
) -> List[Tuple[str, Dict[str, object]]]:
    """
    Prefixa cada chunk com sua trilha estrutural e produz metadados derivados.

    O prefixo torna o chunk auto-descritivo para a busca densa e lexical. Sem
    ele, um bloco tabular composto apenas de rotulos e numeros nao se aproxima
    de nenhuma pergunta em linguagem natural.

    Args:
        chunks: Chunks textuais do documento, na ordem original.
        metadados_documento: Metadados do documento de origem (tipo, numero,
            ano, titulo, ementa).

    Returns:
        Lista de tuplas (texto contextualizado, metadados adicionais). Os
        metadados adicionais contem `secao`, `contem_tabelas` e
        `tem_valores_numericos`.
    """
    if not chunks:
        return []

    identificador = _identificador_documento(metadados_documento)
    ementa = _truncar(
        str(
            metadados_documento.get("ementa")
            or metadados_documento.get("titulo")
            or ""
        ),
        _MAX_CHARS_EMENTA,
    )

    hierarquia: Dict[int, str] = {}
    estrutura_atual = ""
    cabecalho_tabela = ""

    resultado: List[Tuple[str, Dict[str, object]]] = []

    for texto_chunk in chunks:
        texto_chunk = texto_chunk or ""
        (
            hierarquia,
            estrutura_atual,
            cabecalho_tabela,
            continuacao_tabela,
        ) = _analisar_chunk(
            texto_chunk, hierarquia, estrutura_atual, cabecalho_tabela
        )

        trilha = _trilha_de_hierarquia(hierarquia, estrutura_atual)

        campos: List[str] = []
        if identificador:
            campos.append(f"Documento: {identificador}")
        if ementa:
            campos.append(f"Assunto: {ementa}")
        if trilha:
            campos.append(f"Secao: {trilha}")
        # Tabela cortada entre chunks perde o cabecalho; repeti-lo mantem o
        # bloco numerico interpretavel isoladamente.
        if continuacao_tabela and cabecalho_tabela:
            campos.append(f"Continuacao da tabela com colunas: {cabecalho_tabela}")

        prefixo = "[" + " | ".join(campos) + "]\n" if campos else ""
        texto_final = prefixo + texto_chunk

        documento_temporario = Document(page_content=texto_chunk, metadata={})
        metadados_extra: Dict[str, object] = {
            "secao": trilha,
            "contem_tabelas": "Sim" if chunk_tem_tabela(documento_temporario) else "Nao",
            "tem_valores_numericos": (
                "Sim" if _tem_valores_numericos(texto_chunk) else "Nao"
            ),
        }
        resultado.append((texto_final, metadados_extra))

    return resultado


# ---------------------------------------------------------------------------
# Indice lexical BM25
# ---------------------------------------------------------------------------


def chave_documento(documento: Document) -> str:
    """
    Gera a chave de identidade de um chunk (caminho + numero do chunk).

    Args:
        documento: Chunk recuperado do indice.

    Returns:
        Chave estavel usada em deduplicacao e fusao de ranqueamentos.
    """
    try:
        metadados = documento.metadata or {}
    except AttributeError:
        return ""
    return "{0}|{1}".format(
        metadados.get("caminho", ""), metadados.get("chunk", "")
    )


def extrair_documentos_do_vectorstore(vectorstore: object) -> List[Document]:
    """
    Extrai todos os chunks armazenados em um vectorstore FAISS.

    Permite construir o indice lexical e o mapa de irmaos a partir do indice
    vetorial ja carregado em memoria, sem exigir reindexacao.

    Args:
        vectorstore: Instancia FAISS do LangChain.

    Returns:
        Lista de Document presentes no docstore. Lista vazia em caso de falha.
    """
    if vectorstore is None:
        return []

    try:
        docstore = getattr(vectorstore, "docstore", None)
        if docstore is None:
            return []

        interno = getattr(docstore, "_dict", None)
        if isinstance(interno, dict) and interno:
            return [doc for doc in interno.values() if hasattr(doc, "page_content")]

        mapa_ids = getattr(vectorstore, "index_to_docstore_id", {}) or {}
        documentos: List[Document] = []
        for doc_id in mapa_ids.values():
            documento = docstore.search(doc_id)
            if hasattr(documento, "page_content"):
                documentos.append(documento)
        return documentos
    except Exception as exc:
        logger.warning(f"Falha ao extrair documentos do vectorstore: {exc}")
        return []


def _copiar_documento(documento: Document) -> Document:
    """Copia um chunk para evitar mutar os metadados armazenados no docstore."""
    return Document(
        page_content=documento.page_content,
        metadata=dict(documento.metadata or {}),
    )


class IndiceLexical:
    """
    Indice BM25 sobre os chunks do corpus, com mapa de irmaos por documento.

    A busca esparsa complementa a densa em consultas com siglas, numeros e
    referencias normativas. O mapa por caminho viabiliza a expansao por
    documento-pai sem consultas adicionais ao vectorstore.
    """

    def __init__(self, documentos: Sequence[Document]) -> None:
        """
        Constroi o indice a partir de uma colecao de chunks.

        Args:
            documentos: Chunks a indexar.

        Raises:
            RuntimeError: Se a dependencia rank_bm25 nao estiver instalada.
        """
        if not BM25_DISPONIVEL:
            raise RuntimeError("rank_bm25 nao esta instalado")

        self.documentos: List[Document] = [
            doc for doc in documentos if getattr(doc, "page_content", "")
        ]
        corpus_tokens: List[List[str]] = [
            tokenizar(doc.page_content) for doc in self.documentos
        ]

        # Chunks sem token util quebrariam o calculo de IDF do BM25.
        indices_validos = [i for i, tk in enumerate(corpus_tokens) if tk]
        self.documentos = [self.documentos[i] for i in indices_validos]
        corpus_tokens = [corpus_tokens[i] for i in indices_validos]

        if not corpus_tokens:
            raise RuntimeError("corpus vazio: indice lexical nao construido")

        self._bm25 = BM25Okapi(corpus_tokens)

        self.documentos_por_caminho: Dict[str, List[Document]] = {}
        for documento in self.documentos:
            caminho = str((documento.metadata or {}).get("caminho", ""))
            if not caminho:
                continue
            self.documentos_por_caminho.setdefault(caminho, []).append(documento)

        logger.info(
            f"Indice lexical construido: {len(self.documentos)} chunks, "
            f"{len(self.documentos_por_caminho)} documentos"
        )

    def __len__(self) -> int:
        """Retorna a quantidade de chunks indexados."""
        return len(self.documentos)

    def buscar(
        self,
        query: str,
        k: int = 30,
        filtro: Optional[Dict[str, object]] = None,
    ) -> List[Document]:
        """
        Recupera os chunks com maior score BM25 para a consulta.

        Args:
            query: Pergunta do usuario.
            k: Quantidade maxima de chunks a retornar.
            filtro: Filtro opcional de metadados (igualdade exata por campo).

        Returns:
            Lista de copias dos chunks mais relevantes, em ordem decrescente de
            score. Chunks com score zero sao descartados.
        """
        tokens = tokenizar(query)
        if not tokens:
            return []

        try:
            scores = self._bm25.get_scores(tokens)
        except Exception as exc:
            logger.warning(f"Busca lexical falhou: {exc}")
            return []

        # O corte por sobreposicao de termos e mais confiavel que o corte por
        # sinal do score: o IDF do BM25 pode ser negativo para termos muito
        # frequentes, e nesse caso um chunk sem nenhum termo em comum (score
        # zero) ficaria acima de um chunk que contem o termo buscado.
        relevantes = self._indices_com_sobreposicao(tokens)
        if relevantes is None:
            relevantes = {i for i, score in enumerate(scores) if score > 0}

        candidatos = sorted(relevantes, key=lambda i: scores[i], reverse=True)

        selecionados: List[Document] = []
        for indice in candidatos:
            documento = self.documentos[indice]
            if filtro and not self._atende_filtro(documento, filtro):
                continue
            copia = _copiar_documento(documento)
            copia.metadata["score_lexical"] = float(scores[indice])
            selecionados.append(copia)
            if len(selecionados) >= k:
                break

        return selecionados

    def _indices_com_sobreposicao(self, tokens: Sequence[str]) -> Optional[Set[int]]:
        """
        Identifica os chunks que contem ao menos um token da consulta.

        Usa as frequencias por documento mantidas pelo BM25, evitando um indice
        invertido proprio.

        Args:
            tokens: Tokens da consulta.

        Returns:
            Conjunto de indices de chunks com sobreposicao, ou None se a
            estrutura de frequencias nao estiver disponivel.
        """
        frequencias = getattr(self._bm25, "doc_freqs", None)
        if not isinstance(frequencias, list) or len(frequencias) != len(
            self.documentos
        ):
            return None

        # Consulta por token (custo O(1) por busca em dicionario) em vez de
        # intersecao de conjuntos, que construiria um conjunto por chunk do
        # corpus a cada pergunta.
        tokens_unicos = list(dict.fromkeys(tokens))
        return {
            indice
            for indice, frequencia_do_chunk in enumerate(frequencias)
            if any(token in frequencia_do_chunk for token in tokens_unicos)
        }

    def irmaos(self, caminho: str) -> List[Document]:
        """
        Retorna os chunks do documento indicado pelo caminho.

        Args:
            caminho: Caminho do arquivo markdown de origem.

        Returns:
            Lista de chunks do documento, ou lista vazia se desconhecido.
        """
        return self.documentos_por_caminho.get(caminho, [])

    @staticmethod
    def _atende_filtro(documento: Document, filtro: Dict[str, object]) -> bool:
        """Verifica se os metadados do chunk satisfazem o filtro de igualdade."""
        metadados = documento.metadata or {}
        for campo, valor in filtro.items():
            if str(metadados.get(campo, "")) != str(valor):
                return False
        return True


# Cache de indices lexicais. A chave e derivada do arquivo do indice vetorial,
# nao da identidade do objeto: o Streamlit recarrega o vectorstore do disco a
# cada rerun, e uma chave por objeto reconstruiria o BM25 em cada pergunta. O
# timestamp do arquivo invalida o cache automaticamente apos reindexacao.
_CACHE_INDICES: Dict[str, IndiceLexical] = {}


def _chave_cache_vectorstore(vectorstore: object) -> str:
    """
    Deriva a chave de cache de um vectorstore a partir do arquivo do indice.

    Args:
        vectorstore: Instancia FAISS carregada.

    Returns:
        Chave no formato "caminho|timestamp". Recorre a identidade do objeto
        quando o caminho do indice nao esta disponivel.
    """
    caminho = str(getattr(vectorstore, "_vectorstore_path", "") or "")
    if caminho:
        try:
            arquivo_indice = os.path.join(caminho, "index.faiss")
            if os.path.exists(arquivo_indice):
                return f"{caminho}|{os.path.getmtime(arquivo_indice)}"
            return caminho
        except OSError as exc:
            logger.debug(f"Timestamp do indice indisponivel: {exc}")
            return caminho
    return f"objeto|{id(vectorstore)}"


def obter_indice_lexical(vectorstore: object) -> Optional[IndiceLexical]:
    """
    Obtem (com cache) o indice lexical correspondente a um vectorstore.

    Args:
        vectorstore: Instancia FAISS carregada.

    Returns:
        Indice lexical pronto para uso, ou None se indisponivel.
    """
    if vectorstore is None or not BM25_DISPONIVEL:
        return None

    chave = _chave_cache_vectorstore(vectorstore)
    indice = _CACHE_INDICES.get(chave)
    if indice is not None:
        return indice

    documentos = extrair_documentos_do_vectorstore(vectorstore)
    if not documentos:
        logger.warning("Indice lexical nao construido: nenhum chunk encontrado")
        return None

    try:
        indice = IndiceLexical(documentos)
    except Exception as exc:
        logger.warning(f"Indice lexical indisponivel: {exc}")
        return None

    # Mantem apenas o indice do vectorstore corrente para nao acumular memoria
    # apos reindexacoes sucessivas.
    _CACHE_INDICES.clear()
    _CACHE_INDICES[chave] = indice
    return indice


def limpar_cache_indice_lexical() -> None:
    """Descarta indices lexicais em cache (usar apos reindexacao)."""
    _CACHE_INDICES.clear()


# ---------------------------------------------------------------------------
# Fusao de ranqueamentos
# ---------------------------------------------------------------------------


def fundir_rrf(
    listas: Sequence[Sequence[Document]],
    pesos: Optional[Sequence[float]] = None,
    constante: int = 60,
) -> Tuple[List[Document], Dict[str, float]]:
    """
    Combina varias listas ranqueadas por Reciprocal Rank Fusion.

    RRF usa apenas a posicao de cada item em cada lista, dispensando calibracao
    entre scores de similaridade densa e scores BM25, que nao sao comparaveis.

    Args:
        listas: Listas ranqueadas (mais relevante primeiro).
        pesos: Peso de cada lista. Padrao: peso 1.0 para todas.
        constante: Constante de amortecimento do RRF (valor usual: 60).

    Returns:
        Tupla (lista fundida ordenada, mapa de chave para score RRF).
    """
    if not listas:
        return [], {}

    if pesos is None:
        pesos = [1.0] * len(listas)

    scores: Dict[str, float] = {}
    documentos_por_chave: Dict[str, Document] = {}

    for indice_lista, lista in enumerate(listas):
        peso = pesos[indice_lista] if indice_lista < len(pesos) else 1.0
        for posicao, documento in enumerate(lista):
            chave = chave_documento(documento)
            if not chave:
                continue
            scores[chave] = scores.get(chave, 0.0) + peso / (constante + posicao + 1)
            if chave not in documentos_por_chave:
                documentos_por_chave[chave] = documento

    ordenadas = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    fundidos = [documentos_por_chave[chave] for chave, _ in ordenadas]
    return fundidos, scores


# ---------------------------------------------------------------------------
# Expansao por documento-pai
# ---------------------------------------------------------------------------


def _score_irmao(documento: Document) -> float:
    """
    Pontua um chunk irmao como candidato a entrar no contexto.

    Prioriza blocos tabulares e quantitativos, que sao os que a busca por
    similaridade tende a perder, e da leve preferencia a anexos.

    Args:
        documento: Chunk irmao candidato.

    Returns:
        Score de prioridade. Zero significa candidato descartavel.
    """
    metadados = documento.metadata or {}
    score = 0.0

    if chunk_tem_tabela(documento):
        score += 3.0
    if metadados.get("tem_valores_numericos") == "Sim" or _tem_valores_numericos(
        documento.page_content
    ):
        score += 1.5

    secao = normalizar_para_busca(str(metadados.get("secao", "")))
    if "anexo" in secao or "tabela" in secao or "quadro" in secao:
        score += 1.0

    return score


def expandir_com_irmaos_tabulares(
    documentos: Sequence[Document],
    indice: Optional[IndiceLexical],
    max_documentos_pai: int = 3,
    max_por_documento: int = 3,
    max_total: int = 8,
    min_chunks_no_contexto: int = 2,
) -> List[Document]:
    """
    Anexa chunks tabulares dos documentos melhor ranqueados.

    Um anexo com limites numericos raramente tem similaridade textual com uma
    pergunta generica, mas e essencial para responde-la. Se o documento que o
    contem foi considerado relevante, seus blocos tabulares passam a integrar o
    contexto.

    Args:
        documentos: Resultado ranqueado da busca hibrida.
        indice: Indice lexical, usado como fonte dos chunks irmaos.
        max_documentos_pai: Quantos documentos do topo podem ser expandidos.
        max_por_documento: Maximo de irmaos anexados por documento.
        max_total: Maximo de irmaos anexados em toda a expansao.
        min_chunks_no_contexto: Quantos chunks o documento precisa ter no
            resultado para ser expandido. Evita anexar tabelas de documentos
            marginalmente relevantes, que apenas consumiriam contexto. O
            documento melhor colocado e sempre elegivel.

    Returns:
        Lista original acrescida dos irmaos selecionados, marcados com o
        metadado `expandido_por_documento`.
    """
    if not documentos or indice is None:
        return list(documentos)

    presentes = {chave_documento(doc) for doc in documentos}
    caminhos_ordenados: List[str] = []
    contagem_por_caminho: Dict[str, int] = {}
    for documento in documentos:
        caminho = str((documento.metadata or {}).get("caminho", ""))
        if not caminho:
            continue
        if caminho not in caminhos_ordenados:
            caminhos_ordenados.append(caminho)
        contagem_por_caminho[caminho] = contagem_por_caminho.get(caminho, 0) + 1

    # Concentracao de chunks de um mesmo documento no resultado indica que a
    # norma e o assunto da pergunta, e nao uma coincidencia de vocabulario.
    elegiveis = [
        caminho
        for posicao, caminho in enumerate(caminhos_ordenados)
        if posicao == 0
        or contagem_por_caminho.get(caminho, 0) >= min_chunks_no_contexto
    ]

    adicionados: List[Document] = []

    for caminho in elegiveis[:max_documentos_pai]:
        candidatos: List[Tuple[float, Document]] = []
        for irmao in indice.irmaos(caminho):
            if chave_documento(irmao) in presentes:
                continue
            score = _score_irmao(irmao)
            if score <= 0:
                continue
            candidatos.append((score, irmao))

        candidatos.sort(key=lambda item: item[0], reverse=True)
        for score, irmao in candidatos[:max_por_documento]:
            copia = _copiar_documento(irmao)
            copia.metadata["expandido_por_documento"] = True
            adicionados.append(copia)
            presentes.add(chave_documento(irmao))
            if len(adicionados) >= max_total:
                break
        if len(adicionados) >= max_total:
            break

    if adicionados:
        logger.info(
            f"Expansao por documento-pai: {len(adicionados)} chunk(s) "
            f"tabular(es) anexado(s) ao contexto"
        )

    return list(documentos) + adicionados
