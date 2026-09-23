"""
Porta sincrona do RAG, sem HTTP e sem Streamlit.

A API futura e a tela de QA chamam estas funcoes. A geracao em fluxo
(streaming) permanece na interface.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from config import (
    DB_FAISS_PATH,
    get_allowed_llm_providers,
    get_embedding_allowed,
    get_llm_max_tokens,
    get_llm_model,
    get_llm_provider_padrao,
    get_llm_temperature,
    get_max_documentos,
    logger,
)
from antt_rag_unified import (
    _reescrever_query_com_historico,
    carregar_vectorstore_com_provider,
    classificar_falha_de_geracao,
    gerar_resposta,
    mensagem_de_falha_de_geracao,
    pesquisar_documentos,
    reindexacao_ocupada,
    reindexar_base_completa,
)
from llm_providers import (
    create_llm_manager,
    get_available_providers,
    verificar_ollama_disponivel,
)


class RagServiceError(Exception):
    """Falha de negocio do servico de consulta."""


class RagNotReadyError(RagServiceError):
    """Indice ou modelo indisponivel para consultar."""


class RagGenerationError(RagServiceError):
    """A geracao da resposta falhou."""


class ProvedorNaoLiberadoError(RagServiceError):
    """Provedor ou embedding fora da lista liberada no ambiente."""


class PerguntaInvalidaError(RagServiceError):
    """Pergunta ou parametro explicito fora do contrato."""


@dataclass
class DocumentHit:
    """Trecho devolvido junto com a resposta."""

    tipo: str
    numero: str
    ano: str
    trecho: str
    caminho: str
    relevancia: float


@dataclass
class QueryResult:
    """Resultado sincrono de uma consulta. Sem ids HTTP."""

    resposta: str
    modelo_usado: str
    provider: str
    documentos_consultados: List[DocumentHit]
    total_documentos_encontrados: int
    embedding_provider: str
    vectorstore_utilizado: str


@dataclass
class StatusServico:
    """Retrato do indice, do Ollama e do que o ambiente liberou."""

    vectorstore_ok: bool
    vectorstore_path: str
    ollama_ok: bool
    ollama_mensagem: str
    n_docs: int
    provedores_liberados: List[str]
    embeddings_liberados: List[str]
    embedding_provider: str


@dataclass
class DocumentoCatalogo:
    """Item do catalogo compartilhado."""

    tipo: str
    numero: str
    ano: str
    caminho: str
    titulo: str = ""


@dataclass
class ReindexResult:
    """Reindexacao sincrona. A API depois coloca isto em background."""

    job_id: str
    sucesso: bool
    mensagem: str


_CACHE_VECTORSTORE: Dict[str, object] = {}
_PASTA_ENTRADA = os.path.join("dados_antt", "entrada")
_RELATORIO = "relatorio_documentos.json"
_TRECHO_MAXIMO = 500


def limpar_cache_vectorstore() -> None:
    """
    Esvazia o indice mantido neste processo.

    Util apos reindexar e nos testes que trocam o carregamento.
    """
    _CACHE_VECTORSTORE.clear()


def _caminho_indice(embedding_provider: str) -> str:
    """
    Diretorio do indice conforme o embedding.

    Args:
        embedding_provider: Identificador local, free ou openai.

    Returns:
        Caminho relativo do indice.
    """
    if embedding_provider == "openai":
        return DB_FAISS_PATH
    return "vectorstore_local"


def _normalizar_embedding(embedding_provider: Optional[str]) -> str:
    """
    Resolve o embedding e recusa o que o ambiente nao liberou.

    Args:
        embedding_provider: Valor do chamador, ou None para o padrao.

    Returns:
        Identificador aceito por carregar_vectorstore_com_provider.

    Raises:
        ProvedorNaoLiberadoError: Se o valor nao esta na lista.
    """
    permitidos = get_embedding_allowed()
    escolhido = (embedding_provider or permitidos[0]).strip()
    if escolhido == "free" and "local" in permitidos:
        return "free"
    if escolhido not in permitidos:
        raise ProvedorNaoLiberadoError(
            "Embedding '{0}' nao esta em RAG_EMBEDDING_ALLOWED.".format(
                escolhido
            )
        )
    return escolhido


def _resolver_provedor(provider: Optional[str]) -> str:
    """
    Resolve o provedor de geracao.

    Args:
        provider: Valor do chamador, ou None para o padrao.

    Returns:
        Identificador liberado.

    Raises:
        ProvedorNaoLiberadoError: Se a lista do ambiente nao inclui o valor.
    """
    escolhido = (provider or get_llm_provider_padrao()).strip() or "ollama"
    permitidos = get_allowed_llm_providers()
    if permitidos is not None and escolhido not in permitidos:
        raise ProvedorNaoLiberadoError(
            "Provedor '{0}' nao esta em RAG_LLM_ALLOWED_PROVIDERS.".format(
                escolhido
            )
        )
    return escolhido


def _resolver_temperatura(valor: Optional[float]) -> float:
    """
    Usa o ambiente quando a temperatura vem omitida.

    Args:
        valor: Temperatura explicita, ou None.

    Returns:
        Valor entre 0.0 e 1.0.

    Raises:
        PerguntaInvalidaError: Se o chamador mandou um numero fora da faixa.
    """
    if valor is None:
        return get_llm_temperature()
    if valor < 0.0 or valor > 1.0:
        raise PerguntaInvalidaError("Temperatura fora da faixa 0.0 a 1.0.")
    return valor


def _resolver_max_documentos(valor: Optional[int]) -> int:
    """
    Usa 30 (teto 40) quando a quantidade de trechos vem omitida.

    Args:
        valor: Quantidade explicita, ou None.

    Returns:
        Inteiro entre 1 e 40.

    Raises:
        PerguntaInvalidaError: Se o chamador mandou um inteiro fora da faixa.
    """
    if valor is None:
        return get_max_documentos()
    if valor < 1 or valor > 40:
        raise PerguntaInvalidaError("max_documentos fora da faixa 1 a 40.")
    return valor


def _resolver_max_tokens(valor: Optional[int]) -> int:
    """
    Usa 4096 quando o tamanho da resposta vem omitido.

    Args:
        valor: Teto explicito, ou None.

    Returns:
        Inteiro entre 1 e 4096.

    Raises:
        PerguntaInvalidaError: Se o chamador mandou um inteiro fora da faixa.
    """
    if valor is None:
        return get_llm_max_tokens()
    if valor < 1 or valor > 4096:
        raise PerguntaInvalidaError("max_tokens fora da faixa 1 a 4096.")
    return valor


def _obter_vectorstore(embedding_provider: str) -> object:
    """
    Carrega o indice uma vez por processo e por embedding.

    Args:
        embedding_provider: Identificador ja validado.

    Returns:
        Objeto FAISS carregado.

    Raises:
        RagNotReadyError: Se o indice nao puder ser aberto.
    """
    em_cache = _CACHE_VECTORSTORE.get(embedding_provider)
    if em_cache is not None:
        return em_cache
    try:
        vectorstore = carregar_vectorstore_com_provider(embedding_provider)
    except Exception as exc:
        raise RagNotReadyError(
            "Indice indisponivel: {0}".format(exc)
        ) from exc
    _CACHE_VECTORSTORE[embedding_provider] = vectorstore
    return vectorstore


def _texto_meta(meta: Mapping[str, object], *chaves: str) -> str:
    """
    Le a primeira chave de metadado que existir.

    Args:
        meta: Metadados do documento.
        chaves: Nomes em ordem de preferencia.

    Returns:
        Texto, ou string vazia.
    """
    for chave in chaves:
        valor = meta.get(chave)
        if valor is None:
            continue
        texto = str(valor).strip()
        if texto:
            return texto
    return ""


def _hit_de_documento(documento: object) -> DocumentHit:
    """
    Converte um Document do LangChain no hit do contrato.

    Args:
        documento: Documento recuperado.

    Returns:
        Hit com tipo, numero, ano, trecho, caminho e relevancia.
    """
    meta_bruto = getattr(documento, "metadata", None)
    meta: Mapping[str, object]
    if isinstance(meta_bruto, dict):
        meta = meta_bruto
    else:
        meta = {}
    relevancia_bruta = meta.get("relevancia", 0.0)
    if isinstance(relevancia_bruta, bool) or not isinstance(
        relevancia_bruta, (int, float)
    ):
        relevancia = 0.0
    else:
        relevancia = float(relevancia_bruta)
    trecho = getattr(documento, "page_content", "")
    if not isinstance(trecho, str):
        trecho = str(trecho)
    if len(trecho) > _TRECHO_MAXIMO:
        trecho = trecho[:_TRECHO_MAXIMO]
    return DocumentHit(
        tipo=_texto_meta(meta, "nome_tipo", "tipo"),
        numero=_texto_meta(meta, "numero"),
        ano=_texto_meta(meta, "ano"),
        trecho=trecho,
        caminho=_texto_meta(meta, "caminho"),
        relevancia=relevancia,
    )


def _filtro_texto(
    filtros: Optional[Mapping[str, object]],
    chave: str,
) -> Optional[str]:
    """
    Le um filtro textual, tratando vazio e 'Todos' como ausencia.

    Args:
        filtros: Mapa opcional.
        chave: Nome do campo.

    Returns:
        Texto ou None.
    """
    if filtros is None:
        return None
    valor = filtros.get(chave)
    if valor is None:
        return None
    texto = str(valor).strip()
    if not texto or texto == "Todos":
        return None
    return texto


@dataclass
class TrechosRecuperados:
    """Trechos da mesma busca usada pela tela e por consultar."""

    pergunta_busca: str
    documentos: List[object]
    embedding_provider: str
    vectorstore_utilizado: str


def recuperar_trechos(
    pergunta: str,
    filtros: Optional[Mapping[str, object]] = None,
    max_documentos: Optional[int] = None,
    embedding_provider: Optional[str] = None,
    historico: Optional[Sequence[Mapping[str, str]]] = None,
    provider: Optional[str] = None,
    modelo: Optional[str] = None,
    temperatura: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> TrechosRecuperados:
    """
    Busca os trechos da consulta, com o mesmo corte da API.

    k omitido vale 30. k explicito fica entre 1 e 40. A geracao
    posterior nao descarta 31 a 40: o teto interno acompanha 40.

    Args:
        pergunta: Texto da pergunta.
        filtros: tipo_documento, ano e numero, todos opcionais.
        max_documentos: Quantidade de trechos.
        embedding_provider: Como localizar os trechos.
        historico: Trocas anteriores. Quando presente, reescreve a busca.
        provider: Provedor usado so na reescrita.
        modelo: Modelo usado so na reescrita.
        temperatura: Liberdade de redacao da reescrita.
        max_tokens: Teto de tokens da reescrita.

    Returns:
        TrechosRecuperados com a pergunta usada na busca e os documentos.

    Raises:
        PerguntaInvalidaError: Pergunta vazia ou k fora da faixa.
        ProvedorNaoLiberadoError: Embedding nao liberado.
        RagNotReadyError: Indice ou busca indisponivel.
    """
    if not isinstance(pergunta, str) or not pergunta.strip():
        raise PerguntaInvalidaError("Pergunta vazia.")

    k = _resolver_max_documentos(max_documentos)
    embedding = _normalizar_embedding(embedding_provider)
    vectorstore = _obter_vectorstore(embedding)
    caminho_indice = str(
        getattr(vectorstore, "_vectorstore_path", "")
        or _caminho_indice(embedding)
    )

    pergunta_busca = pergunta.strip()
    if historico:
        try:
            provedor = _resolver_provedor(provider)
            modelo_efetivo = (modelo or get_llm_model()).strip() or get_llm_model()
            gerente = create_llm_manager(provedor, modelo_efetivo)
            llm_reescrita = gerente.get_llm(
                temperature=_resolver_temperatura(temperatura),
                max_tokens=_resolver_max_tokens(max_tokens),
            )
            pergunta_busca = _reescrever_query_com_historico(
                pergunta_busca,
                list(historico),
                llm_reescrita,
            )
        except (ProvedorNaoLiberadoError, PerguntaInvalidaError):
            raise
        except Exception as exc:
            logger.warning("Reescrita da pergunta falhou: %s", exc)

    tipo = _filtro_texto(filtros, "tipo_documento")
    numero = _filtro_texto(filtros, "numero")
    ano: Optional[object] = None
    if filtros is not None and filtros.get("ano") is not None:
        ano_bruto = filtros.get("ano")
        if not (isinstance(ano_bruto, str) and ano_bruto.strip() in ("", "Todos")):
            ano = ano_bruto

    try:
        documentos = pesquisar_documentos(
            pergunta_busca,
            vectorstore,
            k=k,
            tipo_documento=tipo,
            ano=ano,
            numero=numero,
            embedding_provider=embedding,
        )
    except Exception as exc:
        raise RagNotReadyError(
            "Busca indisponivel: {0}".format(exc)
        ) from exc

    if not isinstance(documentos, list):
        documentos = list(documentos)
    return TrechosRecuperados(
        pergunta_busca=pergunta_busca,
        documentos=documentos,
        embedding_provider=embedding,
        vectorstore_utilizado=caminho_indice,
    )


def consultar(
    pergunta: str,
    filtros: Optional[Mapping[str, object]] = None,
    provider: Optional[str] = None,
    modelo: Optional[str] = None,
    temperatura: Optional[float] = None,
    max_documentos: Optional[int] = None,
    max_tokens: Optional[int] = None,
    embedding_provider: Optional[str] = None,
    historico: Optional[Sequence[Mapping[str, str]]] = None,
) -> QueryResult:
    """
    Consulta sincrona: busca trechos e redige a resposta.

    Temperatura, trechos e tamanho omitidos saem do ambiente
    (0.1, 30 com teto 40, 4096). O provedor omitido e ollama, ou o
    primeiro de RAG_LLM_ALLOWED_PROVIDERS.

    Args:
        pergunta: Texto da pergunta.
        filtros: tipo_documento, ano e numero, todos opcionais.
        provider: Provedor de geracao.
        modelo: Modelo. Omitido usa RAG_LLM_MODEL.
        temperatura: Liberdade de redacao.
        max_documentos: Quantidade de trechos.
        max_tokens: Teto da resposta.
        embedding_provider: Como localizar os trechos.
        historico: Trocas anteriores reenviadas pelo chamador.

    Returns:
        QueryResult com a resposta e as fontes.

    Raises:
        PerguntaInvalidaError: Pergunta vazia ou parametro fora da faixa.
        ProvedorNaoLiberadoError: Provedor ou embedding nao liberado.
        RagNotReadyError: Indice ou modelo indisponivel.
        RagGenerationError: Falha ao redigir.
    """
    provedor = _resolver_provedor(provider)
    temperatura_efetiva = _resolver_temperatura(temperatura)
    teto_tokens = _resolver_max_tokens(max_tokens)
    modelo_efetivo = (modelo or get_llm_model()).strip() or get_llm_model()
    pacote = recuperar_trechos(
        pergunta,
        filtros=filtros,
        max_documentos=max_documentos,
        embedding_provider=embedding_provider,
        historico=historico,
        provider=provedor,
        modelo=modelo_efetivo,
        temperatura=temperatura_efetiva,
        max_tokens=teto_tokens,
    )
    pergunta_busca = pacote.pergunta_busca
    documentos = pacote.documentos

    try:
        gerente_resposta = create_llm_manager(provedor, modelo_efetivo)
        llm = gerente_resposta.get_llm(
            temperature=temperatura_efetiva,
            max_tokens=teto_tokens,
        )
        resposta, modelo_usado = gerar_resposta(
            pergunta_busca,
            documentos,
            llm,
            provedor,
        )
    except Exception as exc:
        classe = classificar_falha_de_geracao(exc)
        raise RagGenerationError(
            "{0}: {1}".format(classe, mensagem_de_falha_de_geracao(exc))
        ) from exc

    if not isinstance(resposta, str):
        resposta = str(resposta)
    hits = [_hit_de_documento(documento) for documento in documentos]
    return QueryResult(
        resposta=resposta,
        modelo_usado=str(modelo_usado),
        provider=provedor,
        documentos_consultados=hits,
        total_documentos_encontrados=len(hits),
        embedding_provider=pacote.embedding_provider,
        vectorstore_utilizado=pacote.vectorstore_utilizado,
    )


def obter_status() -> StatusServico:
    """
    Retrato leve para a tela e para o futuro GET /api/status.

    Nao abre o indice FAISS: so verifica se o diretorio existe.
    A consulta e que carrega o indice.

    Returns:
        StatusServico.
    """
    embeddings = get_embedding_allowed()
    embedding = embeddings[0]
    caminho = _caminho_indice(embedding)
    ollama_ok, ollama_msg = verificar_ollama_disponivel()
    provedores = list(get_available_providers().keys())
    return StatusServico(
        vectorstore_ok=os.path.isdir(caminho),
        vectorstore_path=caminho,
        ollama_ok=ollama_ok,
        ollama_mensagem=ollama_msg,
        n_docs=len(listar_documentos()),
        provedores_liberados=provedores,
        embeddings_liberados=list(embeddings),
        embedding_provider=embedding,
    )


def listar_documentos(
    relatorio_path: str = _RELATORIO,
) -> List[DocumentoCatalogo]:
    """
    Le o catalogo ja gerado, sem varrer o disco de novo.

    Args:
        relatorio_path: Caminho de relatorio_documentos.json.

    Returns:
        Lista do catalogo. Vazia se o arquivo nao existir.
    """
    import json

    if not os.path.exists(relatorio_path):
        return []
    try:
        with open(relatorio_path, "r", encoding="utf-8") as arquivo:
            bruto = json.load(arquivo)
    except (OSError, ValueError) as exc:
        logger.warning("Catalogo ilegivel (%s): %s", relatorio_path, exc)
        return []
    if not isinstance(bruto, list):
        return []
    itens: List[DocumentoCatalogo] = []
    for entrada in bruto:
        if not isinstance(entrada, dict):
            continue
        caminho = str(entrada.get("arquivo_md") or entrada.get("caminho") or "")
        itens.append(
            DocumentoCatalogo(
                tipo=str(entrada.get("tipo") or ""),
                numero=str(entrada.get("numero") or ""),
                ano=str(entrada.get("ano") or ""),
                caminho=caminho,
                titulo=str(entrada.get("titulo") or ""),
            )
        )
    return itens


def listar_anos(relatorio_path: str = _RELATORIO) -> List[str]:
    """
    Anos de quatro digitos presentes no catalogo, do mais novo ao mais antigo.

    Ano vazio ou fora do seculo 20/21 fica de fora. A tela usa esta lista
    no filtro, no lugar de um intervalo fixo.

    Args:
        relatorio_path: Caminho de relatorio_documentos.json.

    Returns:
        Anos unicos, ordenados de forma decrescente.
    """
    encontrados = {
        item.ano
        for item in listar_documentos(relatorio_path)
        if re.fullmatch(r"(?:19|20)\d{2}", item.ano)
    }
    return sorted(encontrados, reverse=True)


def reindexacao_em_andamento() -> bool:
    """
    True se o lock de reindexacao esta ativo.

    So leitura. Quem adquire o lock continua sendo reindexar_base_completa.
    A tela de QA nao usa esta funcao.

    Returns:
        True quando outra atualizacao da base ainda corre.
    """
    return reindexacao_ocupada()


def aquecer_indice(embedding_provider: Optional[str] = None) -> bool:
    """
    Carrega o indice no cache deste processo.

    Falha de abertura nao derruba o chamador: a API segue no ar e o
    ready responde indisponivel.

    Args:
        embedding_provider: Embedding do indice. Omitido usa o ambiente.

    Returns:
        True se o indice ficou em cache.
    """
    try:
        embedding = _normalizar_embedding(embedding_provider)
        _obter_vectorstore(embedding)
    except Exception as exc:
        logger.warning("Indice nao aquecido: %s", exc)
        return False
    return True


def disparar_reindexacao(
    embedding_provider: Optional[str] = None,
) -> ReindexResult:
    """
    Reindexa a base comum, respeitando o lock ja existente.

    Args:
        embedding_provider: Embedding do job. Omitido usa o ambiente.

    Returns:
        ReindexResult com um id e a mensagem do pipeline.

    Raises:
        ProvedorNaoLiberadoError: Se o embedding nao foi liberado.
    """
    embedding = _normalizar_embedding(embedding_provider)
    job_id = str(uuid.uuid4())
    sucesso, mensagem = reindexar_base_completa(embedding)
    if sucesso:
        limpar_cache_vectorstore()
    return ReindexResult(
        job_id=job_id,
        sucesso=bool(sucesso),
        mensagem=str(mensagem),
    )


def incluir_documento(conteudo: bytes, nome_arquivo: str) -> str:
    """
    Grava um PDF na base comum. A consulta so o ve depois do reindex.

    Args:
        conteudo: Bytes do PDF.
        nome_arquivo: Nome original enviado pelo chamador.

    Returns:
        Caminho relativo do arquivo gravado.

    Raises:
        PerguntaInvalidaError: Se o conteudo nao for um PDF nomeado.
    """
    if not isinstance(conteudo, (bytes, bytearray)) or len(conteudo) < 5:
        raise PerguntaInvalidaError("PDF vazio ou invalido.")
    if not bytes(conteudo).startswith(b"%PDF"):
        raise PerguntaInvalidaError("O arquivo nao parece um PDF.")
    nome = os.path.basename(str(nome_arquivo or "")).strip()
    if not nome.lower().endswith(".pdf"):
        raise PerguntaInvalidaError("O arquivo precisa ter extensao .pdf.")
    seguro = "".join(
        caractere if caractere.isascii() and (
            caractere.isalnum() or caractere in "._-"
        ) else "_"
        for caractere in nome
    )
    if seguro in ("", ".pdf", "_.pdf"):
        seguro = "documento.pdf"
    os.makedirs(_PASTA_ENTRADA, exist_ok=True)
    destino = os.path.join(_PASTA_ENTRADA, seguro)
    if os.path.exists(destino):
        base, ext = os.path.splitext(seguro)
        indice = 2
        while os.path.exists(destino):
            destino = os.path.join(
                _PASTA_ENTRADA,
                "{0}_{1}{2}".format(base, indice, ext),
            )
            indice += 1
    with open(destino, "wb") as arquivo:
        arquivo.write(bytes(conteudo))
    logger.info("PDF incluido na base comum: %s", destino)
    return destino
