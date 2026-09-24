"""
Servidor HTTP do RAG. A tela de QA nao passa por aqui.

Consulta sincrona. Streaming permanece no Streamlit.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional

import rag_service
from antt_rag_unified import mensagem_de_falha_de_geracao
from api.auth import dependencia_api_key
from api.erros import ErroHttp
from api.schemas import (
    DocumentUploadAccepted,
    DocumentHit,
    DocumentListItem,
    ErrorBody,
    HealthResponse,
    IngestionJobResponse,
    LegacyDocumentCreated,
    ProvedorLiberado,
    QueryRequest,
    QueryResponse,
    ReadyResponse,
    ReindexAccepted,
    StatusResponse,
)
from fastapi import BackgroundTasks, Depends, FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from llm_providers import get_available_providers
from rag_service import (
    PerguntaInvalidaError,
    ProvedorNaoLiberadoError,
    RagGenerationError,
    RagNotReadyError,
)

logger = logging.getLogger(__name__)


def _swagger_ligado() -> bool:
    """True salvo quando RAG_SWAGGER esta desligado."""
    bruto = os.environ.get("RAG_SWAGGER", "true").strip().lower()
    return bruto not in ("0", "false", "nao", "no", "off")


def _logar_texto_da_consulta() -> bool:
    """True so se RAG_LOG_QUERY_TEXT pedir o texto da pergunta."""
    return os.environ.get("RAG_LOG_QUERY_TEXT", "false").strip().lower() in (
        "1",
        "true",
        "sim",
        "yes",
        "on",
    )


def _corpo_erro(
    status_code: int,
    detail: str,
    request_id: Optional[str] = None,
) -> JSONResponse:
    """
    Envelope do contrato, sem traceback.

    Args:
        status_code: Codigo HTTP.
        detail: Mensagem curta.
        request_id: Id da consulta, se ja existir.

    Returns:
        Resposta JSON.
    """
    corpo = ErrorBody(detail=detail, request_id=request_id)
    return JSONResponse(status_code=status_code, content=corpo.model_dump())


def _detalhe_geracao(exc: RagGenerationError) -> str:
    """
    Texto amigavel da falha de geracao, sem o JSON cru do provedor.

    Args:
        exc: Erro ja classificado pelo facade.

    Returns:
        Mensagem para o campo detail.
    """
    causa = exc.__cause__
    if isinstance(causa, Exception):
        return mensagem_de_falha_de_geracao(causa)
    return str(exc)


def _e_consulta_longa(exc: RagGenerationError) -> bool:
    """
    True quando o facade marcou a causa como consulta_longa.

    Args:
        exc: Erro de geracao.

    Returns:
        True se a mensagem comeca com essa causa.
    """
    return str(exc).startswith("consulta_longa")


@asynccontextmanager
async def _vida(aplicacao: FastAPI) -> AsyncIterator[None]:
    """
    Tenta aquecer o indice. Falha nao derruba o processo.

    Args:
        aplicacao: Instancia FastAPI.

    Yields:
        Nada. O servidor atende depois do aquecimento.
    """
    del aplicacao
    rag_service.aquecer_indice()
    worker_iniciado = False
    scanner_iniciado = False
    try:
        from config import incremental_upload_enabled, inbox_scan_enabled

        if incremental_upload_enabled():
            from ingestao.geracoes import bootstrap_indice_legado
            from ingestao.jobs import iniciar_worker

            bootstrap_indice_legado()
            iniciar_worker()
            worker_iniciado = True
            if inbox_scan_enabled():
                from ingestao.scanner import iniciar_scanner

                iniciar_scanner()
                scanner_iniciado = True
    except Exception:
        logger.exception("Falha ao iniciar ingestao incremental")
    try:
        yield
    finally:
        if scanner_iniciado:
            from ingestao.scanner import parar_scanner

            parar_scanner()
        if worker_iniciado:
            from ingestao.jobs import parar_worker

            parar_worker()


def criar_aplicacao() -> FastAPI:
    """
    Monta o app com ou sem Swagger, conforme RAG_SWAGGER.

    Returns:
        Aplicacao FastAPI.
    """
    docs = "/api/docs" if _swagger_ligado() else None
    aberto = "/openapi.json" if _swagger_ligado() else None
    aplicacao = FastAPI(
        title="RAG-ANTT",
        docs_url=docs,
        redoc_url=None,
        openapi_url=aberto,
        lifespan=_vida,
    )

    @aplicacao.exception_handler(ErroHttp)
    async def _erro_http(request: Request, exc: ErroHttp) -> JSONResponse:
        """Devolve o envelope sem stacktrace."""
        del request
        return _corpo_erro(exc.status_code, exc.detail, exc.request_id)

    @aplicacao.exception_handler(RequestValidationError)
    async def _pedido_invalido(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Pydantic fora do contrato responde 400."""
        del request
        del exc
        return _corpo_erro(400, "pedido invalido")

    @aplicacao.exception_handler(Exception)
    async def _inesperado(request: Request, exc: Exception) -> JSONResponse:
        """Loga a pilha e esconde o detalhe do cliente."""
        logger.exception("api_erro path=%s", request.url.path)
        del exc
        return _corpo_erro(500, "internal_error")

    @aplicacao.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Processo vivo. Nao abre indice nem Ollama."""
        return HealthResponse(status="ok")

    @aplicacao.get("/api/ready", response_model=None)
    def ready() -> JSONResponse:
        """Indice no disco e Ollama. Nao exige API Key."""
        retrato = rag_service.obter_status()
        corpo = ReadyResponse(
            status="ok" if retrato.vectorstore_ok and retrato.ollama_ok else "unavailable",
            vectorstore=retrato.vectorstore_ok,
            ollama=retrato.ollama_ok,
        )
        codigo = 200 if corpo.status == "ok" else 503
        return JSONResponse(status_code=codigo, content=corpo.model_dump())

    @aplicacao.get(
        "/api/status",
        response_model=StatusResponse,
        dependencies=[Depends(dependencia_api_key)],
    )
    def status() -> StatusResponse:
        """Provedores liberados e quantidade de documentos."""
        retrato = rag_service.obter_status()
        provedores: List[ProvedorLiberado] = []
        for nome, info in get_available_providers().items():
            modelos_brutos = info.get("models") if isinstance(info, dict) else []
            modelos = (
                [str(item) for item in modelos_brutos]
                if isinstance(modelos_brutos, list)
                else []
            )
            provedores.append(ProvedorLiberado(nome=str(nome), modelos=modelos))
        return StatusResponse(
            vectorstore_path=retrato.vectorstore_path,
            llm_acessivel=retrato.ollama_ok,
            n_docs=retrato.n_docs,
            provedores_liberados=provedores,
            embeddings_liberados=list(retrato.embeddings_liberados),
            embedding_provider=retrato.embedding_provider,
            formatos_upload=list(retrato.formatos_upload or []),
            formatos_inbox=list(retrato.formatos_inbox or []),
            indexacao_automatica=retrato.indexacao_automatica,
            reindexacao_em_andamento=retrato.reindexacao_em_andamento,
        )

    @aplicacao.post(
        "/api/query",
        response_model=QueryResponse,
        responses={
            400: {"model": ErrorBody, "description": "Pedido invalido."},
            503: {
                "model": ErrorBody,
                "description": "Indice, modelo ou geracao indisponivel.",
            },
            504: {
                "model": ErrorBody,
                "description": "Consulta excedeu o tempo permitido.",
            },
        },
        dependencies=[Depends(dependencia_api_key)],
    )
    def query(pedido: QueryRequest) -> JSONResponse:
        """Consulta sincrona. Ids e tempo sao desta camada."""
        request_id = str(uuid.uuid4())
        inicio = time.perf_counter()
        codigo = 500
        provider = pedido.provider or ""
        modelo = ""
        n_docs = 0
        try:
            filtros = None
            if pedido.filtros is not None:
                filtros = pedido.filtros.model_dump()
            historico = None
            if pedido.historico is not None:
                historico = [turno.model_dump() for turno in pedido.historico]
            resultado = rag_service.consultar(
                pedido.pergunta,
                filtros=filtros,
                provider=pedido.provider,
                temperatura=pedido.temperatura,
                max_documentos=pedido.max_documentos,
                historico=historico,
            )
            fontes = [
                DocumentHit(
                    tipo=hit.tipo,
                    numero=hit.numero,
                    ano=hit.ano,
                    trecho=hit.trecho,
                    caminho=hit.caminho,
                    relevancia=hit.relevancia,
                )
                for hit in resultado.documentos_consultados
            ]
            corpo = QueryResponse(
                request_id=request_id,
                correlation_id=pedido.correlation_id,
                resposta=resultado.resposta,
                modelo_usado=resultado.modelo_usado,
                provider=resultado.provider,
                documentos_consultados=fontes,
                tempo_processamento_ms=int((time.perf_counter() - inicio) * 1000),
                total_documentos_encontrados=resultado.total_documentos_encontrados,
                embedding_provider=resultado.embedding_provider,
                vectorstore_utilizado=resultado.vectorstore_utilizado,
            )
            codigo = 200
            provider = resultado.provider
            modelo = resultado.modelo_usado
            n_docs = resultado.total_documentos_encontrados
            resposta_http = JSONResponse(
                status_code=200,
                content=corpo.model_dump(),
            )
        except PerguntaInvalidaError as exc:
            codigo = 400
            resposta_http = _corpo_erro(400, str(exc), request_id)
        except ProvedorNaoLiberadoError as exc:
            codigo = 400
            resposta_http = _corpo_erro(400, str(exc), request_id)
        except RagNotReadyError as exc:
            codigo = 503
            resposta_http = _corpo_erro(503, str(exc), request_id)
        except RagGenerationError as exc:
            if _e_consulta_longa(exc):
                codigo = 504
            else:
                codigo = 503
            resposta_http = _corpo_erro(codigo, _detalhe_geracao(exc), request_id)
        finally:
            latencia_ms = int((time.perf_counter() - inicio) * 1000)
            correlation = pedido.correlation_id or "-"
            logger.info(
                "api_query ts=%s request_id=%s correlation_id=%s path=/api/query status=%s latencia_ms=%s provider=%s modelo=%s n_docs=%s",
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                request_id,
                correlation,
                codigo,
                latencia_ms,
                provider,
                modelo,
                n_docs,
            )
            if _logar_texto_da_consulta():
                logger.info(
                    "api_query_text request_id=%s pergunta=%s",
                    request_id,
                    pedido.pergunta.replace("\n", " "),
                )
        return resposta_http

    @aplicacao.get(
        "/api/documents",
        response_model=List[DocumentListItem],
        dependencies=[Depends(dependencia_api_key)],
    )
    def documents() -> List[DocumentListItem]:
        """Catalogo ja gerado. Nao varre o disco de novo."""
        return [
            DocumentListItem(
                tipo=item.tipo,
                numero=item.numero,
                ano=item.ano,
                caminho=item.caminho,
            )
            for item in rag_service.listar_documentos()
        ]

    @aplicacao.post(
        "/api/documents",
        status_code=202,
        response_model=DocumentUploadAccepted,
        responses={
            201: {
                "model": LegacyDocumentCreated,
                "description": "Documento gravado no modo legado.",
            },
            400: {"model": ErrorBody, "description": "Documento invalido."},
            409: {"model": ErrorBody, "description": "Documento duplicado."},
            413: {
                "model": ErrorBody,
                "description": "Documento excede o tamanho permitido.",
            },
        },
        dependencies=[Depends(dependencia_api_key)],
    )
    async def documents_incluir(
        arquivo: UploadFile = File(...),
    ) -> JSONResponse:
        """
        Aceita PDF, DOCX ou XLSX e cria um job incremental.
        """
        conteudo = await arquivo.read()
        nome = arquivo.filename or ""
        from config import incremental_upload_enabled

        if incremental_upload_enabled():
            from ingestao.formatos import DocumentoInvalidoError
            from ingestao.jobs import submeter_upload
            from ingestao.indice_incremental import DocumentoDuplicadoError

            try:
                criado = submeter_upload(conteudo, nome)
            except DocumentoInvalidoError as exc:
                codigo = 413 if "tamanho" in str(exc).lower() else 400
                raise ErroHttp(codigo, str(exc)) from exc
            except DocumentoDuplicadoError as exc:
                raise ErroHttp(409, str(exc)) from exc
            aceito = DocumentUploadAccepted(
                job_id=criado.job_id,
                nome=criado.nome,
                formato=criado.formato,
            )
            return JSONResponse(
                status_code=202,
                content=aceito.model_dump(),
            )
        try:
            caminho = rag_service.incluir_documento(conteudo, nome)
        except PerguntaInvalidaError as exc:
            raise ErroHttp(400, str(exc)) from exc
        return JSONResponse(status_code=201, content={"caminho": caminho})

    @aplicacao.get(
        "/api/jobs/{job_id}",
        response_model=IngestionJobResponse,
        responses={
            404: {
                "model": ErrorBody,
                "description": "Job de ingestao nao encontrado.",
            },
        },
        dependencies=[Depends(dependencia_api_key)],
    )
    def ingestion_job(job_id: str) -> IngestionJobResponse:
        """Retorna o estado persistido de uma ingestao."""
        from ingestao.jobs import JobNaoEncontradoError, obter_job

        try:
            dados = obter_job(job_id)
        except JobNaoEncontradoError as exc:
            raise ErroHttp(404, "job_nao_encontrado") from exc
        return IngestionJobResponse.model_validate(dados)

    @aplicacao.post(
        "/api/reindex",
        dependencies=[Depends(dependencia_api_key)],
    )
    def reindex(
        background: BackgroundTasks,
        request: Request,
        embedding_provider: Optional[str] = None,
    ) -> JSONResponse:
        """
        Aceita a atualizacao da base e roda depois da resposta.

        Lock ativo responde 409. A tela de QA continua sincrona.
        """
        ops_key = os.environ.get("RAG_OPS_API_KEY", "").strip()
        if ops_key and request.headers.get("X-Ops-Key", "") != ops_key:
            raise ErroHttp(403, "reindex_completo_nao_autorizado")
        try:
            embedding = rag_service._normalizar_embedding(embedding_provider)
        except ProvedorNaoLiberadoError as exc:
            raise ErroHttp(400, str(exc)) from exc
        if rag_service.reindexacao_em_andamento():
            raise ErroHttp(409, "reindex_em_andamento")
        job_id = str(uuid.uuid4())

        def _rodar() -> None:
            """Executa a mesma reindexacao sincrona da tela, neste processo."""
            try:
                resultado = rag_service.disparar_reindexacao(embedding)
                logger.info(
                    "api_reindex job_id=%s sucesso=%s",
                    job_id,
                    resultado.sucesso,
                )
            except Exception:
                logger.exception("api_reindex job_id=%s", job_id)

        background.add_task(_rodar)
        aceito = ReindexAccepted(job_id=job_id)
        return JSONResponse(status_code=202, content=aceito.model_dump())

    return aplicacao


app = criar_aplicacao()
