"""
Schemas do contrato HTTP do RAG (Fase 0).

Nao ha rota aqui. O SIGESC valida o JSON da consulta contra estas classes.
A tela omite temperatura, trechos e historico na primeira pergunta.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryFilters(BaseModel):
    """Filtros opcionais da consulta. Ano do pedido e inteiro."""

    model_config = ConfigDict(extra="forbid")

    tipo_documento: Optional[str] = None
    ano: Optional[int] = None
    numero: Optional[str] = None


class TurnoHistorico(BaseModel):
    """Uma troca que o chamador reenvia. O RAG nao grava esta lista."""

    model_config = ConfigDict(extra="forbid")

    pergunta: str
    resposta: str


class QueryRequest(BaseModel):
    """
    Pedido de POST /api/query.

    Campo extra (user_id, question) e rejeitado. Temperatura, trechos
    e historico podem vir omitidos.
    """

    model_config = ConfigDict(extra="forbid")

    pergunta: str = Field(min_length=1)
    filtros: Optional[QueryFilters] = None
    provider: Optional[str] = None
    max_documentos: Optional[int] = Field(default=None, ge=1, le=40)
    temperatura: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    historico: Optional[List[TurnoHistorico]] = None
    correlation_id: Optional[str] = None

    @field_validator("pergunta", mode="before")
    @classmethod
    def _limpar_pergunta(cls, valor: object) -> object:
        """
        Tira espaco nas pontas antes do minimo de 1 caractere.

        Args:
            valor: Texto recebido no JSON.

        Returns:
            Texto sem espaco nas pontas, ou o valor original se nao for str.
        """
        if isinstance(valor, str):
            return valor.strip()
        return valor


class DocumentHit(BaseModel):
    """Uma fonte citada na resposta. Ano aqui e texto, como no catalogo."""

    model_config = ConfigDict(extra="forbid")

    tipo: str
    numero: str
    ano: str
    trecho: str
    caminho: str
    relevancia: float


class QueryResponse(BaseModel):
    """Resposta de POST /api/query. request_id e obrigatorio."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    correlation_id: Optional[str] = None
    resposta: str
    modelo_usado: str
    provider: str
    documentos_consultados: List[DocumentHit]
    tempo_processamento_ms: int
    total_documentos_encontrados: int
    embedding_provider: str
    vectorstore_utilizado: str


class ErrorBody(BaseModel):
    """Envelope de erro. request_id pode faltar quando a falha e antes do id."""

    model_config = ConfigDict(extra="forbid")

    detail: str
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Liveness. So diz se o processo esta vivo."""

    model_config = ConfigDict(extra="forbid")

    status: str


class ReadyResponse(BaseModel):
    """Readiness. Indice e Ollama, sem executar consulta."""

    model_config = ConfigDict(extra="forbid")

    status: str
    vectorstore: bool
    ollama: bool


class ProvedorLiberado(BaseModel):
    """Um servico de geracao que a tela pode mostrar, com os modelos dele."""

    model_config = ConfigDict(extra="forbid")

    nome: str
    modelos: List[str]


class StatusResponse(BaseModel):
    """Snapshot para Servicos de IA e Situacao dos servicos."""

    model_config = ConfigDict(extra="forbid")

    vectorstore_path: str
    llm_acessivel: bool
    n_docs: int
    provedores_liberados: List[ProvedorLiberado]
    embeddings_liberados: List[str]
    embedding_provider: str
    formatos_upload: List[str] = Field(default_factory=list)
    formatos_inbox: List[str] = Field(default_factory=list)
    indexacao_automatica: bool = False
    reindexacao_em_andamento: bool = False


class DocumentListItem(BaseModel):
    """Uma linha do catalogo devolvido por GET /api/documents."""

    model_config = ConfigDict(extra="forbid")

    tipo: str
    numero: str
    ano: str
    caminho: str


class ReindexAccepted(BaseModel):
    """POST /api/reindex aceito. O trabalho segue no mesmo servico."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: Literal["accepted"] = "accepted"


class DocumentUploadAccepted(BaseModel):
    """Upload validado e persistido para processamento assincrono."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: Literal["queued"] = "queued"
    nome: str
    formato: Literal["pdf", "docx", "xlsx"]


class LegacyDocumentCreated(BaseModel):
    """Resposta do modo legado quando o incremental esta desligado."""

    model_config = ConfigDict(extra="forbid")

    caminho: str


class IngestionJobResponse(BaseModel):
    """Estado persistido de um job de ingestao."""

    model_config = ConfigDict(extra="ignore")

    job_id: str
    status: Literal[
        "queued",
        "running",
        "succeeded",
        "succeeded_with_warnings",
        "failed",
    ]
    nome: str
    formato: Literal["pdf", "docx", "xlsx"]
    origem: Literal["api", "inbox"]
    mensagem: str
    avisos: List[str] = Field(default_factory=list)
    geracao: Optional[str] = None
    chunks: Optional[int] = None
    caminho: Optional[str] = None
    created_at: str
    updated_at: str
