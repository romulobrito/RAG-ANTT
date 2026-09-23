"""
API Key de servico para as rotas de negocio.

Health e ready nao usam esta dependencia. Key ausente no ambiente
responde 503, para nao parecer key errada do SIGESC.
"""

from __future__ import annotations

import hmac
import os
from typing import Optional

from fastapi import Request, Security
from fastapi.security import APIKeyHeader

from api.erros import ErroHttp

# Declarado no OpenAPI para o Swagger mostrar o cadeado Authorize.
cabecalho_api_key = APIKeyHeader(name="X-API-Key", auto_error=False)


def chave_configurada() -> str:
    """
    Valor de RAG_API_KEY, sem espaco nas pontas.

    Returns:
        A chave, ou string vazia se o ambiente nao definiu.
    """
    return os.environ.get("RAG_API_KEY", "").strip()


def _chave_recebida(request: Request) -> str:
    """
    Le X-API-Key ou Authorization Bearer.

    Args:
        request: Pedido HTTP.

    Returns:
        Segredo enviado, ou string vazia.
    """
    direta = request.headers.get("X-API-Key", "").strip()
    if direta:
        return direta
    autorizacao = request.headers.get("Authorization", "").strip()
    prefixo = "bearer "
    if autorizacao.lower().startswith(prefixo):
        return autorizacao[len(prefixo) :].strip()
    return ""


def dependencia_api_key(
    request: Request,
    chave_cabecalho: Optional[str] = Security(cabecalho_api_key),
) -> None:
    """
    Exige a API Key de servico.

    O parametro chave_cabecalho existe para o Swagger desenhar o cadeado.
    A leitura real continua em X-API-Key ou Authorization Bearer.

    Args:
        request: Pedido HTTP.
        chave_cabecalho: Valor injetado pelo Swagger. Nao substitui o Bearer.

    Raises:
        ErroHttp: 503 se a env estiver vazia; 401 se a key nao bater.
    """
    del chave_cabecalho
    configurada = chave_configurada()
    if not configurada:
        raise ErroHttp(503, "api_key_not_configured")
    recebida = _chave_recebida(request)
    if not recebida or not hmac.compare_digest(recebida, configurada):
        raise ErroHttp(401, "api_key_invalid")
