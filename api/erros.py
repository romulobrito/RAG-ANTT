"""Erro HTTP com o envelope do contrato, sem stacktrace na resposta."""

from __future__ import annotations

from typing import Optional


class ErroHttp(Exception):
    """
    Falha ja classificada para o cliente.

    Attributes:
        status_code: Codigo HTTP.
        detail: Texto curto do envelope.
        request_id: Id da consulta, ou None se ainda nao existe.
    """

    def __init__(
        self,
        status_code: int,
        detail: str,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Args:
            status_code: Codigo HTTP.
            detail: Mensagem do campo detail.
            request_id: Id opcional.
        """
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.request_id = request_id
