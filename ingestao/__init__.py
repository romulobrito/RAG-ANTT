"""Ingestao segura e incremental da base compartilhada."""

from .formatos import (
    DocumentoInvalidoError,
    DocumentoValidado,
    FormatoDocumento,
    validar_documento,
)

__all__ = [
    "DocumentoInvalidoError",
    "DocumentoValidado",
    "FormatoDocumento",
    "validar_documento",
]
