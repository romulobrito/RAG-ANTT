"""Validacao estrita dos documentos aceitos na ingestao."""

from __future__ import annotations

import hashlib
import io
import os
import re
import zipfile
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from config import get_upload_formats, get_upload_max_bytes


class DocumentoInvalidoError(ValueError):
    """Documento recusado antes de qualquer escrita canonica."""


class FormatoDocumento(str, Enum):
    """Formatos modernos suportados."""

    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"


@dataclass(frozen=True)
class DocumentoValidado:
    """Resultado imutavel da validacao binaria."""

    nome: str
    formato: FormatoDocumento
    conteudo: bytes
    sha256: str
    mime: str


_MIMES = {
    FormatoDocumento.PDF: "application/pdf",
    FormatoDocumento.DOCX: (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
    FormatoDocumento.XLSX: (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ),
}


def _nome_seguro(nome: str) -> str:
    """Normaliza nome sem permitir diretorios ou caracteres nao ASCII."""
    base = os.path.basename(str(nome or "")).strip()
    seguro = "".join(
        caractere
        if caractere.isascii() and (
            caractere.isalnum() or caractere in "._-"
        )
        else "_"
        for caractere in base
    )
    seguro = re.sub(r"_+", "_", seguro).strip("._")
    if not seguro or "." not in seguro:
        raise DocumentoInvalidoError("Nome de arquivo invalido.")
    return seguro


def _validar_ooxml(conteudo: bytes, formato: FormatoDocumento) -> None:
    """Confere se o ZIP contem a estrutura minima do formato declarado."""
    esperado = (
        "word/document.xml"
        if formato == FormatoDocumento.DOCX
        else "xl/workbook.xml"
    )
    try:
        with zipfile.ZipFile(io.BytesIO(conteudo)) as pacote:
            nomes = set(pacote.namelist())
            if "[Content_Types].xml" not in nomes or esperado not in nomes:
                raise DocumentoInvalidoError(
                    "O arquivo OOXML nao corresponde a extensao informada."
                )
            ruim = pacote.testzip()
            if ruim is not None:
                raise DocumentoInvalidoError(
                    "Arquivo compactado corrompido."
                )
    except zipfile.BadZipFile as exc:
        raise DocumentoInvalidoError(
            "O arquivo nao e um DOCX/XLSX valido."
        ) from exc


def validar_documento(
    conteudo: bytes,
    nome_arquivo: str,
    formatos: Iterable[str] | None = None,
) -> DocumentoValidado:
    """Valida tamanho, extensao, assinatura e hash do documento."""
    if not isinstance(conteudo, (bytes, bytearray)):
        raise DocumentoInvalidoError("Conteudo precisa ser binario.")
    bytes_documento = bytes(conteudo)
    if not bytes_documento:
        raise DocumentoInvalidoError("Arquivo vazio.")
    if len(bytes_documento) > get_upload_max_bytes():
        raise DocumentoInvalidoError("Arquivo excede o tamanho permitido.")

    nome = _nome_seguro(nome_arquivo)
    extensao = os.path.splitext(nome)[1].lower().lstrip(".")
    liberados = set(formatos or get_upload_formats())
    if extensao not in liberados:
        aceitos = ", ".join(sorted(liberados))
        raise DocumentoInvalidoError(
            "Formato nao suportado. Aceitos: {0}.".format(aceitos)
        )
    try:
        formato = FormatoDocumento(extensao)
    except ValueError as exc:
        raise DocumentoInvalidoError("Formato nao suportado.") from exc

    if formato == FormatoDocumento.PDF:
        if not bytes_documento.startswith(b"%PDF"):
            raise DocumentoInvalidoError("O arquivo nao parece um PDF.")
    else:
        _validar_ooxml(bytes_documento, formato)

    return DocumentoValidado(
        nome=nome,
        formato=formato,
        conteudo=bytes_documento,
        sha256=hashlib.sha256(bytes_documento).hexdigest(),
        mime=_MIMES[formato],
    )
