"""Validacao binaria dos formatos aceitos."""

from __future__ import annotations

import io
import zipfile

import pytest

from ingestao.formatos import (
    DocumentoInvalidoError,
    FormatoDocumento,
    validar_documento,
)


def _ooxml(entrada: str) -> bytes:
    """Cria um container OOXML minimo para validar assinatura."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as pacote:
        pacote.writestr("[Content_Types].xml", "<Types/>")
        pacote.writestr(entrada, "<root/>")
    return buffer.getvalue()


def test_pdf_valido() -> None:
    """PDF coerente retorna hash e MIME."""
    resultado = validar_documento(b"%PDF-1.4\nconteudo", "norma.pdf")
    assert resultado.formato == FormatoDocumento.PDF
    assert len(resultado.sha256) == 64
    assert resultado.mime == "application/pdf"


def test_docx_e_xlsx_validos() -> None:
    """Os dois containers OOXML sao diferenciados pelo conteudo."""
    docx = validar_documento(
        _ooxml("word/document.xml"),
        "nota.docx",
    )
    xlsx = validar_documento(
        _ooxml("xl/workbook.xml"),
        "dados.xlsx",
    )
    assert docx.formato == FormatoDocumento.DOCX
    assert xlsx.formato == FormatoDocumento.XLSX


def test_extensao_e_assinatura_divergentes() -> None:
    """ZIP do Word nao pode se passar por planilha."""
    with pytest.raises(DocumentoInvalidoError):
        validar_documento(
            _ooxml("word/document.xml"),
            "dados.xlsx",
        )


def test_path_traversal_e_normalizado() -> None:
    """Diretorios do nome enviado nunca chegam ao destino."""
    resultado = validar_documento(
        b"%PDF-1.4\nconteudo",
        "../../INM 34.pdf",
    )
    assert resultado.nome == "INM_34.pdf"
    assert "/" not in resultado.nome


def test_formato_fora_da_lista() -> None:
    """Formato moderno pode ser restringido pelo chamador."""
    with pytest.raises(DocumentoInvalidoError):
        validar_documento(
            _ooxml("word/document.xml"),
            "nota.docx",
            formatos=["pdf"],
        )
