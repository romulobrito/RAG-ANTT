"""Conversao de tabelas nativas para markdown contextualizado."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from ingestao.conversores import converter_documento
from ingestao.formatos import validar_documento


def _xlsx() -> bytes:
    """Monta planilha com percentual, data textual e formula."""
    import openpyxl

    pasta = openpyxl.Workbook()
    aba = pasta.active
    aba.title = "IRI"
    aba.append(["Pista", "Limite", "Unidade"])
    aba.append(["Principal", 2.7, "m/km"])
    aba.append(["Percentual", 0.25, "%"])
    aba["B3"].number_format = "0%"
    aba["B5"] = "=SUM(B2:B3)"
    buffer = io.BytesIO()
    pasta.save(buffer)
    return buffer.getvalue()


def test_xlsx_preserva_aba_tabela_e_formula(tmp_path: Path) -> None:
    """Planilha vira secao e tabela sem perder unidade."""
    documento = validar_documento(_xlsx(), "limites.xlsx")
    resultado = converter_documento(documento, str(tmp_path))
    assert "## Aba: IRI" in resultado.markdown
    assert "| Pista | Limite | Unidade |" in resultado.markdown
    assert "Principal" in resultado.markdown
    assert "m/km" in resultado.markdown
    assert "25%" in resultado.markdown
    assert "[formula: =SUM(B2:B3)]" in resultado.markdown
    assert resultado.tabelas == 1
    assert resultado.avisos


def test_docx_preserva_ordem_e_tabela(tmp_path: Path) -> None:
    """Paragrafo, tabela e paragrafo mantem a ordem do DOCX."""
    docx = pytest.importorskip("docx")
    arquivo = docx.Document()
    arquivo.add_heading("Limites de manutencao", level=1)
    tabela = arquivo.add_table(rows=2, cols=2)
    tabela.cell(0, 0).text = "Pista"
    tabela.cell(0, 1).text = "IRI"
    tabela.cell(1, 0).text = "Principal"
    tabela.cell(1, 1).text = "2,7 m/km"
    arquivo.add_paragraph("Fim da tabela.")
    buffer = io.BytesIO()
    arquivo.save(buffer)
    documento = validar_documento(buffer.getvalue(), "limites.docx")
    resultado = converter_documento(documento, str(tmp_path))
    assert resultado.markdown.index("Limites") < resultado.markdown.index("| Pista")
    assert resultado.markdown.index("| Pista") < resultado.markdown.index("Fim")
    assert "2,7 m/km" in resultado.markdown
    assert resultado.tabelas == 1
