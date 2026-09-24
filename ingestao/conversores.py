"""Conversores robustos para o markdown canonico da base."""

from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable, List, Sequence

from ingestao.formatos import DocumentoValidado, FormatoDocumento


class ConversaoDocumentoError(RuntimeError):
    """Falha que impede promover o documento."""


@dataclass
class ResultadoConversao:
    """Markdown e diagnosticos produzidos por um conversor."""

    markdown: str
    avisos: List[str] = field(default_factory=list)
    tabelas: int = 0
    imagens_processadas: int = 0


def _escapar_celula(valor: object) -> str:
    """Converte uma celula para markdown sem quebrar a tabela."""
    if valor is None:
        return ""
    if isinstance(valor, (datetime, date)):
        texto = valor.isoformat()
    else:
        texto = str(valor)
    return texto.replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def _tabela_markdown(linhas: Sequence[Sequence[object]]) -> str:
    """Gera tabela markdown retangular e repete o primeiro registro como header."""
    if not linhas:
        return ""
    largura = max(len(linha) for linha in linhas)
    normalizadas = [
        [_escapar_celula(valor) for valor in list(linha) + [""] * (largura - len(linha))]
        for linha in linhas
    ]
    if not any(any(celula for celula in linha) for linha in normalizadas):
        return ""
    cabecalho = normalizadas[0]
    corpo = normalizadas[1:]
    partes = [
        "| " + " | ".join(cabecalho) + " |",
        "| " + " | ".join(["---"] * largura) + " |",
    ]
    partes.extend("| " + " | ".join(linha) + " |" for linha in corpo)
    return "\n".join(partes)


def _ocr_imagem(conteudo: bytes) -> str:
    """Extrai primeiro tabelas e depois texto de uma imagem incorporada."""
    caminho = ""
    try:
        from img2table.document import Image as ImagemTabela
        from img2table.ocr import TesseractOCR

        descritor, caminho = tempfile.mkstemp(suffix=".png")
        with os.fdopen(descritor, "wb") as arquivo:
            arquivo.write(conteudo)
        imagem_tabela = ImagemTabela(src=caminho)
        ocr_tabela = TesseractOCR(n_threads=1, lang="por")
        tabelas = imagem_tabela.extract_tables(
            ocr=ocr_tabela,
            implicit_rows=True,
            borderless_tables=True,
            min_confidence=50,
        )
        blocos = []
        for tabela in tabelas:
            quadro = getattr(tabela, "df", None)
            if quadro is None or getattr(quadro, "empty", True):
                continue
            linhas = [list(quadro.columns)] + quadro.values.tolist()
            markdown = _tabela_markdown(linhas)
            if markdown:
                blocos.append(markdown)
        if blocos:
            return "\n\n".join(blocos)
    except Exception:
        # Imagem sem grade ou backend tabular indisponivel cai no OCR textual.
        pass
    finally:
        if caminho:
            try:
                os.remove(caminho)
            except OSError:
                pass
    try:
        import pytesseract
        from PIL import Image

        imagem = Image.open(io.BytesIO(conteudo))
        texto = pytesseract.image_to_string(imagem, lang="por")
        return "\n".join(
            linha.strip() for linha in texto.splitlines() if linha.strip()
        )
    except Exception as exc:
        raise ConversaoDocumentoError(
            "Falha ao executar OCR de imagem incorporada: {0}".format(exc)
        ) from exc


def _converter_pdf(documento: DocumentoValidado, destino_temporario: str) -> ResultadoConversao:
    """Reutiliza o pipeline PDF existente e valida sua saida."""
    caminho_pdf = os.path.join(destino_temporario, documento.nome)
    with open(caminho_pdf, "wb") as arquivo:
        arquivo.write(documento.conteudo)
    try:
        from antt_rag_unified import _converter_pdf_para_md

        markdown = _converter_pdf_para_md(caminho_pdf)
    except Exception as exc:
        raise ConversaoDocumentoError(
            "Falha ao converter PDF: {0}".format(exc)
        ) from exc
    if not markdown or len(markdown.strip()) < 20:
        raise ConversaoDocumentoError("PDF sem conteudo extraivel.")
    return ResultadoConversao(
        markdown=markdown.strip() + "\n",
        tabelas=markdown.count("| ---"),
    )


def _imagens_do_paragrafo(paragrafo: object, documento_docx: object) -> Iterable[bytes]:
    """Retorna blobs de imagens referenciadas por um paragrafo DOCX."""
    elemento = getattr(paragrafo, "_p")
    for blip in elemento.xpath(".//a:blip"):
        rel_id = blip.get(
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
        )
        if not rel_id:
            continue
        parte = documento_docx.part.related_parts.get(rel_id)
        blob = getattr(parte, "blob", None)
        if isinstance(blob, bytes):
            yield blob


def _linhas_tabela_docx(tabela: object) -> List[List[str]]:
    """Extrai linhas DOCX, preservando celulas mescladas sem duplicar texto."""
    linhas: List[List[str]] = []
    for linha in tabela.rows:
        valores: List[str] = []
        ids: set[int] = set()
        for celula in linha.cells:
            identificador = id(celula._tc)
            if identificador in ids:
                continue
            ids.add(identificador)
            valores.append(" ".join(celula.text.split()))
        linhas.append(valores)
    return linhas


def _converter_docx(documento: DocumentoValidado) -> ResultadoConversao:
    """Converte DOCX na ordem real de paragrafos, tabelas e imagens."""
    try:
        from docx import Document as DocxDocument
        from docx.table import Table
        from docx.text.paragraph import Paragraph
        from docx.oxml.table import CT_Tbl
        from docx.oxml.text.paragraph import CT_P
    except ImportError as exc:
        raise ConversaoDocumentoError("Dependencia python-docx ausente.") from exc

    try:
        origem = DocxDocument(io.BytesIO(documento.conteudo))
    except Exception as exc:
        raise ConversaoDocumentoError(
            "DOCX corrompido ou protegido: {0}".format(exc)
        ) from exc

    blocos: List[str] = []
    avisos: List[str] = []
    tabelas = 0
    imagens = 0
    corpo = origem.element.body
    for filho in corpo.iterchildren():
        if isinstance(filho, CT_P):
            paragrafo = Paragraph(filho, origem)
            texto = " ".join(paragrafo.text.split())
            estilo = str(getattr(paragrafo.style, "name", "") or "")
            if texto:
                if estilo.lower().startswith("heading"):
                    blocos.append("## " + texto)
                else:
                    blocos.append(texto)
            for blob in _imagens_do_paragrafo(paragrafo, origem):
                imagens += 1
                try:
                    ocr = _ocr_imagem(blob)
                except ConversaoDocumentoError as exc:
                    avisos.append(str(exc))
                    continue
                if ocr:
                    blocos.append("### Conteudo extraido de imagem\n\n" + ocr)
                else:
                    avisos.append("Imagem incorporada sem texto OCR legivel.")
        elif isinstance(filho, CT_Tbl):
            tabela = Table(filho, origem)
            markdown = _tabela_markdown(_linhas_tabela_docx(tabela))
            if not markdown:
                raise ConversaoDocumentoError("Tabela DOCX sem estrutura legivel.")
            tabelas += 1
            blocos.append(markdown)

    markdown_final = "\n\n".join(bloco for bloco in blocos if bloco.strip())
    if len(markdown_final.strip()) < 20:
        raise ConversaoDocumentoError("DOCX sem conteudo extraivel.")
    return ResultadoConversao(
        markdown=markdown_final.strip() + "\n",
        avisos=avisos,
        tabelas=tabelas,
        imagens_processadas=imagens,
    )


def _valor_excel(formula: object, calculado: object, avisos: List[str]) -> object:
    """Escolhe valor exibido e registra formula sem cache."""
    if isinstance(formula, str) and formula.startswith("="):
        if calculado is not None:
            return calculado
        avisos.append("Formula sem valor calculado: {0}".format(formula))
        return "[formula: {0}]".format(formula)
    return formula


def _valor_mesclado(planilha: object, linha: int, coluna: int) -> object:
    """Repete o valor superior esquerdo de uma celula mesclada."""
    coordenada = planilha.cell(linha, coluna).coordinate
    for intervalo in planilha.merged_cells.ranges:
        if coordenada in intervalo:
            return planilha.cell(
                intervalo.min_row,
                intervalo.min_col,
            ).value
    return planilha.cell(linha, coluna).value


def _formatar_numero_excel(celula: object, valor: object) -> object:
    """Preserva percentual exibido quando o formato da celula o informa."""
    formato = str(getattr(celula, "number_format", "") or "")
    if (
        isinstance(valor, (int, float))
        and not isinstance(valor, bool)
        and "%" in formato
    ):
        return "{0:g}%".format(float(valor) * 100.0)
    return valor


def _converter_xlsx(documento: DocumentoValidado) -> ResultadoConversao:
    """Converte abas visiveis, formulas e imagens de uma planilha."""
    try:
        import openpyxl
    except ImportError as exc:
        raise ConversaoDocumentoError("Dependencia openpyxl ausente.") from exc
    try:
        formulas = openpyxl.load_workbook(
            io.BytesIO(documento.conteudo),
            data_only=False,
            read_only=False,
        )
        valores = openpyxl.load_workbook(
            io.BytesIO(documento.conteudo),
            data_only=True,
            read_only=False,
        )
    except Exception as exc:
        raise ConversaoDocumentoError(
            "XLSX corrompido ou protegido: {0}".format(exc)
        ) from exc

    blocos: List[str] = []
    avisos: List[str] = []
    tabelas = 0
    imagens = 0
    max_linhas = 10000
    max_colunas = 200
    for planilha in formulas.worksheets:
        if planilha.sheet_state != "visible":
            continue
        planilha_valores = valores[planilha.title]
        blocos.append("## Aba: {0}".format(planilha.title))
        limite_linhas = min(planilha.max_row, max_linhas)
        limite_colunas = min(planilha.max_column, max_colunas)
        if planilha.max_row > max_linhas or planilha.max_column > max_colunas:
            avisos.append(
                "Aba {0} truncada no limite operacional.".format(planilha.title)
            )
        linhas: List[List[object]] = []
        for numero_linha in range(1, limite_linhas + 1):
            linha: List[object] = []
            for numero_coluna in range(1, limite_colunas + 1):
                celula = planilha.cell(numero_linha, numero_coluna)
                formula = _valor_mesclado(
                    planilha,
                    numero_linha,
                    numero_coluna,
                )
                calculado = planilha_valores.cell(
                    numero_linha,
                    numero_coluna,
                ).value
                valor = _valor_excel(formula, calculado, avisos)
                linha.append(_formatar_numero_excel(celula, valor))
            while linha and linha[-1] in (None, ""):
                linha.pop()
            linhas.append(linha)
        while linhas and not linhas[-1]:
            linhas.pop()
        tabela = _tabela_markdown(linhas)
        if tabela:
            tabelas += 1
            blocos.append(tabela)
        for imagem in getattr(planilha, "_images", []):
            imagens += 1
            try:
                ocr = _ocr_imagem(imagem._data())
            except Exception as exc:
                avisos.append(
                    "Falha no OCR de imagem da aba {0}: {1}".format(
                        planilha.title,
                        exc,
                    )
                )
                continue
            if ocr:
                blocos.append(
                    "### Imagem da aba {0}\n\n{1}".format(
                        planilha.title,
                        ocr,
                    )
                )
        if getattr(planilha, "_charts", []):
            avisos.append(
                "Aba {0} possui graficos; dados fonte foram priorizados.".format(
                    planilha.title
                )
            )

    markdown_final = "\n\n".join(bloco for bloco in blocos if bloco.strip())
    if len(markdown_final.strip()) < 20:
        raise ConversaoDocumentoError("XLSX sem conteudo extraivel.")
    return ResultadoConversao(
        markdown=markdown_final.strip() + "\n",
        avisos=avisos,
        tabelas=tabelas,
        imagens_processadas=imagens,
    )


def converter_documento(
    documento: DocumentoValidado,
    destino_temporario: str,
) -> ResultadoConversao:
    """Converte um documento validado sem alterar a base canonica."""
    os.makedirs(destino_temporario, exist_ok=True)
    if documento.formato == FormatoDocumento.PDF:
        return _converter_pdf(documento, destino_temporario)
    if documento.formato == FormatoDocumento.DOCX:
        return _converter_docx(documento)
    if documento.formato == FormatoDocumento.XLSX:
        return _converter_xlsx(documento)
    raise ConversaoDocumentoError("Formato sem conversor.")
