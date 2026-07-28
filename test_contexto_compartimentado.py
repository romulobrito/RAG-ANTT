"""
Testes da montagem de contexto compartimentado por documento-fonte.
"""

from langchain_core.documents import Document

from contexto_compartimentado import (
    id_fonte_documento,
    montar_contexto_compartimentado,
)


def test_id_fonte_documento():
    """ID legivel combina tipo, numero e ano."""
    assert id_fonte_documento(
        {"nome_tipo": "INM", "numero": "34", "ano": "2024"}
    ) == "INM 34/2024"


def test_contexto_agrupa_e_delimita_fontes():
    """Chunks do mesmo doc ficam no mesmo bloco FONTE INICIO/FIM."""
    docs = [
        Document(
            page_content="IRI 3,5 m/km",
            metadata={
                "nome_tipo": "INM",
                "numero": "34",
                "ano": "2024",
                "chunk": 1,
                "total_chunks": 2,
                "caminho": "a.md",
            },
        ),
        Document(
            page_content="flexibilizacao FSO",
            metadata={
                "nome_tipo": "SEI",
                "numero": "1320",
                "ano": "2025",
                "chunk": 1,
                "total_chunks": 1,
                "caminho": "b.md",
            },
        ),
        Document(
            page_content="Dadm 50",
            metadata={
                "nome_tipo": "INM",
                "numero": "34",
                "ano": "2024",
                "chunk": 2,
                "total_chunks": 2,
                "caminho": "a.md",
            },
        ),
    ]

    texto, tipos, contagem = montar_contexto_compartimentado(docs)

    assert "===== FONTE INICIO: INM 34/2024 =====" in texto
    assert "===== FONTE FIM: INM 34/2024 =====" in texto
    assert "===== FONTE INICIO: SEI 1320/2025 =====" in texto
    assert "===== FONTE FIM: SEI 1320/2025 =====" in texto
    assert "Escopo: as regras, valores e formalizacoes abaixo aplicam-se SOMENTE" in texto
    assert "IRI 3,5 m/km" in texto
    assert "Dadm 50" in texto
    assert texto.index("FONTE INICIO: INM") < texto.index("FONTE INICIO: SEI")
    bloco_inm = texto.split("FONTE INICIO: SEI")[0]
    assert bloco_inm.index("IRI 3,5") < bloco_inm.index("Dadm 50")
    assert "INM" in tipos
    assert contagem["INM"] == 2
    assert contagem["SEI"] == 1
