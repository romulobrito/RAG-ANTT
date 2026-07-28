"""
Testes unitarios do harness de avaliacao de retrieval.

Nao carregam o indice FAISS nem chamam o LLM: validam pontuacao de
gabarito, montagem do relatorio e helpers RAGAS offline.
"""

from langchain_core.documents import Document

from avaliar_retrieval import (
    METRICAS_RAGAS,
    CasoAvaliacao,
    _float_ou_nulo,
    avaliar_caso,
    avaliar_completude_resposta,
    cobertura_valores,
    documento_bate,
    gerar_relatorio_markdown,
    montar_referencia,
    normalizar_texto,
)


def test_normalizar_texto_remove_acentos():
    """Gabarito e contexto devem colidir apesar de acentuacao."""
    assert normalizar_texto("Pavimento Rígido") == "pavimento rigido"
    assert normalizar_texto("Seção") == "secao"


def test_cobertura_valores_parcial():
    """Cobertura e a fracao de itens do gabarito encontrados no contexto."""
    cobertura, encontrados, ausentes = cobertura_valores(
        "IRI 3,5 m/km e Dadm 50",
        ("3,5 m/km", "2,7 m/km", "Dadm"),
    )
    assert cobertura == 2 / 3
    assert encontrados == ["3,5 m/km", "Dadm"]
    assert ausentes == ["2,7 m/km"]


def test_documento_bate_ignora_zeros_a_esquerda():
    """Numeros indexados com padding devem casar com o gabarito curto."""
    meta = {"tipo_documento": "INM", "numero": "00000034", "ano": "2024"}
    assert documento_bate(meta, ("INM", "34", "2024")) is True


def test_montar_referencia_inclui_valores_e_documento():
    """Referencia RAGAS deve listar gabarito e documento-alvo."""
    caso = CasoAvaliacao(
        identificador="x",
        dominio="pavimento",
        pergunta="teste",
        valores_esperados=("3,5 m/km", "12 mm"),
        doc_esperado=("INM", "34", "2024"),
    )
    ref = montar_referencia(caso)
    assert "3,5 m/km" in ref
    assert "12 mm" in ref
    assert "INM 34/2024" in ref


def test_avaliar_completude_resposta():
    """Completude da resposta usa o mesmo matching do retrieval."""
    caso = CasoAvaliacao(
        identificador="x",
        dominio="pavimento",
        pergunta="teste",
        valores_esperados=("3,5 m/km", "2,7 m/km"),
        doc_esperado=None,
    )
    metrica = avaliar_completude_resposta(
        caso,
        "O limite superior e 3,5 m/km na pista principal.",
    )
    assert metrica["completude_resposta"] == 0.5
    assert metrica["encontrados_resposta"] == ["3,5 m/km"]
    assert metrica["ausentes_resposta"] == ["2,7 m/km"]


def test_avaliar_caso_conta_estruturados_e_cobertura():
    """Metrica combina gabarito textual e classificacao de fonte."""
    caso = CasoAvaliacao(
        identificador="x",
        dominio="pavimento",
        pergunta="teste",
        valores_esperados=("3,5 m/km", "12 mm"),
        doc_esperado=("INM", "34", "2024"),
        min_estruturados=1,
    )
    docs = [
        Document(
            page_content=(
                "## Tabelas auxiliares estruturadas (transcricao)\n"
                "| Pista | IRI |\n| --- | --- |\n| Principal | 3,5 m/km |"
            ),
            metadata={
                "tipo_documento": "INM",
                "numero": "00000034",
                "ano": "2024",
                "caminho": "a.md",
                "chunk": 1,
            },
        ),
        Document(
            page_content="## Anexo OCR - 1.png\n12 mm",
            metadata={
                "tipo_documento": "INM",
                "numero": "00000034",
                "ano": "2024",
                "caminho": "a.md",
                "chunk": 2,
                "secao": "Anexo OCR - 1.png",
            },
        ),
    ]

    metrica = avaliar_caso(caso, docs)

    assert metrica["cobertura"] == 1.0
    assert metrica["hit_documento"] is True
    assert metrica["n_estruturada"] >= 1
    assert metrica["ok_estruturados"] is True
    assert "referencia" in metrica
    assert isinstance(metrica["contextos"], list)


def test_float_ou_nulo_trata_nan():
    """Scores NaN do RAGAS viram None no relatorio."""
    assert _float_ou_nulo(0.85) == 0.85
    assert _float_ou_nulo("x") is None
    assert _float_ou_nulo(float("nan")) is None


def test_gerar_relatorio_contem_resumo():
    """O relatorio precisa expor cobertura media de forma legivel."""
    resultados = [
        {
            "identificador": "a",
            "dominio": "pavimento",
            "cobertura": 1.0,
            "encontrados": ["3,5"],
            "ausentes": [],
            "hit_documento": True,
            "n_estruturada": 2,
            "n_ocr": 0,
            "ok_estruturados": True,
            "tempo_retrieval_s": 0.4,
            "tempo_geracao_s": 1.2,
            "completude_resposta": 0.75,
            "ausentes_resposta": ["2,7"],
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.85,
        }
    ]
    md = gerar_relatorio_markdown(
        resultados,
        {
            "data": "hoje",
            "k": 16,
            "embeddings": "local",
            "com_geracao": True,
            "com_ragas": True,
        },
    )
    assert "Cobertura media" in md
    assert "100%" in md
    assert "Completude media" in md
    assert "faithfulness" in md
    for metrica in METRICAS_RAGAS:
        assert metrica in md
