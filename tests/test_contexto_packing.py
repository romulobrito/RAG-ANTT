"""
Testes do packing estrutural de contexto (sem hardcode de dominio).
"""

from langchain_core.documents import Document

from contexto_compartimentado import (
    id_fonte_documento,
    montar_contexto_compartimentado,
)
from contexto_packing import (
    cortar_preservando_tabela,
    eh_identificador_ollama,
    selecionar_chunks_para_orcamento,
)


def _doc_prosa(fonte_num: str, tamanho: int, rank_meta: int = 1) -> Document:
    """Cria chunk de prosa longa sem tabela."""
    corpo = ("Texto legal generico sem estrutura tabular. " * 40)[:tamanho]
    return Document(
        page_content=corpo,
        metadata={
            "nome_tipo": "RES",
            "numero": fonte_num,
            "ano": "2022",
            "chunk": rank_meta,
            "total_chunks": 10,
            "fonte_qualidade": "documento",
        },
    )


def _doc_tabela(fonte_num: str = "34") -> Document:
    """Cria chunk tabular estruturado (markdown)."""
    tabela = (
        "| Parametro | Limite |\n"
        "| --- | --- |\n"
        "| Alpha | 1,5 |\n"
        "| Beta | 2,0 |\n"
        "| Gamma | 3,5 |\n"
    )
    return Document(
        page_content=tabela,
        metadata={
            "nome_tipo": "INM",
            "numero": fonte_num,
            "ano": "2024",
            "chunk": 1,
            "total_chunks": 2,
            "fonte_qualidade": "estruturada",
            "injetado_auxiliar": True,
            "contem_tabelas": "Sim",
        },
    )


def test_identidade_quando_cabe_no_orcamento():
    """
    Invariante API: abaixo do limite, lista intacta (ordem e conteudo).
    """
    docs = [
        _doc_prosa("6000", 500, 1),
        _doc_tabela("34"),
        _doc_prosa("6053", 400, 2),
    ]
    # Limite folgado (equivalente a cloud tipico abaixo de 50k).
    resultado = selecionar_chunks_para_orcamento(
        docs,
        limite_chars=50000,
        overhead_instrucoes=1000,
    )
    assert len(resultado) == len(docs)
    assert resultado is not docs
    for original, obtido in zip(docs, resultado):
        assert obtido.page_content == original.page_content
        assert obtido.metadata == original.metadata


def test_tabela_entra_com_orcamento_apertado():
    """
    Prosa longa primeiro no ranking + tabela estruturada: tabela deve entrar
    quando o slice linear descartaria o final.
    """
    prosa = _doc_prosa("6000", 3500, 1)
    tabela = _doc_tabela("34")
    docs = [prosa, tabela]

    corpo_full, _, _ = montar_contexto_compartimentado(docs)
    overhead = 200
    # Orcamento insuficiente para prosa+tabela, mas suficiente para a tabela
    # (e eventualmente um pouco de prosa se o score permitir).
    limite = overhead + 2 + len(tabela.page_content) + 400

    resultado = selecionar_chunks_para_orcamento(
        docs,
        limite_chars=limite,
        overhead_instrucoes=overhead,
    )
    assert len(resultado) >= 1
    textos = [d.page_content for d in resultado]
    assert any("| Alpha | 1,5 |" in t for t in textos)


def test_teto_por_fonte_impede_monopolio():
    """
    Duas fontes grandes: nenhuma fonte consome todo o orcamento sozinha.
    """
    fonte_a = [
        _doc_prosa("100", 800, i + 1)
        for i in range(4)
    ]
    for doc in fonte_a:
        doc.metadata["numero"] = "100"
        doc.metadata["nome_tipo"] = "RES"

    fonte_b = [
        Document(
            page_content=("Dados estruturados fonte B com numeros 12 34 56. " * 20),
            metadata={
                "nome_tipo": "INM",
                "numero": "200",
                "ano": "2024",
                "chunk": i + 1,
                "total_chunks": 4,
                "fonte_qualidade": "estruturada",
                "injetado_auxiliar": True,
            },
        )
        for i in range(4)
    ]

    docs = fonte_a + fonte_b
    overhead = 100
    corpo, _, _ = montar_contexto_compartimentado(docs)
    # Limite que forca packing, mas permite trechos das duas fontes.
    limite = max(2500, int((overhead + 2 + len(corpo)) * 0.45))

    resultado = selecionar_chunks_para_orcamento(
        docs,
        limite_chars=limite,
        overhead_instrucoes=overhead,
        fracao_teto_fonte=0.40,
    )
    fontes = {id_fonte_documento(d.metadata or {}) for d in resultado}
    assert "RES 100/2022" in fontes
    assert "INM 200/2024" in fontes

    chars_por_fonte = {}
    for doc in resultado:
        fid = id_fonte_documento(doc.metadata or {})
        chars_por_fonte[fid] = chars_por_fonte.get(fid, 0) + len(
            doc.page_content or ""
        )
    teto = int(limite * 0.40)
    for fid, uso in chars_por_fonte.items():
        # Primeiro chunk pode exceder o teto; demais respeitam. Com varios
        # chunks, o uso agregado nao deve monopolizar quase todo o limite.
        assert uso <= max(teto + 800, int(limite * 0.85)), fid


def test_cortar_preservando_tabela():
    """Corte prefere linhas | quando o chunk unico nao cabe."""
    texto = (
        "Introducao longa sem valor.\n"
        "| A | B |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n"
        "| 3 | 4 |\n"
    )
    cortado = cortar_preservando_tabela(texto, max_chars=60)
    assert "|" in cortado
    assert len(cortado) <= 60


def test_eh_identificador_ollama():
    """Detecta Ollama sem classificar deepseek/openai como local."""
    assert eh_identificador_ollama("ollama")
    assert eh_identificador_ollama("qwen2.5:7b")
    assert eh_identificador_ollama("llama3.2:3b")
    assert not eh_identificador_ollama("deepseek")
    assert not eh_identificador_ollama("gpt-4")
    assert not eh_identificador_ollama("openai")
