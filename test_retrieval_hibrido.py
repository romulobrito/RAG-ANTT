"""
Testes da recuperacao hibrida e da contextualizacao de chunks.

O caso que motivou este modulo: perguntas amplas ("quais sao os parametros
tecnicos para pavimentos rodoviarios?") recuperavam apenas os artigos
conceituais das normas, nunca os anexos tabulares com os limites numericos. A
causa era estrutural, nao especifica de um assunto: um bloco de tabela contem
somente rotulos curtos e numeros, sem nenhum termo da pergunta, e por isso fica
distante dela no espaco de embeddings e ausente dos candidatos.

Os testes cobrem as quatro defesas independentes de dominio: contextualizacao
de chunk, tokenizacao que preserva numeros, fusao de ranqueamentos e expansao
por documento-pai.
"""

import pytest

from langchain_core.documents import Document

from retrieval_hibrido import (
    BM25_DISPONIVEL,
    IndiceLexical,
    chave_documento,
    chunk_tem_tabela,
    classificar_fonte_qualidade,
    contextualizar_chunks,
    expandir_com_irmaos_tabulares,
    fundir_rrf,
    normalizar_para_busca,
    priorizar_fontes_estruturadas,
    tokenizar,
)

METADADOS_NORMA = {
    "nome_tipo": "Instrucao Normativa",
    "numero": "00000034",
    "ano": "2024",
    "ementa": "Estabelece parametros tecnicos de desempenho do pavimento.",
    "caminho": "dados_antt/INM/2024/INM-00000034-2024.md",
}

CHUNK_CONCEITUAL = (
    "# INSTRUCAO NORMATIVA 34/2024\n\n"
    "Art. 2 Para os efeitos desta norma, considera-se irregularidade "
    "longitudinal o indice que mede as variacoes da superficie."
)

CHUNK_TABELA = (
    "## ANEXO I - PARAMETROS DE DESEMPENHO DO PAVIMENTO\n\n"
    "### Pavimento flexivel\n\n"
    "| Pista | Trabalhos Iniciais | Recuperacao |\n"
    "| --- | --- | --- |\n"
    "| Principal | 3,5 | 2,7 |\n"
)

CHUNK_CONTINUACAO_TABELA = "| Marginal | 4,0 | 3,0 |\n| Acostamento | 4,5 | 3,5 |"


# ---------------------------------------------------------------------------
# Normalizacao e tokenizacao
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entrada, esperado",
    [
        ("SEÇÃO II", "secao ii"),
        ("Instrução Normativa", "instrucao normativa"),
        ("Índice de Condição", "indice de condicao"),
        ("", ""),
    ],
)
def test_normalizacao_remove_acentos(entrada, esperado):
    """A normalizacao deve casar grafias com e sem acento."""
    assert normalizar_para_busca(entrada) == esperado


def test_tokenizacao_preserva_valores_decimais():
    """Valores como 3,5 sao a informacao central de uma tabela de limites."""
    tokens = tokenizar("O limite de IRI e 3,5 m/km na pista principal")
    assert "3.5" in tokens
    assert "iri" in tokens


def test_tokenizacao_unifica_separador_decimal():
    """A norma escreve 3,5 e o usuario pode digitar 3.5; devem colidir."""
    assert tokenizar("valor 2,7") == tokenizar("valor 2.7")


def test_tokenizacao_descarta_stopwords():
    """Palavras funcionais degradam o BM25 sem agregar discriminacao."""
    tokens = tokenizar("Quais sao os limites de deflexao para a pista?")
    assert "limites" in tokens
    assert "deflexao" in tokens
    assert "sao" not in tokens
    assert "para" not in tokens


# ---------------------------------------------------------------------------
# Contextualizacao de chunks
# ---------------------------------------------------------------------------


def test_contextualizacao_identifica_norma_e_anexo():
    """
    O prefixo precisa dizer de que norma e de que anexo o bloco veio.

    Sem essa informacao o chunk tabular nao tem como ser recuperado por uma
    pergunta que menciona a norma ou o assunto, e nao a tabela.
    """
    resultado = contextualizar_chunks([CHUNK_CONCEITUAL, CHUNK_TABELA], METADADOS_NORMA)
    texto_tabela, metadados_tabela = resultado[1]

    assert "Instrucao Normativa 34/2024" in texto_tabela
    assert "ANEXO I" in metadados_tabela["secao"]
    assert "parametros tecnicos de desempenho" in texto_tabela
    assert metadados_tabela["contem_tabelas"] == "Sim"
    assert metadados_tabela["tem_valores_numericos"] == "Sim"


def test_contextualizacao_preserva_conteudo_original():
    """O prefixo e acrescentado; o texto da norma nao pode ser alterado."""
    (texto, _), = contextualizar_chunks([CHUNK_TABELA], METADADOS_NORMA)
    assert texto.endswith(CHUNK_TABELA)


def test_contextualizacao_repete_cabecalho_em_continuacao_de_tabela():
    """
    Tabela cortada entre chunks perde o cabecalho na segunda metade.

    Sem repeti-lo, o bloco "| Marginal | 4,0 | 3,0 |" nao informa a que
    colunas os valores pertencem, nem para a busca nem para o modelo.
    """
    resultado = contextualizar_chunks(
        [CHUNK_TABELA, CHUNK_CONTINUACAO_TABELA], METADADOS_NORMA
    )
    texto_continuacao, metadados_continuacao = resultado[1]

    assert "Continuacao da tabela" in texto_continuacao
    assert "Trabalhos Iniciais" in texto_continuacao
    assert metadados_continuacao["contem_tabelas"] == "Sim"


def test_contextualizacao_herda_secao_de_chunk_sem_titulo():
    """Chunks seguintes permanecem no anexo aberto pelo chunk anterior."""
    resultado = contextualizar_chunks(
        [CHUNK_TABELA, "Paragrafo unico. A verificacao sera semestral."],
        METADADOS_NORMA,
    )
    _, metadados_paragrafo = resultado[1]
    assert "ANEXO I" in metadados_paragrafo["secao"]


def test_contextualizacao_de_lista_vazia():
    """Documento sem chunks nao deve gerar excecao."""
    assert contextualizar_chunks([], METADADOS_NORMA) == []


# ---------------------------------------------------------------------------
# Deteccao de blocos tabulares
# ---------------------------------------------------------------------------


def test_deteccao_de_tabela_por_conteudo():
    """
    Indices criados antes desta mudanca nao tem o metadado `contem_tabelas`.

    A deteccao precisa funcionar por inspecao do conteudo para que a base
    existente se beneficie sem reindexacao.
    """
    doc = Document(page_content=CHUNK_TABELA, metadata={})
    assert chunk_tem_tabela(doc) is True


def test_deteccao_de_tabela_ignora_texto_corrido():
    """Texto sem estrutura tabular nao deve ser classificado como tabela."""
    doc = Document(page_content=CHUNK_CONCEITUAL, metadata={})
    assert chunk_tem_tabela(doc) is False


# ---------------------------------------------------------------------------
# Fusao de ranqueamentos
# ---------------------------------------------------------------------------


def _documento(caminho, chunk, conteudo="conteudo"):
    """Cria um Document minimo para os testes de fusao e expansao."""
    return Document(
        page_content=conteudo,
        metadata={"caminho": caminho, "chunk": chunk},
    )


def test_fusao_privilegia_consenso_entre_buscas():
    """
    Um chunk recuperado pelas duas buscas deve superar o primeiro de uma so.

    E o que faz a tabela emergir: ela e fraca na busca densa e forte na
    lexical, e o consenso a coloca acima de chunks que apenas uma das buscas
    considerou relevante.
    """
    consenso = _documento("a.md", 1)
    isolado = _documento("b.md", 1)
    outro = _documento("c.md", 1)

    fundidos, scores = fundir_rrf([[isolado, consenso], [outro, consenso]])

    assert fundidos[0] is consenso
    assert scores[chave_documento(consenso)] > scores[chave_documento(isolado)]


def test_fusao_deduplica_por_chunk():
    """O mesmo chunk vindo de varias listas aparece uma unica vez."""
    documento = _documento("a.md", 1)
    fundidos, _ = fundir_rrf([[documento], [documento], [documento]])
    assert len(fundidos) == 1


def test_fusao_respeita_pesos():
    """Listas com peso maior devem influenciar mais a ordenacao final."""
    preferido = _documento("a.md", 1)
    secundario = _documento("b.md", 1)

    fundidos, _ = fundir_rrf([[preferido], [secundario]], pesos=[1.0, 0.1])
    assert fundidos[0] is preferido


def test_fusao_de_listas_vazias():
    """Ausencia de candidatos nao deve gerar excecao."""
    assert fundir_rrf([]) == ([], {})
    assert fundir_rrf([[], []]) == ([], {})


# ---------------------------------------------------------------------------
# Expansao por documento-pai
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not BM25_DISPONIVEL, reason="rank_bm25 nao instalado")
def test_expansao_anexa_tabela_do_documento_relevante():
    """
    O nucleo da correcao: se a norma e relevante, seus anexos numericos entram.

    A pergunta ampla recupera o artigo conceitual da norma. O anexo com os
    limites nao tem similaridade com ela, mas pertence ao mesmo documento e e
    justamente o que responde a pergunta.
    """
    caminho = METADADOS_NORMA["caminho"]
    conceitual = _documento(caminho, 1, CHUNK_CONCEITUAL)
    tabular = _documento(caminho, 2, CHUNK_TABELA)
    indice = IndiceLexical([conceitual, tabular])

    expandido = expandir_com_irmaos_tabulares([conceitual], indice)

    assert len(expandido) == 2
    assert expandido[1].metadata["expandido_por_documento"] is True
    assert "3,5" in expandido[1].page_content


@pytest.mark.skipif(not BM25_DISPONIVEL, reason="rank_bm25 nao instalado")
def test_expansao_nao_duplica_chunk_ja_presente():
    """Chunk que ja esta no contexto nao deve ser anexado outra vez."""
    caminho = METADADOS_NORMA["caminho"]
    tabular = _documento(caminho, 2, CHUNK_TABELA)
    indice = IndiceLexical([tabular])

    expandido = expandir_com_irmaos_tabulares([tabular], indice)
    assert len(expandido) == 1


@pytest.mark.skipif(not BM25_DISPONIVEL, reason="rank_bm25 nao instalado")
def test_expansao_ignora_irmaos_sem_dados_quantitativos():
    """Texto corrido do mesmo documento nao deve consumir o orcamento."""
    caminho = METADADOS_NORMA["caminho"]
    conceitual = _documento(caminho, 1, CHUNK_CONCEITUAL)
    outro_texto = _documento(
        caminho, 3, "Art. 9 Esta norma entra em vigor na data de sua publicacao."
    )
    indice = IndiceLexical([conceitual, outro_texto])

    expandido = expandir_com_irmaos_tabulares([conceitual], indice)
    assert len(expandido) == 1


@pytest.mark.skipif(not BM25_DISPONIVEL, reason="rank_bm25 nao instalado")
def test_expansao_respeita_teto_de_chunks():
    """O orcamento de contexto do modelo limita quantos irmaos entram."""
    caminho = METADADOS_NORMA["caminho"]
    conceitual = _documento(caminho, 1, CHUNK_CONCEITUAL)
    tabulares = [
        _documento(caminho, indice, CHUNK_TABELA) for indice in range(2, 12)
    ]
    indice = IndiceLexical([conceitual] + tabulares)

    expandido = expandir_com_irmaos_tabulares(
        [conceitual], indice, max_por_documento=10, max_total=3
    )
    assert len(expandido) == 4


def test_expansao_sem_indice_retorna_entrada_intacta():
    """Sem indice lexical, a busca segue funcionando apenas com a parte densa."""
    documento = _documento("a.md", 1)
    assert expandir_com_irmaos_tabulares([documento], None) == [documento]


# ---------------------------------------------------------------------------
# Indice lexical
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not BM25_DISPONIVEL, reason="rank_bm25 nao instalado")
def test_busca_lexical_recupera_bloco_numerico():
    """
    A busca lexical e a defesa contra a fraqueza da densa em numeros e siglas.

    Uma consulta pelo valor exato da tabela precisa retornar o bloco tabular.
    """
    tabular = _documento("a.md", 1, CHUNK_TABELA)
    conceitual = _documento("a.md", 2, CHUNK_CONCEITUAL)
    indice = IndiceLexical([conceitual, tabular])

    resultados = indice.buscar("pista principal 3,5 recuperacao", k=2)

    assert resultados
    assert "3,5" in resultados[0].page_content


@pytest.mark.skipif(not BM25_DISPONIVEL, reason="rank_bm25 nao instalado")
def test_busca_lexical_aplica_filtro_de_metadados():
    """Os filtros da barra lateral devem valer tambem na busca lexical."""
    doc_2024 = Document(
        page_content=CHUNK_TABELA,
        metadata={"caminho": "a.md", "chunk": 1, "ano": "2024"},
    )
    doc_2019 = Document(
        page_content=CHUNK_TABELA,
        metadata={"caminho": "b.md", "chunk": 1, "ano": "2019"},
    )
    indice = IndiceLexical([doc_2024, doc_2019])

    resultados = indice.buscar("pista principal", k=5, filtro={"ano": "2019"})

    assert len(resultados) == 1
    assert resultados[0].metadata["ano"] == "2019"


@pytest.mark.skipif(not BM25_DISPONIVEL, reason="rank_bm25 nao instalado")
def test_busca_lexical_nao_muta_documentos_indexados():
    """
    O indice referencia os chunks do docstore do FAISS.

    Escrever metadados nesses objetos contaminaria o indice para as consultas
    seguintes, portanto a busca deve devolver copias.
    """
    original = _documento("a.md", 1, CHUNK_TABELA)
    indice = IndiceLexical([original])

    resultados = indice.buscar("pista principal", k=1)
    resultados[0].metadata["prioritario"] = True

    assert "prioritario" not in original.metadata


@pytest.mark.skipif(not BM25_DISPONIVEL, reason="rank_bm25 nao instalado")
def test_busca_lexical_com_consulta_sem_tokens_uteis():
    """Consulta formada apenas por stopwords nao deve gerar excecao."""
    indice = IndiceLexical([_documento("a.md", 1, CHUNK_TABELA)])
    assert indice.buscar("os de para a", k=5) == []


# ---------------------------------------------------------------------------
# Preferencia de fonte estruturada sobre OCR
# ---------------------------------------------------------------------------


def test_classifica_tabela_auxiliar_como_estruturada():
    """O separador de mesclagem e o sinal canonico de transcricao limpa."""
    texto = (
        "## Tabelas auxiliares estruturadas (transcricao) - INM-34\n\n"
        "| Pista | Limite |\n| --- | --- |\n| Principal | 3,5 |"
    )
    assert classificar_fonte_qualidade(texto, {}) == "estruturada"


def test_classifica_anexo_ocr_como_ocr():
    """Cabecalhos injetados pelo enriquecimento de imagem sao OCR."""
    texto = "## Anexo OCR - 1871662.png\n\nPrincipal 12mm 7mm"
    assert classificar_fonte_qualidade(
        texto, {"secao": "Anexo OCR - 1871662.png"}
    ) == "ocr"


def test_priorizar_sobe_estruturada_e_rebaixa_ocr_da_mesma_norma():
    """
    Na mesma norma, a tabela auxiliar deve preceder o OCR no contexto.

    E a defesa contra o modelo preencher fases a partir de grades OCR
    ruidosas quando ja existe transcricao limpa.
    """
    caminho = "dados_antt/INM/2024/INM-00000034-2024.md"
    ocr = Document(
        page_content="## Anexo OCR - 1.png\n12mm 7mm",
        metadata={"caminho": caminho, "chunk": 1, "secao": "Anexo OCR - 1.png"},
    )
    estruturada = Document(
        page_content=(
            "## Tabelas auxiliares estruturadas (transcricao)\n\n"
            "| Pista | Limite |\n| --- | --- |\n| Principal | 12 mm |"
        ),
        metadata={"caminho": caminho, "chunk": 2},
    )
    legal = Document(
        page_content="Art. 2 Definicões.",
        metadata={"caminho": caminho, "chunk": 3},
    )

    ordenados = priorizar_fontes_estruturadas([ocr, legal, estruturada])

    assert ordenados[0].metadata["fonte_qualidade"] == "estruturada"
    assert ordenados[-1].metadata["fonte_qualidade"] == "ocr"
    assert "12 mm" in ordenados[0].page_content
