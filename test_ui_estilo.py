"""
Guarda de regressao visual da camada de apresentacao (Fase 1).

Estes testes nao validam comportamento de negocio: eles impedem que o padrao
visual institucional definido em planning/frontend-institucional.md regrida em
manutencoes futuras (retorno de emojis, cores Bootstrap ou feedback ludico).

Escopo: funcao interface_usuario_unificada() de antt_rag_unified.py, os
helpers de apresentacao e as strings devolvidas ao usuario final.
"""

import re
import textwrap
import unicodedata
from pathlib import Path
from typing import List, Tuple

import pytest

ARQUIVO_PRINCIPAL = Path(__file__).parent / "antt_rag_unified.py"
ARQUIVO_CONFIG = Path(__file__).parent / "config.py"
ARQUIVO_PROVIDERS = Path(__file__).parent / "llm_providers.py"
ARQUIVO_TEMA = Path(__file__).parent / "ui" / "theme.py"
ARQUIVO_TEMA_NATIVO = Path(__file__).parent / ".streamlit" / "config.toml"

# Faixas Unicode de pictogramas, simbolos diversos, setas decorativas e
# seletores de variacao usados por emojis.
_PADRAO_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # pictogramas, transportes, simbolos suplementares
    "\u2600-\u27BF"          # simbolos diversos e dingbats
    "\u2B00-\u2BFF"          # setas e formas geometricas suplementares
    "\u2190-\u21FF"          # setas
    "\uFE0F"                 # seletor de variacao (emoji presentation)
    "\u20E3"                 # keycap
    "]"
)

# Paleta Bootstrap substituida pela paleta gov.br na Fase 2. Inclui tambem o
# gradiente decorativo do cabecalho antigo e cores neutras arbitrarias.
_CORES_BOOTSTRAP = (
    "#007bff",
    "#28a745",
    "#d4edda",
    "#f8d7da",
    "#fff3cd",
    "#e9ecef",
    "#f8f9fa",
    "#1e3c72",
    "#2a5298",
    "#666666",
)

# Tokens oficiais do Design System gov.br (@govbr-ds/core@3.7.0). Qualquer cor
# usada na interface precisa pertencer a esta paleta.
_TOKENS_GOVBR = (
    "#071D41",  # --blue-warm-vivid-90
    "#0C326F",  # --blue-warm-vivid-80
    "#1351B4",  # --blue-warm-vivid-70
    "#155BCB",  # --blue-warm-vivid-60
    "#5992ED",  # --blue-warm-vivid-40
    "#EDF5FF",  # --blue-warm-vivid-5
    "#333333",  # --gray-80
    "#636363",  # --gray-60
    "#CCCCCC",  # --gray-20
    "#E6E6E6",  # --gray-10
    "#F8F8F8",  # --gray-2
    "#FFFFFF",  # --pure-0
    "#168821",  # --green-cool-vivid-50
    "#E3F5E1",  # --green-cool-vivid-5
    "#19311E",  # --green-cool-vivid-80
    "#E52207",  # --red-vivid-50
    "#FFF3F2",  # --red-vivid-5
    "#5C1111",  # --red-vivid-80
    "#FFCD07",  # --yellow-vivid-20
    "#FFF5C2",  # --yellow-vivid-5
    "#352313",  # --orange-vivid-80
)

_PADRAO_HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")

# Termos de linguagem comercial que nao devem aparecer na interface do
# usuario final (preocupacao de operacao, nao de quem consulta a norma).
_TERMOS_COMERCIAIS = (
    "gratuito",
    "gratuita",
    "consome creditos",
    "consome créditos",
    "modo pago",
    "sem custos",
)


def _ler(caminho: Path) -> str:
    """
    Le um arquivo de codigo em UTF-8.

    Args:
        caminho: Caminho do arquivo a ser lido.

    Returns:
        Conteudo textual do arquivo.

    Raises:
        FileNotFoundError: Se o arquivo nao existir.
    """
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {caminho}")
    return caminho.read_text(encoding="utf-8")


def _extrair_bloco_ui() -> List[Tuple[int, str]]:
    """
    Extrai as linhas da camada de apresentacao do arquivo principal.

    O bloco comeca na definicao de interface_usuario_unificada() e vai ate o
    fim do arquivo, cobrindo tambem main().

    Returns:
        Lista de tuplas (numero_da_linha, conteudo) da camada de UI.

    Raises:
        AssertionError: Se a funcao de interface nao for localizada.
    """
    linhas = _ler(ARQUIVO_PRINCIPAL).splitlines()
    indice_inicio = None
    for indice, linha in enumerate(linhas):
        if linha.startswith("def interface_usuario_unificada"):
            indice_inicio = indice
            break

    assert indice_inicio is not None, (
        "Funcao interface_usuario_unificada() nao encontrada em "
        f"{ARQUIVO_PRINCIPAL.name}. O teste de estilo precisa ser ajustado."
    )

    return [
        (indice_inicio + deslocamento + 1, texto)
        for deslocamento, texto in enumerate(linhas[indice_inicio:])
    ]


def _remover_acentos(texto: str) -> str:
    """
    Remove acentuacao para comparacao de termos insensivel a diacriticos.

    Args:
        texto: Texto de entrada.

    Returns:
        Texto sem marcas de combinacao.
    """
    decomposto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in decomposto if not unicodedata.combining(c))


def test_ui_sem_emojis():
    """A camada de apresentacao nao deve conter emojis."""
    ocorrencias = [
        f"linha {numero}: {conteudo.strip()[:100]}"
        for numero, conteudo in _extrair_bloco_ui()
        if _PADRAO_EMOJI.search(conteudo)
    ]

    assert not ocorrencias, (
        "Emojis encontrados na camada de UI (padrao institucional exige "
        "rotulos textuais):\n" + "\n".join(ocorrencias)
    )


def test_page_icon_sem_emoji():
    """O favicon configurado nao deve ser um emoji."""
    conteudo = _ler(ARQUIVO_CONFIG)
    linha_icone = [
        linha
        for linha in conteudo.splitlines()
        if linha.startswith("STREAMLIT_PAGE_ICON")
    ]

    assert linha_icone, "Constante STREAMLIT_PAGE_ICON nao encontrada."
    assert not _PADRAO_EMOJI.search(linha_icone[0]), (
        f"STREAMLIT_PAGE_ICON ainda usa emoji: {linha_icone[0]}"
    )


def test_rotulos_de_provedores_seguem_padrao():
    """
    Rotulos definidos fora da funcao de UI tambem chegam a tela.

    Os campos "name" de get_available_embedding_providers e de LLM_PROVIDERS
    sao renderizados na barra lateral e no painel de informacoes, portanto
    valem para eles as mesmas regras de estilo.
    """
    from llm_providers import get_available_embedding_providers

    problemas = []
    for chave, dados in get_available_embedding_providers().items():
        rotulo = dados.get("name", "")
        if _PADRAO_EMOJI.search(rotulo):
            problemas.append(f"{chave}: emoji em '{rotulo}'")
        texto = _remover_acentos(rotulo).lower()
        for termo in _TERMOS_COMERCIAIS:
            if _remover_acentos(termo).lower() in texto:
                problemas.append(f"{chave}: termo comercial '{termo}' em '{rotulo}'")

    assert not problemas, (
        "Rotulos de provedores exibidos na interface fora do padrao:\n"
        + "\n".join(problemas)
    )


def test_ui_sem_feedback_ludico():
    """Efeitos decorativos nao sao adequados a um sistema institucional."""
    proibidos = ("st.balloons(", "st.snow(")
    ocorrencias = [
        f"linha {numero}: {efeito}"
        for numero, conteudo in _extrair_bloco_ui()
        for efeito in proibidos
        if efeito in conteudo
    ]

    assert not ocorrencias, (
        "Feedback ludico encontrado na UI:\n" + "\n".join(ocorrencias)
    )


def test_ui_sem_linguagem_comercial():
    """
    A tela do usuario final nao deve tratar de custo ou gratuidade.

    Custo de API e assunto de operacao e pertence ao painel administrativo
    previsto na Fase 3, nao a interface de consulta normativa.
    """
    ocorrencias = []
    for numero, conteudo in _extrair_bloco_ui():
        texto = _remover_acentos(conteudo).lower()
        for termo in _TERMOS_COMERCIAIS:
            if _remover_acentos(termo).lower() in texto:
                ocorrencias.append(f"linha {numero} ({termo}): {conteudo.strip()[:100]}")

    assert not ocorrencias, (
        "Linguagem comercial encontrada na UI:\n" + "\n".join(ocorrencias)
    )


# O balao de ajuda do Streamlit e estreito e corta textos longos.
# Orientacao extensa pertence a secao Ajuda da barra lateral.
_LIMITE_TOOLTIP = 120

_PADRAO_HELP = re.compile(r'help=\(?\s*((?:"[^"]*"\s*)+)')


def test_tooltips_sao_curtos():
    """
    Nenhum tooltip deve exceder o limite que o balao do Streamlit comporta.

    Textos longos aparecem truncados para o usuario. Explicacao detalhada
    deve ir para a secao Ajuda, que tem espaco adequado.
    """
    conteudo = _ler(ARQUIVO_PRINCIPAL)
    inicio_ui = conteudo.index("def interface_usuario_unificada")

    excedentes = []
    for casamento in _PADRAO_HELP.finditer(conteudo[inicio_ui:]):
        texto = "".join(re.findall(r'"([^"]*)"', casamento.group(1)))
        if len(texto) > _LIMITE_TOOLTIP:
            excedentes.append(f"{len(texto)} caracteres: {texto[:80]}...")

    assert not excedentes, (
        f"Tooltips acima de {_LIMITE_TOOLTIP} caracteres serao truncados "
        "na interface. Mova a explicacao para a secao Ajuda:\n"
        + "\n".join(excedentes)
    )


def test_existe_secao_ajuda():
    """A barra lateral deve manter a secao Ajuda com orientacao detalhada."""
    conteudo = _ler(ARQUIVO_PRINCIPAL)

    assert 'st.expander("Ajuda")' in conteudo, (
        "Secao Ajuda ausente. Ela concentra a orientacao que nao cabe "
        "nos tooltips."
    )


def test_tooltip_com_css_ampliado():
    """O CSS deve ampliar o balao de ajuda padrao do Streamlit."""
    from ui.theme import montar_css_institucional

    css = montar_css_institucional()

    assert '[data-testid="stTooltipContent"]' in css, (
        "Regra de CSS do tooltip ausente; o balao voltara a largura padrao."
    )
    assert "max-width: 360px" in css


def test_badge_tipo_documento_gera_html_valido():
    """O badge institucional deve produzir HTML com faixa lateral colorida."""
    from antt_rag_unified import badge_tipo_documento

    html_gerado = badge_tipo_documento("Instrucao Normativa")

    assert 'class="badge-tipo"' in html_gerado
    assert "border-left:3px solid #1351B4" in html_gerado
    assert "Instrucao Normativa" in html_gerado
    assert not _PADRAO_EMOJI.search(html_gerado)


def test_badge_tipo_documento_aceita_acentuacao():
    """Rotulos acentuados devem casar com a mesma cor institucional."""
    from antt_rag_unified import badge_tipo_documento

    com_acento = badge_tipo_documento("Instrução Normativa")
    sem_acento = badge_tipo_documento("Instrucao Normativa")

    assert "#1351B4" in com_acento
    assert "#1351B4" in sem_acento


def test_badge_tipo_documento_usa_cor_padrao_para_tipo_desconhecido():
    """Tipos nao mapeados recebem a cor neutra, sem quebrar a renderizacao."""
    from antt_rag_unified import badge_tipo_documento
    from ui.theme import CINZA_SECUNDARIO

    html_gerado = badge_tipo_documento("Tipo Inexistente")

    assert CINZA_SECUNDARIO in html_gerado
    assert "Tipo Inexistente" in html_gerado


def test_badge_tipo_documento_trata_entrada_vazia():
    """Entrada vazia nao deve gerar badge sem rotulo."""
    from antt_rag_unified import badge_tipo_documento

    assert "Documento" in badge_tipo_documento("")
    assert "Documento" in badge_tipo_documento("   ")


def test_badge_tipo_documento_escapa_html():
    """O rotulo deve ser escapado para evitar injecao de marcacao."""
    from antt_rag_unified import badge_tipo_documento

    html_gerado = badge_tipo_documento("<script>alert(1)</script>")

    assert "<script>" not in html_gerado
    assert "&lt;script&gt;" in html_gerado


def test_ui_sem_cores_bootstrap():
    """A camada de UI nao deve conter cores fora da paleta gov.br."""
    fontes = [
        (ARQUIVO_PRINCIPAL.name, _extrair_bloco_ui()),
        (
            ARQUIVO_TEMA.name,
            list(enumerate(_ler(ARQUIVO_TEMA).splitlines(), start=1)),
        ),
    ]

    ocorrencias = [
        f"{arquivo} linha {numero}: {cor}"
        for arquivo, linhas in fontes
        for numero, conteudo in linhas
        for cor in _CORES_BOOTSTRAP
        if cor in conteudo.lower()
    ]

    assert not ocorrencias, (
        "Cores fora da paleta gov.br encontradas:\n" + "\n".join(ocorrencias)
    )


def test_tema_usa_apenas_tokens_oficiais():
    """
    Todo valor hexadecimal do tema deve ser um token do gov.br.

    Impede que manutencoes futuras introduzam cores arbitrarias, o que
    quebraria a consistencia com o Design System e o contraste verificado.
    """
    tokens_validos = {cor.lower() for cor in _TOKENS_GOVBR}
    tokens_validos.add("#fff")  # abreviacao aceita para --pure-0

    encontrados = {
        cor.lower() for cor in _PADRAO_HEX.findall(_ler(ARQUIVO_TEMA))
    }
    invalidos = sorted(encontrados - tokens_validos)

    assert not invalidos, (
        "Cores fora do Design System gov.br em ui/theme.py: "
        + ", ".join(invalidos)
    )


def test_tema_nativo_do_streamlit_existe():
    """
    O tema nativo deve estar versionado e alinhado ao CSS injetado.

    Sem o config.toml, os componentes nativos do Streamlit (botoes, campos,
    barra lateral) voltam a paleta padrao vermelha, que destoa do gov.br.
    """
    assert ARQUIVO_TEMA_NATIVO.exists(), (
        "Arquivo .streamlit/config.toml ausente. Verifique tambem se a "
        "excecao ao padrao *.toml continua no .gitignore."
    )

    conteudo = _ler(ARQUIVO_TEMA_NATIVO)

    assert 'primaryColor = "#1351B4"' in conteudo, (
        "A cor primaria do tema nativo deve ser o token gov.br #1351B4."
    )
    assert 'textColor = "#333333"' in conteudo, (
        "A cor de texto do tema nativo deve ser o token gov.br #333333."
    )


def test_campo_de_pergunta_e_redimensionavel():
    """
    O CSS deve liberar o arraste vertical e manter a largura total.

    O arraste horizontal (resize: both) quebrava o botao de envio. A altura
    vem do preset e ainda pode ser refinada com a alca do navegador.
    """
    from ui.theme import montar_css_institucional

    css = montar_css_institucional()

    assert "resize: vertical" in css, (
        "Campo de pergunta precisa de resize: vertical para arrastar a altura."
    )
    assert "resize: both" not in css, (
        "resize: both regressou: a largura deve permanecer total."
    )
    assert "min-height: 8.0rem" in css or "min-height: 8rem" in css, (
        "Altura minima do preset padrao ausente."
    )
    assert "align-items: flex-end" in css, (
        "Botao de envio deve permanecer alinhado a base do campo."
    )
    assert '[data-testid="stChatInput"]' in css


def test_presets_de_altura_do_campo_de_pergunta():
    """Presets Compacto, Padrao e Amplo devem gerar alturas distintas."""
    from ui.theme import (
        ALTURAS_CAMPO_PERGUNTA,
        montar_css_institucional,
        resolver_altura_campo_pergunta,
    )

    assert resolver_altura_campo_pergunta("compacto") < resolver_altura_campo_pergunta(
        "padrao"
    )
    assert resolver_altura_campo_pergunta("padrao") < resolver_altura_campo_pergunta(
        "amplo"
    )
    assert resolver_altura_campo_pergunta("inexistente") == ALTURAS_CAMPO_PERGUNTA[
        "padrao"
    ]

    css_amplo = montar_css_institucional(
        altura_campo_rem=ALTURAS_CAMPO_PERGUNTA["amplo"]
    )
    assert f"min-height: {ALTURAS_CAMPO_PERGUNTA['amplo']}rem" in css_amplo


def test_campo_de_pergunta_fica_fixo_no_rodape():
    """
    A pergunta deve usar st.chat_input, que o Streamlit fixa no rodape.

    Com st.text_area o campo ficava no meio da pagina, acima da ultima
    resposta: a cada nova pergunta o usuario precisava rolar para cima para
    encontrar onde escrever.
    """
    import ast

    arvore = ast.parse(_ler(ARQUIVO_PRINCIPAL))

    chamadas_com_rotulo = []
    for no in ast.walk(arvore):
        if not isinstance(no, ast.Call) or not isinstance(no.func, ast.Attribute):
            continue
        primeiro = no.args[0] if no.args else None
        if not isinstance(primeiro, ast.Constant):
            continue
        if not isinstance(primeiro.value, str):
            continue
        if "Digite sua pergunta" in primeiro.value:
            chamadas_com_rotulo.append((no.func.attr, no.lineno))

    assert chamadas_com_rotulo, "Campo de pergunta nao encontrado."

    fora_do_padrao = [
        f"linha {linha}: st.{funcao}"
        for funcao, linha in chamadas_com_rotulo
        if funcao != "chat_input"
    ]

    assert not fora_do_padrao, (
        "O campo de pergunta deve usar st.chat_input, que o Streamlit fixa "
        "no rodape:\n" + "\n".join(fora_do_padrao)
    )


def test_conversa_ocupa_largura_total():
    """
    A conversa nao deve ficar espremida em uma coluna de dois tercos.

    A coluna lateral "Informacoes" repetia dados da barra lateral e reduzia
    a area util de leitura da resposta.
    """
    conteudo = "\n".join(texto for _, texto in _extrair_bloco_ui())

    assert "col_main, col_info" not in conteudo, (
        "A divisao em duas colunas voltou; a conversa deve ocupar a largura "
        "total da area principal."
    )


def test_estilo_e_reinjetado_a_cada_execucao():
    """
    O CSS nao pode ser condicionado a uma flag de sessao.

    O Streamlit reconstroi a arvore de elementos a cada interacao e descarta
    o que nao for reemitido. Uma guarda do tipo "injetar so uma vez" faz o
    tema desaparecer no primeiro clique do usuario.
    """
    import ast
    import inspect

    from ui.theme import aplicar_estilo_institucional

    arvore = ast.parse(
        textwrap.dedent(inspect.getsource(aplicar_estilo_institucional))
    )

    # Remove a docstring: ela explica justamente por que a guarda nao existe,
    # e a mencao ao termo nao pode reprovar o teste.
    referencias = [
        no.attr
        for no in ast.walk(arvore)
        if isinstance(no, ast.Attribute) and no.attr == "session_state"
    ]

    assert not referencias, (
        "aplicar_estilo_institucional() nao deve depender de session_state; "
        "o estilo precisa ser reemitido em toda execucao do script."
    )


def test_cabecalho_sem_gradiente_decorativo():
    """O cabecalho institucional usa fundo solido, nao gradiente."""
    conteudo = _ler(ARQUIVO_TEMA)

    assert "linear-gradient" not in conteudo, (
        "Gradiente decorativo encontrado no cabecalho; o padrao gov.br "
        "preve fundo solido com faixa na cor primaria."
    )
    assert ".app-header" in conteudo, "Classe .app-header ausente no tema."


def test_montar_cabecalho_escapa_conteudo():
    """Titulo e subtitulo devem ser escapados antes de ir para o HTML."""
    from ui.theme import montar_cabecalho

    html_gerado = montar_cabecalho("<b>Titulo</b>", "<i>Sub</i>")

    assert "<b>" not in html_gerado
    assert "&lt;b&gt;Titulo&lt;/b&gt;" in html_gerado
    assert "&lt;i&gt;Sub&lt;/i&gt;" in html_gerado


def test_montar_cabecalho_rejeita_titulo_vazio():
    """Cabecalho sem titulo indica erro de programacao, nao estado valido."""
    from ui.theme import montar_cabecalho

    with pytest.raises(ValueError):
        montar_cabecalho("   ")


def test_montar_cabecalho_omite_subtitulo_ausente():
    """Sem subtitulo, o paragrafo descritivo nao deve ser renderizado."""
    from ui.theme import montar_cabecalho

    assert "<p>" not in montar_cabecalho("Titulo")


def _luminancia_relativa(cor_hex: str) -> float:
    """
    Calcula a luminancia relativa de uma cor conforme a WCAG 2.1.

    Args:
        cor_hex: Cor no formato "#RRGGBB".

    Returns:
        Luminancia relativa no intervalo [0, 1].

    Raises:
        ValueError: Se a cor nao estiver no formato hexadecimal de 6 digitos.
    """
    valor = cor_hex.lstrip("#")
    if len(valor) != 6:
        raise ValueError(f"Cor fora do formato #RRGGBB: {cor_hex}")

    canais = []
    for inicio in (0, 2, 4):
        proporcao = int(valor[inicio:inicio + 2], 16) / 255.0
        if proporcao <= 0.03928:
            canais.append(proporcao / 12.92)
        else:
            canais.append(((proporcao + 0.055) / 1.055) ** 2.4)

    return 0.2126 * canais[0] + 0.7152 * canais[1] + 0.0722 * canais[2]


def _razao_de_contraste(frente: str, fundo: str) -> float:
    """
    Calcula a razao de contraste entre duas cores conforme a WCAG 2.1.

    Args:
        frente: Cor do texto no formato "#RRGGBB".
        fundo: Cor do fundo no formato "#RRGGBB".

    Returns:
        Razao de contraste no intervalo [1, 21].
    """
    luminancias = sorted(
        (_luminancia_relativa(frente), _luminancia_relativa(fundo)),
        reverse=True,
    )
    return (luminancias[0] + 0.05) / (luminancias[1] + 0.05)


# Contraste minimo da WCAG 2.1 nivel AA para texto normal.
_CONTRASTE_MINIMO_AA = 4.5


def test_pares_de_cor_atendem_contraste_aa():
    """
    Cada combinacao texto/fundo da interface deve atingir WCAG AA.

    Requisito de acessibilidade do padrao gov.br e criterio de homologacao
    do sistema. Texto normal exige razao de contraste de pelo menos 4,5:1.
    """
    from ui import theme

    pares = (
        ("texto corrente", theme.CINZA_TEXTO, theme.BRANCO),
        ("texto secundario", theme.CINZA_SECUNDARIO, theme.BRANCO),
        ("texto sobre bloco", theme.CINZA_TEXTO, theme.CINZA_FUNDO),
        ("legenda sobre bloco", theme.CINZA_SECUNDARIO, theme.CINZA_FUNDO),
        ("cabecalho", theme.BRANCO, theme.AZUL_INSTITUCIONAL),
        ("acao primaria", theme.BRANCO, theme.AZUL_PRIMARIO),
        ("situacao positiva", theme.VERDE_TEXTO, theme.VERDE_FUNDO),
        ("situacao de erro", theme.VERMELHO_TEXTO, theme.VERMELHO_FUNDO),
        ("situacao de alerta", theme.AMARELO_TEXTO, theme.AMARELO_FUNDO),
    )

    reprovados = []
    for descricao, frente, fundo in pares:
        razao = _razao_de_contraste(frente, fundo)
        if razao < _CONTRASTE_MINIMO_AA:
            reprovados.append(
                f"{descricao}: {frente} sobre {fundo} = {razao:.2f}:1"
            )

    assert not reprovados, (
        f"Pares de cor abaixo de {_CONTRASTE_MINIMO_AA}:1 (WCAG AA):\n"
        + "\n".join(reprovados)
    )
