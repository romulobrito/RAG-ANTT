"""
Tema institucional alinhado ao Design System do gov.br (DSGov).

Centraliza os tokens de cor e o CSS da aplicacao para que a identidade visual
seja definida em um unico lugar, em vez de ficar espalhada em blocos de estilo
embutidos na funcao de interface.

Todos os valores hexadecimais deste modulo sao tokens oficiais extraidos de
@govbr-ds/core@3.7.0/dist/core-tokens.css. Ao alterar qualquer cor, use um
token existente da paleta; nao introduza cores arbitrarias.

Referencia: https://www.gov.br/ds/fundamentos-visuais/cor
"""

import html
from typing import Dict, Final, Mapping

import streamlit as st

# ---------------------------------------------------------------------------
# Tokens de cor (DSGov)
# ---------------------------------------------------------------------------

# Familia Blue Warm Vivid: identidade primaria do governo federal.
AZUL_INSTITUCIONAL: Final[str] = "#071D41"  # --blue-warm-vivid-90
AZUL_ESCURO: Final[str] = "#0C326F"         # --blue-warm-vivid-80
AZUL_PRIMARIO: Final[str] = "#1351B4"       # --blue-warm-vivid-70
AZUL_HOVER: Final[str] = "#155BCB"          # --blue-warm-vivid-60
AZUL_MEDIO: Final[str] = "#5992ED"          # --blue-warm-vivid-40
AZUL_CLARO: Final[str] = "#EDF5FF"          # --blue-warm-vivid-5

# Familia Gray: texto, bordas e superficies neutras.
CINZA_TEXTO: Final[str] = "#333333"         # --gray-80
CINZA_SECUNDARIO: Final[str] = "#636363"    # --gray-60
CINZA_BORDA: Final[str] = "#CCCCCC"         # --gray-20
CINZA_SEPARADOR: Final[str] = "#E6E6E6"     # --gray-10
CINZA_FUNDO: Final[str] = "#F8F8F8"         # --gray-2
BRANCO: Final[str] = "#FFFFFF"              # --pure-0

# Cores de estado. Cada situacao usa um trio (fundo claro, borda saturada,
# texto escuro) para garantir contraste minimo AA sem depender apenas da cor.
VERDE_SUCESSO: Final[str] = "#168821"       # --green-cool-vivid-50
VERDE_FUNDO: Final[str] = "#E3F5E1"         # --green-cool-vivid-5
VERDE_TEXTO: Final[str] = "#19311E"         # --green-cool-vivid-80

VERMELHO_ERRO: Final[str] = "#E52207"       # --red-vivid-50
VERMELHO_FUNDO: Final[str] = "#FFF3F2"      # --red-vivid-5
VERMELHO_TEXTO: Final[str] = "#5C1111"      # --red-vivid-80

AMARELO_ALERTA: Final[str] = "#FFCD07"      # --yellow-vivid-20
AMARELO_FUNDO: Final[str] = "#FFF5C2"       # --yellow-vivid-5
AMARELO_TEXTO: Final[str] = "#352313"       # --orange-vivid-80

# ---------------------------------------------------------------------------
# Escala tipografica
# ---------------------------------------------------------------------------

TAMANHO_TITULO_APP: Final[str] = "1.5rem"
TAMANHO_TITULO_SECAO: Final[str] = "1.15rem"
TAMANHO_SUBSECAO: Final[str] = "1rem"
TAMANHO_CORPO: Final[str] = "0.95rem"
TAMANHO_LEGENDA: Final[str] = "0.8rem"

# Espacamento vertical padrao entre blocos de conteudo.
ESPACAMENTO_BLOCO: Final[str] = "1.5rem"

# Alturas do campo de pergunta no rodape (em rem). A largura permanece
# sempre 100%: o arraste horizontal quebrava o alinhamento do botao de
# envio no layout do Streamlit.
ALTURAS_CAMPO_PERGUNTA: Final[Mapping[str, float]] = {
    "compacto": 4.0,
    "padrao": 8.0,
    "amplo": 14.0,
}
ALTURA_CAMPO_PADRAO: Final[str] = "padrao"
CHAVE_ALTURA_CAMPO: Final[str] = "altura_campo_pergunta"


def resolver_altura_campo_pergunta(chave: str) -> float:
    """
    Converte o identificador do preset de altura em valor em rem.

    Args:
        chave: Um dos identificadores em ALTURAS_CAMPO_PERGUNTA.

    Returns:
        Altura em rem. Valores desconhecidos caem no preset padrao.
    """
    if chave in ALTURAS_CAMPO_PERGUNTA:
        return ALTURAS_CAMPO_PERGUNTA[chave]
    return ALTURAS_CAMPO_PERGUNTA[ALTURA_CAMPO_PADRAO]


def rotulos_altura_campo_pergunta() -> Dict[str, str]:
    """
    Rotulos em portugues para os presets de altura do campo de pergunta.

    Returns:
        Mapa chave -> rotulo exibido na barra lateral.
    """
    return {
        "compacto": "Compacto",
        "padrao": "Padrao",
        "amplo": "Amplo",
    }


def montar_css_institucional(altura_campo_rem: float | None = None) -> str:
    """
    Monta a folha de estilo institucional da aplicacao.

    Funcao pura: nao depende do estado do Streamlit, o que permite validar o
    CSS em testes automatizados sem subir a interface.

    Args:
        altura_campo_rem: Altura minima do campo de pergunta, em rem.
            Quando omitida, usa o preset padrao.

    Returns:
        Bloco <style> completo, pronto para injecao via st.markdown.

    Raises:
        ValueError: Se a altura for menor ou igual a zero.
    """
    if altura_campo_rem is None:
        altura = ALTURAS_CAMPO_PERGUNTA[ALTURA_CAMPO_PADRAO]
    else:
        if altura_campo_rem <= 0:
            raise ValueError("A altura do campo de pergunta deve ser positiva.")
        altura = altura_campo_rem

    return f"""
    <style>
    /* ------------------------------------------------------------------
       Cabecalho institucional
       Fundo solido na cor da barra superior do gov.br, com faixa inferior
       na cor primaria. Substitui o gradiente decorativo anterior.
       ------------------------------------------------------------------ */
    .app-header {{
        background-color: {AZUL_INSTITUCIONAL};
        border-bottom: 4px solid {AZUL_PRIMARIO};
        padding: 1.25rem 1.5rem;
        margin-bottom: {ESPACAMENTO_BLOCO};
        color: {BRANCO};
    }}
    .app-header h1 {{
        font-size: {TAMANHO_TITULO_APP};
        font-weight: 600;
        line-height: 1.3;
        margin: 0;
        padding: 0;
        color: {BRANCO};
        letter-spacing: 0.2px;
    }}
    .app-header p {{
        font-size: {TAMANHO_CORPO};
        font-weight: 400;
        margin: 0.35rem 0 0 0;
        color: {BRANCO};
        opacity: 0.85;
    }}

    /* ------------------------------------------------------------------
       Barra decorativa do topo
       O Streamlit desenha uma faixa em degrade rosa/laranja acima do
       conteudo. Substitui pela cor primaria institucional.
       ------------------------------------------------------------------ */
    /* O !important e necessario porque o Streamlit aplica o degrade em um
       estilo de maior especificidade, gerado em tempo de execucao. */
    [data-testid="stDecoration"], #stDecoration {{
        background-image: none !important;
        background-color: {AZUL_PRIMARIO} !important;
    }}

    /* ------------------------------------------------------------------
       Escala tipografica
       Aplicada por CSS para manter consistencia mesmo quando o codigo usa
       st.header, st.subheader ou markdown com sustenidos.
       ------------------------------------------------------------------ */
    .stApp h2 {{
        font-size: {TAMANHO_TITULO_SECAO};
        font-weight: 600;
        color: {CINZA_TEXTO};
        margin-top: {ESPACAMENTO_BLOCO};
    }}
    .stApp h3, .stApp h4 {{
        font-size: {TAMANHO_SUBSECAO};
        font-weight: 600;
        color: {CINZA_TEXTO};
        margin-top: 1rem;
    }}

    /* ------------------------------------------------------------------
       Situacao dos servicos
       ------------------------------------------------------------------ */
    .provider-status {{
        padding: 0.5rem 0.75rem;
        margin: 0.35rem 0;
        border-radius: 2px;
        border-left: 3px solid transparent;
        font-size: {TAMANHO_LEGENDA};
        font-weight: 600;
    }}
    .status-ok {{
        background-color: {VERDE_FUNDO};
        border-left-color: {VERDE_SUCESSO};
        color: {VERDE_TEXTO};
    }}
    .status-error {{
        background-color: {VERMELHO_FUNDO};
        border-left-color: {VERMELHO_ERRO};
        color: {VERMELHO_TEXTO};
    }}
    .status-warning {{
        background-color: {AMARELO_FUNDO};
        border-left-color: {AMARELO_ALERTA};
        color: {AMARELO_TEXTO};
    }}

    /* ------------------------------------------------------------------
       Blocos de conteudo
       ------------------------------------------------------------------ */
    .metric-card {{
        background-color: {CINZA_FUNDO};
        border: 1px solid {CINZA_SEPARADOR};
        border-left: 3px solid {AZUL_PRIMARIO};
        border-radius: 2px;
        padding: 0.85rem 1rem;
        margin: 0.5rem 0 {ESPACAMENTO_BLOCO} 0;
    }}
    .metric-card h4 {{
        font-size: {TAMANHO_SUBSECAO};
        font-weight: 600;
        color: {CINZA_TEXTO};
        margin: 0 0 0.4rem 0;
    }}
    .metric-card p {{
        font-size: {TAMANHO_LEGENDA};
        color: {CINZA_SECUNDARIO};
        margin: 0.15rem 0;
    }}
    .citation-box, .bloco-fonte {{
        background-color: {CINZA_FUNDO};
        border: 1px solid {CINZA_BORDA};
        border-left: 3px solid {AZUL_PRIMARIO};
        border-radius: 2px;
        padding: 0.75rem 1rem;
        margin: 1rem 0;
    }}
    .citation-box h4 {{
        font-size: {TAMANHO_SUBSECAO};
        font-weight: 600;
        color: {CINZA_TEXTO};
        margin: 0;
    }}

    /* ------------------------------------------------------------------
       Etiqueta de tipo de documento
       ------------------------------------------------------------------ */
    .badge-tipo {{
        display: inline-block;
        padding-left: 8px;
        margin-bottom: 0.35rem;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: {CINZA_TEXTO};
    }}

    /* ------------------------------------------------------------------
       Balao de ajuda
       O balao padrao do Streamlit e estreito e corta textos mais longos.
       Amplia a caixa e libera a altura para o conteudo.
       ------------------------------------------------------------------ */
    [data-testid="stTooltipContent"] {{
        max-width: 360px;
        max-height: none;
        white-space: normal;
        overflow: visible;
        font-size: 0.85rem;
        line-height: 1.45;
    }}
    [data-testid="stTooltipContent"] p {{
        margin-bottom: 0.35rem;
    }}

    /* ------------------------------------------------------------------
       Campo de pergunta fixo no rodape
       Largura sempre total (o arraste horizontal quebrava o botao de
       envio). Altura vem do preset da barra lateral e ainda pode ser
       refinada arrastando a borda inferior (resize: vertical).
       ------------------------------------------------------------------ */
    [data-testid="stBottom"] {{
        background-color: {BRANCO};
        border-top: 2px solid {AZUL_PRIMARIO};
        padding-top: 0.5rem;
        padding-bottom: 0.65rem;
    }}
    [data-testid="stBottom"]::before {{
        content: "Arraste a borda inferior do campo para ajustar a altura.";
        display: block;
        font-size: {TAMANHO_LEGENDA};
        color: {CINZA_SECUNDARIO};
        padding: 0 0.35rem 0.4rem 0.35rem;
    }}
    [data-testid="stBottom"] > div {{
        background-color: {BRANCO};
    }}
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] form,
    [data-testid="stChatInput"] [data-baseweb="base-input"],
    [data-testid="stChatInput"] [data-baseweb="textarea"] {{
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
        background-color: {BRANCO} !important;
    }}
    /* Manter o botao de envio alinhado a base quando a altura cresce. */
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] form {{
        display: flex !important;
        align-items: flex-end !important;
        width: 100% !important;
        gap: 0.35rem;
    }}
    [data-testid="stChatInput"] {{
        border: 2px solid {AZUL_PRIMARIO};
        border-radius: 4px;
        box-shadow: 0 1px 4px rgba(7, 29, 65, 0.12);
        width: 100% !important;
        max-width: 100% !important;
    }}
    [data-testid="stChatInput"]:focus-within {{
        border-color: {AZUL_HOVER};
        outline: 3px solid {AZUL_CLARO};
        box-shadow: 0 2px 8px rgba(19, 81, 180, 0.18);
    }}
    [data-testid="stChatInput"] textarea {{
        resize: vertical !important;
        /* So min-height (nao height) para o arraste vertical continuar
           funcionando: height com !important impede a alca do navegador. */
        min-height: {altura}rem !important;
        max-height: 60vh !important;
        width: 100% !important;
        max-width: 100% !important;
        flex: 1 1 auto !important;
        font-size: 1rem !important;
        line-height: 1.5 !important;
        color: {CINZA_TEXTO} !important;
        background-color: {BRANCO} !important;
        padding: 0.85rem 1rem !important;
        overflow-y: auto !important;
        box-sizing: border-box !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder {{
        color: {CINZA_SECUNDARIO};
        opacity: 1;
        font-size: 0.95rem;
    }}

    /* ------------------------------------------------------------------
       Acessibilidade: indicador de foco visivel para navegacao por teclado
       ------------------------------------------------------------------ */
    .stApp :focus-visible {{
        outline: 2px solid {AZUL_PRIMARIO};
        outline-offset: 2px;
    }}
    </style>
    """


def montar_cabecalho(titulo: str, subtitulo: str = "") -> str:
    """
    Monta o HTML do cabecalho institucional.

    Args:
        titulo: Titulo principal da aplicacao.
        subtitulo: Linha descritiva opcional exibida abaixo do titulo.

    Returns:
        Bloco HTML do cabecalho, com o conteudo textual escapado.

    Raises:
        ValueError: Se o titulo for vazio ou contiver apenas espacos.
    """
    titulo_limpo = titulo.strip()
    if not titulo_limpo:
        raise ValueError("O cabecalho institucional exige um titulo.")

    linhas = [
        '<div class="app-header">',
        f"    <h1>{html.escape(titulo_limpo)}</h1>",
    ]

    subtitulo_limpo = subtitulo.strip()
    if subtitulo_limpo:
        linhas.append(f"    <p>{html.escape(subtitulo_limpo)}</p>")

    linhas.append("</div>")
    return "\n".join(linhas)


def aplicar_estilo_institucional(altura_campo_rem: float | None = None) -> None:
    """
    Injeta a folha de estilo institucional na pagina.

    Precisa ser chamada em toda execucao do script. O Streamlit reconstroi a
    arvore de elementos a cada interacao e descarta o que nao for reemitido:
    condicionar a injecao a uma flag em st.session_state faria o estilo
    desaparecer no primeiro clique. Reemitir nao duplica nada, porque o
    Streamlit reconcilia os elementos por posicao.

    Args:
        altura_campo_rem: Altura minima do campo de pergunta, em rem.
    """
    st.markdown(
        montar_css_institucional(altura_campo_rem=altura_campo_rem),
        unsafe_allow_html=True,
    )


def renderizar_cabecalho(titulo: str, subtitulo: str = "") -> None:
    """
    Renderiza o cabecalho institucional na pagina.

    Args:
        titulo: Titulo principal da aplicacao.
        subtitulo: Linha descritiva opcional exibida abaixo do titulo.

    Raises:
        ValueError: Se o titulo for vazio ou contiver apenas espacos.
    """
    st.markdown(montar_cabecalho(titulo, subtitulo), unsafe_allow_html=True)
