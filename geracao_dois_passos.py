"""
Geracao em dois passos para modelos locais (Ollama).

Passo 1a: colheita deterministica de tabelas markdown do contexto (estrutural).
Passo 1b: extracao complementar via LLM (valores/prazos em prosa).
Passo 2: redigir a resposta final com as evidencias hibridas.

Prompts e colheita sao genericos (sem listas de topicos ou siglas de dominio).
"""

from __future__ import annotations

import re
from typing import Iterator, List, Optional, Tuple

from config import logger

# Teto de chars das tabelas colhidas deterministicamente (deixa espaco ao LLM).
_MAX_CHARS_TABELAS_DETERMINISTICAS = 14000

# Extracao: forca copia fiel; nao pede redacao narrativa.
_PROMPT_EXTRACAO = """Voce e um extrator factual. Com base SOMENTE no CONTEXTO abaixo,
liste evidencias uteis para responder a PERGUNTA que NAO estejam apenas em
tabelas markdown (as tabelas ja serao anexadas automaticamente).

REGRAS OBRIGATORIAS:
- Copie valores, limites, unidades, prazos e criterios exatamente como aparecem.
- Se ja houver tabela markdown no contexto, NAO a reescreva; foque em prosa
  complementar (definicoes, escopo, excecoes, periodicidade fora da tabela).
- Para cada item, indique a fonte (texto apos FONTE INICIO, se existir).
- Nao invente valores, unidades nem definicoes de siglas ausentes no contexto.
- Nao expanda siglas a menos que o proprio contexto as defina.
- Se nao houver evidencia complementar, responda exatamente: NENHUMA_EVIDENCIA
- Nao escreva a resposta final ao usuario; apenas a lista de evidencias.

PERGUNTA:
{question}

CONTEXTO:
{context}

EVIDENCIAS:
"""

# Redacao: usa so o material ja filtrado no passo 1.
_PROMPT_REDACAO = """Com base nas EVIDENCIAS abaixo, responda a PERGUNTA.

REGRAS OBRIGATORIAS:
- Use APENAS as evidencias. Nao invente fatos.
- Se existir a secao BLOCOS_TABULARES, ela JA E evidencia suficiente: reproduza
  as tabelas e valores numericos no inicio da resposta (nao omita numeros).
- NUNCA diga que faltam informacoes, que os trechos sao insuficientes, ou que
  o usuario deve consultar o documento original, se BLOCOS_TABULARES estiver
  presente ou se houver qualquer tabela/valor numerico nas evidencias.
- Cite as fontes indicadas (FONTE INICIO / rotulos nas evidencias).
- So declare insuficiencia se as evidencias forem exatamente NENHUMA_EVIDENCIA
  e nao houver BLOCOS_TABULARES.
- Nao invente expansoes de siglas; use apenas o que estiver nas evidencias.

PERGUNTA:
{question}

EVIDENCIAS:
{evidence}

RESPOSTA:
"""

_RE_FONTE_INICIO = re.compile(
    r"===== FONTE INICIO:\s*(.+?) =====",
    re.MULTILINE,
)


def texto_da_resposta_llm(resposta: object) -> str:
    """
    Normaliza a saida do LLM LangChain para string.

    Args:
        resposta: Objeto retornado por invoke/stream (AIMessage ou str).

    Returns:
        Conteudo textual sem espacos extremos.
    """
    if resposta is None:
        return ""
    if hasattr(resposta, "content"):
        conteudo = getattr(resposta, "content")
        if isinstance(conteudo, str):
            return conteudo.strip()
        return str(conteudo).strip()
    return str(resposta).strip()


def _linha_tabela(linha: str) -> bool:
    """Indica se a linha parece celula de tabela markdown."""
    texto = linha.strip()
    return texto.startswith("|") and texto.count("|") >= 2


def _extrair_tabelas_de_bloco(bloco: str, rotulo_fonte: str) -> List[str]:
    """
    Extrai tabelas markdown de um bloco, com titulo imediatamente anterior.

    Args:
        bloco: Texto de uma fonte ou do contexto inteiro.
        rotulo_fonte: Identificador da fonte para cabecalho.

    Returns:
        Lista de blocos de tabela ja rotulados.
    """
    linhas = bloco.split("\n")
    saidas: List[str] = []
    i = 0
    while i < len(linhas):
        if not _linha_tabela(linhas[i]):
            i += 1
            continue
        inicio_tab = i
        while i < len(linhas) and _linha_tabela(linhas[i]):
            i += 1
        # Ate 3 linhas acima como titulo (## ...), sem atravessar outra tabela.
        titulo_ini = inicio_tab
        olhares = 0
        j = inicio_tab - 1
        while j >= 0 and olhares < 3:
            candidata = linhas[j].strip()
            if not candidata:
                j -= 1
                olhares += 1
                continue
            if _linha_tabela(candidata):
                break
            if candidata.startswith("#") or len(candidata) < 120:
                titulo_ini = j
                j -= 1
                olhares += 1
                continue
            break
        trecho = "\n".join(linhas[titulo_ini:i]).strip()
        if trecho:
            saidas.append(
                "Fonte: {0}\n{1}".format(rotulo_fonte, trecho)
            )
    return saidas


def extrair_blocos_tabulares_do_contexto(
    contexto: str,
    max_chars: int = _MAX_CHARS_TABELAS_DETERMINISTICAS,
) -> str:
    """
    Colhe tabelas markdown do contexto de forma deterministica.

    Independente de tema: qualquer tabela | ... | e capturada com titulo
    local e rotulo de fonte quando houver cercas FONTE INICIO.

    Args:
        contexto: Corpo do contexto montado.
        max_chars: Limite de caracteres do texto colhido.

    Returns:
        Texto com tabelas ou string vazia.
    """
    texto = contexto or ""
    if "|" not in texto:
        return ""

    partes: List[str] = []
    matches = list(_RE_FONTE_INICIO.finditer(texto))
    if matches:
        for indice, match in enumerate(matches):
            inicio = match.end()
            fim = (
                matches[indice + 1].start()
                if indice + 1 < len(matches)
                else len(texto)
            )
            rotulo = match.group(1).strip() or "Fonte"
            partes.extend(_extrair_tabelas_de_bloco(texto[inicio:fim], rotulo))
    else:
        partes.extend(_extrair_tabelas_de_bloco(texto, "Contexto"))

    if not partes:
        return ""

    acumulado: List[str] = []
    tamanho = 0
    for parte in partes:
        custo = len(parte) + (2 if acumulado else 0)
        if tamanho + custo > max_chars:
            break
        acumulado.append(parte)
        tamanho += custo

    logger.info(
        "Dois passos Ollama: colhidas %s tabela(s) deterministicas (%s chars)",
        len(acumulado),
        tamanho,
    )
    return "\n\n".join(acumulado)


def montar_evidencias_hibridas(
    blocos_tabulares: str,
    evidencias_llm: str,
) -> str:
    """
    Combina tabelas deterministicas com a extracao do LLM.

    Args:
        blocos_tabulares: Saida de extrair_blocos_tabulares_do_contexto.
        evidencias_llm: Saida do passo de extracao LLM.

    Returns:
        Pacote de evidencias para a redacao.
    """
    llm_txt = (evidencias_llm or "").strip()
    if llm_txt.upper() == "NENHUMA_EVIDENCIA":
        llm_txt = ""

    if blocos_tabulares.strip():
        secoes = [
            "=== BLOCOS_TABULARES (copia fiel do contexto; obrigatorios) ===",
            blocos_tabulares.strip(),
        ]
        if llm_txt:
            secoes.extend(
                [
                    "=== EVIDENCIAS_COMPLEMENTARES (extracao LLM) ===",
                    llm_txt,
                ]
            )
        return "\n\n".join(secoes)

    if llm_txt:
        return llm_txt
    return "NENHUMA_EVIDENCIA"


def extrair_evidencias(pergunta: str, contexto: str, llm: object) -> str:
    """
    Passo 1b: extrai evidencias complementares via LLM.

    Args:
        pergunta: Pergunta do usuario.
        contexto: Corpo do contexto (preferencialmente sem instrucoes longas).
        llm: Instancia LangChain com .invoke.

    Returns:
        Texto de evidencias ou NENHUMA_EVIDENCIA.
    """
    prompt = _PROMPT_EXTRACAO.format(
        question=pergunta or "",
        context=contexto or "",
    )
    logger.info("Dois passos Ollama: iniciando extracao LLM complementar")
    resposta = llm.invoke(prompt)
    evidencias = texto_da_resposta_llm(resposta)
    logger.info(
        "Dois passos Ollama: extracao LLM concluida (%s chars)",
        len(evidencias),
    )
    if not evidencias:
        return "NENHUMA_EVIDENCIA"
    return evidencias


def coletar_evidencias(pergunta: str, contexto: str, llm: object) -> str:
    """
    Monta evidencias hibridas (tabelas deterministicas + LLM).

    Args:
        pergunta: Pergunta do usuario.
        contexto: Contexto documental empacotado.
        llm: Instancia LangChain.

    Returns:
        Pacote de evidencias para o passo 2.
    """
    tabelas = extrair_blocos_tabulares_do_contexto(contexto)
    try:
        evid_llm = extrair_evidencias(pergunta, contexto, llm)
    except Exception as exc:
        logger.warning(
            "Dois passos Ollama: extracao LLM falhou (%s); "
            "seguindo so com tabelas deterministicas",
            exc,
        )
        evid_llm = "NENHUMA_EVIDENCIA"

    pacote = montar_evidencias_hibridas(tabelas, evid_llm)
    logger.info(
        "Dois passos Ollama: evidencias hibridas (%s chars); "
        "tem_blocos_tabulares=%s",
        len(pacote),
        "BLOCOS_TABULARES" in pacote,
    )
    logger.info(
        "Dois passos Ollama: preview evidencias: %s",
        pacote[:400].replace("\n", " "),
    )
    return pacote


def montar_prompt_redacao(pergunta: str, evidencias: str) -> str:
    """
    Monta o prompt do passo 2 (redacao).

    Args:
        pergunta: Pergunta do usuario.
        evidencias: Saida do passo 1 (hibrido).

    Returns:
        Prompt completo para o LLM.
    """
    return _PROMPT_REDACAO.format(
        question=pergunta or "",
        evidence=evidencias or "NENHUMA_EVIDENCIA",
    )


def redigir_com_evidencias(pergunta: str, evidencias: str, llm: object) -> str:
    """
    Passo 2: redige a resposta final a partir das evidencias.

    Args:
        pergunta: Pergunta do usuario.
        evidencias: Texto do passo 1.
        llm: Instancia LangChain com .invoke.

    Returns:
        Resposta final em texto.
    """
    prompt = montar_prompt_redacao(pergunta, evidencias)
    logger.info("Dois passos Ollama: iniciando redacao")
    resposta = llm.invoke(prompt)
    texto = texto_da_resposta_llm(resposta)
    logger.info("Dois passos Ollama: redacao concluida (%s chars)", len(texto))
    return texto


def gerar_com_dois_passos(pergunta: str, contexto: str, llm: object) -> str:
    """
    Executa colheita/extracao e redacao em sequencia (modo nao streaming).

    Args:
        pergunta: Pergunta do usuario.
        contexto: Contexto documental empacotado.
        llm: Instancia LangChain.

    Returns:
        Resposta final.
    """
    evidencias = coletar_evidencias(pergunta, contexto, llm)
    return redigir_com_evidencias(pergunta, evidencias, llm)


def iter_redacao_streaming(
    pergunta: str,
    evidencias: str,
    llm: object,
) -> Iterator[str]:
    """
    Passo 2 em streaming (apos coleta de evidencias).

    Args:
        pergunta: Pergunta do usuario.
        evidencias: Texto do passo 1.
        llm: Instancia LangChain com .stream.

    Yields:
        Tokens de texto da resposta final.
    """
    from langchain_core.messages import HumanMessage

    prompt = montar_prompt_redacao(pergunta, evidencias)
    logger.info("Dois passos Ollama: redacao em streaming")
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        if hasattr(chunk, "content"):
            bruto = getattr(chunk, "content")
            token = bruto if isinstance(bruto, str) else str(bruto)
        else:
            token = str(chunk)
        if token:
            yield token


def deve_usar_dois_passos(modelo_usado: Optional[str]) -> bool:
    """
    Indica se a geracao deve usar o fluxo de dois passos.

    Ativado para Ollama; pode ser desligado com RAG_OLLAMA_TWO_PASS=0.

    Args:
        modelo_usado: Identificador do provedor/modelo.

    Returns:
        True se dois passos devem rodar.
    """
    import os

    from contexto_packing import eh_identificador_ollama

    flag = os.environ.get("RAG_OLLAMA_TWO_PASS", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    return eh_identificador_ollama(modelo_usado)
