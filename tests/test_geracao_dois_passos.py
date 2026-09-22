"""
Testes do fluxo de geracao em dois passos (Ollama).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Iterator, List

import geracao_dois_passos as gdp


class _FakeLLM:
    """LLM minimo para testes (invoke + stream)."""

    def __init__(self, respostas_invoke: List[str]) -> None:
        self._respostas = list(respostas_invoke)
        self.invoke_prompts: List[str] = []
        self.stream_prompts: List[str] = []

    def invoke(self, prompt: object) -> SimpleNamespace:
        """Consome a proxima resposta configurada."""
        texto = prompt if isinstance(prompt, str) else str(prompt)
        self.invoke_prompts.append(texto)
        if not self._respostas:
            return SimpleNamespace(content="")
        return SimpleNamespace(content=self._respostas.pop(0))

    def stream(self, messages: object) -> Iterator[SimpleNamespace]:
        """Simula streaming token a token."""
        conteudo = ""
        if messages and isinstance(messages, list):
            msg = messages[0]
            conteudo = getattr(msg, "content", str(msg))
        self.stream_prompts.append(conteudo)
        for token in ["Resp", "osta", " final"]:
            yield SimpleNamespace(content=token)


def test_texto_da_resposta_llm():
    """Normaliza AIMessage e string."""
    assert gdp.texto_da_resposta_llm(SimpleNamespace(content="  ok  ")) == "ok"
    assert gdp.texto_da_resposta_llm("direto") == "direto"


def test_extrair_blocos_tabulares_do_contexto():
    """Colheita deterministica pega tabela e titulo, com rotulo de fonte."""
    contexto = (
        "===== FONTE INICIO: INM 1/2024 =====\n"
        "ID: INM 1/2024\n"
        "## Limites do parametro\n"
        "| Fase | Valor |\n"
        "| --- | --- |\n"
        "| A | 2,7 |\n"
        "| B | 3,5 |\n"
        "===== FONTE FIM: INM 1/2024 =====\n"
        "===== FONTE INICIO: RES 2/2022 =====\n"
        "So prosa sem tabela.\n"
        "===== FONTE FIM: RES 2/2022 =====\n"
    )
    tabelas = gdp.extrair_blocos_tabulares_do_contexto(contexto)
    assert "BLOCOS" not in tabelas  # so o miolo; cabecalho vem no hibrido
    assert "INM 1/2024" in tabelas
    assert "| A | 2,7 |" in tabelas
    assert "2,7" in tabelas
    assert "So prosa" not in tabelas


def test_montar_evidencias_hibridas_prioriza_tabelas():
    """Com tabelas, nunca fica NENHUMA_EVIDENCIA mesmo se LLM vazio."""
    pacote = gdp.montar_evidencias_hibridas(
        "| X | 1 |\n| --- | --- |\n| Y | 2 |",
        "NENHUMA_EVIDENCIA",
    )
    assert "BLOCOS_TABULARES" in pacote
    assert "| Y | 2 |" in pacote
    assert pacote != "NENHUMA_EVIDENCIA"


def test_coletar_evidencias_injeta_tabelas_mesmo_com_llm_fraco():
    """
    Simula o bug observado: LLM ignora tabelas; colheita deterministica salva.
    """
    contexto = (
        "===== FONTE INICIO: DOC A =====\n"
        "## Parametro Alpha\n"
        "| Pista | Limite |\n"
        "| --- | --- |\n"
        "| Principal | 2,7 |\n"
        "===== FONTE FIM: DOC A =====\n"
    )
    llm = _FakeLLM(
        [
            "Apenas mencao a fase de recuperacao na RES, sem numeros.",
        ]
    )
    pacote = gdp.coletar_evidencias("Quais os limites?", contexto, llm)
    assert "BLOCOS_TABULARES" in pacote
    assert "2,7" in pacote
    assert "Principal" in pacote


def test_gerar_com_dois_passos_usa_evidencias_na_redacao():
    """Passo 2 recebe pacote hibrido com tabela."""
    contexto = (
        "===== FONTE INICIO: DOC X =====\n"
        "| Limite | 2,7 |\n"
        "| --- | --- |\n"
        "| A | 2,7 |\n"
        "===== FONTE FIM: DOC X =====\n"
    )
    llm = _FakeLLM(
        [
            "Complemento: periodicidade anual.",
            "Tabela: Limite 2,7 (DOC X)",
        ]
    )
    saida = gdp.gerar_com_dois_passos("Quais os limites?", contexto, llm)
    assert "2,7" in saida
    assert len(llm.invoke_prompts) == 2
    assert "BLOCOS_TABULARES" in llm.invoke_prompts[1] or "2,7" in llm.invoke_prompts[1]


def test_extracao_vazia_vira_nenhuma_evidencia():
    """Invoke vazio no passo LLM vira marcador NENHUMA_EVIDENCIA."""
    llm = _FakeLLM([""])
    evid = gdp.extrair_evidencias("pergunta", "contexto sem pipes", llm)
    assert evid == "NENHUMA_EVIDENCIA"


def test_deve_usar_dois_passos_so_ollama(monkeypatch):
    """API cloud nao usa dois passos; Ollama sim (flag padrao)."""
    monkeypatch.delenv("RAG_OLLAMA_TWO_PASS", raising=False)
    assert gdp.deve_usar_dois_passos("ollama") is True
    assert gdp.deve_usar_dois_passos("qwen2.5:7b") is True
    assert gdp.deve_usar_dois_passos("deepseek") is False
    assert gdp.deve_usar_dois_passos("gpt-4") is False
    monkeypatch.setenv("RAG_OLLAMA_TWO_PASS", "0")
    assert gdp.deve_usar_dois_passos("ollama") is False


def test_iter_redacao_streaming():
    """Streaming do passo 2 concatena tokens."""
    llm = _FakeLLM([])
    partes = list(
        gdp.iter_redacao_streaming("pergunta", "evidencia X", llm)
    )
    assert "".join(partes) == "Resposta final"
    assert len(llm.stream_prompts) == 1
    assert "evidencia X" in llm.stream_prompts[0]


def test_prompts_sem_hardcode_de_dominio():
    """Prompts nao devem citar temas/siglas de pavimento."""
    proibidos = ("iri", "fwd", "dadm", "pavimento", "inm 34", "qwen")
    blob = (gdp._PROMPT_EXTRACAO + gdp._PROMPT_REDACAO).lower()
    for termo in proibidos:
        assert termo not in blob, termo


def test_prompt_redacao_proibe_insuficiencia_com_tabelas():
    """Redacao deve proibir 'faltam informacoes' quando ha BLOCOS_TABULARES."""
    prompt = gdp.montar_prompt_redacao(
        "pergunta",
        "=== BLOCOS_TABULARES ===\n| A | 1 |",
    )
    assert "BLOCOS_TABULARES" in prompt
    assert "NUNCA diga que faltam informacoes" in prompt
