"""
Testes do tratamento de falhas do provedor de IA.

Antes desta guarda, uma falha do provedor era exibida ao usuario como texto
bruto da excecao: um JSON de milhares de caracteres, com identificador de
conta e nomes de provedores. Estes testes garantem que a interface mostre
orientacao em linguagem clara e que o detalhe tecnico fique apenas no log.
"""

import pytest

from antt_rag_unified import (
    classificar_falha_de_geracao,
    mensagem_de_falha_de_geracao,
    resposta_indica_falha,
)

# Trecho real capturado no log em 2026-07-27, reduzido ao essencial. Reproduz
# o formato do OpenRouter, que acumula as falhas de cada provedor tentado.
ERRO_402_REAL = (
    "Error code: 402 - {'error': {'message': 'This request requires more "
    "credits, or fewer max_tokens. You requested up to 2048 tokens, but can "
    "only afford 1680. To increase, visit "
    "https://openrouter.ai/settings/credits and upgrade to a paid account', "
    "'code': 402, 'metadata': {'provider_name': None, 'previous_errors': "
    "[{'code': 402, 'message': 'Prompt tokens limit exceeded: 10845 > 4800.'}"
    "]}}, 'user_id': 'user_2tUcoElKMl2LNHe6XLQD7JBaaIF'}"
)


@pytest.mark.parametrize(
    "texto_do_erro, causa_esperada",
    [
        (ERRO_402_REAL, "credito"),
        ("Error code: 402 - Payment Required", "credito"),
        ("Error code: 401 - Unauthorized", "credencial"),
        ("Invalid API key provided", "credencial"),
        ("Error code: 429 - Rate limit exceeded", "limite_de_uso"),
        ("This model's maximum context length is 8192 tokens", "consulta_longa"),
        ("Prompt tokens limit exceeded: 10845 > 4800", "consulta_longa"),
        ("Connection timed out", "comunicacao"),
        ("Error code: 503 - Service Unavailable", "comunicacao"),
        ("Algo totalmente inesperado aconteceu", "desconhecida"),
    ],
)
def test_classificacao_da_falha(texto_do_erro: str, causa_esperada: str):
    """Cada formato de erro conhecido deve cair na causa correspondente."""
    assert classificar_falha_de_geracao(Exception(texto_do_erro)) == causa_esperada


def test_erro_de_credito_tem_prioridade_sobre_tamanho():
    """
    Um 402 que tambem cita limite de tokens e falta de credito.

    O OpenRouter reduz o teto de prompt conforme o saldo cai, entao a
    mensagem de tamanho e consequencia, nao causa. Orientar o usuario a
    encurtar a pergunta seria enganoso.
    """
    assert classificar_falha_de_geracao(Exception(ERRO_402_REAL)) == "credito"


def test_mensagem_nao_vaza_detalhe_tecnico():
    """A orientacao ao usuario nao deve conter dados internos do provedor."""
    mensagem = mensagem_de_falha_de_geracao(Exception(ERRO_402_REAL))

    proibidos = (
        "user_id",
        "user_2tUcoElKMl2LNHe6XLQD7JBaaIF",
        "402",
        "max_tokens",
        "openrouter",
        "provider_name",
        "{",
    )
    encontrados = [
        termo for termo in proibidos if termo.lower() in mensagem.lower()
    ]

    assert not encontrados, (
        "Detalhe tecnico vazou para a mensagem do usuario: "
        + ", ".join(encontrados)
    )


def test_mensagem_e_curta_e_orienta_acao():
    """A orientacao deve ser legivel e dizer o que fazer."""
    mensagem = mensagem_de_falha_de_geracao(Exception(ERRO_402_REAL))

    assert len(mensagem) < 400, (
        f"Mensagem longa demais ({len(mensagem)} caracteres) para um aviso."
    )
    assert "equipe" in mensagem.lower(), (
        "Falha de credito depende de acao da equipe tecnica; a mensagem "
        "precisa indicar isso."
    )


def test_mensagem_de_limite_orienta_esperar():
    """Limite por minuto se resolve sozinho; a orientacao deve refletir isso."""
    mensagem = mensagem_de_falha_de_geracao(
        Exception("Error code: 429 - Rate limit exceeded")
    )

    assert "minuto" in mensagem.lower()


def test_resposta_indica_falha_reconhece_avisos():
    """A interface precisa distinguir aviso de falha de resposta legitima."""
    aviso = mensagem_de_falha_de_geracao(Exception(ERRO_402_REAL))

    assert resposta_indica_falha(aviso) is True
    assert resposta_indica_falha("") is True
    assert resposta_indica_falha("   ") is True


def test_resposta_indica_falha_aceita_resposta_legitima():
    """Texto normativo real nao pode ser confundido com falha."""
    resposta = (
        "Conforme o Art. 22 da Resolucao 6057/2024, o reequilibrio "
        "economico-financeiro dos contratos de concessao rodoviaria "
        "ocorrera nos termos ali previstos."
    )

    assert resposta_indica_falha(resposta) is False


def test_aviso_aparece_quando_streaming_falha(monkeypatch):
    """
    O gerador de streaming deve entregar orientacao, nao a excecao bruta.

    Simula um provedor que falha na primeira chamada, reproduzindo o cenario
    observado em producao.
    """
    import antt_rag_unified

    class LLMQueFalha:
        """Duble de LLM que sempre falha, como um provedor sem credito."""

        def stream(self, _mensagens):
            raise RuntimeError(ERRO_402_REAL)

    monkeypatch.setattr(
        antt_rag_unified,
        "_preparar_contexto_resposta",
        lambda *args, **kwargs: ("prompt qualquer", "normativa"),
    )

    documento = type("Doc", (), {"page_content": "texto", "metadata": {}})()
    saida = "".join(
        antt_rag_unified.gerar_resposta_streaming(
            "pergunta", [documento], LLMQueFalha(), "deepseek"
        )
    )

    assert "user_id" not in saida
    assert "402" not in saida
    assert resposta_indica_falha(saida) is True
