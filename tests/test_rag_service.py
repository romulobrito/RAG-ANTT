"""
Testes do facade sincrono, com indice e LLM substituidos.

Nao abrem FAISS nem chamam Ollama.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Dict, List

import pytest
from langchain_core.documents import Document

from rag_service import (
    PerguntaInvalidaError,
    ProvedorNaoLiberadoError,
    RagGenerationError,
    consultar,
    incluir_documento,
    limpar_cache_vectorstore,
    listar_anos,
)

RAIZ = Path(__file__).resolve().parent.parent


def _documento() -> Document:
    """Trecho minimo para o facade montar um hit."""
    return Document(
        page_content="IRI 2,7 m/km",
        metadata={
            "nome_tipo": "INM",
            "numero": "34",
            "ano": "2024",
            "caminho": "dados_antt/INM/2024/INM-00000034-2024.md",
            "relevancia": 0.9,
        },
    )


def _instalar_dublagens(monkeypatch: pytest.MonkeyPatch) -> Dict[str, object]:
    """
    Substitui indice, busca e geracao por registros do que foi pedido.

    Args:
        monkeypatch: Fixture do pytest.

    Returns:
        Dicionario preenchido na hora da consulta.
    """
    capturado: Dict[str, object] = {}

    def falso_carregar(provedor: str) -> SimpleNamespace:
        return SimpleNamespace(
            _vectorstore_path="vectorstore_local",
            _embedding_provider=provedor,
        )

    def falsa_busca(
        query: str,
        vectorstore: object,
        k: int = 16,
        tipo_documento: object = None,
        ano: object = None,
        numero: object = None,
        embedding_provider: str = "free",
    ) -> List[Document]:
        capturado["k"] = k
        capturado["query"] = query
        capturado["embedding"] = embedding_provider
        return [_documento()]

    def falsa_resposta(
        pergunta: str,
        documentos: object,
        llm: object,
        modelo_usado: str = "gpt-4",
    ) -> tuple:
        capturado["modelo"] = modelo_usado
        return ("resposta de teste", "qwen2.5:7b")

    class GerenteFalso:
        """LLM manager que nao abre rede."""

        provider = "ollama"

        def __init__(self, provider: str, model: object = None) -> None:
            self.provider = provider
            capturado["provider"] = provider
            capturado["model"] = model

        def get_llm(
            self,
            temperature: float = 0.1,
            max_tokens: object = None,
        ) -> object:
            capturado["temperature"] = temperature
            capturado["max_tokens"] = max_tokens
            return object()

    monkeypatch.setattr(
        "rag_service.carregar_vectorstore_com_provider",
        falso_carregar,
    )
    monkeypatch.setattr("rag_service.pesquisar_documentos", falsa_busca)
    monkeypatch.setattr("rag_service.gerar_resposta", falsa_resposta)
    monkeypatch.setattr("rag_service.create_llm_manager", GerenteFalso)
    limpar_cache_vectorstore()
    return capturado


def test_publicar_modulo_substitui_alias_antigo() -> None:
    """Rerun do Streamlit nao pode manter o modulo da execucao anterior."""
    import antt_rag_unified

    real = sys.modules["antt_rag_unified"]
    try:
        antigo = ModuleType("__main__")
        sys.modules["antt_rag_unified"] = antigo
        atual = ModuleType("__main__")
        setattr(atual, "reindexacao_ocupada", lambda: False)
        antt_rag_unified._publicar_modulo_em_execucao(atual)
        assert sys.modules["antt_rag_unified"] is atual
        assert hasattr(sys.modules["antt_rag_unified"], "reindexacao_ocupada")
    finally:
        sys.modules["antt_rag_unified"] = real


def test_publicar_modulo_ignora_import_normal() -> None:
    """Importar o nucleo como biblioteca nao troca o modulo carregado."""
    import antt_rag_unified

    real = sys.modules["antt_rag_unified"]
    outro = ModuleType("antt_rag_unified")
    antt_rag_unified._publicar_modulo_em_execucao(outro)
    assert sys.modules["antt_rag_unified"] is real


def test_import_do_nucleo_nao_carrega_streamlit() -> None:
    """Importar o nucleo nao pode puxar a tela."""
    codigo = (
        "import sys\n"
        "import antt_rag_unified\n"
        "assert 'streamlit' not in sys.modules, sorted(sys.modules)\n"
    )
    concluido = subprocess.run(
        [sys.executable, "-c", codigo],
        cwd=str(RAIZ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert concluido.returncode == 0, concluido.stderr


def test_pergunta_vazia_falha() -> None:
    """Pergunta em branco nao chega na busca."""
    with pytest.raises(PerguntaInvalidaError):
        consultar("   ")


def test_provedor_nao_liberado_falha(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provedor fora da lista do ambiente e recusado."""
    monkeypatch.setenv("RAG_LLM_ALLOWED_PROVIDERS", "ollama")
    with pytest.raises(ProvedorNaoLiberadoError):
        consultar("Qual o IRI?", provider="openai")


def test_omissao_de_trechos_usa_30(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sem max_documentos, a consulta pede os 30 trechos do ambiente."""
    monkeypatch.delenv("RAG_MAX_DOCUMENTOS", raising=False)
    monkeypatch.delenv("RAG_LLM_TEMPERATURE", raising=False)
    monkeypatch.delenv("RAG_LLM_MAX_TOKENS", raising=False)
    monkeypatch.delenv("RAG_LLM_MODEL", raising=False)
    monkeypatch.delenv("RAG_LLM_ALLOWED_PROVIDERS", raising=False)
    capturado = _instalar_dublagens(monkeypatch)

    resultado = consultar("Qual o IRI maximo?")

    assert capturado["k"] == 30
    assert capturado["temperature"] == 0.1
    assert capturado["max_tokens"] == 4096
    assert capturado["provider"] == "ollama"
    assert capturado["model"] == "llama3.2:3b"
    assert resultado.resposta == "resposta de teste"
    assert resultado.documentos_consultados[0].numero == "34"
    assert resultado.total_documentos_encontrados == 1


def test_deepseek_usa_o_modelo_dele(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provedor DeepSeek nao herda o nome do modelo local."""
    monkeypatch.delenv("RAG_LLM_MODEL", raising=False)
    monkeypatch.delenv("RAG_LLM_ALLOWED_PROVIDERS", raising=False)
    capturado = _instalar_dublagens(monkeypatch)

    consultar("Qual o IRI maximo?", provider="deepseek")

    assert capturado["provider"] == "deepseek"
    assert capturado["model"] == "deepseek-v4-flash"


def test_resposta_vazia_do_provedor_falha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resposta vazia nunca pode ser publicada como consulta bem-sucedida."""
    _instalar_dublagens(monkeypatch)
    monkeypatch.setattr(
        "rag_service.gerar_resposta",
        lambda *args, **kwargs: ("   ", "deepseek-chat"),
    )

    with pytest.raises(
        RagGenerationError,
        match="resposta_vazia_do_provedor",
    ):
        consultar("Resuma a planilha", provider="deepseek")


def test_incluir_documento_grava_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """O PDF entra na pasta comum e nao e indexado neste passo."""
    monkeypatch.chdir(tmp_path)
    caminho = incluir_documento(b"%PDF-1.4 conteudo", "INM 34.pdf")
    assert caminho.endswith(".pdf")
    assert Path(caminho).is_file()
    assert Path(caminho).read_bytes().startswith(b"%PDF")


def test_listar_anos_ignora_vazio_e_ordena(tmp_path: Path) -> None:
    """O filtro de ano sai do catalogo, do mais novo para o mais antigo."""
    relatorio = tmp_path / "relatorio_documentos.json"
    relatorio.write_text(
        json.dumps([
            {"ano": "2020", "tipo": "RES", "arquivo_md": "a.md"},
            {"ano": "", "tipo": "OUTROS", "arquivo_md": "b.md"},
            {"ano": "2024", "tipo": "INM", "arquivo_md": "c.md"},
            {"ano": "2024", "tipo": "RES", "arquivo_md": "d.md"},
            {"ano": "antigo", "tipo": "SEI", "arquivo_md": "e.md"},
        ]),
        encoding="utf-8",
    )
    assert listar_anos(str(relatorio)) == ["2024", "2020"]


def test_incluir_documento_rejeita_nao_pdf() -> None:
    """Arquivo que nao comeca com a marca de PDF e recusado."""
    with pytest.raises(PerguntaInvalidaError):
        incluir_documento(b"nao e pdf", "nota.pdf")
