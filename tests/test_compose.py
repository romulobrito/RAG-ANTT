"""
O compose local nao publica o Ollama e nao manda a base para a imagem.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent


def test_dockerignore_deixa_base_e_venv_fora_da_imagem() -> None:
    """O contexto de build nao leva indice, documentos nem o venv."""
    texto = (RAIZ / ".dockerignore").read_text(encoding="utf-8")
    for nome in ("venv", "dados_antt", "vectorstore_local", ".env"):
        assert nome in texto.splitlines()


def test_compose_padrao_nao_publica_ollama() -> None:
    """11434 fica na rede interna. O profile de debug e que abre a porta."""
    concluido = subprocess.run(
        ["docker", "compose", "config", "--format", "json"],
        cwd=str(RAIZ),
        capture_output=True,
        text=True,
        check=False,
    )
    if concluido.returncode != 0:
        pytest.skip(concluido.stderr.strip() or "docker compose indisponivel")
    documento = json.loads(concluido.stdout)
    servicos = documento.get("services", {})
    assert "ollama" in servicos
    assert "rag-api" in servicos
    assert "streamlit" not in servicos
    assert "ollama-debug" not in servicos
    for nome, servico in servicos.items():
        for porta in servico.get("ports") or []:
            publicado = str(porta.get("published", ""))
            assert publicado != "11434", nome


def test_compose_debug_publica_ollama_so_no_localhost() -> None:
    """O profile debug-ollama abre 11434 apenas em 127.0.0.1."""
    concluido = subprocess.run(
        [
            "docker",
            "compose",
            "--profile",
            "debug-ollama",
            "config",
            "--format",
            "json",
        ],
        cwd=str(RAIZ),
        capture_output=True,
        text=True,
        check=False,
    )
    if concluido.returncode != 0:
        pytest.skip(concluido.stderr.strip() or "docker compose indisponivel")
    documento = json.loads(concluido.stdout)
    debug = documento.get("services", {}).get("ollama-debug", {})
    portas = debug.get("ports") or []
    assert portas
    assert str(portas[0].get("published", "")) == "11434"
    assert str(portas[0].get("host_ip", "")) == "127.0.0.1"
