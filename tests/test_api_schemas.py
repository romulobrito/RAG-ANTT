"""
Testes do contrato HTTP congelado.

Validam o JSON da INM 34. Nao sobem servidor.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from api.schemas import QueryRequest, QueryResponse

RAIZ = Path(__file__).resolve().parent.parent

_PEDIDO_INM_34 = {
    "pergunta": (
        "Qual o IRI maximo da pista principal na manutencao "
        "segundo a INM 34/2024?"
    ),
    "filtros": {"tipo_documento": "INM", "ano": 2024, "numero": "34"},
    "correlation_id": "sigesc-ticket-8891",
}

_RESPOSTA_INM_34 = {
    "request_id": "a3f2c1e0-9b44-4c21-8d10-11aa22bb33cc",
    "correlation_id": "sigesc-ticket-8891",
    "resposta": (
        "Segundo a INM 34/2024, o IRI maximo na pista principal "
        "na fase de manutencao e 2,7 m/km (...citacao...)"
    ),
    "modelo_usado": "qwen2.5:7b",
    "provider": "ollama",
    "documentos_consultados": [
        {
            "tipo": "INM",
            "numero": "34",
            "ano": "2024",
            "trecho": (
                "Irregularidade Longitudinal Maxima - IRI | "
                "Principal | ... | 2,7 m/km"
            ),
            "caminho": "dados_antt/INM/2024/INM-00000034-2024.md",
            "relevancia": 0.91,
        }
    ],
    "tempo_processamento_ms": 28000,
    "total_documentos_encontrados": 30,
    "embedding_provider": "local",
    "vectorstore_utilizado": "vectorstore_local",
}


def test_pedido_inm_34_entra_no_schema() -> None:
    """O POST ilustrativo da INM 34 e um QueryRequest valido."""
    pedido = QueryRequest.model_validate(_PEDIDO_INM_34)
    assert pedido.pergunta.startswith("Qual o IRI")
    assert pedido.filtros is not None
    assert pedido.filtros.tipo_documento == "INM"
    assert pedido.filtros.ano == 2024
    assert pedido.filtros.numero == "34"
    assert pedido.temperatura is None
    assert pedido.max_documentos is None
    assert pedido.historico is None


def test_resposta_inm_34_entra_no_schema() -> None:
    """O JSON ilustrativo de resposta valida, com request_id."""
    resposta = QueryResponse.model_validate(_RESPOSTA_INM_34)
    assert resposta.request_id == "a3f2c1e0-9b44-4c21-8d10-11aa22bb33cc"
    assert resposta.documentos_consultados[0].tipo == "INM"
    assert resposta.tempo_processamento_ms == 28000


def test_pergunta_vazia_e_rejeitada() -> None:
    """Pergunta vazia ou so com espaco nao passa."""
    with pytest.raises(ValidationError):
        QueryRequest.model_validate({"pergunta": ""})
    with pytest.raises(ValidationError):
        QueryRequest.model_validate({"pergunta": "   "})


def test_temperatura_e_trechos_fora_da_faixa() -> None:
    """Temperatura acima de 1 e zero trechos respondem como pedido invalido."""
    with pytest.raises(ValidationError):
        QueryRequest.model_validate({"pergunta": "IRI?", "temperatura": 2})
    with pytest.raises(ValidationError):
        QueryRequest.model_validate({"pergunta": "IRI?", "max_documentos": 0})


def test_campo_extra_e_rejeitado() -> None:
    """user_id e question nao pertencem ao contrato."""
    with pytest.raises(ValidationError):
        QueryRequest.model_validate({"pergunta": "IRI?", "user_id": "fiscal"})
    with pytest.raises(ValidationError):
        QueryRequest.model_validate({"pergunta": "IRI?", "question": "IRI?"})


def test_import_nao_carrega_streamlit_nem_fastapi() -> None:
    """O pacote de schemas nao puxa a tela nem o servidor."""
    codigo = (
        "import sys\n"
        "import api.schemas\n"
        "assert 'streamlit' not in sys.modules, sorted(sys.modules)\n"
        "assert 'fastapi' not in sys.modules, sorted(sys.modules)\n"
    )
    concluido = subprocess.run(
        [sys.executable, "-c", codigo],
        cwd=str(RAIZ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert concluido.returncode == 0, concluido.stderr
