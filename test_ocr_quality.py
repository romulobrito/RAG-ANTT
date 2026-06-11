#!/usr/bin/env python3
"""
Testes sandbox para pipeline OCR v3 (qualidade, pos-processamento, cache).

Executar:
    cd RAG-ANTT
    source venv/bin/activate
    python test_ocr_quality.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from antt_rag_unified import (
    _OCR_PIPELINE_VERSION,
    _OCR_QUALIDADE_MINIMA,
    _cache_ocr_deve_ser_invalidado,
    _calcular_score_qualidade_ocr,
    _formatar_cache_ocr,
    _parse_meta_cache_ocr,
    _pos_processar_tabela_ocr,
    _tentar_corrigir_notacao_cientifica_celula,
)


TABELA_FWD_CORROMPIDA = """\
| VDM min | VDM max | Nestimado | Dadm |
| --- | --- | --- | --- |
| 0 |  |  | 70 |
| 500 |  |  | 60 |
| tooo | 2500 |  | so /] |
|  |  | 600807 | as /] |
| 5000 |  |  | 40 |
"""

TABELA_FWD_BOA = """\
| VDM min | VDM max | Nestimado | Dadm |
| --- | --- | --- | --- |
| 0 | 500 | 6,00E+06 | 70 |
| 500 | 1000 | 1,20E+07 | 60 |
| 1000 | 2500 | 3,00E+07 | 50 |
| 2500 | 5000 | 6,00E+07 | 45 |
| 5000 | 10700 | 1,30E+08 | 40 |
"""


def test_score_tabela_corrompida_baixo():
    score = _calcular_score_qualidade_ocr(TABELA_FWD_CORROMPIDA)
    assert score < _OCR_QUALIDADE_MINIMA, f"Score alto demais: {score}"
    print(f"OK: tabela corrompida score={score:.2f} (< {_OCR_QUALIDADE_MINIMA})")


def test_score_tabela_boa_alto():
    score = _calcular_score_qualidade_ocr(TABELA_FWD_BOA)
    assert score >= _OCR_QUALIDADE_MINIMA, f"Score baixo demais: {score}"
    print(f"OK: tabela boa score={score:.2f} (>= {_OCR_QUALIDADE_MINIMA})")


def test_correcao_notacao_cientifica():
    assert _tentar_corrigir_notacao_cientifica_celula("600807") == "6,00E+07"
    assert _tentar_corrigir_notacao_cientifica_celula("120007") == "1,20E+07"
    assert _tentar_corrigir_notacao_cientifica_celula("6,00E+06") == "6,00E+06"
    print("OK: correcao de notacao cientifica")


def test_pos_processamento_aplica_correcoes():
    entrada = "| x | Nestimado |\n| --- | --- |\n| a | 600807 |"
    saida = _pos_processar_tabela_ocr(entrada)
    assert "6,00E+07" in saida, f"Correcao nao aplicada: {saida}"
    print("OK: pos-processamento corrige notacao cientifica em tabela")


def test_cache_invalida_versao_antiga():
    cache_antigo = "# ocr_pipeline_v=1\n# quality=0.90\n" + TABELA_FWD_BOA
    assert _cache_ocr_deve_ser_invalidado(cache_antigo) is True
    print("OK: cache versao antiga invalidado")


def test_cache_invalida_qualidade_baixa():
    payload = _formatar_cache_ocr(TABELA_FWD_CORROMPIDA, 0.30)
    assert _cache_ocr_deve_ser_invalidado(payload) is True
    print("OK: cache baixa qualidade invalidado")


def test_cache_valido_versao_atual():
    payload = _formatar_cache_ocr(TABELA_FWD_BOA, 0.95)
    meta, conteudo = _parse_meta_cache_ocr(payload)
    assert meta.get("ocr_pipeline_v") == _OCR_PIPELINE_VERSION
    assert _cache_ocr_deve_ser_invalidado(payload) is False
    assert "6,00E+06" in conteudo
    print("OK: cache valido mantido")


def test_cache_real_inm34_fwd_invalidado():
    cache_path = os.path.join(
        os.path.dirname(__file__),
        "dados_antt",
        ".ocr_cache",
        "78ca7ae7d6ebda54.txt",
    )
    if not os.path.exists(cache_path):
        print("SKIP: cache 78ca7ae7d6ebda54.txt nao encontrado")
        return

    with open(cache_path, "r", encoding="utf-8") as f:
        conteudo_cache = f.read()

    score = _calcular_score_qualidade_ocr(
        _pos_processar_tabela_ocr(conteudo_cache)
    )
    invalido = _cache_ocr_deve_ser_invalidado(conteudo_cache)
    assert invalido is True, (
        f"Cache corrompido deveria ser invalidado (score={score:.2f})"
    )
    print(f"OK: cache real INM34 FWD invalidado (score={score:.2f})")


def main():
    print("=" * 60)
    print("TESTES OCR QUALIDADE v3 (sandbox)")
    print("=" * 60)

    test_score_tabela_corrompida_baixo()
    test_score_tabela_boa_alto()
    test_correcao_notacao_cientifica()
    test_pos_processamento_aplica_correcoes()
    test_cache_invalida_versao_antiga()
    test_cache_invalida_qualidade_baixa()
    test_cache_valido_versao_atual()
    test_cache_real_inm34_fwd_invalidado()

    print("=" * 60)
    print("TODOS OS TESTES PASSARAM")
    print("=" * 60)


if __name__ == "__main__":
    main()
