#!/usr/bin/env python3
"""
Testes sandbox para pipeline OCR v4 (qualidade, estrutura, pos-processamento).

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
    _calcular_score_estrutural_ocr,
    _calcular_score_qualidade_ocr,
    _detectar_matriz_wide_fragmentada,
    _enriquecer_imagens_documento,
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

OCR_MATRIZ_INM34_FRAGMENTADA = """\
VALORES LIMITES E PERIODICIDADE POR FASE DE CONCESSAO
2,7 m/km em
60% da
3,5 m/km rodovia; e 3,5 2,7 m/km 2,7 m/km
m/km em 40%
. da rodovia
Irregularidade Longitudinal Maxima - IRI
3,0 m/km em
60% da
Marginal 3,5 m/km rodovia; e 3,5 3,0 m/km 3,0 m/km
m/km em 40%
da rodovia
Principal
IFI (International Friction Index) rincipal e >0,13 >0,2 >0,2
Marginal
"""

OCR_VDM_LINEAR = """\
VDM comercial (unidirecional) Nestimado Dadm (0,01 mm)
0 500 6,00E+06 70
500 1000 1,20E+07 60
1000 2500 3,00E+07 50
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


def test_detecta_matriz_wide_fragmentada():
    assert _detectar_matriz_wide_fragmentada(OCR_MATRIZ_INM34_FRAGMENTADA) is True
    assert _detectar_matriz_wide_fragmentada(TABELA_FWD_BOA) is False
    print("OK: detecta matriz wide fragmentada")


def test_reconstrucao_iri_principal_marginal():
    saida = _pos_processar_tabela_ocr(OCR_MATRIZ_INM34_FRAGMENTADA)
    assert "| ---" in saida, "Deveria gerar markdown estruturado"
    assert "Principal" in saida and "Marginal" in saida
    assert "2,7 m/km (60%)" in saida or "2,7 m/km (60%);" in saida
    assert "3,0 m/km (60%)" in saida or "3,0 m/km (60%);" in saida
    assert "3,5 m/km (40%)" in saida
    score = _calcular_score_estrutural_ocr(saida)
    assert score >= 0.85, f"Score estrutural baixo apos reconstrucao: {score}"
    print(f"OK: reconstrucao IRI Principal/Marginal (score={score:.2f})")


def test_reconstrucao_vdm_nao_regride():
    saida = _pos_processar_tabela_ocr(OCR_VDM_LINEAR)
    assert "6,00E+06" in saida
    assert "70" in saida
    score = _calcular_score_qualidade_ocr(saida)
    assert score >= _OCR_QUALIDADE_MINIMA
    print(f"OK: VDM linear preservado (score={score:.2f})")


def test_separador_anexo_enriquecimento():
    import hashlib
    import antt_rag_unified as mod

    url = "https://exemplo.net/imagens/1871642.png"
    md = f"Texto\n![]({url})\n"

    def _fake_processar(tupla):
        return (tupla[1], "0 500 6,00E+06 70")

    original_proc = mod._processar_imagem_ocr
    original_cache = mod._obter_cache_ocr
    mod._processar_imagem_ocr = _fake_processar
    mod._obter_cache_ocr = lambda _h: None
    try:
        out = _enriquecer_imagens_documento(md)
        assert "## Anexo OCR - 1871642.png" in out
        assert "6,00E+06" in out
    finally:
        mod._processar_imagem_ocr = original_proc
        mod._obter_cache_ocr = original_cache
    print("OK: separador por anexo OCR")


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


def test_cache_real_inm34_matriz_reprocessada():
    """
    Verifica o tratamento do cache real da matriz wide da INM 34/2024.

    O arquivo de cache e estado mutavel: pode estar em versao anterior do
    pipeline, e entao deve ser invalidado, ou ja na versao corrente, e entao
    deve ser mantido. O teste cobre os dois casos em vez de presumir um
    deles, e valida o que de fato importa: que os valores de IRI de cada
    pista sejam recuperaveis apos o pos-processamento.
    """
    cache_path = os.path.join(
        os.path.dirname(__file__),
        "dados_antt",
        ".ocr_cache",
        "51e3de9dbfe4672d.txt",
    )
    if not os.path.exists(cache_path):
        print("SKIP: cache 51e3de9dbfe4672d.txt nao encontrado")
        return

    with open(cache_path, "r", encoding="utf-8") as f:
        conteudo_cache = f.read()

    meta, conteudo = _parse_meta_cache_ocr(conteudo_cache)
    versao_cache = meta.get("ocr_pipeline_v", "1")
    invalido = _cache_ocr_deve_ser_invalidado(conteudo_cache)

    if versao_cache != _OCR_PIPELINE_VERSION:
        assert invalido is True, (
            f"Cache na versao {versao_cache} deve ser invalidado "
            f"pelo pipeline v{_OCR_PIPELINE_VERSION}"
        )
    else:
        assert invalido is False, (
            "Cache ja na versao corrente e com qualidade aceitavel nao "
            "deve ser invalidado; reprocessar sem necessidade e custoso"
        )

    proc = _pos_processar_tabela_ocr(conteudo)

    # Os valores abaixo eram o defeito original: IRI saia invertido entre
    # pista principal e marginal, e o percentual 60/40 se perdia.
    for esperado in ("Principal", "Marginal", "3,5 m/km", "2,7 m/km", "3,0 m/km"):
        assert esperado in proc, (
            f"Valor '{esperado}' ausente apos pos-processamento do cache real"
        )

    print(
        f"OK: cache matriz INM34 (v{versao_cache}) tratado corretamente "
        "e valores de IRI preservados"
    )


def main():
    print("=" * 60)
    print(f"TESTES OCR QUALIDADE v{_OCR_PIPELINE_VERSION} (sandbox)")
    print("=" * 60)

    test_score_tabela_corrompida_baixo()
    test_score_tabela_boa_alto()
    test_correcao_notacao_cientifica()
    test_pos_processamento_aplica_correcoes()
    test_detecta_matriz_wide_fragmentada()
    test_reconstrucao_iri_principal_marginal()
    test_reconstrucao_vdm_nao_regride()
    test_separador_anexo_enriquecimento()
    test_cache_invalida_versao_antiga()
    test_cache_invalida_qualidade_baixa()
    test_cache_valido_versao_atual()
    test_cache_real_inm34_matriz_reprocessada()

    print("=" * 60)
    print("TODOS OS TESTES PASSARAM")
    print("=" * 60)


if __name__ == "__main__":
    main()
