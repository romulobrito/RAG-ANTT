#!/usr/bin/env python3
"""
Testes sandbox P0 dados_antt - tabelas auxiliares estruturadas.

Padrao replicavel:
  dados_antt/tabelas_auxiliares/{TIPO}/{ANO}/{TIPO}-{NUM}-{ANO}-{slug}.md
  Frontmatter YAML com documento_pai, tipo, numero, ano.
  Mescladas automaticamente ao carregar/indexar o documento pai.

Executar:
    cd RAG-ANTT
    source venv/bin/activate
    python test_dados_antt_p0.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gerar_relatorio import _varrer_filesystem
from antt_rag_unified import (
    _SEPARADOR_TABELAS_AUX,
    _carregar_documento_markdown,
    _id_documento_regulatorio,
    _listar_md_em_dados_antt,
    _listar_tabelas_auxiliares,
    _mesclar_tabelas_auxiliares,
    _remover_frontmatter_yaml,
    detectar_documentos_novos,
)


def test_id_documento_regulatorio():
    assert _id_documento_regulatorio("INM", "34", "2024") == "INM-00000034-2024"
    print("OK: id documento regulatorio padronizado")


def test_listar_tabelas_aux_inm34():
    paths = _listar_tabelas_auxiliares("INM", "34", "2024")
    assert len(paths) >= 1
    assert any("parametros-pavimento" in p for p in paths)
    print(f"OK: listar aux INM 34 ({len(paths)} arquivo(s))")


def test_listar_tabelas_aux_inm18_inm33_res6000():
    assert len(_listar_tabelas_auxiliares("INM", "18", "2023")) >= 1
    assert len(_listar_tabelas_auxiliares("INM", "33", "2024")) >= 1
    assert len(_listar_tabelas_auxiliares("RES", "6000", "2022")) >= 1
    print("OK: auxiliares INM 18, INM 33 e RES 6000 encontrados")


def test_remover_frontmatter():
    bruto = "---\ndocumento_pai: X\n---\n# Titulo\n| A | B |\n"
    limpo = _remover_frontmatter_yaml(bruto)
    assert limpo.startswith("# Titulo")
    assert "documento_pai" not in limpo
    print("OK: frontmatter YAML removido")


def test_mesclar_inm34_contem_iri_dadm_sem_ocr():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    caminho = os.path.join(
        base_dir, "dados_antt", "INM", "2024", "INM-00000034-2024.md"
    )
    with open(caminho, "r", encoding="utf-8") as f:
        conteudo = f.read()

    mesclado = _mesclar_tabelas_auxiliares(conteudo, "INM", "34", "2024")
    assert _SEPARADOR_TABELAS_AUX in mesclado
    assert "2,7 m/km (60%)" in mesclado
    assert "6,00E+06" in mesclado and "| 70 |" in mesclado
    assert "International Friction Index" in mesclado or "IFI" in mesclado
    print("OK: mescla INM 34 com IRI, Dadm/FWD e IFI (sem depender de OCR)")


def test_mesclar_documento_sem_aux_inalterado():
    conteudo = "Art. 1 Texto qualquer.\n"
    saida = _mesclar_tabelas_auxiliares(conteudo, "RES", "9999", "2099")
    assert saida == conteudo
    print("OK: documento sem aux permanece inalterado")


def test_carregar_documento_inm34_chunks_com_tabela():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    caminho = os.path.join(
        base_dir, "dados_antt", "INM", "2024", "INM-00000034-2024.md"
    )

    import antt_rag_unified as mod

    original_enriquecer = mod._enriquecer_imagens_documento
    mod._enriquecer_imagens_documento = lambda c: c
    try:
        docs = _carregar_documento_markdown(
            caminho, "INM", "Instrucao Normativa", "34", "2024"
        )
    finally:
        mod._enriquecer_imagens_documento = original_enriquecer

    assert len(docs) >= 1
    texto_total = "\n".join(d.page_content for d in docs)
    assert "| ---" in texto_total
    assert "2,7 m/km" in texto_total
    assert any(d.metadata.get("contem_tabelas") == "Sim" for d in docs)
    print(f"OK: carregar INM 34 gera {len(docs)} chunk(s) com tabelas aux")


def test_varredura_catalogo_ignora_tabelas_auxiliares():
    md_map = _varrer_filesystem(
        os.path.join(os.path.dirname(__file__), "dados_antt")
    )
    paths = list(md_map.values())
    assert not any("tabelas_auxiliares" in p.replace("\\", "/") for p in paths)
    assert "INM-00000034-2024-parametros-pavimento.md" not in md_map
    print("OK: catalogo nao indexa arquivos em tabelas_auxiliares/")


def test_detector_pendencias_ignora_tabelas_auxiliares():
    """
    O aviso "documento ainda nao incluido" nao deve citar tabelas auxiliares.

    Elas nao sao documentos autonomos; se o detector as listar, o usuario ve
    um falso positivo e clica em Atualizar base sem necessidade.
    """
    raiz = os.path.dirname(os.path.abspath(__file__))
    md_map = _listar_md_em_dados_antt(os.path.join(raiz, "dados_antt"))
    assert "INM-00000034-2024-parametros-pavimento.md" not in md_map
    assert not any(
        "tabelas_auxiliares" in caminho.replace("\\", "/")
        for caminho in md_map.values()
    )

    pendentes = detectar_documentos_novos(
        diretorio=os.path.join(raiz, "dados_antt"),
        relatorio_path=os.path.join(raiz, "relatorio_documentos.json"),
    )
    auxiliares = (
        "INM-00000018-2023-cronograma-revisao-ordinaria.md",
        "INM-00000033-2024-prazos-recomposicao.md",
        "INM-00000034-2024-parametros-pavimento.md",
        "RES-00006000-2022-fases-rcr-parametros.md",
    )
    listados = [nome for nome in pendentes if nome in auxiliares]
    assert not listados, (
        "Falso positivo: tabelas auxiliares apareceram como pendentes: "
        + ", ".join(listados)
    )
    print("OK: detector de pendencias ignora tabelas_auxiliares/")


def main():
    print("=" * 60)
    print("TESTES P0 DADOS_ANTT - TABELAS AUXILIARES (sandbox)")
    print("=" * 60)

    test_id_documento_regulatorio()
    test_listar_tabelas_aux_inm34()
    test_listar_tabelas_aux_inm18_inm33_res6000()
    test_remover_frontmatter()
    test_mesclar_inm34_contem_iri_dadm_sem_ocr()
    test_mesclar_documento_sem_aux_inalterado()
    test_carregar_documento_inm34_chunks_com_tabela()
    test_varredura_catalogo_ignora_tabelas_auxiliares()
    test_detector_pendencias_ignora_tabelas_auxiliares()

    print("=" * 60)
    print("TODOS OS TESTES PASSARAM")
    print("=" * 60)


if __name__ == "__main__":
    main()
