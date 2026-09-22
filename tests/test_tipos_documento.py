"""
Testes do catalogo de tipos gerado a partir da base de conhecimento.
"""

import json

from tipos_documento import (
    atualizar_catalogo_tipos,
    detectar_referencias_documento,
    extrair_nome_do_cabecalho,
    limpar_cache_catalogo_tipos,
    listar_siglas_tipo,
    montar_catalogo_de_tipos,
    montar_catalogo_tipos,
    montar_regex_tipos,
    nome_amigavel_tipo,
    normalizar_alias_tipo,
    resolver_sigla_tipo,
    varrer_tipos_na_base,
)


def test_normalizar_alias_remove_acentos():
    """Aliases com acento devem colidir com a forma ASCII."""
    assert normalizar_alias_tipo("Instrucao Normativa") == "instrucao normativa"
    assert normalizar_alias_tipo("Resolução") == "resolucao"


def test_extrair_nome_do_cabecalho_instrucao():
    """Cabecalho tipico de INM rende nome por extenso."""
    texto = (
        "MINISTERIO DOS TRANSPORTES\n\n"
        "INSTRUCAO NORMATIVA No 34, DE 14 DE NOVEMBRO DE 2024\n"
    )
    assert extrair_nome_do_cabecalho(texto) == "instrucao normativa"


def test_extrair_nome_resolucao_com_acento():
    """Titulo acentuado e normalizado antes do match."""
    texto = "RESOLUÇÃO Nº 6.053, DE 31 DE OUTUBRO DE 2024\n"
    assert extrair_nome_do_cabecalho(texto) == "resolucao"


def test_base_vazia_catalogo_vazio(tmp_path):
    """Base zerada nao inventa tipos do config antigo."""
    base = tmp_path / "dados_antt"
    base.mkdir()
    tipos = varrer_tipos_na_base(str(base))
    assert tipos == {}
    cat = montar_catalogo_de_tipos(tipos)
    assert cat.siglas == ()
    assert listar_siglas_tipo(cat) == []


def test_varrer_extrai_aliases_do_md(tmp_path):
    """Arquivo INM gera sigla + aliases do cabecalho + alias curto 'in'."""
    base = tmp_path / "dados_antt"
    pasta = base / "INM" / "2024"
    pasta.mkdir(parents=True)
    (pasta / "INM-00000034-2024.md").write_text(
        "INSTRUCAO NORMATIVA No 34, DE 14 DE NOVEMBRO DE 2024\nArt. 1\n",
        encoding="utf-8",
    )
    tipos = varrer_tipos_na_base(str(base))
    assert "INM" in tipos
    aliases = set(tipos["INM"]["aliases"])
    assert "instrucao normativa" in aliases
    assert "inm" in aliases
    assert "in" in aliases  # TIPOS_DOCUMENTO_ALIASES_CURTOS


def test_pasta_sem_md_nao_cria_tipo(tmp_path):
    """Pasta vazia nao entra no catalogo (existencia = arquivos)."""
    base = tmp_path / "dados_antt"
    (base / "RES").mkdir(parents=True)
    tipos = varrer_tipos_na_base(str(base))
    assert "RES" not in tipos


def test_ignora_tabelas_auxiliares(tmp_path):
    """tabelas_auxiliares nao vira tipo mesmo com .md."""
    base = tmp_path / "dados_antt"
    aux = base / "tabelas_auxiliares" / "INM" / "2024"
    aux.mkdir(parents=True)
    (aux / "INM-00000034-2024-parametros.md").write_text(
        "INSTRUCAO NORMATIVA\n", encoding="utf-8"
    )
    tipos = varrer_tipos_na_base(str(base))
    assert tipos == {}


def test_atualizar_persiste_json_e_invalida_cache(tmp_path):
    """atualizar_catalogo_tipos grava JSON e limpa lru_cache."""
    base = tmp_path / "dados_antt"
    pasta = base / "XYZ"
    pasta.mkdir(parents=True)
    (pasta / "XYZ-00000001-2024.md").write_text(
        "RESOLUCAO No 1, DE 1 DE JANEIRO DE 2024\n",
        encoding="utf-8",
    )
    limpar_cache_catalogo_tipos()
    cat = atualizar_catalogo_tipos(str(base))
    assert "XYZ" in cat.siglas
    caminho = base / ".catalogo_tipos.json"
    assert caminho.is_file()
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    assert "XYZ" in dados["tipos"]
    assert resolver_sigla_tipo("resolucao", cat) == "XYZ"


def test_detectar_in_34_com_catalogo_vivo():
    """IN 34/2024 resolve via alias curto quando INM esta no catalogo."""
    cat = montar_catalogo_de_tipos(
        {
            "INM": {
                "nome": "instrucao normativa",
                "aliases": ["instrucao normativa", "inm", "in"],
                "n_docs": 1,
            }
        }
    )
    refs = detectar_referencias_documento("Limites na IN 34/2024", cat)
    assert refs
    assert refs[0]["tipo"] == "INM"
    assert refs[0]["numero"] == "34"


def test_detectar_sem_tipo_na_base():
    """Sem tipos na base, nenhuma referencia e detectada."""
    cat = montar_catalogo_de_tipos({})
    refs = detectar_referencias_documento("IN 34/2024", cat)
    assert refs == []


def test_regex_prefere_alias_longo():
    """Alias longo vem antes de 'in' no alternation."""
    cat = montar_catalogo_de_tipos(
        {
            "INM": {
                "nome": "instrucao normativa",
                "aliases": ["instrucao normativa", "inm", "in"],
            }
        }
    )
    partes = montar_regex_tipos(cat).split("|")
    idx_longo = next(i for i, p in enumerate(partes) if "instrucao" in p)
    idx_in = partes.index("in")
    assert idx_longo < idx_in


def test_compat_montar_catalogo_tipos_override():
    """API de override ainda funciona para testes legados."""
    cat = montar_catalogo_tipos(
        entradas_config=[
            {
                "sigla": "RES",
                "nome": "Resolucao",
                "aliases": ("resolucao", "res"),
            }
        ],
        siglas_disco=["XYZ"],
    )
    assert resolver_sigla_tipo("resolucao", cat) == "RES"
    assert "XYZ" in listar_siglas_tipo(cat)
    assert nome_amigavel_tipo("RES", cat) == "Resolucao"
