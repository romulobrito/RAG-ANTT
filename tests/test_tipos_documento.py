"""
Testes do catalogo de tipos gerado a partir da base de conhecimento.
"""

import json
from typing import Sequence

from scripts.gerar_relatorio import _extrair_metadados_do_nome, _varrer_filesystem
from tipos_documento import (
    atualizar_catalogo_tipos,
    classificar_documento_sem_padrao,
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


def test_alias_repetido_fica_com_maior_n_docs():
    """Alias compartilhado fica com a sigla que tem mais documentos."""
    cat = montar_catalogo_de_tipos(
        {
            "INC": {
                "nome": "instrucao normativa",
                "aliases": [
                    "inc",
                    "instrucao normativa",
                    "instrucao normativa complementar",
                ],
                "n_docs": 2,
            },
            "INM": {
                "nome": "instrucao normativa",
                "aliases": ["in", "inm", "instrucao normativa"],
                "n_docs": 11,
            },
        }
    )
    assert resolver_sigla_tipo("instrucao normativa", cat) == "INM"
    assert resolver_sigla_tipo("inc", cat) == "INC"
    assert resolver_sigla_tipo("instrucao normativa complementar", cat) == "INC"


def _modelo_indisponivel(trecho: str, siglas: Sequence[str]) -> str:
    """Substitui o modelo local por uma falha."""
    raise RuntimeError("sem modelo")


def test_entrada_sem_padrao_entra_como_outros(tmp_path):
    """Arquivo da entrada sem nome padrao e indexado como OUTROS."""
    base = tmp_path / "dados_antt"
    entrada = base / "entrada"
    entrada.mkdir(parents=True)
    rascunho = entrada / "rascunho.md"
    rascunho.write_text(
        "INSTRUCAO NORMATIVA\n",
        encoding="utf-8",
    )
    (entrada / "INM-00000034-2024.md").write_text(
        "INSTRUCAO NORMATIVA No 34\n",
        encoding="utf-8",
    )
    tipos = varrer_tipos_na_base(str(base), sugerir_sigla=_modelo_indisponivel)
    assert "ENTRADA" not in tipos
    assert "INM" in tipos
    assert "OUTROS" in tipos
    encontrados = _varrer_filesystem(str(base))
    assert "rascunho.md" in encontrados
    assert "INM-00000034-2024.md" in encontrados
    meta = _extrair_metadados_do_nome(
        "rascunho.md",
        str(rascunho),
        sugerir_sigla=_modelo_indisponivel,
    )
    assert meta["tipo"] == "OUTROS"
    assert meta["numero"] == ""
    assert meta["ano"] == ""


def test_cabecalho_classifica_sem_modelo(tmp_path):
    """Cabecalho conhecido vira sigla, numero e ano sem chamar o modelo."""
    def _proibido(trecho: str, siglas: Sequence[str]) -> str:
        raise AssertionError("modelo nao deveria ser chamado")

    base = tmp_path / "dados_antt" / "entrada"
    base.mkdir(parents=True)
    caminho = base / "ato.md"
    caminho.write_text(
        "Resolucao No 6.057, de 1 de janeiro de 2024\n",
        encoding="utf-8",
    )
    catalogo = montar_catalogo_de_tipos(
        {
            "RES": {
                "nome": "resolucao",
                "aliases": ["resolucao", "res"],
                "n_docs": 4,
            }
        }
    )
    resultado = classificar_documento_sem_padrao(
        str(caminho),
        catalogo=catalogo,
        sugerir_sigla=_proibido,
    )
    assert resultado.tipo == "RES"
    assert resultado.numero == "6057"
    assert resultado.ano == "2024"


def test_modelo_invalido_ou_falho_cai_em_outros(tmp_path):
    """Sigla fora da lista ou excecao do modelo vira OUTROS."""
    base = tmp_path / "dados_antt" / "entrada"
    base.mkdir(parents=True)
    caminho = base / "solto.md"
    caminho.write_text("Memorando interno sem ato.\n", encoding="utf-8")
    catalogo = montar_catalogo_de_tipos(
        {
            "RES": {
                "nome": "resolucao",
                "aliases": ["resolucao", "res"],
                "n_docs": 1,
            }
        }
    )

    def _invalida(trecho: str, siglas: Sequence[str]) -> str:
        return "ZZZ"

    invalido = classificar_documento_sem_padrao(
        str(caminho),
        catalogo=catalogo,
        sugerir_sigla=_invalida,
    )
    assert invalido.tipo == "OUTROS"
    assert invalido.numero == ""

    falho = classificar_documento_sem_padrao(
        str(caminho),
        catalogo=catalogo,
        sugerir_sigla=_modelo_indisponivel,
    )
    assert falho.tipo == "OUTROS"
