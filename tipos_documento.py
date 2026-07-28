"""
Catalogo de tipos de documento gerado a partir da base de conhecimento.

Fonte de verdade da existencia: arquivos .md sob dados_antt/ (nao config estatico).
Aliases: extraidos do cabecalho/metadados no ingest; aliases curtos opcionais
em config so se aplicam se a sigla existir na base.
Persistencia: dados_antt/.catalogo_tipos.json
Atualizacao: atualizar_catalogo_tipos() no reindex/upload.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from config import TIPOS_DOCUMENTO_ALIASES_CURTOS, TIPOS_DOCUMENTO_IGNORAR_DIRS

logger = logging.getLogger(__name__)

_PADRAO_SIGLA_DIR = re.compile(r"^[A-Z]{2,4}$")
_PADRAO_NOME_ARQUIVO = re.compile(
    r"^([A-Z]{2,4})-(\d+)-(\d{4})\.md$",
    re.IGNORECASE,
)
# Titulos tipicos no cabecalho de atos ANTT (apos normalizacao sem acento).
_PADRAO_TITULO_ATO = re.compile(
    r"(?P<nome>"
    r"instrucao\s+normativa(?:\s+complementar)?"
    r"|resolucao"
    r"|deliberacao"
    r"|portaria"
    r"|decreto"
    r"|lei"
    r"|voto"
    r"|consulta"
    r"|oficio"
    r"|nota\s+tecnica(?:\s+sei)?"
    r"|sei"
    r")\b",
    re.IGNORECASE,
)

_NOME_CATALOGO = ".catalogo_tipos.json"
_MAX_BYTES_CABECALHO = 4000


@dataclass(frozen=True)
class CatalogoTiposDocumento:
    """
    Snapshot do catalogo derivado da base.

    Attributes:
        alias_para_sigla: Alias normalizado -> sigla.
        sigla_para_nome: Sigla -> nome amigavel.
        siglas: Lista ordenada de siglas presentes na base.
    """

    alias_para_sigla: Mapping[str, str]
    sigla_para_nome: Mapping[str, str]
    siglas: Tuple[str, ...]


def normalizar_alias_tipo(valor: str) -> str:
    """Normaliza texto de tipo (sem acento, minusculas, espacos unicos)."""
    if not isinstance(valor, str):
        return ""
    decomposto = unicodedata.normalize("NFKD", valor)
    sem_acento = "".join(c for c in decomposto if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", sem_acento.strip().lower())


def caminho_arquivo_catalogo(base_dir: str = "dados_antt") -> str:
    """Caminho do JSON persistido do catalogo."""
    return os.path.join(base_dir, _NOME_CATALOGO)


def _dirs_ignorados() -> set:
    return {d.lower() for d in TIPOS_DOCUMENTO_IGNORAR_DIRS}


def _caminho_ignorado(caminho: str, base_dir: str) -> bool:
    """True se o caminho esta sob pasta ignorada (ex.: tabelas_auxiliares)."""
    try:
        rel = os.path.relpath(caminho, base_dir)
    except ValueError:
        rel = caminho
    partes = {p.lower() for p in rel.replace("\\", "/").split("/") if p}
    return bool(partes & _dirs_ignorados())


def _inferir_sigla(caminho: str, base_dir: str, nome_arquivo: str) -> Optional[str]:
    """
    Infere a sigla do documento pela pasta sob dados_antt/ ou pelo nome do arquivo.

    Args:
        caminho: Caminho completo do .md.
        base_dir: Raiz dados_antt.
        nome_arquivo: Basename.

    Returns:
        Sigla (ex.: INM) ou None.
    """
    match = _PADRAO_NOME_ARQUIVO.match(nome_arquivo)
    if match:
        return match.group(1).upper()

    try:
        rel = os.path.relpath(caminho, base_dir)
    except ValueError:
        rel = nome_arquivo
    partes = [p for p in rel.replace("\\", "/").split("/") if p]
    if partes and _PADRAO_SIGLA_DIR.match(partes[0].upper()):
        return partes[0].upper()
    return None


def extrair_nome_do_cabecalho(texto: str) -> Optional[str]:
    """
    Extrai o nome por extenso do ato a partir do cabecalho do markdown.

    Args:
        texto: Conteudo (preferencialmente inicio do arquivo).

    Returns:
        Nome normalizado (ex.: "instrucao normativa") ou None.
    """
    if not texto:
        return None
    amostra = texto[:_MAX_BYTES_CABECALHO]
    normalizado = normalizar_alias_tipo(amostra)
    match = _PADRAO_TITULO_ATO.search(normalizado)
    if not match:
        return None
    return normalizar_alias_tipo(match.group("nome"))


def _aliases_de_nome(nome: str, sigla: str) -> List[str]:
    """Gera aliases a partir do nome por extenso e da sigla."""
    aliases: List[str] = []
    nome_n = normalizar_alias_tipo(nome)
    sigla_n = normalizar_alias_tipo(sigla)
    if nome_n:
        aliases.append(nome_n)
    if sigla_n:
        aliases.append(sigla_n)
    return aliases


def _aplicar_aliases_curtos(
    sigla: str,
    aliases: List[str],
) -> List[str]:
    """Acrescenta aliases curtos do config se a sigla existir na base."""
    extras = TIPOS_DOCUMENTO_ALIASES_CURTOS.get(sigla) or ()
    saida = list(aliases)
    for alias in extras:
        chave = normalizar_alias_tipo(str(alias))
        if chave and chave not in saida:
            saida.append(chave)
    return saida


def _sugerir_aliases_llm(sigla: str, trecho: str) -> Tuple[Optional[str], List[str]]:
    """
    Fallback opcional: pede ao LLM nome e aliases quando o regex falha.

    Desligado por padrao (TIPOS_DOCUMENTO_USAR_LLM_ALIASES=False).
    Em falha devolve (None, []).
    """
    try:
        from config import TIPOS_DOCUMENTO_USAR_LLM_ALIASES
    except ImportError:
        return None, []
    if not TIPOS_DOCUMENTO_USAR_LLM_ALIASES:
        return None, []
    if not trecho.strip():
        return None, []

    try:
        from llm_providers import create_llm_manager

        llm = create_llm_manager("deepseek").get_llm(temperature=0.0, max_tokens=200)
        prompt = (
            "Voce classifica documentos regulatorios da ANTT.\n"
            f"Sigla da pasta: {sigla}\n"
            f"Trecho do documento:\n{trecho[:800]}\n\n"
            "Responda em uma linha JSON ASCII com chaves nome e aliases "
            '(lista de strings curtas em portugues sem acento). '
            'Exemplo: {"nome":"nota tecnica","aliases":["nota tecnica","nt"]}'
        )
        resposta = llm.invoke(prompt)
        conteudo = getattr(resposta, "content", str(resposta))
        inicio = conteudo.find("{")
        fim = conteudo.rfind("}")
        if inicio < 0 or fim <= inicio:
            return None, []
        dados = json.loads(conteudo[inicio : fim + 1])
        nome = normalizar_alias_tipo(str(dados.get("nome") or ""))
        aliases_raw = dados.get("aliases") or []
        aliases: List[str] = []
        if isinstance(aliases_raw, list):
            for item in aliases_raw:
                chave = normalizar_alias_tipo(str(item))
                if chave:
                    aliases.append(chave)
        if nome and nome not in aliases:
            aliases.insert(0, nome)
        return (nome or None), aliases
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fallback LLM de aliases falhou para %s: %s", sigla, exc)
        return None, []


def _ler_cabecalho(caminho: str) -> str:
    """Le os primeiros bytes do arquivo para extracao de titulo."""
    try:
        with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(_MAX_BYTES_CABECALHO)
    except OSError:
        return ""


def varrer_tipos_na_base(base_dir: str = "dados_antt") -> Dict[str, Dict[str, object]]:
    """
    Varre .md da base e agrega tipos presentes (com pelo menos 1 arquivo).

    Args:
        base_dir: Raiz dos documentos.

    Returns:
        Dict sigla -> {nome, aliases, n_docs}.
    """
    abs_base = os.path.abspath(base_dir)
    if not os.path.isdir(abs_base):
        return {}

    agregados: Dict[str, Dict[str, object]] = {}

    for dirpath, dirnames, filenames in os.walk(abs_base):
        dirnames[:] = [
            d for d in dirnames
            if d.lower() not in _dirs_ignorados() and not d.startswith(".")
        ]
        if _caminho_ignorado(dirpath, abs_base):
            continue
        for fname in filenames:
            if not fname.endswith(".md"):
                continue
            if fname.startswith("."):
                continue
            caminho = os.path.join(dirpath, fname)
            sigla = _inferir_sigla(caminho, abs_base, fname)
            if not sigla:
                continue

            cabecalho = _ler_cabecalho(caminho)
            nome = extrair_nome_do_cabecalho(cabecalho)
            aliases = _aliases_de_nome(nome or sigla, sigla)

            if nome is None:
                nome_llm, aliases_llm = _sugerir_aliases_llm(sigla, cabecalho)
                if nome_llm:
                    nome = nome_llm
                for alias in aliases_llm:
                    if alias not in aliases:
                        aliases.append(alias)

            if nome is None:
                nome = sigla

            aliases = _aplicar_aliases_curtos(sigla, aliases)
            alias_set = {normalizar_alias_tipo(a) for a in aliases if a}

            if sigla not in agregados:
                agregados[sigla] = {
                    "nome": nome,
                    "aliases": sorted(alias_set),
                    "n_docs": 1,
                }
                continue

            entrada = agregados[sigla]
            entrada["n_docs"] = int(entrada["n_docs"]) + 1
            nome_atual = str(entrada.get("nome") or "")
            if nome and (
                nome_atual.upper() == sigla or len(nome) > len(nome_atual)
            ):
                entrada["nome"] = nome
            existentes = {
                normalizar_alias_tipo(a)
                for a in (entrada.get("aliases") or [])
                if isinstance(a, str)
            }
            existentes.update(alias_set)
            entrada["aliases"] = sorted(a for a in existentes if a)

    return agregados


def montar_catalogo_de_tipos(
    tipos: Mapping[str, Mapping[str, object]],
) -> CatalogoTiposDocumento:
    """
    Constroi CatalogoTiposDocumento a partir do dict agregado.

    Args:
        tipos: sigla -> {nome, aliases, ...}.

    Returns:
        Catalogo tipado para parser/UI.
    """
    alias_para_sigla: Dict[str, str] = {}
    sigla_para_nome: Dict[str, str] = {}

    for sigla_raw, meta in tipos.items():
        sigla = str(sigla_raw).strip().upper()
        if not sigla:
            continue
        nome = str(meta.get("nome") or sigla).strip() or sigla
        sigla_para_nome[sigla] = nome
        alias_para_sigla[normalizar_alias_tipo(sigla)] = sigla
        aliases = meta.get("aliases") or ()
        if isinstance(aliases, (list, tuple)):
            for alias in aliases:
                chave = normalizar_alias_tipo(str(alias))
                if chave:
                    # Em colisao, primeira sigla vista mantem o alias.
                    alias_para_sigla.setdefault(chave, sigla)

    return CatalogoTiposDocumento(
        alias_para_sigla=alias_para_sigla,
        sigla_para_nome=sigla_para_nome,
        siglas=tuple(sorted(sigla_para_nome.keys())),
    )


def salvar_catalogo_tipos(
    tipos: Mapping[str, Mapping[str, object]],
    base_dir: str = "dados_antt",
) -> str:
    """
    Grava o catalogo em dados_antt/.catalogo_tipos.json.

    Args:
        tipos: Agregado de varrer_tipos_na_base.
        base_dir: Raiz dos dados.

    Returns:
        Caminho do arquivo gravado.
    """
    caminho = caminho_arquivo_catalogo(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    payload = {
        "gerado_em": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tipos": {
            sigla: {
                "nome": meta.get("nome", sigla),
                "aliases": list(meta.get("aliases") or []),
                "n_docs": int(meta.get("n_docs") or 0),
            }
            for sigla, meta in sorted(tipos.items())
        },
    }
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    return caminho


def carregar_catalogo_do_arquivo(
    base_dir: str = "dados_antt",
) -> Optional[CatalogoTiposDocumento]:
    """Le o JSON persistido; None se ausente ou invalido."""
    caminho = caminho_arquivo_catalogo(base_dir)
    if not os.path.isfile(caminho):
        return None
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            dados = json.load(f)
        tipos = dados.get("tipos") or {}
        if not isinstance(tipos, dict):
            return None
        return montar_catalogo_de_tipos(tipos)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        logger.warning("Falha ao ler %s: %s", caminho, exc)
        return None


def atualizar_catalogo_tipos(base_dir: str = "dados_antt") -> CatalogoTiposDocumento:
    """
    Regenera o catalogo a partir dos arquivos da base, grava JSON e limpa cache.

    Deve ser chamado apos upload/reindex/conversao de PDFs.

    Args:
        base_dir: Raiz dados_antt.

    Returns:
        Catalogo atualizado.
    """
    tipos = varrer_tipos_na_base(base_dir)
    salvar_catalogo_tipos(tipos, base_dir=base_dir)
    limpar_cache_catalogo_tipos()
    catalogo = montar_catalogo_de_tipos(tipos)
    logger.info(
        "Catalogo de tipos atualizado: %d sigla(s) em %s",
        len(catalogo.siglas),
        caminho_arquivo_catalogo(base_dir),
    )
    return catalogo


@lru_cache(maxsize=8)
def carregar_catalogo_tipos(base_dir: str = "dados_antt") -> CatalogoTiposDocumento:
    """
    Carrega catalogo (JSON se existir; senao varre a base e persiste).

    Args:
        base_dir: Caminho relativo ou absoluto de dados_antt.

    Returns:
        CatalogoTiposDocumento.
    """
    abs_dir = os.path.abspath(base_dir)
    existente = carregar_catalogo_do_arquivo(abs_dir)
    if existente is not None:
        return existente
    tipos = varrer_tipos_na_base(abs_dir)
    try:
        salvar_catalogo_tipos(tipos, base_dir=abs_dir)
    except OSError as exc:
        logger.warning("Nao foi possivel persistir catalogo: %s", exc)
    return montar_catalogo_de_tipos(tipos)


def limpar_cache_catalogo_tipos() -> None:
    """Invalida o cache em memoria de carregar_catalogo_tipos."""
    carregar_catalogo_tipos.cache_clear()


def resolver_sigla_tipo(
    tipo_raw: str,
    catalogo: Optional[CatalogoTiposDocumento] = None,
) -> Optional[str]:
    """Resolve alias/sigla para a sigla canonica presente na base."""
    cat = catalogo if catalogo is not None else carregar_catalogo_tipos()
    chave = normalizar_alias_tipo(tipo_raw)
    if not chave:
        return None
    return cat.alias_para_sigla.get(chave)


def nome_amigavel_tipo(
    sigla: str,
    catalogo: Optional[CatalogoTiposDocumento] = None,
) -> str:
    """Nome por extenso da sigla, ou a propria sigla."""
    cat = catalogo if catalogo is not None else carregar_catalogo_tipos()
    chave = str(sigla or "").strip().upper()
    return cat.sigla_para_nome.get(chave, chave or "Documento")


def listar_siglas_tipo(
    catalogo: Optional[CatalogoTiposDocumento] = None,
) -> List[str]:
    """Siglas presentes na base, ordenadas (filtro UI)."""
    cat = catalogo if catalogo is not None else carregar_catalogo_tipos()
    return list(cat.siglas)


def montar_regex_tipos(
    catalogo: Optional[CatalogoTiposDocumento] = None,
) -> str:
    """Alternation regex dos aliases (mais longos primeiro)."""
    cat = catalogo if catalogo is not None else carregar_catalogo_tipos()
    termos = sorted(cat.alias_para_sigla.keys(), key=len, reverse=True)
    partes: List[str] = []
    vistos = set()
    for termo in termos:
        if termo in vistos:
            continue
        vistos.add(termo)
        partes.append(re.escape(termo).replace(r"\ ", r"\s+"))
    if not partes:
        # Catalogo vazio: nao casa nenhum tipo.
        return r"(?!)"
    return "|".join(partes)


def aliases_ordenados(
    catalogo: Optional[CatalogoTiposDocumento] = None,
) -> Iterable[str]:
    """Itera aliases conhecidos."""
    cat = catalogo if catalogo is not None else carregar_catalogo_tipos()
    return cat.alias_para_sigla.keys()


def detectar_referencias_documento(
    query: str,
    catalogo: Optional[CatalogoTiposDocumento] = None,
) -> List[Dict[str, str]]:
    """Detecta referencias tipo/numero/ano na pergunta."""
    cat = catalogo if catalogo is not None else carregar_catalogo_tipos()
    tipos_regex = montar_regex_tipos(cat)
    query_n = normalizar_alias_tipo(query) if isinstance(query, str) else ""
    if not query_n or tipos_regex == r"(?!)":
        return []

    padrao = re.compile(
        rf"({tipos_regex})\s*(?:n[o.]\s*)?(\d[\d.]*)\s*(?:/|,?\s*de\s+)(\d{{4}})",
        re.IGNORECASE,
    )

    resultados: List[Dict[str, str]] = []
    for match in padrao.finditer(query_n):
        tipo_raw = normalizar_alias_tipo(match.group(1))
        numero_raw = match.group(2).replace(".", "")
        ano = match.group(3)
        sigla = resolver_sigla_tipo(tipo_raw, cat)
        if sigla is None:
            continue
        resultados.append({
            "tipo": sigla,
            "numero": numero_raw,
            "ano": ano,
        })
    return resultados


# Compatibilidade com testes/chamadas antigas.
def montar_catalogo_tipos(
    base_dir: str = "dados_antt",
    entradas_config: Optional[Sequence[Mapping[str, object]]] = None,
    siglas_disco: Optional[Sequence[str]] = None,
) -> CatalogoTiposDocumento:
    """
    Compat: monta catalogo em memoria.

    Preferir montar_catalogo_de_tipos / atualizar_catalogo_tipos.
    Se entradas_config/siglas_disco forem passados, nao varre disco.
    """
    if entradas_config is not None or siglas_disco is not None:
        tipos: Dict[str, Dict[str, object]] = {}
        if entradas_config:
            for item in entradas_config:
                sigla = str(item.get("sigla") or "").strip().upper()
                if not sigla:
                    continue
                tipos[sigla] = {
                    "nome": item.get("nome") or sigla,
                    "aliases": list(item.get("aliases") or []),
                    "n_docs": 1,
                }
        if siglas_disco:
            for sigla_raw in siglas_disco:
                sigla = str(sigla_raw).strip().upper()
                if sigla and sigla not in tipos:
                    tipos[sigla] = {
                        "nome": sigla,
                        "aliases": [normalizar_alias_tipo(sigla)],
                        "n_docs": 1,
                    }
        return montar_catalogo_de_tipos(tipos)
    return montar_catalogo_de_tipos(varrer_tipos_na_base(base_dir))
