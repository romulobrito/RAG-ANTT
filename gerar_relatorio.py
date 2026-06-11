import os
import json
import glob
import re

DIR_DADOS = "dados_antt"
TIPOS_DOCUMENTO = ["RES", "POR", "INM", "DLB", "INC", "VTO", "DEC", "LEI", "CON"]

_PADRAO_NOME = re.compile(
    r"^([A-Z]{2,4})-(\d+)-(\d{4})\.md$"
)


def _extrair_metadados_do_nome(nome_arquivo: str, caminho: str) -> dict:
    """
    Extrai tipo, numero e ano a partir do nome do arquivo .md.

    Suporta dois formatos:
    - Padrao regulatorio: RES-00006053-2024.md
    - Processo SEI/generico: 50500.0124472025-25.md, SEI_34503555_Nota_Tecnica.md

    Args:
        nome_arquivo: Nome do arquivo.
        caminho: Caminho completo do arquivo.

    Returns:
        dict com campos compativeis com o formato do catalogo.
    """
    match = _PADRAO_NOME.match(nome_arquivo)
    if match:
        tipo = match.group(1)
        numero = str(int(match.group(2)))
        ano = match.group(3)
        return {
            "titulo": f"{tipo} {numero}/{ano}",
            "ementa": "",
            "data": "",
            "tipo": tipo,
            "numero": numero,
            "ano": ano,
            "orgao": "ANTT",
            "url": "",
            "arquivo_html": "",
            "arquivo_md": caminho,
        }

    # Fallback: inferir tipo pela pasta pai ou pelo nome do arquivo
    nome_base = os.path.splitext(nome_arquivo)[0]
    pasta_pai = os.path.basename(os.path.dirname(caminho))

    # Detectar tipo SEI pelo padrao de processo (50500.XXXXXXX-XX)
    if re.match(r"^\d{5}\.\d{7}", nome_base):
        tipo = "SEI"
        numero = nome_base
    elif "nota_tecnica" in nome_base.lower() or "Nota_Tecnica" in nome_base:
        tipo = "NT"
        numero = nome_base
    else:
        tipo = pasta_pai.upper() if pasta_pai else "DOC"
        numero = nome_base

    return {
        "titulo": nome_base.replace("_", " ").replace("-", "/"),
        "ementa": "",
        "data": "",
        "tipo": tipo,
        "numero": numero,
        "ano": "",
        "orgao": "ANTT",
        "url": "",
        "arquivo_html": "",
        "arquivo_md": caminho,
    }


def _varrer_filesystem(diretorio: str) -> dict:
    """
    Varre recursivamente o diretorio e retorna dict {basename: caminho}
    de todos os .md encontrados (deduplicado por basename, pega o primeiro).

    Args:
        diretorio: Raiz da varredura.

    Returns:
        dict mapeando nome_arquivo -> caminho_relativo.
    """
    resultado: dict = {}
    for dirpath, _dirs, filenames in os.walk(diretorio):
        for fname in filenames:
            if fname.endswith(".md") and fname not in resultado:
                resultado[fname] = os.path.join(dirpath, fname)
    return resultado


def gerar_relatorio_documentos():
    """
    Gera relatorio_documentos.json mesclando:
    1. Entradas do documentos.json (fonte primaria com metadados ricos)
    2. Varredura do filesystem para capturar .md nao registrados

    Deduplicacao por nome de arquivo garante que nenhum documento
    seja indexado em duplicidade.

    Returns:
        list[dict]: Lista de documentos catalogados.
    """
    print(f"Gerando relatorio de documentos em {DIR_DADOS}...")

    if not os.path.exists(DIR_DADOS):
        print(f"Erro: Diretorio {DIR_DADOS} nao encontrado.")
        return []

    # 1) Carregar documentos.json (metadados ricos) se existir
    documentos_por_nome: dict = {}
    documentos_json_path = os.path.join(DIR_DADOS, "documentos.json")
    if os.path.exists(documentos_json_path):
        try:
            with open(documentos_json_path, "r", encoding="utf-8") as f:
                docs_json = json.load(f)
            for doc in docs_json:
                arq = doc.get("arquivo_md", "")
                if arq:
                    basename = os.path.basename(arq)
                    documentos_por_nome[basename] = doc
            print(f"Carregados {len(documentos_por_nome)} documentos do arquivo principal.")
        except Exception as e:
            print(f"Erro ao carregar documentos.json: {e}")

    # 2) Varrer filesystem para capturar .md nao registrados
    md_no_disco = _varrer_filesystem(DIR_DADOS)
    novos = 0
    for nome_arquivo, caminho in md_no_disco.items():
        if nome_arquivo not in documentos_por_nome:
            meta = _extrair_metadados_do_nome(nome_arquivo, caminho)
            if meta:
                documentos_por_nome[nome_arquivo] = meta
                novos += 1

    if novos > 0:
        print(f"Descobertos {novos} documentos novos via varredura do filesystem.")

    # 3) Gerar lista final e salvar
    documentos = list(documentos_por_nome.values())
    print(f"Total: {len(documentos)} documentos no catalogo.")

    with open("relatorio_documentos.json", "w", encoding="utf-8") as f:
        json.dump(documentos, f, ensure_ascii=False, indent=4)

    print("Relatorio salvo em relatorio_documentos.json")
    return documentos


if __name__ == "__main__":
    documentos = gerar_relatorio_documentos()
    print(f"Processamento concluido. {len(documentos)} documentos no relatorio.")
