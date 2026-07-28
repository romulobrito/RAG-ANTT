"""
Comparacao lado a lado de modelos de LLM sobre um conjunto de referencia.

Executa as mesmas perguntas normativas contra varios modelos, reutilizando
exatamente os mesmos trechos recuperados, de modo que a unica variavel seja
o modelo. Gera um relatorio em markdown com as respostas completas e uma
tabela de cobertura dos valores esperados.

O escore e um indicador de triagem, nao um oraculo: ele apenas verifica se os
valores de referencia aparecem no texto. A leitura das respostas continua
sendo necessaria antes de decidir a troca de modelo.

Uso:
    python comparar_modelos.py
    python comparar_modelos.py --modelos deepseek-v4-flash,deepseek-chat
    python comparar_modelos.py --perguntas iri_principal,prazos_eef
    python comparar_modelos.py --trechos 20 --saida relatorio.md
"""

import argparse
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from config import LLM_PROVIDERS, logger
from llm_providers import create_llm_manager


# ---------------------------------------------------------------------------
# Conjunto de referencia
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CasoReferencia:
    """
    Pergunta de referencia com os valores que a resposta deve conter.

    Attributes:
        identificador: Chave curta usada na linha de comando.
        pergunta: Texto enviado ao sistema de consulta.
        esperados: Trechos que devem aparecer na resposta.
        origem: Documento de onde o gabarito foi extraido.
    """

    identificador: str
    pergunta: str
    esperados: Tuple[str, ...]
    origem: str


# Gabaritos extraidos das tabelas auxiliares em dados_antt/tabelas_auxiliares.
CASOS_REFERENCIA: Tuple[CasoReferencia, ...] = (
    CasoReferencia(
        identificador="iri_principal",
        pergunta=(
            "Quais sao os limites de irregularidade longitudinal (IRI) da "
            "pista principal em cada fase, conforme a Instrucao Normativa "
            "34 de 2024?"
        ),
        esperados=("3,5 m/km", "2,7 m/km", "60%", "40%"),
        origem="INM 34/2024 - Anexo, IRI pista principal",
    ),
    CasoReferencia(
        identificador="iri_marginal",
        pergunta=(
            "Qual o limite de IRI da pista marginal nas fases de manutencao "
            "e de recebimento final na Instrucao Normativa 34 de 2024?"
        ),
        esperados=("3,0 m/km",),
        origem="INM 34/2024 - Anexo, IRI pista marginal",
    ),
    CasoReferencia(
        identificador="dadm_fwd",
        pergunta=(
            "Qual a deflexao admissivel (Dadm) para VDM comercial entre "
            "1000 e 2500 veiculos, conforme a Instrucao Normativa 34 de 2024?"
        ),
        esperados=("50",),
        origem="INM 34/2024 - Anexo, Dadm por VDM comercial",
    ),
    CasoReferencia(
        identificador="ifi_manutencao",
        pergunta=(
            "Qual o valor de IFI exigido na fase de manutencao pela "
            "Instrucao Normativa 34 de 2024?"
        ),
        esperados=("0,2",),
        origem="INM 34/2024 - Anexo, IFI",
    ),
    CasoReferencia(
        identificador="cronograma_revisao",
        pergunta=(
            "Quais sao os prazos do cronograma de revisao ordinaria em "
            "relacao a data-base, conforme o Anexo I da Instrucao Normativa "
            "18 de 2023?"
        ),
        esperados=("140", "90", "75", "35"),
        origem="INM 18/2023 - Anexo I, cronograma",
    ),
    CasoReferencia(
        identificador="prazos_eef",
        pergunta=(
            "Quais os prazos maximos para analise de admissibilidade e para "
            "a decisao da Diretoria no pedido de recomposicao do equilibrio "
            "economico-financeiro, conforme a Instrucao Normativa 33 de 2024?"
        ),
        esperados=("60 dias", "180 dias"),
        origem="INM 33/2024 - Art. 17",
    ),
    CasoReferencia(
        identificador="recurso_admissibilidade",
        pergunta=(
            "Qual o prazo para recurso contra o indeferimento da "
            "admissibilidade na Instrucao Normativa 33 de 2024?"
        ),
        esperados=("15 dias",),
        origem="INM 33/2024 - Art. 11",
    ),
)


MODELOS_PADRAO: Tuple[str, ...] = (
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-chat",
)


# gerar_resposta captura excecoes internamente e devolve texto de aviso em vez
# de propagar o erro. Sem reconhecer essas mensagens, uma falha de rede ou de
# credito seria contabilizada como resposta de baixa qualidade.
_MENSAGENS_DE_FALHA: Tuple[str, ...] = (
    "nao foi possivel gerar a resposta",
    "nao foi possivel gerar uma resposta",
    "nao encontrei documentos relevantes",
    "limitacoes temporarias",
)


def detectar_falha_de_geracao(resposta: str) -> Optional[str]:
    """
    Identifica se a resposta e, na verdade, um aviso de falha.

    Args:
        resposta: Texto devolvido por gerar_resposta.

    Returns:
        Descricao da falha, ou None se a resposta for legitima.
    """
    if not resposta or not resposta.strip():
        return "resposta vazia"

    resposta_norm = normalizar_texto(resposta)
    for marcador in _MENSAGENS_DE_FALHA:
        if marcador in resposta_norm:
            return "geracao nao concluida (indisponibilidade ou credito)"

    return None


# ---------------------------------------------------------------------------
# Normalizacao e escore
# ---------------------------------------------------------------------------


def normalizar_texto(texto: str) -> str:
    """
    Normaliza texto para comparacao tolerante a formatacao.

    Remove acentuacao, reduz a caixa baixa, colapsa espacos e uniformiza o
    separador decimal, de modo que "2,7 m/km" e "2.7 m/km" sejam
    equivalentes.

    Args:
        texto: Texto de entrada.

    Returns:
        Texto normalizado.
    """
    if not isinstance(texto, str):
        return ""

    decomposto = unicodedata.normalize("NFKD", texto)
    sem_acento = "".join(c for c in decomposto if not unicodedata.combining(c))
    minusculo = sem_acento.lower()
    # Uniformiza separador decimal apenas entre digitos.
    decimal_uniforme = re.sub(r"(?<=\d),(?=\d)", ".", minusculo)
    return re.sub(r"\s+", " ", decimal_uniforme).strip()


def avaliar_cobertura(
    resposta: str, esperados: Sequence[str]
) -> Tuple[int, List[str]]:
    """
    Conta quantos valores de referencia aparecem na resposta.

    Args:
        resposta: Texto gerado pelo modelo.
        esperados: Valores que deveriam constar na resposta.

    Returns:
        Tupla (quantidade_encontrada, lista_de_ausentes).
    """
    resposta_norm = normalizar_texto(resposta)
    ausentes: List[str] = []
    encontrados = 0

    for termo in esperados:
        if normalizar_texto(termo) in resposta_norm:
            encontrados += 1
        else:
            ausentes.append(termo)

    return encontrados, ausentes


# ---------------------------------------------------------------------------
# Execucao
# ---------------------------------------------------------------------------


@dataclass
class ResultadoModelo:
    """Resultado de um modelo para uma pergunta especifica."""

    modelo: str
    resposta: str = ""
    segundos: float = 0.0
    encontrados: int = 0
    total_esperado: int = 0
    ausentes: List[str] = field(default_factory=list)
    erro: Optional[str] = None

    @property
    def cobertura(self) -> float:
        """Fracao dos valores de referencia presentes na resposta."""
        if self.total_esperado == 0:
            return 0.0
        return self.encontrados / self.total_esperado


def validar_modelos(modelos: Sequence[str]) -> None:
    """
    Verifica se os modelos informados estao registrados em config.

    Args:
        modelos: Chaves de modelo a validar.

    Raises:
        SystemExit: Se algum modelo nao existir na configuracao.
    """
    disponiveis = set(LLM_PROVIDERS["deepseek"]["models"].keys())
    desconhecidos = [m for m in modelos if m not in disponiveis]

    if desconhecidos:
        print(
            f"Modelos nao configurados: {', '.join(desconhecidos)}\n"
            f"Disponiveis: {', '.join(sorted(disponiveis))}",
            file=sys.stderr,
        )
        raise SystemExit(2)


def executar_caso(
    caso: CasoReferencia,
    documentos: List[object],
    modelos: Sequence[str],
    temperatura: float,
    max_tokens: int,
) -> List[ResultadoModelo]:
    """
    Executa uma pergunta contra todos os modelos usando o mesmo contexto.

    Args:
        caso: Pergunta de referencia.
        documentos: Trechos recuperados, identicos para todos os modelos.
        modelos: Chaves dos modelos a comparar.
        temperatura: Temperatura de geracao.
        max_tokens: Limite de tokens da resposta.

    Returns:
        Lista de resultados, um por modelo.
    """
    from antt_rag_unified import gerar_resposta

    resultados: List[ResultadoModelo] = []

    for modelo in modelos:
        resultado = ResultadoModelo(
            modelo=modelo, total_esperado=len(caso.esperados)
        )
        inicio = time.time()

        try:
            manager = create_llm_manager("deepseek", modelo)
            llm = manager.get_llm(
                temperature=temperatura, max_tokens=max_tokens
            )
            # O nome do modelo e repassado para que o template adaptativo
            # continue selecionando a variante DeepSeek.
            resposta, _ = gerar_resposta(
                caso.pergunta, documentos, llm, modelo
            )
            resultado.resposta = resposta or ""

            falha = detectar_falha_de_geracao(resultado.resposta)
            if falha:
                resultado.erro = falha
                resultado.ausentes = list(caso.esperados)
            else:
                resultado.encontrados, resultado.ausentes = avaliar_cobertura(
                    resultado.resposta, caso.esperados
                )
        except Exception as exc:  # noqa: BLE001 - relatorio deve seguir
            resultado.erro = f"{type(exc).__name__}: {exc}"
            resultado.ausentes = list(caso.esperados)
            logger.error(f"Falha no modelo {modelo}: {exc}")

        resultado.segundos = time.time() - inicio
        resultados.append(resultado)

        estado = "erro" if resultado.erro else f"{resultado.cobertura:.0%}"
        print(
            f"    {modelo:22s} {resultado.segundos:6.1f}s  cobertura={estado}"
        )

    return resultados


def montar_relatorio(
    execucoes: List[Tuple[CasoReferencia, List[ResultadoModelo]]],
    modelos: Sequence[str],
    trechos: int,
) -> str:
    """
    Monta o relatorio em markdown com resumo e respostas completas.

    Args:
        execucoes: Pares (caso, resultados) na ordem de execucao.
        modelos: Modelos comparados.
        trechos: Quantidade de trechos recuperados por pergunta.

    Returns:
        Conteudo do relatorio em markdown.
    """
    agora = datetime.now().strftime("%d/%m/%Y %H:%M")
    linhas: List[str] = [
        "# Comparacao de modelos - RAG ANTT",
        "",
        f"Execucao: {agora}",
        f"Trechos recuperados por pergunta: {trechos}",
        f"Modelos: {', '.join(modelos)}",
        "",
        "O escore indica apenas se os valores de referencia aparecem no",
        "texto. Leia as respostas antes de decidir a troca de modelo.",
        "",
        "## Resumo de cobertura",
        "",
        "| Pergunta | " + " | ".join(modelos) + " |",
        "| --- |" + " --- |" * len(modelos),
    ]

    totais: Dict[str, List[float]] = {m: [] for m in modelos}
    tempos: Dict[str, List[float]] = {m: [] for m in modelos}

    for caso, resultados in execucoes:
        celulas: List[str] = []
        for resultado in resultados:
            if resultado.erro:
                celulas.append("erro")
            else:
                celulas.append(
                    f"{resultado.encontrados}/{resultado.total_esperado}"
                )
                totais[resultado.modelo].append(resultado.cobertura)
            tempos[resultado.modelo].append(resultado.segundos)
        linhas.append(f"| {caso.identificador} | " + " | ".join(celulas) + " |")

    linhas.extend(["", "## Media por modelo", "", "| Modelo | Cobertura media | Tempo medio |", "| --- | --- | --- |"])
    for modelo in modelos:
        valores = totais[modelo]
        tempo = sum(tempos[modelo]) / len(tempos[modelo]) if tempos[modelo] else 0.0
        if valores:
            media = f"{sum(valores) / len(valores):.0%} ({len(valores)} respostas)"
        else:
            media = "n/d (nenhuma resposta obtida)"
        linhas.append(f"| {modelo} | {media} | {tempo:.1f}s |")

    linhas.extend(
        [
            "",
            "Modelos marcados como erro nao chegaram a responder. Verifique",
            "credito da conta, disponibilidade do servico e o limite de tokens",
            "antes de interpretar o resultado como qualidade do modelo.",
        ]
    )

    linhas.extend(["", "## Respostas completas", ""])

    for caso, resultados in execucoes:
        linhas.extend(
            [
                f"### {caso.identificador}",
                "",
                f"**Pergunta:** {caso.pergunta}",
                "",
                f"**Gabarito ({caso.origem}):** {', '.join(caso.esperados)}",
                "",
            ]
        )
        for resultado in resultados:
            linhas.append(f"#### {resultado.modelo}")
            linhas.append("")
            if resultado.erro:
                linhas.extend([f"Falha: {resultado.erro}", ""])
                continue
            ausentes = ", ".join(resultado.ausentes) if resultado.ausentes else "nenhum"
            linhas.extend(
                [
                    f"Cobertura: {resultado.encontrados}/{resultado.total_esperado} "
                    f"| Tempo: {resultado.segundos:.1f}s | Ausentes: {ausentes}",
                    "",
                    resultado.resposta.strip() or "(resposta vazia)",
                    "",
                ]
            )

    return "\n".join(linhas) + "\n"


def main() -> int:
    """
    Ponto de entrada da comparacao.

    Returns:
        Codigo de saida: 0 em sucesso, 1 em falha de carga do indice.
    """
    parser = argparse.ArgumentParser(
        description="Compara modelos de LLM sobre perguntas normativas de referencia."
    )
    parser.add_argument(
        "--modelos",
        default=",".join(MODELOS_PADRAO),
        help="Lista separada por virgula das chaves de modelo.",
    )
    parser.add_argument(
        "--perguntas",
        default="",
        help="Subconjunto de perguntas por identificador, separado por virgula.",
    )
    parser.add_argument(
        "--trechos", type=int, default=20, help="Trechos recuperados por pergunta."
    )
    parser.add_argument(
        "--temperatura", type=float, default=0.1, help="Temperatura de geracao."
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048, help="Limite de tokens da resposta."
    )
    parser.add_argument(
        "--embeddings",
        default="local",
        help="Provedor de embeddings usado na recuperacao.",
    )
    parser.add_argument(
        "--saida", default="", help="Caminho do relatorio markdown."
    )
    args = parser.parse_args()

    modelos = [m.strip() for m in args.modelos.split(",") if m.strip()]
    validar_modelos(modelos)

    casos = CASOS_REFERENCIA
    if args.perguntas:
        selecionados = {p.strip() for p in args.perguntas.split(",") if p.strip()}
        casos = tuple(c for c in CASOS_REFERENCIA if c.identificador in selecionados)
        if not casos:
            print("Nenhuma pergunta corresponde ao filtro.", file=sys.stderr)
            return 2

    from antt_rag_unified import (
        carregar_vectorstore_com_provider,
        pesquisar_documentos,
    )

    print(f"Carregando indice (embeddings: {args.embeddings})...")
    try:
        vectorstore = carregar_vectorstore_com_provider(args.embeddings)
    except Exception as exc:  # noqa: BLE001 - falha de carga encerra execucao
        print(f"Falha ao carregar o indice: {exc}", file=sys.stderr)
        return 1

    execucoes: List[Tuple[CasoReferencia, List[ResultadoModelo]]] = []

    for indice, caso in enumerate(casos, start=1):
        print(f"\n[{indice}/{len(casos)}] {caso.identificador}")
        documentos = pesquisar_documentos(
            caso.pergunta,
            vectorstore,
            k=args.trechos,
            embedding_provider=args.embeddings,
        )
        print(f"    trechos recuperados: {len(documentos)}")

        if not documentos:
            print("    nenhum trecho recuperado; pergunta ignorada")
            continue

        resultados = executar_caso(
            caso, documentos, modelos, args.temperatura, args.max_tokens
        )
        execucoes.append((caso, resultados))

    if not execucoes:
        print("Nenhuma pergunta pode ser executada.", file=sys.stderr)
        return 1

    relatorio = montar_relatorio(execucoes, modelos, args.trechos)

    destino = args.saida
    if not destino:
        pasta = "relatorios_comparacao"
        os.makedirs(pasta, exist_ok=True)
        carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
        destino = os.path.join(pasta, f"comparacao_{carimbo}.md")

    with open(destino, "w", encoding="utf-8") as arquivo:
        arquivo.write(relatorio)

    print(f"\nRelatorio salvo em: {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
