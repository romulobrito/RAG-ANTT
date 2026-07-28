# Avaliacao de qualidade do RAG-ANTT

Guia do harness offline usado para medir retrieval, geracao e metricas RAGAS.

## Visao geral

O script `avaliar_retrieval.py` percorre um conjunto fixo de perguntas (gabarito multi-dominio) e gera relatorios em `relatorios_avaliacao/` (Markdown + JSON).

Ha duas camadas de metrica:

1. **Deterministicas (sempre)** — matching textual contra o gabarito normativo; baratas e estaveis.
2. **RAGAS (opcional)** — juiz LLM (`ragas==0.1.21`); mede aderencia ao contexto e relevancia.

No dominio ANTT (limites numericos de normas), o gabarito **nao e substituido** pelo juiz: uma resposta plausivel com valor errado pode passar em relevancy e falhar na completude factual.

## Dependencia

```
ragas==0.1.21
datasets>=2.14.0,<3.0.0
pysbd==0.3.4
```

Usar RAGAS 0.1.x. Versoes 0.2+ puxam LangChain 1.x e quebram o pin `langchain==0.2.15` do projeto.

## Comandos

```bash
cd RAG-ANTT

# Retrieval apenas
python avaliar_retrieval.py
python avaliar_retrieval.py --k 16 --casos pavimento_generico,iri_principal

# + geracao (completude da resposta + latencia de geracao)
python avaliar_retrieval.py --com-geracao --casos iri_principal

# + RAGAS (implica geracao; usa o mesmo LLM como juiz)
python avaliar_retrieval.py --com-ragas --casos iri_principal,dadm_vdm \
  --llm-provider deepseek --llm-model deepseek-v4-flash
```

Saida tipica:

- `relatorios_avaliacao/retrieval_YYYYMMDD_HHMMSS.md`
- `relatorios_avaliacao/retrieval_YYYYMMDD_HHMMSS.json`

## Metricas

| Metrica | Origem | Significado |
|---|---|---|
| `cobertura` | Gabarito | Fracao dos `valores_esperados` presentes nos chunks recuperados |
| `completude_resposta` | Gabarito | Fracao dos valores presentes na resposta gerada |
| `hit_documento` | Gabarito | Documento-alvo (tipo/numero/ano) aparece no top-k |
| `n_estruturada` / `n_ocr` | Classificador de fonte | Preferencia por tabelas auxiliares vs OCR |
| `tempo_retrieval_s` | Relogio | Latencia da busca hibrida |
| `tempo_geracao_s` | Relogio | Latencia do LLM de resposta |
| `faithfulness` | RAGAS | Afirmacoes da resposta sustentadas pelo contexto |
| `answer_relevancy` | RAGAS | Resposta endereca a pergunta |
| `context_precision` | RAGAS | Contexto recuperado e focado (usa referencia) |
| `context_recall` | RAGAS | Contexto cobre a referencia (usa ground_truth) |

## Casos de teste

Definidos em `CASOS_AVALIACAO` dentro de `avaliar_retrieval.py`. Dominios atuais:

- pavimento (INM 34/2024)
- prazos (INM 18/2023)
- eef (INM 33/2024)

Para incluir um dominio novo: adicione um `CasoAvaliacao` com pergunta, valores esperados e documento-alvo.

## Testes automaticos

Sem carregar FAISS nem chamar LLM:

```bash
python -m pytest test_avaliar_retrieval.py test_retrieval_hibrido.py -q
```

## Relacao com a arquitetura

No monolitico Streamlit atual, o harness chama as mesmas funcoes de producao:

- `carregar_vectorstore_com_provider`
- `pesquisar_documentos` (via `retrieval_hibrido`)
- `gerar_resposta`

Na arquitetura-alvo (API + Kubernetes), o harness continua como ferramenta de CI/homologacao; a API nao precisa expor RAGAS em tempo de consulta. Ver `docs/arquitetura_rag_api.tex`, secao de avaliacao.
