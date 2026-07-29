# Regra: manter atualizado o LaTeX do piloto LLM local

## Documento obrigatorio

Arquivo-fonte: `docs/piloto_llm_local_avaliacao.tex`

Espelho Cursor: `.cursor/rules/atualizar-piloto-llm-local.mdc`

Estilo de referencia: `docs/arquitetura_rag_api.tex`

## Quando atualizar

Sempre que ocorrer qualquer um dos itens abaixo no projeto RAG-ANTT, atualizar o `.tex` **na mesma entrega** (nao deixar para depois):

1. Nova **abordagem** implementada, testada, priorizada ou descartada (modelo, truncamento, template, API, etc.).
2. Nova **dificuldade** operacional ou de engenharia relevante (instalacao, disco, RAM, GPU, permissao, dependencia).
3. Novo **gargalo** identificado no pipeline (recuperacao, contexto, geracao, latencia).
4. Novo **resultado** qualitativo ou quantitativo (comparativo de modelos, pergunta-gabarito, harness `avaliar_retrieval.py`, RAGAS).
5. Mudanca de **defaults** (modelo Ollama padrao, `_LIMITE_CONTEXTO_OLLAMA`, fallback cloud, perfil `RAG_LLM_ALLOWED_PROVIDERS`).

## O que registrar no LaTeX

- Data/revisao no rodape ou secao de controle.
- Tabela ou item em: Abordagens, Dificuldades, Gargalos e Achados/Resultados.
- Status claro: Testado / Planejado / Descartado / Em andamento.
- Evidencia minima: pergunta usada, modelo, veredito em 1--3 frases (e caminho de relatorio se houver).

## O que nao fazer

- Nao criar documento paralelo com o mesmo conteudo sem apontar para este `.tex`.
- Nao atualizar so o chat/README e esquecer o LaTeX.
- Nao apagar historico de achados; marcar status antigo e acrescentar linha nova.

## Checklist rapido (agente ou humano)

- [ ] Editei `docs/piloto_llm_local_avaliacao.tex`?
- [ ] Inclui abordagem e/ou dificuldade e/ou resultado?
- [ ] Tabelas de status ainda coerentes?
- [ ] Mencionei artefatos tocados (`config.py`, `llm_providers.py`, `antt_rag_unified.py`, testes, relatorios)?
