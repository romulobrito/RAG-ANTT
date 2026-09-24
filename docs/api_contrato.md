# Contrato HTTP do RAG para o SIGESCANTT

Contrato congelado na Fase 0. Nao ha servidor nesta entrega. A porta 8000 passa a existir na Fase 2. O Streamlit na porta 8501 e bancada interna de QA e nao e a API.

## 1. Papel do RAG

Caixa-preta HTTP. Nao autentica fiscal. Nao grava o banco do SIGESC. Nao executa regra operacional. O SIGESCANTT envia a pergunta e renderiza a resposta e as fontes.

## 2. Base URL

- No cluster: `http://rag-api-svc:8000`
- Local, a partir da Fase 2: `http://localhost:8000`

O browser do fiscal nao chama essa URL. Quem chama e o backend do SIGESCANTT.

## 3. Auth de servico

Header `X-API-Key`. Alternativa: `Authorization: Bearer` com o mesmo valor. Sem key, ou key errada, responde 401. Nao ha OAuth nem SSO no RAG.

Livres de key: `GET /api/health`, `GET /api/ready` e `GET /api/docs` quando `RAG_SWAGGER=true`. Consulta, status, documents e reindex exigem a key.

## 4. Timeouts

O cliente SIGESC usa timeout de pelo menos 120 segundos. A consulta e sincrona. Nesta entrega nao ha streaming. Em CPU, o tempo tipico fica entre 20 e 50 segundos. O exemplo abaixo usa cerca de 28 segundos e nao e promessa de latencia.

## 5. Rotas

| Metodo | Path | Auth | Sucesso | Proposito |
| --- | --- | --- | --- | --- |
| GET | `/api/health` | nao | 200 | Processo vivo |
| GET | `/api/ready` | nao | 200 ou 503 | Indice e Ollama. Probe do cluster. Nao e consulta |
| GET | `/api/status` | sim | 200 | Provedores, embeddings, formatos de upload/inbox, jobs e quantidade de documentos |
| POST | `/api/query` | sim | 200 | Consulta. Provedor fora de `RAG_LLM_ALLOWED_PROVIDERS` responde 400 |
| GET | `/api/documents` | sim | 200 | Catalogo (`relatorio_documentos.json`) |
| POST | `/api/documents` | sim | 202 | Recebe PDF, DOCX ou XLSX e devolve `job_id` |
| GET | `/api/jobs/{job_id}` | sim | 200 ou 404 | Estado persistido da conversao/indexacao |
| POST | `/api/reindex` | servico + ops | 202 | Rebuild completo tecnico. Lock ativo responde 409 |
| GET | `/api/docs` | flag | 200 | Swagger, so se `RAG_SWAGGER=true`. Desligado em producao |

### 5.1 Upload e job incremental

O backend envia multipart no campo `arquivo`. A extensao e a assinatura
binaria precisam concordar. Limite padrao: 50 MiB.

```bash
curl -s -X POST http://rag-api-svc:8000/api/documents \
  -H "X-API-Key: <segredo-do-servico>" \
  -F "arquivo=@nota.docx"
```

Resposta 202:

```json
{
  "job_id": "f97bb134-170d-4894-8056-403c5a470e60",
  "status": "queued",
  "nome": "nota.docx",
  "formato": "docx"
}
```

O SIGESC consulta `GET /api/jobs/{job_id}` ate `succeeded`,
`succeeded_with_warnings` ou `failed`. A pergunta seguinte ao sucesso ja
consulta documentos antigos e novos na mesma geracao.

A pasta `dados_antt/entrada` e uma inbox exclusiva para PDF. O produtor
grava `<nome>.pdf.part` e renomeia para `<nome>.pdf` quando terminar. DOCX
e XLSX entram somente pela API. Mesmo nome ou mesmo SHA-256 responde conflito
e nao gera novos vetores.

## 6. POST /api/query — request

Exemplo da INM 34. Temperatura, quantidade de trechos e tamanho da resposta nao vao neste JSON. `provider` so entra se a tela tiver mais de um servico liberado. `historico` fica de fora na primeira pergunta.

```http
POST /api/query HTTP/1.1
Host: rag-api-svc:8000
X-API-Key: <segredo-do-servico>
Content-Type: application/json

{
  "pergunta": "Qual o IRI maximo da pista principal na manutencao segundo a INM 34/2024?",
  "filtros": { "tipo_documento": "INM", "ano": 2024, "numero": "34" },
  "correlation_id": "sigesc-ticket-8891"
}
```

`tipo_documento` e uma sigla do catalogo. `OUTROS` e a sigla dos arquivos que nao casaram com um ato conhecido.

## 7. POST /api/query — response

O texto de `resposta` abaixo e ilustrativo. O modelo de 7B pode errar o rotulo. As fontes em `documentos_consultados` sao obrigatorias.

```json
{
  "request_id": "a3f2c1e0-9b44-4c21-8d10-11aa22bb33cc",
  "correlation_id": "sigesc-ticket-8891",
  "resposta": "Segundo a INM 34/2024, o IRI maximo na pista principal na fase de manutencao e 2,7 m/km (...citacao...)",
  "modelo_usado": "qwen2.5:7b",
  "provider": "ollama",
  "documentos_consultados": [
    {
      "tipo": "INM",
      "numero": "34",
      "ano": "2024",
      "trecho": "Irregularidade Longitudinal Maxima - IRI | Principal | ... | 2,7 m/km",
      "caminho": "dados_antt/INM/2024/INM-00000034-2024.md",
      "relevancia": 0.91
    }
  ],
  "tempo_processamento_ms": 28000,
  "total_documentos_encontrados": 30,
  "embedding_provider": "local",
  "vectorstore_utilizado": "vectorstore_local"
}
```

## 8. Validacao

- `pergunta`: pelo menos 1 caractere depois do trim. Vazia responde 400.
- `temperatura`: 0.0 a 1.0. Omissa vale `RAG_LLM_TEMPERATURE` (0.1). Fora da faixa responde 400.
- `max_documentos`: 1 a 40. Omisso vale `RAG_MAX_DOCUMENTOS` (30). Acima de 40 responde 400.
- `provider`: ausente vale `ollama`. Fora de `RAG_LLM_ALLOWED_PROVIDERS` responde 400.
- `historico`: omisso ou vazio e a primeira pergunta. O servico corta por `RAG_HISTORICO_TURNOS` (10), `RAG_HISTORICO_REESCRITA` (8) e `RAG_HISTORICO_CHARS_RESPOSTA` (1000). Excesso nao responde 400. O RAG nao grava essa lista.
- Campo `user_id` ou `question` no JSON responde 400.

Quando o SIGESC omite temperatura, trechos e tokens, a API preenche 0.1, 30 (teto 40) e 4096.

## 9. Erros

Corpo unico, sem stacktrace:

```json
{ "detail": "mensagem", "request_id": null }
```

| Codigo | Quando |
| --- | --- |
| 400 | Pergunta/arquivo invalido, valor fora da faixa ou provedor nao liberado |
| 401 | Sem key ou key errada |
| 403 | Rebuild completo sem `X-Ops-Key` quando configurada |
| 409 | Nome/hash duplicado ou reindex com lock ativo |
| 413 | Upload acima do limite |
| 503 | Ready falho, ou consulta sem indice ou sem Ollama |
| 504 | O modelo estourou o tempo |
| 500 | Falha generica |

## 10. Fora de escopo

SSO, `user_id`, crawler, parametros vivos do SIGESC, streaming e GPU. Processar documento nao esta fora: e `POST /api/documents`.

## 11. Como o SIGESC renderiza

Mostrar `resposta` e `documentos_consultados` (tipo, numero, ano e trecho). Guardar `request_id` se quiser historico. A frase amigavel de timeout ou de servico indisponivel e texto do SIGESC.
