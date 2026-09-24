# Matriz de integracoes SIGESCANTT e RAG

Recorte do oficio para o componente de IA. Nao ha acesso a banco. Nao ha sincronizacao periodica nesta entrega.

## Consulta

| Campo | Valor |
| --- | --- |
| Origem | Backend do SIGESCANTT |
| Destino | RAG-API `POST /api/query` |
| Finalidade | Consulta normativa |
| Dados | Pergunta, filtros, resposta, fontes e ids de correlacao |
| Frequencia | Sob demanda |
| Auth | API Key de servico |
| PII | Nao. Sem identificador de fiscal |
| Responsavel origem | OTI / SIGESC |
| Responsavel destino | DeepFeed |

## Rebuild completo (operacao tecnica)

| Campo | Valor |
| --- | --- |
| Origem | Equipe tecnica autorizada |
| Destino | RAG-API `POST /api/reindex` |
| Finalidade | Reindexar a base comum |
| Dados | Disparo do job. Embedding so se estiver em `RAG_EMBEDDING_ALLOWED` (inicial `local`) |
| Frequencia | Sob demanda, operacao |
| Auth | API Key de servico e `X-Ops-Key`, quando configurada |
| Base | Compartilhada, nao por fiscal. Lock ativo responde 409 |
| Responsavel origem | OTI / SIGESC |
| Responsavel destino | DeepFeed |

## Processar documento

| Campo | Valor |
| --- | --- |
| Origem | Tela do SIGESC, via backend, ou inbox PDF por volume |
| Destino | RAG-API `POST /api/documents` |
| Finalidade | Converter e indexar automaticamente na base comum |
| Dados | PDF, DOCX ou XLSX pela API; somente PDF na inbox |
| Frequencia | Sob demanda |
| Auth | A mesma API Key de servico |
| Efeito | Retorna `job_id`; no sucesso a proxima consulta usa a nova geracao |
| Responsavel origem | OTI / SIGESC |
| Responsavel destino | DeepFeed |
