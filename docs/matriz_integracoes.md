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

## Atualizar base

| Campo | Valor |
| --- | --- |
| Origem | Tela do SIGESC, via backend |
| Destino | RAG-API `POST /api/reindex` |
| Finalidade | Reindexar a base comum |
| Dados | Disparo do job. Embedding so se estiver em `RAG_EMBEDDING_ALLOWED` (inicial `local`) |
| Frequencia | Sob demanda, operacao |
| Auth | A mesma API Key de servico |
| Base | Compartilhada, nao por fiscal. Lock ativo responde 409 |
| Responsavel origem | OTI / SIGESC |
| Responsavel destino | DeepFeed |

## Processar documento

| Campo | Valor |
| --- | --- |
| Origem | Tela do SIGESC, via backend |
| Destino | RAG-API `POST /api/documents` |
| Finalidade | Gravar um PDF na base comum |
| Dados | Bytes do PDF e o nome do arquivo |
| Frequencia | Sob demanda |
| Auth | A mesma API Key de servico |
| Efeito | O arquivo so entra na consulta depois de Atualizar base |
| Responsavel origem | OTI / SIGESC |
| Responsavel destino | DeepFeed |
