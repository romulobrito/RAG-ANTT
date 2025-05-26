# 🚛 Sistema RAG Unificado - ANTT

## 📋 Visão Geral

O Sistema RAG Unificado combina **todas as funcionalidades avançadas** do `chat-RAG.py` com o **sistema de múltiplos provedores** (OpenAI e DeepSeek), oferecendo a melhor experiência possível para consultas sobre regulamentações da ANTT.

## 🆚 Comparação dos Sistemas

### 📊 Funcionalidades Disponíveis

| Funcionalidade | antt_rag_deepseek.py | antt_rag_unified.py |
|---|---|---|
| **Múltiplos Provedores** | ✅ OpenAI + DeepSeek | ✅ OpenAI + DeepSeek |
| **Templates Especializados** | ❌ Básico | ✅ 4 Templates Avançados |
| **Busca Híbrida** | ❌ Simples | ✅ Semântica + Keywords + MMR |
| **Reranking Inteligente** | ❌ | ✅ Score por Relevância |
| **Extração Agressiva** | ❌ | ✅ Fallback Inteligente |
| **Filtros Avançados** | ✅ Básicos | ✅ Tipo + Ano + Número |
| **Processamento PDF** | ❌ | ✅ Tabelas + OCR |
| **Interface Avançada** | ✅ Boa | ✅ Completa |
| **Citações Automáticas** | ❌ | ✅ Extração Inteligente |
| **Análise de Contexto** | ❌ | ✅ Detecção Automática |

## 🚀 Como Executar

### Sistema Unificado (Recomendado)
```bash
source .env && streamlit run antt_rag_unified.py --server.port 8502
```
**Acesse:** http://localhost:8502

### Sistema Original (Simples)
```bash
source .env && streamlit run antt_rag_deepseek.py --server.port 8501
```
**Acesse:** http://localhost:8501

## 🎯 Principais Vantagens do Sistema Unificado

### 1. **Templates Especializados**
- **Parâmetros Técnicos**: Para consultas sobre especificações técnicas
- **Análise Normativa**: Para aspectos jurídicos e regulamentares
- **Extração Agressiva**: Para maximizar informações encontradas
- **Resposta com Citações**: Template padrão com referências

### 2. **Busca Inteligente**
- **Busca Semântica**: Usando embeddings para similaridade
- **Busca por Keywords**: Complementar para termos específicos
- **MMR (Maximal Marginal Relevance)**: Evita redundância
- **Reranking**: Ordena por relevância calculada

### 3. **Processamento Avançado**
- **Análise de Contexto**: Detecta automaticamente o tipo de consulta
- **Fallback Inteligente**: Se a primeira resposta for insatisfatória, tenta extração agressiva
- **Processamento de PDF**: Extrai e analisa tabelas automaticamente

### 4. **Interface Completa**
- **Status das APIs**: Verificação visual das conexões
- **Configurações Avançadas**: Temperatura, tokens, número de documentos
- **Filtros Detalhados**: Por tipo, ano e número de documento
- **Exemplos Interativos**: Consultas pré-definidas
- **Detalhes da Busca**: Informações sobre o processamento

## 🔧 Configurações Avançadas

### Parâmetros Ajustáveis
- **Temperatura**: 0.0 (preciso) a 1.0 (criativo)
- **Max Tokens**: 100 a 4000 tokens
- **Documentos**: 5 a 20 documentos para busca
- **Filtros**: Tipo, ano e número específico

### Provedores Suportados
- **OpenAI**: GPT-4, GPT-3.5-turbo
- **DeepSeek**: deepseek-r1:free (gratuito)

## 📚 Exemplos de Uso

### Consultas Técnicas
```
Quais são os parâmetros técnicos para pavimentos rodoviários?
```
→ Usa template especializado para parâmetros técnicos

### Consultas Normativas
```
Resolução 6057 de 2024 - principais pontos
```
→ Usa template de análise normativa

### Consultas Gerais
```
Como funciona o processo de fiscalização da ANTT?
```
→ Usa template padrão com citações

## 🎨 Interface Visual

### Elementos Visuais
- **Header Gradiente**: Visual moderno e profissional
- **Status Cards**: Indicadores visuais das APIs
- **Caixas de Citação**: Destaque para documentos referenciados
- **Métricas Cards**: Informações organizadas
- **Botões Interativos**: Ações claras e intuitivas

### Cores e Status
- 🟢 **Verde**: Funcionando corretamente
- 🔴 **Vermelho**: Erro ou problema
- 🟡 **Amarelo**: Aviso ou atenção
- 🔵 **Azul**: Informação ou neutro

## 🔍 Algoritmo de Busca

### Processo de Busca Híbrida
1. **Busca Semântica com MMR**: Primeira tentativa
2. **Busca por Keywords**: Complementar se necessário
3. **Busca Direta**: Fallback se poucos resultados
4. **Busca Ampla**: Última tentativa com termos simplificados
5. **Reranking**: Ordenação final por relevância

### Critérios de Relevância
- **Correspondência de Keywords**: Peso alto
- **Densidade de Termos**: Proporção no documento
- **Metadados Especiais**: Relevância técnica, tabelas
- **Posição no Documento**: Primeiro chunk tem bonus

## 🚀 Performance

### Otimizações
- **Processamento Paralelo**: Para análise de tabelas
- **Cache Inteligente**: Reutilização de embeddings
- **Retry com Backoff**: Para rate limits
- **Timeout Configurável**: Evita travamentos

### Métricas Típicas
- **Busca**: 1-3 segundos
- **Geração**: 3-10 segundos (dependendo do modelo)
- **Processamento PDF**: 30-120 segundos

## 🛠️ Manutenção

### Logs e Debug
- Logs detalhados em tempo real
- Informações sobre busca e processamento
- Rastreamento de erros e fallbacks

### Monitoramento
- Status das APIs em tempo real
- Métricas de uso e performance
- Alertas visuais para problemas

## 📈 Próximas Melhorias

### Funcionalidades Planejadas
- [ ] Cache de respostas frequentes
- [ ] Histórico de consultas
- [ ] Export de respostas (PDF/Word)
- [ ] Análise de sentimento
- [ ] Sugestões automáticas
- [ ] API REST para integração

### Otimizações Técnicas
- [ ] Embeddings locais (Sentence Transformers)
- [ ] Quantização de modelos
- [ ] Streaming de respostas
- [ ] Compressão de vectorstore

## 🎯 Recomendação

**Use o Sistema Unificado** (`antt_rag_unified.py`) para:
- ✅ Consultas complexas e detalhadas
- ✅ Análise técnica e normativa
- ✅ Processamento de novos documentos
- ✅ Interface completa e profissional
- ✅ Máxima qualidade de respostas

**Use o Sistema Original** (`antt_rag_deepseek.py`) para:
- ✅ Consultas rápidas e simples
- ✅ Testes básicos
- ✅ Menor uso de recursos
- ✅ Interface minimalista

---

## 🔗 Links Úteis

- **Sistema Unificado**: http://localhost:8502
- **Sistema Original**: http://localhost:8501
- **Documentação APIs**: `CONFIGURACAO_APIS.md`
- **Testes**: `test_llm_providers.py` 