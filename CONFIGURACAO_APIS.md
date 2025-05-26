# Configuração de APIs para RAG ANTT

Este documento explica como configurar as APIs necessárias para usar o sistema RAG ANTT com diferentes provedores de LLM.

## Provedores Suportados

### 1. OpenAI
- **Modelos**: GPT-4o, GPT-4, GPT-3.5-turbo
- **Embeddings**: text-embedding-ada-002
- **Custo**: Pago por uso

### 2. DeepSeek (via OpenRouter)
- **Modelos**: DeepSeek-R1 (gratuito), DeepSeek-Chat
- **Embeddings**: Usa OpenAI (necessária chave OpenAI)
- **Custo**: DeepSeek-R1 é gratuito via OpenRouter

## Configuração das Chaves de API

### Método 1: Arquivo .env (Recomendado)

1. Crie um arquivo `.env` na raiz do projeto:
```bash
# OpenAI API (necessária para embeddings)
OPENAI_API_KEY=sua-chave-openai-aqui

# OpenRouter API (para DeepSeek)
OPENROUTER_API_KEY=sua-chave-openrouter-aqui
```

2. Substitua pelas suas chaves reais

### Método 2: Variáveis de Ambiente

```bash
export OPENAI_API_KEY="sua-chave-openai-aqui"
export OPENROUTER_API_KEY="sua-chave-openrouter-aqui"
```

## Como Obter as Chaves

### OpenAI API Key

1. Acesse [https://platform.openai.com](https://platform.openai.com)
2. Faça login ou crie uma conta
3. Vá em "API Keys" no menu
4. Clique em "Create new secret key"
5. Copie a chave (formato: sk-...)

### OpenRouter API Key (para DeepSeek)

1. Acesse [https://openrouter.ai](https://openrouter.ai)
2. Faça login ou crie uma conta gratuita
3. Vá em "Keys" no menu
4. Clique em "Create Key"
5. Copie a chave

**Importante**: O OpenRouter oferece DeepSeek-R1 gratuitamente!

## Configuração no Sistema

### Para usar DeepSeek (Recomendado - Gratuito)

1. Configure apenas a chave do OpenRouter:
```bash
OPENROUTER_API_KEY=sua-chave-openrouter-aqui
```

2. Configure também a chave OpenAI para embeddings:
```bash
OPENAI_API_KEY=sua-chave-openai-aqui
```

### Para usar apenas OpenAI

1. Configure apenas a chave OpenAI:
```bash
OPENAI_API_KEY=sua-chave-openai-aqui
```

## Testando a Configuração

Execute o sistema e verifique na barra lateral:
- ✅ Verde: Chave configurada corretamente
- ❌ Vermelho: Chave não encontrada

## Custos Estimados

### DeepSeek via OpenRouter
- **DeepSeek-R1**: Gratuito
- **Embeddings OpenAI**: ~$0.0001 por 1K tokens

### OpenAI
- **GPT-4o**: ~$0.005 por 1K tokens de entrada
- **Embeddings**: ~$0.0001 por 1K tokens

## Solução de Problemas

### Erro: "Chave de API não encontrada"
- Verifique se o arquivo `.env` está na raiz do projeto
- Verifique se as variáveis estão nomeadas corretamente
- Reinicie a aplicação após alterar as variáveis

### Erro: "Rate limit exceeded"
- Para OpenAI: Aguarde ou aumente seu limite
- Para OpenRouter: Verifique se não excedeu o limite gratuito

### Erro: "Invalid API key"
- Verifique se a chave foi copiada corretamente
- Verifique se a chave não expirou
- Para OpenAI: Certifique-se que começa com "sk-"

## Recomendação

Para começar rapidamente e sem custos:
1. Crie uma conta no OpenRouter (gratuita)
2. Obtenha uma chave OpenAI (necessária para embeddings)
3. Use DeepSeek-R1 via OpenRouter (gratuito)

Isso permite usar o sistema completo com custo mínimo apenas para embeddings. 