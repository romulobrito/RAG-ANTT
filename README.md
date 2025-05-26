# Sistema RAG-ANTT

Sistema de Retrieval Augmented Generation (RAG) para consulta a documentos normativos da ANTT (Agência Nacional de Transportes Terrestres).

## Descrição

Este sistema permite a consulta inteligente aos documentos regulatórios e normativos da ANTT, como Resoluções, Instruções Normativas, Deliberações, Portarias e outros documentos oficiais. O sistema utiliza tecnologia de Inteligência Artificial para processar consultas em linguagem natural e recuperar informações relevantes dos documentos.

## Características

- Busca semântica em documentos normativos
- **Suporte a múltiplos provedores de LLM**: OpenAI e DeepSeek
- **DeepSeek-R1 gratuito** via OpenRouter
- Interface de usuário intuitiva com Streamlit
- Processamento inteligente de consultas
- Exibição estruturada de informações técnicas
- Citação precisa das fontes documentais
- Visualização dos trechos relevantes dos documentos
- Filtros por tipo de documento, ano e número

## Provedores de LLM Suportados

### OpenAI
- **Modelos**: GPT-4o, GPT-4, GPT-3.5-turbo
- **Custo**: Pago por uso
- **Qualidade**: Excelente

### DeepSeek (via OpenRouter)
- **Modelos**: DeepSeek-R1 (gratuito), DeepSeek-Chat
- **Custo**: DeepSeek-R1 é completamente gratuito
- **Qualidade**: Competitivo com GPT-4

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/RAG-ANTT.git
cd RAG-ANTT
```

2. Crie e ative um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure as chaves de API no arquivo `.env`:
```bash
# Para usar DeepSeek (recomendado - gratuito)
OPENROUTER_API_KEY=sua-chave-openrouter-aqui
OPENAI_API_KEY=sua-chave-openai-aqui  # Necessária para embeddings

# Ou apenas OpenAI
OPENAI_API_KEY=sua-chave-openai-aqui
```

## Configuração de APIs

### Opção 1: DeepSeek (Recomendado - Gratuito)

1. Crie uma conta gratuita em [OpenRouter](https://openrouter.ai)
2. Obtenha sua chave de API
3. Configure no arquivo `.env`:
```bash
OPENROUTER_API_KEY=sua-chave-openrouter-aqui
OPENAI_API_KEY=sua-chave-openai-aqui  # Para embeddings
```

### Opção 2: OpenAI

1. Crie uma conta em [OpenAI Platform](https://platform.openai.com)
2. Obtenha sua chave de API
3. Configure no arquivo `.env`:
```bash
OPENAI_API_KEY=sua-chave-openai-aqui
```

**📖 Para instruções detalhadas, consulte [CONFIGURACAO_APIS.md](CONFIGURACAO_APIS.md)**

## Uso

### Interface Principal (com suporte a múltiplos LLMs)

Execute a nova interface com suporte ao DeepSeek:
```bash
streamlit run antt_rag_deepseek.py
```

### Interface Original (apenas OpenAI)

Execute a interface original:
```bash
streamlit run chat-RAG.py
```

A interface permite:
- **Escolher o provedor de LLM** (OpenAI ou DeepSeek)
- **Selecionar o modelo específico**
- Fazer perguntas em linguagem natural sobre documentos da ANTT
- Filtrar por tipo de documento, ano e número
- Visualizar os trechos relevantes dos documentos citados
- Explorar todas as fontes consultadas

### Reconstruir o vectorstore

Para processar novos documentos e reconstruir o vectorstore:
```bash
python reconstruir_vectorstore.py
```

### Rastreamento de documentos

Para rastrear e processar documentos da ANTT:
```bash
python antt_crawler.py --diretorio dados_antt
```

## Estrutura de Arquivos

- `antt_rag_deepseek.py`: Interface principal com suporte a múltiplos LLMs
- `chat-RAG.py`: Interface original (apenas OpenAI)
- `config.py`: Configurações centralizadas do sistema
- `llm_providers.py`: Gerenciador de provedores de LLM
- `antt_crawler.py`: Rastreador de documentos da ANTT
- `reconstruir_vectorstore.py`: Script para reconstruir o vectorstore
- `vectorstore/`: Diretório contendo o banco de vetores FAISS
- `dados_antt/`: Diretório contendo os documentos normativas da ANTT
- `CONFIGURACAO_APIS.md`: Guia detalhado de configuração de APIs

## Custos Estimados

### DeepSeek via OpenRouter (Recomendado)
- **DeepSeek-R1**: Gratuito
- **Embeddings OpenAI**: ~$0.0001 por 1K tokens
- **Custo total**: Praticamente gratuito

### OpenAI
- **GPT-4o**: ~$0.005 por 1K tokens
- **Embeddings**: ~$0.0001 por 1K tokens
- **Custo total**: Moderado

## Segurança

- **IMPORTANTE**: Nunca compartilhe ou commite o arquivo `.env` contendo suas chaves de API.
- Caso precise compartilhar o código, certifique-se de que o arquivo `.env` está listado no `.gitignore`.
- Em ambientes de produção, use variáveis de ambiente para configurar as chaves de API em vez de arquivos locais.

## Contribuições

Contribuições são bem-vindas! Por favor, abra um issue para discutir as mudanças ou envie um pull request. 