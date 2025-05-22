# Sistema RAG-ANTT

Sistema de Retrieval Augmented Generation (RAG) para consulta a documentos normativos da ANTT (Agência Nacional de Transportes Terrestres).

## Descrição

Este sistema permite a consulta inteligente aos documentos regulatórios e normativos da ANTT, como Resoluções, Instruções Normativas, Deliberações, Portarias e outros documentos oficiais. O sistema utiliza tecnologia de Inteligência Artificial para processar consultas em linguagem natural e recuperar informações relevantes dos documentos.

## Características

- Busca semântica em documentos normativos
- Interface de usuário intuitiva com Streamlit
- Processamento inteligente de consultas
- Exibição estruturada de informações técnicas
- Citação precisa das fontes documentais
- Visualização dos trechos relevantes dos documentos

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

4. Configure sua chave de API da OpenAI no arquivo `.env`:
```
OPENAI_API_KEY=sua-chave-aqui
```

## Configuração

As configurações do sistema estão centralizadas no arquivo `config.py`. Este arquivo gerencia variáveis de ambiente, caminhos de armazenamento, parâmetros de processamento e outras configurações.

Para configurar o sistema:

1. Crie um arquivo `.env` na raiz do projeto com sua chave da API OpenAI:
```
OPENAI_API_KEY=sua-chave-aqui
```

2. Ajuste os parâmetros no arquivo `config.py` conforme necessário:
```python
# Tamanho dos chunks para processamento de documentos
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150

# Modelo LLM padrão
DEFAULT_LLM_MODEL = "gpt-4o"

# Diretório do vectorstore
DB_FAISS_PATH = "vectorstore/db_faiss"
```

## Uso

### Interface de usuário

Execute a interface Streamlit:
```bash
streamlit run chat-RAG.py
```

A interface permite:
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

- `chat-RAG.py`: Interface principal do sistema RAG
- `config.py`: Configurações centralizadas do sistema
- `antt_crawler.py`: Rastreador de documentos da ANTT
- `reconstruir_vectorstore.py`: Script para reconstruir o vectorstore
- `vectorstore/`: Diretório contendo o banco de vetores FAISS
- `dados_antt/`: Diretório contendo os documentos normativas da ANTT

## Segurança

- **IMPORTANTE**: Nunca compartilhe ou commite o arquivo `.env` contendo sua chave da API OpenAI.
- Caso precise compartilhar o código, certifique-se de que o arquivo `.env` está listado no `.gitignore`.
- Em ambientes de produção, use variáveis de ambiente para configurar a chave da API em vez de arquivos locais.

## Contribuições

Contribuições são bem-vindas! Por favor, abra um issue para discutir as mudanças ou envie um pull request. 