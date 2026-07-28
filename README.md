# RAG-ANTT - Sistema de Consulta Inteligente a Documentos

> **Proprietário**: DEEPFEED SOLUTION
> 
> **Uso**: Sistema proprietário para consulta inteligente a documentos da ANTT

Este projeto implementa um sistema RAG (Retrieval-Augmented Generation) para consulta inteligente a documentos da ANTT (Agência Nacional de Transportes Terrestres). O sistema utiliza múltiplos provedores de IA, embeddings locais e uma interface web para fornecer respostas precisas sobre regulamentações de transporte terrestre.

## Índice
- [RAG-ANTT - Sistema de Consulta Inteligente a Documentos](#rag-antt---sistema-de-consulta-inteligente-a-documentos)
  - [Índice](#índice)
  - [Estrutura do Projeto](#estrutura-do-projeto)
  - [Arquitetura da Solução](#arquitetura-da-solução)
    - [Fluxo de Dados](#fluxo-de-dados)
    - [Componentes e Responsabilidades](#componentes-e-responsabilidades)
  - [Avaliacao de Qualidade](#avaliacao-de-qualidade)
  - [Pré-requisitos](#pré-requisitos)
    - [Sistema Operacional](#sistema-operacional)
    - [Software](#software)
    - [Recursos do Sistema](#recursos-do-sistema)
  - [Instalação e Execução](#instalação-e-execução)
    - [1. Clone do Repositório](#1-clone-do-repositório)
    - [2. Configuração Inicial](#2-configuração-inicial)
    - [3. Execução](#3-execução)
    - [4. Acesso à Interface Web](#4-acesso-à-interface-web)
  - [Configuração](#configuração)
    - [Variáveis de Ambiente](#variáveis-de-ambiente)
  - [Monitoramento](#monitoramento)
    - [Logs](#logs)
    - [Interface Web](#interface-web)
  - [Integração com Outros Sistemas](#integração-com-outros-sistemas)
    - [API REST](#api-rest)
      - [Endpoints Disponíveis](#endpoints-disponíveis)
      - [Exemplo de Uso com Python](#exemplo-de-uso-com-python)
  - [Troubleshooting](#troubleshooting)
    - [Problemas Comuns](#problemas-comuns)
    - [Solução de Erros](#solução-de-erros)
  - [Contribuição](#contribuição)
  - [Licença](#licença)
    - [Termos de Uso](#termos-de-uso)
  - [Suporte](#suporte)

## Estrutura do Projeto

```
RAG-ANTT/
├── antt_rag_unified.py          # App Streamlit + pipeline RAG
├── retrieval_hibrido.py         # FAISS + BM25 + RRF + prioridade estruturada
├── llm_providers.py             # LLM (DeepSeek/OpenAI) e embeddings locais
├── config.py                    # Constantes e provedores
├── avaliar_retrieval.py         # Harness: latencia, completude, precisao, RAGAS
├── test_avaliar_retrieval.py
├── test_retrieval_hibrido.py
├── requirements.txt
├── docs/
│   ├── arquitetura_rag_api.tex  # Arquitetura de referencia (API + K8s)
│   └── avaliacao_rag.md         # Guia do harness e metricas
├── dados_antt/
│   ├── ...                      # Normas indexadas (INM, RES, etc.)
│   └── tabelas_auxiliares/      # Transcricoes estruturadas (preferidas ao OCR)
├── vectorstore_local/           # Indice FAISS (embeddings locais)
├── relatorios_avaliacao/        # Saida do harness (md/json)
└── planning/                    # Notas de planejamento (nao operacional)
```

Embedding padrao: `intfloat/multilingual-e5-small` (`LOCAL_EMBEDDING_MODEL` em `config.py`).
Trocar o modelo exige reindexacao completa.

## Arquitetura da Solução

```mermaid
graph TB
    subgraph "Sistema RAG-ANTT"
        A[Interface Web :8501] --> B[antt_rag_unified]
        B --> C[Provedores de IA]
        C --> D[OpenAI]
        C --> E[DeepSeek via OpenRouter]
        B --> F[Embeddings locais e5-small]
        B --> G[retrieval_hibrido]
        G --> H[FAISS semantico]
        G --> I[BM25 lexical]
        G --> J[RRF + rerank + tabelas auxiliares]
        B --> K[gerar_resposta]
    end

    subgraph "Qualidade offline"
        L[avaliar_retrieval] --> G
        L --> K
        L --> M[RAGAS juiz]
        L --> N[relatorios_avaliacao]
    end

    subgraph "Ingestao"
        O[PDF / Markdown] --> P[OCR e tabelas]
        O --> Q[tabelas_auxiliares]
        P --> R[vectorstore_local]
        Q --> R
        Q --> G
    end
```

Documentacao de deploy futuro (API FastAPI + Ollama + Rancher): `docs/arquitetura_rag_api.tex`.

### Fluxo de Dados

```mermaid
sequenceDiagram
    participant User as Usuario
    participant UI as Streamlit
    participant RAG as antt_rag_unified
    participant Ret as retrieval_hibrido
    participant VS as FAISS + BM25
    participant LLM as DeepSeek / OpenAI

    User->>UI: Pergunta
    UI->>RAG: Processa consulta
    RAG->>Ret: pesquisar_documentos
    Ret->>VS: Semantico + lexical (RRF)
    Ret-->>RAG: Chunks (prioriza auxiliar estruturada)
    RAG->>LLM: gerar_resposta + contexto
    LLM-->>RAG: Resposta
    RAG-->>UI: Resposta + citacoes
    UI-->>User: Resultado
```

### Componentes e Responsabilidades

| Componente | Responsabilidade |
|---|---|
| `antt_rag_unified.py` | UI Streamlit, ingestao, OCR, prompts, orquestracao |
| `retrieval_hibrido.py` | Busca hibrida, RRF, expansao por documento-pai, boost de fonte estruturada |
| `llm_providers.py` | ChatOpenAI (OpenRouter/OpenAI) e `LocalEmbeddings` |
| `avaliar_retrieval.py` | Metricas offline (gabarito + latencia + RAGAS opcional) |
| `tipos_documento.py` | Catalogo de tipos gerado da base (aliases do cabecalho; refresh no reindex/upload) |
| `dados_antt/tabelas_auxiliares/` | Tabelas normativas em Markdown (preferidas ao OCR) |

## Avaliacao de Qualidade

O harness `avaliar_retrieval.py` mede qualidade sem depender da UI.

| Modo | O que mede |
|---|---|
| Padrao | Cobertura do gabarito no retrieval, hit do documento, estruturado vs OCR, latencia |
| `--com-geracao` | + completude factual da resposta e latencia de geracao |
| `--com-ragas` | + faithfulness, answer_relevancy, context_precision, context_recall (RAGAS 0.1.21) |

```bash
# Apenas retrieval (rapido, sem custo de LLM juiz)
python avaliar_retrieval.py --casos iri_principal

# Retrieval + geracao
python avaliar_retrieval.py --com-geracao --casos iri_principal

# Completo com RAGAS (gasta tokens do juiz)
python avaliar_retrieval.py --com-ragas --casos iri_principal,dadm_vdm
```

Detalhes, interpretacao das metricas e pin de dependencia: [`docs/avaliacao_rag.md`](docs/avaliacao_rag.md).

Testes unitarios (sem FAISS/LLM):

```bash
python -m pytest test_avaliar_retrieval.py test_retrieval_hibrido.py -q
```

## Pré-requisitos

### Sistema Operacional
- Linux (Ubuntu 20.04+)
- macOS (10.15+)
- Windows 10/11 (com WSL2)

### Software
- Python 3.9+
- Tesseract OCR
- Git 2.30+

### Recursos do Sistema
- CPU: 2 cores mínimo
- RAM: 4GB mínimo
- Disco: 10GB livre
- Rede: Conexão estável com internet

## Instalação e Execução

### 1. Clone do Repositório

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/RAG-ANTT.git

# Entre no diretório
cd RAG-ANTT
```

### 2. Configuração Inicial

```bash
# Crie e ative o ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
# ou
.\venv\Scripts\activate  # Windows

# Instale as dependências
pip install -r requirements.txt

# Instale o Tesseract OCR
sudo apt-get install tesseract-ocr  # Ubuntu/Debian
# ou
brew install tesseract  # macOS
```

### 3. Execução

```bash
# Execute o sistema
streamlit run antt_rag_unified.py
```

### 4. Acesso à Interface Web

A interface web estará disponível em: `http://localhost:8501`

## Configuração

### Variáveis de Ambiente

O sistema usa as seguintes variáveis de ambiente:

```bash
# Configurações da OpenAI
OPENAI_API_KEY=seu_api_key_aqui

# Configurações do OpenRouter (DeepSeek)
OPENROUTER_API_KEY=seu_api_key_aqui

# Configurações do Sistema
CHUNK_SIZE=500
CHUNK_OVERLAP=150
```

## Monitoramento

### Logs

O sistema utiliza logging para monitoramento:

```python
import logging

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

### Interface Web

A interface web fornece:
- Consulta a documentos
- Visualização de respostas
- Citações e referências
- Configurações do sistema

## Integração com Outros Sistemas

### API REST

O sistema pode ser integrado via API REST:

#### Endpoints Disponíveis

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/query` | POST | Realiza consulta |
| `/api/documents` | GET | Lista documentos |
| `/api/status` | GET | Status do sistema |

#### Exemplo de Uso com Python

```python
import requests

class RAGClient:
    def __init__(self, base_url="http://localhost:8501"):
        self.base_url = base_url

    def query(self, question):
        response = requests.post(
            f"{self.base_url}/api/query",
            json={"question": question}
        )
        return response.json()

    def get_documents(self):
        response = requests.get(f"{self.base_url}/api/documents")
        return response.json()
```

## Troubleshooting

### Problemas Comuns

1. **Erro de API Key**
   - Verifique se as variáveis de ambiente estão configuradas
   - Confirme se as chaves são válidas

2. **Erro de Tesseract**
   - Verifique se o Tesseract está instalado
   - Confirme o caminho de instalação

3. **Erro de Memória**
   - Aumente a memória disponível
   - Reduza o tamanho dos chunks

### Solução de Erros

1. **Logs de Erro**
```bash
   # Verifique os logs
   tail -f rag_antt.log
   ```

2. **Reinicialização**
```bash
   # Pare o processo
   pkill -f streamlit
   
   # Reinicie
   streamlit run antt_rag_unified.py
   ```

## Contribuição

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Faça commit das suas alterações (`git commit -m 'feat: adiciona nova feature'`)
4. Faça push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## Licença

Este projeto é proprietário da DEEPFEED SOLUTION e seu uso é restrito à consulta de documentos públicos dos órgãos de transporte brasileiros.

### Termos de Uso

- O uso deste software é restrito à DEEPFEED SOLUTION e seus clientes autorizados
- Não é permitida a distribuição, modificação ou uso comercial sem autorização expressa
- Todos os direitos reservados © DEEPFEED SOLUTION

## Suporte

Para suporte, entre em contato:
- Email: romulobrito@deepfeedsolutions.com
- Issues: GitHub Issues
- Documentação: `/docs` (`arquitetura_rag_api.tex`, `avaliacao_rag.md`) 