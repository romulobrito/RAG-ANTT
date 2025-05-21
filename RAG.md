# RAG (Retrieval-Augmented Generation) para Análise de Documentos ANTT

## Introdução
O sistema RAG desenvolvido é uma aplicação especializada para análise automatizada de documentos da ANTT (Agência Nacional de Transportes Terrestres), focando especificamente na identificação e análise de descumprimentos de parâmetros de desempenho em contratos de concessão rodoviária.

## Objetivos
- Automatizar a extração de informações de documentos PDF
- Identificar descumprimentos contratuais de forma precisa
- Fornecer análises contextualizadas dos parâmetros de desempenho
- Permitir consultas em linguagem natural sobre os descumprimentos
- Gerar respostas fundamentadas em evidências documentais

## Arquitetura do Sistema

### 1. Componentes Principais
- **Frontend**: Interface Streamlit para interação com usuário
- **Backend**: Sistema de processamento em Python
- **Banco de Dados Vetorial**: FAISS para armazenamento e recuperação eficiente
- **Modelo de Linguagem**: GPT-4 para análise e geração de respostas
- **OCR**: Tesseract para extração de texto de imagens

### 2. Pipeline de Processamento

A[Upload PDF] --> B[Extração de Tabelas]
B --> C[OCR & Processamento]
C --> D[Análise de Descumprimentos]
D --> E[Vetorização]
E --> F[Armazenamento FAISS]

### 3. Sistema de Consulta

A[Pergunta do Usuário] --> B[Recuperação Contextual]
B --> C[Processamento LLM]
C --> D[Geração de Resposta]
D --> E[Apresentação dos Resultados]



## Funcionalidades Principais

### 1. Processamento de Documentos
- **Extração de Tabelas**: Utiliza `img2table` para identificar e extrair tabelas
- **OCR Avançado**: Processamento paralelo com Tesseract
- **Análise de Contexto**: Extração do contexto ao redor das tabelas
- **Validação de Dados**: Verificação de integridade e qualidade

### 2. Análise de Descumprimentos
- Identificação automática de parâmetros não conformes
- Classificação de gravidade dos descumprimentos
- Análise de impacto na segurança/qualidade
- Geração de relatórios detalhados

### 3. Sistema de Consulta
- Interface intuitiva para perguntas em linguagem natural
- Sugestões de perguntas comuns
- Respostas contextualizadas com evidências
- Visualização de tabelas relevantes

### 4. Métricas e Monitoramento
- Acompanhamento em tempo real do processamento
- Indicadores de progresso
- Logs detalhados
- Tratamento de erros e exceções

## Tecnologias Utilizadas

### 1. Processamento de Dados
- **Python**: Linguagem principal
- **img2table**: Extração de tabelas
- **Tesseract**: OCR
- **NumPy/PIL**: Processamento de imagens

### 2. Machine Learning
- **LangChain**: Framework para aplicações LLM
- **OpenAI GPT-4**: Modelo de linguagem
- **FAISS**: Banco de dados vetorial
- **Embeddings**: OpenAI Ada

### 3. Interface e Visualização
- **Streamlit**: Interface web
- **Markdown**: Formatação de texto
- **Logging**: Sistema de logs

## Aplicações

### 1. Análise Contratual
- Verificação de conformidade
- Identificação de violações
- Monitoramento de desempenho

### 2. Auditoria
- Rastreamento de descumprimentos
- Geração de relatórios
- Evidências documentais

### 3. Suporte à Decisão
- Análise de impacto
- Recomendações baseadas em dados
- Histórico de conformidade

## Benefícios

1. **Eficiência Operacional**
   - Redução do tempo de análise
   - Automatização de processos
2. **Precisão**
   - Análises baseadas em dados concretos
   - Redução de erros humanos
3. **Acessibilidade**
   - Interface amigável para usuários não técnicos
   - Consultas em linguagem natural
4. **Escalabilidade**
   - Capacidade de processar grandes volumes de documentos
   - Adaptabilidade a diferentes tipos de contratos

## Conclusão
O sistema RAG para análise de documentos da ANTT representa uma solução inovadora e eficiente para a identificação de descumprimentos contratuais. Combinando tecnologias avançadas de processamento de linguagem natural e extração de dados, ele oferece uma ferramenta poderosa para auditores, analistas e gestores, promovendo a conformidade e a transparência nos contratos de concessão rodoviária.