"""
Sistema RAG Unificado - ANTT
Combina funcionalidades avançadas com suporte a múltiplos provedores de LLM
"""

import logging
import time
from datetime import datetime
import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from PIL import Image
import numpy as np
from img2table.document import PDF
from img2table.ocr import TesseractOCR
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from openai import RateLimitError
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import json
import re
from pathlib import Path
from types import SimpleNamespace

# Importar configurações e gerenciador de LLM
from config import (
    get_openai_api_key,
    get_openrouter_api_key,
    DB_FAISS_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    STREAMLIT_PAGE_TITLE,
    STREAMLIT_PAGE_ICON,
    STREAMLIT_LAYOUT,
    setup_logging,
    logger
)

from llm_providers import LLMManager, get_available_providers, create_llm_manager, get_available_embedding_providers

# Inicializa o logger globalmente
logger = setup_logging()

# Templates de prompts otimizados (copiados do chat-RAG.py)
# Templates base (versão padrão)
TEMPLATE_RESPOSTA_COM_CITACOES_BASE = """
Você é um assistente especializado em regulamentação da ANTT (Agência Nacional de Transportes Terrestres).
Para a consulta: "{question}"

INSTRUÇÕES IMPORTANTES PARA GERAÇÃO DE RESPOSTA:
1. ANALISE PROFUNDAMENTE os documentos fornecidos para extrair todas as informações relevantes
2. Sua resposta deve ser COMPLETA, PRECISA e ESTRUTURADA - use listas, marcadores e formatação quando apropriado
3. Inclua TODOS os detalhes relevantes, como datas, números, prazos, valores e requisitos específicos
4. Organize informações complexas em seções claras com subtítulos quando necessário
5. Ao citar regulamentações:
   - Mencione o tipo de documento, número e ano EXATAMENTE como aparecem no original
   - Destaque artigos, parágrafos e incisos específicos
   - Explique claramente as implicações práticas das normas
6. Quando houver dados numericos ou tecnicos, apresente-os de forma estruturada
7. NAO INVENTE INFORMACOES! Se algo nao estiver nos documentos, indique claramente a limitacao
8. TABELAS: Os documentos podem conter tabelas em formato markdown (linhas com |).
   Extraia TODOS os valores numericos, limites e criterios presentes nessas tabelas
9. DEMANDAS DISTINTAS: Quando os trechos fornecidos tratarem de demandas, processos ou
   assuntos distintos (ex: notas tecnicas diferentes, processos SEI diferentes), identifique-os
   separadamente e NAO os funda em uma unica narrativa. Quando tratarem do mesmo assunto, consolide.
10. Após sua resposta, SEMPRE adicione uma seção "TRECHOS DOS DOCUMENTOS CITADOS" estruturada assim:

### TRECHOS DOS DOCUMENTOS CITADOS:

**Documento: [TIPO] [NÚMERO]/[ANO]**
"[Trecho exato usado]"

**Documento: [TIPO] [NÚMERO]/[ANO]**
"[Trecho exato usado]"

Contextos:
{context}

Sua resposta em português:
"""

# Templates otimizados para GPT-4 (mais estruturados e detalhados)
TEMPLATE_RESPOSTA_COM_CITACOES_GPT4 = """
Você é um assistente especializado em regulamentação da ANTT (Agência Nacional de Transportes Terrestres).

## CONTEXTO DA CONSULTA
Pergunta: "{question}"

## INSTRUÇÕES DETALHADAS DE ANÁLISE
### 1. ANÁLISE DOCUMENTAL
- Examine minuciosamente cada documento fornecido
- Identifique conexões entre diferentes fontes
- Priorize informações mais recentes e específicas

### 2. ESTRUTURAÇÃO DA RESPOSTA
- Use hierarquia clara: títulos, subtítulos, listas
- Organize cronologicamente quando relevante
- Separe aspectos técnicos, jurídicos e práticos

### 3. CITAÇÕES E REFERÊNCIAS
- Para cada informação, cite: [TIPO] [NÚMERO]/[ANO]
- Inclua artigos, parágrafos e incisos específicos
- Destaque mudanças normativas e atualizações

### 4. VALIDAÇÃO DE INFORMAÇÕES
- Base-se EXCLUSIVAMENTE nos documentos fornecidos
- Se o contexto contiver limites ou tabelas, apresente-os diretamente; não negue sua existência
- Lacuna pontual para campos ausentes; omita seções sem evidência textual
- Não faça inferências além do que está documentado
- TABELAS: Os documentos podem conter tabelas em formato markdown (linhas com |).
  Extraia TODOS os valores numéricos, limites e critérios presentes nessas tabelas.

### 4.1 DEMANDAS DISTINTAS
- Quando os trechos tratarem de demandas, processos ou assuntos distintos
  (ex: notas técnicas diferentes, processos SEI diferentes), identifique-os
  separadamente e NÃO os funda em uma única narrativa
- Quando tratarem do mesmo assunto, consolide normalmente

### 5. FORMATAÇÃO FINAL
Estruture sua resposta com:
- **Resumo Executivo** (2-3 linhas)
- **Desenvolvimento Detalhado** (seções organizadas)
- **Implicações Práticas** (quando aplicável)
- **Documentos Citados** (lista estruturada)

## DOCUMENTOS ANALISADOS
{context}

## SUA RESPOSTA ESTRUTURADA
"""

# Templates otimizados para DeepSeek (mais diretos e concisos)
TEMPLATE_RESPOSTA_COM_CITACOES_DEEPSEEK = """
Voce e um especialista em regulamentacao da ANTT. Responda de forma direta e precisa.

PERGUNTA: "{question}"

INSTRUCOES:
- Analise os documentos e extraia informacoes relevantes
- Seja objetivo e direto na resposta
- Se o contexto contiver limites ou tabelas, apresente-os no inicio; nao declare ausencia de dados
- Relacione equipamento/metodo da pergunta ao parametro normativo correspondente quando aplicavel
- Lacuna pontual apenas para campos ausentes; omita secoes sem evidencia no contexto
- Use listas e marcadores para organizar informacoes
- Cite sempre: [TIPO DOCUMENTO] [NUMERO]/[ANO]
- Inclua artigos e paragrafos especificos
- Se nao encontrar informacao, diga claramente
- IMPORTANTE: Os documentos podem conter TABELAS em formato markdown (linhas com |).
  Extraia TODOS os valores numericos, limites, faixas e criterios presentes nessas tabelas.
  Apresente esses dados de forma estruturada na resposta.
- DEMANDAS DISTINTAS: Quando os trechos tratarem de demandas, processos ou assuntos
  distintos (ex: notas tecnicas diferentes, processos SEI diferentes), identifique-os
  separadamente. Quando tratarem do mesmo assunto, consolide normalmente.

FORMATO DA RESPOSTA:
1. Resposta direta a pergunta
2. Valores numericos e limites (quando existirem nos documentos)
3. Detalhes tecnicos/normativos (se aplicavel)
4. Fontes citadas

DOCUMENTOS:
{context}

RESPOSTA:
"""

TEMPLATE_EXTRACAO_AGRESSIVA_BASE = """
Você é um especialista em análise documental com foco em regulamentações da ANTT.
Nos documentos abaixo, ANALISE MINUCIOSAMENTE todas as informações relevantes para: "{question}"

INSTRUÇÕES DE EXTRAÇÃO:
1. EXAMINE cada documento com atenção - informações importantes podem estar em qualquer parte
2. CONECTE informações fragmentadas de diferentes documentos para formar uma resposta coesa
3. EXTRAIA todos os detalhes relevantes: datas, números, requisitos, procedimentos, exceções
4. ORGANIZE as informações de forma lógica e estruturada, usando:
   - Listas para sequências ou itens relacionados
   - Seções com subtítulos para diferentes aspectos da resposta
5. Se houver conteudo parcialmente relevante, extraia o que for possivel e indique as lacunas restantes. Se NENHUMA informacao relevante existir nos documentos, informe claramente que nao ha dados disponiveis
6. Se os documentos apresentarem informações contraditórias ou ambíguas, EXPLIQUE as diferentes interpretações
6.1 Quando os trechos tratarem de demandas ou processos DISTINTOS, apresente-os separadamente. Quando tratarem do mesmo assunto, consolide
7. TABELAS: Os documentos podem conter tabelas em formato markdown (linhas com |).
   Extraia TODOS os valores numericos, limites e criterios presentes nessas tabelas.
8. Para cada informação, CITE A FONTE EXATA de onde a extraiu
8. Ao final, SEMPRE adicione uma seção "TRECHOS DOS DOCUMENTOS CITADOS" estruturada assim:

### TRECHOS DOS DOCUMENTOS CITADOS:

**Documento: [TIPO] [NÚMERO]/[ANO]**
"[Trecho exato usado]"

**Documento: [TIPO] [NÚMERO]/[ANO]**
"[Trecho exato usado]"

Documentos analisados:
{context}

Sua resposta completa em português:
"""

# Templates adaptativos para extração agressiva
TEMPLATE_EXTRACAO_AGRESSIVA_GPT4 = """
Você é um analista especializado em documentos regulatórios da ANTT.

## MISSÃO DE EXTRAÇÃO
Extrair TODAS as informações relevantes para: "{question}"

## METODOLOGIA DE ANÁLISE
### Fase 1: Mapeamento Documental
- Identifique todos os documentos e suas hierarquias
- Mapeie conexões entre diferentes fontes
- Priorize informações por relevância e atualidade

### Fase 2: Extração Sistemática
- Examine cada parágrafo, artigo e anexo
- Conecte informações fragmentadas
- Identifique padrões e exceções

### Fase 3: Síntese Inteligente
- Organize informações por temas
- Resolva contradições aparentes
- Construa narrativa coerente

## CRITERIOS DE QUALIDADE
- Precisao factual absoluta
- Completude informacional
- Clareza na apresentacao
- Rastreabilidade das fontes
- Se algum dado solicitado NAO constar nos documentos, indique a lacuna explicitamente
- TABELAS: Os documentos podem conter tabelas em formato markdown (linhas com |).
  Extraia TODOS os valores numericos, limites e criterios presentes nessas tabelas.
- Quando os trechos tratarem de demandas ou processos DISTINTOS, apresente-os separadamente

## BASE DOCUMENTAL
{context}

## EXTRAÇÃO COMPLETA
"""

TEMPLATE_EXTRACAO_AGRESSIVA_DEEPSEEK = """
MISSAO: Extrair TODAS as informacoes sobre "{question}" dos documentos.

METODO:
- Leia cada documento completamente
- Extraia TODOS os detalhes relevantes
- Conecte informacoes de diferentes fontes
- Organize por importancia
- Cite sempre a fonte exata: [DOCUMENTO] [NUMERO]/[ANO]

REGRAS:
- Nao ignore nenhuma informacao util
- Se ha contradicoes, explique ambas
- Use formato claro e direto
- Se o contexto NAO contiver a informacao solicitada, declare a lacuna. NAO invente dados.
- TABELAS: Os documentos podem conter tabelas em formato markdown (linhas com |).
  Extraia TODOS os valores numericos, limites e criterios presentes nessas tabelas.
- Quando os trechos tratarem de demandas ou processos DISTINTOS (ex: notas tecnicas
  diferentes, processos SEI diferentes), apresente-os SEPARADAMENTE. Quando tratarem
  do mesmo assunto, consolide.

DOCUMENTOS:
{context}

EXTRACAO COMPLETA:
"""

TEMPLATE_PARAMETROS_TECNICOS_BASE = """
Você é um especialista técnico em regulamentações da ANTT.
Para a consulta sobre parâmetros ou especificações técnicas: "{question}"

INSTRUÇÕES DE ANÁLISE TÉCNICA:
0. RESPOSTA DIRETA: Se o contexto contiver limites ou tabelas, apresente-os no inicio.
   Nao declare ausencia de dados se esses valores estiverem no contexto.
   Relacione equipamento/metodo da pergunta ao parametro normativo correspondente quando aplicavel.
1. IDENTIFIQUE todos os parâmetros técnicos, especificações, limites ou critérios mencionados
2. Para cada parâmetro técnico COM EVIDENCIA NO CONTEXTO, DETALHE:
   - Nome/tipo do parâmetro
   - Valores numéricos, faixas ou limites especificados
   - Unidades de medida
   - Metodologias de verificação ou equipamentos
   - Frequência de monitoramento ou avaliação
   - Critérios de conformidade ou aceitação
   - Exceções ou condições especiais de aplicação

3. ORGANIZE os parâmetros técnicos em formato estruturado:
   - Use tabelas conceituais para apresentar dados numéricos
   - Agrupe parâmetros relacionados em seções lógicas
   - Destaque visualmente valores críticos ou limites importantes

4. EXPLIQUE o contexto técnico e a finalidade de cada parâmetro
5. COMPARE diferentes requisitos quando existirem variações por categoria, situação ou período
6. CITE a fonte específica de cada parâmetro (documento, artigo, anexo)
7. TABELAS: Os documentos podem conter tabelas em formato markdown (linhas com |).
   Extraia TODOS os valores numericos, limites e criterios presentes nessas tabelas.
8. NAO INVENTE DADOS! Lacuna apenas para campos especificos ausentes; omita secoes sem evidencia.
9. Quando os trechos tratarem de demandas ou processos DISTINTOS, apresente-os separadamente.
   Quando tratarem do mesmo assunto, consolide normalmente.
10. SEMPRE adicione uma seção "TRECHOS DOS DOCUMENTOS CITADOS" estruturada assim:

### TRECHOS DOS DOCUMENTOS CITADOS:

**Documento: [TIPO] [NÚMERO]/[ANO]**
"[Trecho exato usado]"

**Documento: [TIPO] [NÚMERO]/[ANO]**
"[Trecho exato usado]"

Contextos técnicos analisados:
{context}

Sua resposta técnica detalhada em português:
"""

# Templates adaptativos para parâmetros técnicos
TEMPLATE_PARAMETROS_TECNICOS_GPT4 = """
Você é um engenheiro especialista em normas técnicas da ANTT.

## ANÁLISE TÉCNICA SOLICITADA
Parâmetros/especificações para: "{question}"

## PROTOCOLO DE ANÁLISE TÉCNICA

REGRAS TRANSVERSAIS (OBRIGATÓRIAS):
- Resposta direta primeiro: limites e tabelas no início quando existirem no contexto.
- Nunca negue ausência de dados se o contexto contiver valores ou tabelas relevantes.
- Equipamento/método na pergunta -> identifique o parâmetro normativo e seus limites no contexto.
- Lacuna pontual (campo ausente), nunca lacuna global se parte dos dados existir.
- Omita seções sem evidência textual; não invente detalhes operacionais.

### 0. RESPOSTA DIRETA À PERGUNTA
- Apresente objetivamente os limites/valores encontrados (tabela ou lista).
- Só então desenvolva detalhes complementares com evidência no contexto.

### 1. IDENTIFICAÇÃO DE PARÂMETROS
- Mapeie todos os parâmetros técnicos mencionados
- Classifique por categoria (segurança, qualidade, desempenho)
- Identifique hierarquia de importância

### 2. ESPECIFICAÇÃO DETALHADA
Para cada parâmetro COM EVIDÊNCIA NO CONTEXTO, documente:
- **Denominação técnica oficial**
- **Valores numéricos e tolerâncias**
- **Unidades de medida padronizadas**
- **Metodologia de verificação/ensaio**
- **Equipamentos de medição requeridos**
- **Frequência de monitoramento**
- **Critérios de aceitação/rejeição**
- **Condições especiais de aplicação**

### 3. APRESENTAÇÃO ESTRUTURADA
- Tabelas técnicas organizadas
- Agrupamento por sistemas/subsistemas
- Destaque para valores críticos
- Comparações entre diferentes categorias

### 3.1 ATENCAO A TABELAS
- Os documentos podem conter TABELAS em formato markdown (linhas com |)
- Extraia TODOS os valores numéricos dessas tabelas: limites, faixas, percentuais, unidades
- Inclua na resposta os valores exatos como aparecem nas tabelas

### 4. CONTEXTUALIZAÇÃO NORMATIVA
- Finalidade técnica de cada parâmetro
- Impacto na segurança operacional
- Relação com outras normas
- Evolução histórica dos requisitos
- Quando os trechos tratarem de demandas ou processos DISTINTOS, apresente-os separadamente

## DOCUMENTAÇÃO TÉCNICA
{context}

## ANÁLISE TÉCNICA COMPLETA
"""

TEMPLATE_PARAMETROS_TECNICOS_DEEPSEEK = """
ANALISE TECNICA: "{question}"

OBJETIVO: Responder com os limites/valores normativos presentes no contexto, em qualquer
disciplina (pavimento, sinalizacao, estruturas, seguranca, etc.).

REGRAS DE FUNDAMENTACAO (OBRIGATORIAS):
- Resposta direta primeiro: se o contexto tiver limites ou tabelas, apresente-os no inicio.
- NUNCA diga que nao ha valores/limites se o contexto contiver esses dados.
- Pergunta por equipamento ou metodo de ensaio: identifique o parametro normativo correspondente
  (ex: FWD -> Dadm; perfilometro -> IRI) e responda com os limites desse parametro.
- Lacuna PONTUAL apenas para campos especificos ausentes; entregue o que existir no contexto.
- NAO fabrique numeros, unidades, espacamentos, exclusoes ou frequencias sem evidencia textual.
- Omita secoes do template sem evidencia; nao preencha com suposicoes.

ATENCAO A TABELAS:
- Os documentos podem conter TABELAS em formato markdown (linhas delimitadas por |) ou blocos
  tabulares em texto (OCR). Extraia TODOS os valores numericos: limites, faixas, percentuais,
  unidades (m/km, mm, %, etc.). Preserve dimensoes (fase, pista, faixa, categoria, VDM).
- Inclua na resposta os valores exatos como aparecem no contexto.

ESTRUTURA DA RESPOSTA:
0. **RESPOSTA DIRETA** - limites/valores solicitados (tabela ou lista)
1. **PARAMETROS IDENTIFICADOS** (somente com evidencia no contexto)
   Para cada parametro:
   - Nome do parametro
   - Valor/limite numerico exato (ex: 2,7 m/km, 7mm, >0,2)
   - Unidade de medida
   - Condicoes de aplicacao (ex: pista principal vs marginal, faixa de VDM)
   - Periodicidade (se constar no contexto)
   - Fonte: [DOC] [NUM]/[ANO], Art./Anexo

2. **METODOLOGIAS DE VERIFICACAO** (somente se constar no contexto)
   - Equipamento ou metodo
   - Area monitorada
   - Frequencia

3. **CRITERIOS DE CLASSIFICACAO/CONFORMIDADE** (somente se constar no contexto)

4. **LACUNAS PONTUAIS** - apenas campos solicitados e ausentes no contexto

5. **DEMANDAS DISTINTAS**: processos ou assuntos distintos -> apresente separadamente

DOCUMENTOS TECNICOS:
{context}

ANALISE:
"""

TEMPLATE_ANALISE_NORMATIVA_BASE = """
Você é um especialista jurídico em regulamentações da ANTT.
Para a consulta sobre normativas, regulamentações ou aspectos legais: "{question}"

INSTRUÇÕES DE ANÁLISE JURÍDICA:
1. IDENTIFIQUE todas as normativas relevantes nos documentos (resoluções, instruções, deliberações, etc.)
2. Para cada normativa importante, FORNEÇA:
   - Tipo, número e data da normativa
   - Objetivo principal e escopo de aplicação
   - Estrutura básica (capítulos/seções principais)
   - Dispositivos específicos relacionados à consulta (artigos, parágrafos, incisos)
   - Requisitos, prazos, procedimentos ou obrigações estabelecidos
   - Penalidades ou consequências pelo descumprimento (quando mencionadas)
   - Relação com outras normativas citadas

3. ANALISE as implicações práticas das normativas para os diferentes atores
   
4. ESTRUTURE sua resposta de forma clara:
   - Use seções e subtítulos para diferentes aspectos ou normativas
   - Cite textualmente trechos importantes, destacando-os adequadamente
   - Explique termos técnico-jurídicos quando necessário
   
5. CONTEXTUALIZE a evolução normativa quando relevante (alterações, revogações, etc.)
6. TABELAS: Os documentos podem conter tabelas em formato markdown (linhas com |).
   Extraia TODOS os valores numericos, limites e criterios presentes nessas tabelas.
7. NAO INVENTE INFORMACOES! Se algo nao estiver nos documentos, indique claramente a lacuna.
8. Quando os trechos tratarem de demandas ou processos DISTINTOS, apresente-os separadamente.
   Quando tratarem do mesmo assunto, consolide normalmente.
9. SEMPRE adicione uma seção "TRECHOS DOS DOCUMENTOS CITADOS" estruturada assim:

### TRECHOS DOS DOCUMENTOS CITADOS:

**Documento: [TIPO] [NÚMERO]/[ANO]**
"[Trecho exato usado]"

**Documento: [TIPO] [NÚMERO]/[ANO]**
"[Trecho exato usado]"

Documentos normativos analisados:
{context}

Sua análise jurídica em português:
"""

# Templates adaptativos para análise normativa
TEMPLATE_ANALISE_NORMATIVA_GPT4 = """
Voce e um jurista especializado em direito regulatorio da ANTT.

Objetivo: produzir uma resposta abrangente e bem fundamentada sobre: "{question}"

Diretrizes (sem formato fixo):
- Cubra identificacao da norma, objetivo/escopo, dispositivos pertinentes, condicoes/excecoes, procedimentos e prazos, sancoes, relacoes com outras normas e implicacoes praticas.
- Seja completo, preciso e rastreavel.
- Cite sempre no formato [TIPO NUMERO/ANO, Art. X, paragrafo Y, inciso Z].
- Ao final, inclua uma secao com trechos textuais exatos usados.
- Se houver lacunas nos documentos, indique explicitamente.
- TABELAS: Os documentos podem conter tabelas em formato markdown (linhas com |).
  Extraia TODOS os valores numericos, limites e criterios presentes nessas tabelas.
- Quando os trechos tratarem de demandas ou processos DISTINTOS, apresente-os separadamente.

Base documental:
{context}

Resposta completa:
"""

TEMPLATE_ANALISE_NORMATIVA_DEEPSEEK = """
Analise juridica sobre: "{question}"

Requisitos (sem impor estrutura fixa):
- Traga todos os pontos relevantes (identificacao, escopo, dispositivos, prazos, procedimentos, sancoes, relacoes normativas, efeitos praticos).
- Responda de forma direta e completa, com citacoes precisas no formato [TIPO NUMERO/ANO, Art. X, paragrafo Y, inciso Z].
- Inclua ao final trechos textuais exatos usados.
- Se faltar informacao nos documentos, declare a lacuna.
- TABELAS: Os documentos podem conter tabelas em formato markdown (linhas com |).
  Extraia TODOS os valores numericos, limites e criterios presentes nessas tabelas.
- Quando os trechos tratarem de demandas ou processos DISTINTOS, apresente-os separadamente.

Documentos:
{context}

Analise:
"""

def selecionar_template_adaptativo(template_base, modelo_usado):
    """
    Seleciona o template mais adequado baseado no modelo de LLM usado.
    
    Args:
        template_base (str): Nome do template base ('resposta', 'extracao', 'parametros', 'normativa')
        modelo_usado (str): Nome do modelo (ex: 'gpt-4', 'deepseek-r1')
    
    Returns:
        str: Template otimizado para o modelo
    """
    # Detectar tipo de modelo
    is_gpt4 = any(gpt in modelo_usado.lower() for gpt in ['gpt-4', 'gpt4', 'openai'])
    is_deepseek = any(ds in modelo_usado.lower() for ds in ['deepseek', 'deep-seek'])
    
    # Mapear templates
    template_map = {
        'resposta': {
            'gpt4': TEMPLATE_RESPOSTA_COM_CITACOES_GPT4,
            'deepseek': TEMPLATE_RESPOSTA_COM_CITACOES_DEEPSEEK,
            'base': TEMPLATE_RESPOSTA_COM_CITACOES_BASE
        },
        'extracao': {
            'gpt4': TEMPLATE_EXTRACAO_AGRESSIVA_GPT4,
            'deepseek': TEMPLATE_EXTRACAO_AGRESSIVA_DEEPSEEK,
            'base': TEMPLATE_EXTRACAO_AGRESSIVA_BASE
        },
        'parametros': {
            'gpt4': TEMPLATE_PARAMETROS_TECNICOS_GPT4,
            'deepseek': TEMPLATE_PARAMETROS_TECNICOS_DEEPSEEK,
            'base': TEMPLATE_PARAMETROS_TECNICOS_BASE
        },
        'normativa': {
            'gpt4': TEMPLATE_ANALISE_NORMATIVA_GPT4,
            'deepseek': TEMPLATE_ANALISE_NORMATIVA_DEEPSEEK,
            'base': TEMPLATE_ANALISE_NORMATIVA_BASE
        }
    }
    
    # Selecionar template apropriado
    if template_base in template_map:
        if is_gpt4:
            return template_map[template_base]['gpt4']
        elif is_deepseek:
            return template_map[template_base]['deepseek']
        else:
            return template_map[template_base]['base']
    
    # Fallback para template base se não encontrar
    return template_map.get('resposta', {}).get('base', TEMPLATE_RESPOSTA_COM_CITACOES_BASE)

# Manter compatibilidade com código existente
TEMPLATE_RESPOSTA_COM_CITACOES = TEMPLATE_RESPOSTA_COM_CITACOES_BASE
TEMPLATE_EXTRACAO_AGRESSIVA = TEMPLATE_EXTRACAO_AGRESSIVA_BASE
TEMPLATE_PARAMETROS_TECNICOS = TEMPLATE_PARAMETROS_TECNICOS_BASE
TEMPLATE_ANALISE_NORMATIVA = TEMPLATE_ANALISE_NORMATIVA_BASE

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(RateLimitError)
)
def call_llm_with_retry(llm, prompt, context):
    """
    Chama o LLM com retry em caso de rate limit.
    """
    try:
        return llm.invoke(prompt.format(context=context))
    except RateLimitError as e:
        logger.warning(f"Rate limit atingido, tentando novamente: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Erro na chamada do LLM: {str(e)}")
        raise 

def create_processing_metrics():
    """Cria containers para métricas de processamento"""
    metrics = {
        'status': st.empty(),
        'progress': st.progress(0),
        'info': st.empty(),
        'error': st.empty(),
        'time': st.empty()
    }
    return metrics

def update_metrics(metrics, status, progress=None, info=None, error=None):
    """
    Atualiza as métricas na interface de forma thread-safe.
    """
    try:
        if status:
            metrics['status'].text(f"Status: {status}")
        if progress is not None:
            progress = max(0, min(100, progress))
            normalized_progress = progress / 100.0
            metrics['progress'].progress(normalized_progress)
        if info:
            metrics['info'].info(info)
        if error:
            metrics['error'].error(error)
        metrics['time'].text(f"Última atualização: {datetime.now().strftime('%H:%M:%S')}")
    except Exception as e:
        logger.error(f"Erro ao atualizar métricas: {str(e)}")

def get_cropped_table(table, page, pdf):
    """Extrai uma imagem recortada da tabela do PDF."""
    bbox = table.bbox
    table_array = np.array(pdf.images[page][bbox.y1:bbox.y2, bbox.x1:bbox.x2])
    return Image.fromarray(table_array.astype('uint8'))

def process_single_table(table_info):
    """Processa uma única tabela com foco em descumprimentos."""
    try:
        table = table_info['table']
        page_num = table_info['page_num']
        table_num = table_info['table_num']
        pdf_doc = table_info['pdf_doc']
        llm = table_info['llm']
        
        table_text = table.html_repr()
        table_image = get_cropped_table(table, page_num, pdf_doc)
        inferred_context = infer_table_context(table_text, llm)
        
        if "Nenhum descumprimento identificado" not in inferred_context:
            return {
                'page_number': page_num,
                'table_number': table_num,
                'table_content': table_text,
                'descumprimentos': inferred_context,
                'table_image': table_image,
                'full_text': f"""
                Página {page_num}, Tabela {table_num}:
                
                # Análise de Descumprimentos:
                # {inferred_context}
                
                Conteúdo da Tabela:
                {table_text}
                """
            }
        return None
    except Exception as e:
        logger.error(f"Erro ao processar tabela {table_num} da página {page_num}: {str(e)}")
        return None

def extract_tables_with_context(pdf_path, metrics=None):
    """Extrai tabelas e infere seu contexto usando LLM e OCR com processamento paralelo."""
    tables_with_context = []
    start_time = time.time()
    
    if metrics:
        update_metrics(metrics, "Iniciando processamento do PDF", progress=0, info="Configurando ambiente...")
    
    try:
        # Usar o LLMManager para criar o LLM
        llm_manager = create_llm_manager("openai", "gpt-4")  # Usar OpenAI para processamento de tabelas
        llm = llm_manager.get_llm()
        
        if metrics:
            update_metrics(metrics, "Configurando OCR", progress=10, info="Inicializando OCR...")
        
        num_cores = multiprocessing.cpu_count()
        ocr = TesseractOCR(n_threads=min(num_cores, 4), lang="por")
        
        if metrics:
            update_metrics(metrics, "Carregando PDF", progress=20)
        
        pdf_doc = PDF(src=pdf_path, detect_rotation=False, pdf_text_extraction=True)
        
        if metrics:
            update_metrics(metrics, "Extraindo tabelas", progress=30)
        
        extracted_tables = pdf_doc.extract_tables(
            ocr=ocr,
            implicit_rows=True,
            borderless_tables=True,
            min_confidence=50
        )
        
        tables_to_process = []
        total_tables = 0
        
        for page_num, tables in extracted_tables.items():
            if not tables:
                continue
            total_tables += len(tables)
            for idx, table in enumerate(tables):
                if table and hasattr(table, 'html_repr'):
                    tables_to_process.append({
                        'table': table,
                        'page_num': page_num,
                        'table_num': idx + 1,
                        'pdf_doc': pdf_doc,
                        'llm': llm
                    })
        
        if metrics:
            update_metrics(metrics, f"Encontradas {total_tables} tabelas", progress=40, 
                          info=f"Iniciando processamento paralelo...")
        
        processed_tables = 0
        max_workers = min(num_cores, 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_table = {
                executor.submit(process_single_table, table_info): table_info
                for table_info in tables_to_process
            }
            
            for future in as_completed(future_to_table):
                try:
                    result = future.result()
                    if result:
                        tables_with_context.append(result)
                        processed_tables += 1
                        
                        if metrics:
                            progress = 40 + (processed_tables/total_tables * 50)
                            update_metrics(metrics, "Processando tabelas", progress=progress,
                                        info=f"Processadas {processed_tables}/{total_tables} tabelas")
                except Exception as e:
                    if metrics:
                        update_metrics(metrics, "Erro no processamento", 
                                     error=f"Erro ao processar tabela: {str(e)}")
        
        tables_with_context.sort(key=lambda x: (x['page_number'], x['table_number']))
        
        processing_time = time.time() - start_time
        if metrics:
            update_metrics(metrics, "Processamento concluído", progress=100,
                          info=f"Processamento concluído em {processing_time:.2f} segundos")
        
        return tables_with_context
        
    except Exception as e:
        if metrics:
            update_metrics(metrics, "Erro no processamento", progress=0,
                          error=f"Erro durante o processamento: {str(e)}")
        raise

def infer_table_context(table_text, llm):
    """Analisa especificamente tabelas de parâmetros de desempenho da ANTT."""
    prompt = ChatPromptTemplate.from_template("""
    Você é um especialista em análise de parâmetros de desempenho da ANTT.
    
    Analise a tabela de parâmetros de desempenho abaixo e identifique APENAS os casos onde há descumprimento dos parâmetros estabelecidos:

    {table_content}

    Regras de análise:
    1. Identifique parâmetros com valores fora dos limites estabelecidos
    2. Compare os valores encontrados com os valores de referência
    3. Considere "X" como parâmetro obrigatório
    4. Analise valores numéricos em relação aos limites (maior que, menor que, entre)
    5. Foque em indicadores como IRI, IGG, ATT, deflexão, etc.

    Para cada descumprimento encontrado, forneça:
    1. Nome do parâmetro descumprido
    2. Valor de referência
    3. Gravidade do descumprimento (Alta/Média/Baixa)
    4. Impacto na segurança/qualidade da rodovia

    Se não houver descumprimentos claros, responda apenas: "Nenhum descumprimento identificado nesta tabela."

    Resposta (apenas descumprimentos):
    """)
    
    try:
        chain = prompt | llm | StrOutputParser()
        response = call_llm_with_retry(chain, prompt, {"table_content": table_text})
        
        if not response or response.strip() == "":
            return "Nenhum descumprimento identificado nesta tabela."
        return response
        
    except Exception as e:
        logger.error(f"Erro ao analisar parâmetros de desempenho: {str(e)}")
        return "Erro na análise dos parâmetros de desempenho. Por favor, tente novamente."

def process_pdf(uploaded_file):
    """Processa o PDF e cria/atualiza a base de conhecimento"""
    metrics = create_processing_metrics()
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            update_metrics(metrics, status="Salvando arquivo temporário", progress=5)
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

            try:
                tables_data = extract_tables_with_context(tmp_file_path, metrics)
                
                if not tables_data:
                    update_metrics(metrics, status="Nenhuma tabela encontrada", progress=100,
                                 info="O documento não contém tabelas para processar")
                    return None, []

                update_metrics(metrics, status="Preparando documentos para indexação", progress=90)
                
                documents = []
                for table_data in tables_data:
                    documents.append(table_data['full_text'])

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP
                )
                splits = text_splitter.create_documents(documents)

                update_metrics(metrics, status="Criando embeddings", progress=95)
                
                try:
                    embeddings = OpenAIEmbeddings(
                        api_key=get_openai_api_key(),
                        max_retries=3,
                        timeout=30
                    )
                    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
                except RateLimitError:
                    update_metrics(metrics, status="Limite de requisições atingido", progress=95,
                                 error="Aguarde alguns segundos e tente novamente")
                    raise
                
                update_metrics(metrics, status="Salvando base de conhecimento", progress=98)
                
                os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
                vectorstore.save_local(DB_FAISS_PATH)
                
                update_metrics(metrics, status="Processamento finalizado", progress=100,
                             info=f"Processadas {len(tables_data)} tabelas com sucesso")
                
                os.remove(tmp_file_path)
                return vectorstore, tables_data
                
            except Exception as e:
                update_metrics(metrics, status="Erro no processamento", progress=0,
                             error=f"Erro durante o processamento: {str(e)}")
                raise
    except Exception as e:
        update_metrics(metrics, status="Erro fatal", progress=0,
                      error=f"Erro ao processar arquivo: {str(e)}")
        raise 

def format_docs(docs):
    """Formata os documentos para o contexto."""
    return "\n\n".join(doc.page_content for doc in docs)

def verificar_documentos_importantes():
    """Verifica se documentos importantes estão no vectorstore e os adiciona se necessário."""
    documentos_importantes = [
        "dados_antt/INM/2024/INM-00000034-2024.md",
        "dados_antt/RES/2024/RES-00006057-2024.md",
        "dados_antt/DLB/2024/DLB-00000092-2024.md"
    ]
    
    documentos_indexados = []
    
    for caminho in documentos_importantes:
        if os.path.exists(caminho):
            with open(caminho, 'r', encoding='utf-8') as f:
                conteudo = f.read()
            
            partes_caminho = caminho.split('/')
            tipo_documento = partes_caminho[1] if len(partes_caminho) > 1 else None
            ano = partes_caminho[2] if len(partes_caminho) > 2 else None
            numero = None
            
            nome_arquivo = os.path.basename(caminho)
            if '-' in nome_arquivo:
                partes = nome_arquivo.split('-')
                if len(partes) >= 2:
                    numero = partes[1]
            
            tipo_nome = {
                "RES": "Resolução",
                "POR": "Portaria",
                "INM": "Instrução Normativa",
                "DLB": "Deliberação",
                "INC": "Instrução Normativa Complementar"
            }.get(tipo_documento, tipo_documento)
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=200,
                length_function=len,
            )
            
            chunks = text_splitter.split_text(conteudo)
            for i, chunk in enumerate(chunks):
                documento = Document(
                    page_content=chunk,
                    metadata={
                        "tipo_documento": tipo_documento,
                        "nome_tipo": tipo_nome,
                        "ano": ano,
                        "numero": numero,
                        "caminho": caminho,
                        "chunk": i + 1,
                        "total_chunks": len(chunks)
                    }
                )
                documentos_indexados.append(documento)
    
    return documentos_indexados

def atualizar_vectorstore_com_documentos_importantes(vectorstore, embeddings):
    """Atualiza o vectorstore com documentos importantes se necessário."""
    documentos = verificar_documentos_importantes()
    if documentos:
        logger.info(f"Atualizando vectorstore com {len(documentos)} documentos importantes...")
        vectorstore.add_documents(documentos)
        vectorstore.save_local(DB_FAISS_PATH)
        logger.info("Vectorstore atualizado e salvo.")
    return vectorstore

def carregar_vectorstore():
    """Carrega o vectorstore ANTT."""
    logger.info(f"Carregando vectorstore de {DB_FAISS_PATH}...")
    
    # Tentar usar DeepSeek primeiro (gratuito) para embeddings
    try:
        logger.info("Tentando usar OpenAI para embeddings...")
        llm_manager = create_llm_manager("openai")  # Tentar OpenAI primeiro
        embeddings = llm_manager.get_embeddings()
    except Exception as e:
        logger.warning(f"OpenAI falhou (provavelmente cota excedida): {e}")
        logger.info("Usando fallback: carregando vectorstore sem recriar embeddings...")
        
        # Fallback: usar OpenAI embeddings básicos sem verificação de cota
        try:
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(
                openai_api_key=get_openai_api_key(),
                model="text-embedding-ada-002"
            )
        except Exception as e2:
            logger.error(f"Erro crítico ao criar embeddings: {e2}")
            raise Exception("Não foi possível carregar embeddings. Verifique as chaves de API.")
    
    try:
        vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        logger.info("Vectorstore carregado com sucesso")
        
        # Não tentar atualizar com documentos importantes se OpenAI falhou
        # vectorstore = atualizar_vectorstore_com_documentos_importantes(vectorstore, embeddings)
        
        return vectorstore
    except Exception as e:
        logger.error(f"Erro ao carregar vectorstore: {e}")
        raise


def _criar_embeddings_local():
    """Cria embeddings locais via sentence-transformers.

    Returns:
        LocalEmbeddings ou None em caso de falha.
    """
    try:
        llm_manager = create_llm_manager("deepseek", embedding_provider="local")
        emb = llm_manager.get_embeddings()
        try:
            _ = emb.embed_query("teste-local")
        except Exception:
            pass
        logger.info("Embeddings locais configurados com sucesso")
        return emb
    except Exception as exc:
        logger.warning(f"Embeddings locais falharam: {exc}")
        return None


def _criar_embeddings_openai():
    """Cria embeddings OpenAI.

    Returns:
        OpenAIEmbeddings ou None em caso de falha.
    """
    try:
        from langchain_openai import OpenAIEmbeddings
        emb = OpenAIEmbeddings(
            openai_api_key=get_openai_api_key(),
            model="text-embedding-ada-002",
            max_retries=1,
            timeout=10,
        )
        _ = emb.embed_query("teste")
        logger.info("OpenAI embeddings configurados com sucesso")
        return emb
    except Exception as exc:
        logger.warning(f"OpenAI embeddings falharam: {exc}")
        return None


def carregar_vectorstore_com_provider(embedding_provider="local"):
    """Carrega o vectorstore ANTT com suporte a diferentes provedores.

    Ordem de tentativa de embeddings (sem recursao):
      - local/free: tenta local primeiro, depois OpenAI como fallback
      - openai: tenta OpenAI primeiro, depois local como fallback

    Ordem de tentativa de vectorstore:
      - vectorstore_local (quando embeddings locais) ou vectorstore/db_faiss (OpenAI)
      - Se nao existir e embeddings locais disponiveis, tenta criar automaticamente
    """
    logger.info(f"Carregando vectorstore com provedor: {embedding_provider}")

    # --- 1. Resolver embeddings (sem recursao) -------------------------
    embeddings = None
    vectorstore_path = None

    if embedding_provider in ("local", "free"):
        embeddings = _criar_embeddings_local()
        if embeddings is not None:
            vectorstore_path = "vectorstore_local"
        else:
            logger.info("Tentando fallback para OpenAI...")
            embeddings = _criar_embeddings_openai()
            if embeddings is not None:
                vectorstore_path = DB_FAISS_PATH
    elif embedding_provider == "openai":
        embeddings = _criar_embeddings_openai()
        if embeddings is not None:
            vectorstore_path = DB_FAISS_PATH
        else:
            logger.info("Tentando fallback para embeddings locais...")
            embeddings = _criar_embeddings_local()
            if embeddings is not None:
                vectorstore_path = "vectorstore_local"
    else:
        logger.warning(f"Provedor '{embedding_provider}' nao reconhecido, usando local")
        embeddings = _criar_embeddings_local()
        if embeddings is not None:
            vectorstore_path = "vectorstore_local"

    if embeddings is None:
        raise Exception(
            "Nao foi possivel configurar embeddings. "
            "Instale sentence-transformers ou configure chave OpenAI."
        )

    # --- 2. Carregar ou criar vectorstore ------------------------------
    if os.path.exists(vectorstore_path):
        logger.info(f"Carregando vectorstore de {vectorstore_path}...")
        try:
            vectorstore = FAISS.load_local(
                vectorstore_path, embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("Vectorstore carregado com sucesso")
            vectorstore._embedding_provider = embedding_provider
            vectorstore._vectorstore_path = vectorstore_path
            return vectorstore
        except Exception as exc:
            logger.error(f"Erro ao carregar vectorstore: {exc}")
            raise Exception(f"Erro ao carregar base de conhecimento: {exc}")

    # Vectorstore nao existe -- tentar criar automaticamente (so local)
    if "local" not in vectorstore_path:
        raise Exception(
            f"Vectorstore nao encontrado em {vectorstore_path}. "
            "Execute a reindexacao pela barra lateral."
        )

    logger.warning(f"Vectorstore nao encontrado em {vectorstore_path}")

    if not os.path.exists("relatorio_documentos.json"):
        raise Exception(
            "Arquivo relatorio_documentos.json nao encontrado. "
            "Clique em Reindexar Base na barra lateral."
        )

    if not _adquirir_lock_reindexacao():
        raise Exception(
            "Criacao do vectorstore em andamento em outra instancia. "
            "Recarregue a pagina em alguns minutos."
        )

    try:
        logger.info("Criando vectorstore local automaticamente...")
        sucesso = criar_vectorstore_local(embeddings)
        if not sucesso:
            raise Exception("Falha na criacao automatica do vectorstore local")
        logger.info("Vectorstore local criado com sucesso!")
    except Exception as exc:
        logger.error(f"Erro ao criar vectorstore local: {exc}")
        raise
    finally:
        _liberar_lock_reindexacao()

    vectorstore = FAISS.load_local(
        vectorstore_path, embeddings,
        allow_dangerous_deserialization=True,
    )
    vectorstore._embedding_provider = embedding_provider
    vectorstore._vectorstore_path = vectorstore_path
    return vectorstore

def _listar_md_em_dados_antt(diretorio: str = "dados_antt") -> dict:
    """
    Varre dados_antt/ e retorna um dict {nome_arquivo: caminho_completo}
    contendo todos os .md encontrados (ja deduplicados por nome).

    Args:
        diretorio: Raiz da pasta de documentos.

    Returns:
        dict mapeando basename -> caminho absoluto do primeiro .md encontrado.
    """
    resultado: dict = {}
    if not os.path.isdir(diretorio):
        return resultado
    for dirpath, _dirnames, filenames in os.walk(diretorio):
        for fname in filenames:
            if fname.endswith(".md") and fname not in resultado:
                resultado[fname] = os.path.join(dirpath, fname)
    return resultado


def _listar_pdfs_sem_md(diretorio: str = "dados_antt") -> list:
    """
    Retorna lista de nomes de PDFs que nao possuem .md correspondente.

    Args:
        diretorio: Raiz da pasta de documentos.

    Returns:
        list[str]: Nomes dos arquivos PDF sem markdown.
    """
    pdfs_pendentes: list = []
    for dirpath, _dirs, filenames in os.walk(diretorio):
        for fname in filenames:
            if not fname.lower().endswith(".pdf"):
                continue
            nome_base = os.path.splitext(fname)[0]
            md_correspondente = os.path.join(dirpath, f"{nome_base}.md")
            if not os.path.exists(md_correspondente):
                pdfs_pendentes.append(fname)
    return sorted(pdfs_pendentes)


def detectar_documentos_novos(diretorio: str = "dados_antt",
                              relatorio_path: str = "relatorio_documentos.json") -> list:
    """
    Compara os arquivos .md e .pdf existentes em dados_antt/ com o catalogo
    registrado em relatorio_documentos.json.

    Detecta:
    - Arquivos .md presentes no disco mas ausentes do catalogo
    - Arquivos .pdf que nao possuem .md correspondente (nao convertidos)

    Args:
        diretorio: Pasta raiz dos documentos.
        relatorio_path: Caminho do JSON de catalogo.

    Returns:
        list[str]: Nomes de arquivos pendentes (novos .md + PDFs nao convertidos).
    """
    md_no_disco = _listar_md_em_dados_antt(diretorio)

    catalogados: set = set()
    if os.path.exists(relatorio_path):
        try:
            with open(relatorio_path, "r", encoding="utf-8") as f:
                import json as _json
                for doc in _json.load(f):
                    arq = doc.get("arquivo_md", "")
                    if arq:
                        catalogados.add(os.path.basename(arq))
        except Exception as exc:
            logger.warning(f"Erro ao ler catalogo {relatorio_path}: {exc}")

    novos_md = sorted(set(md_no_disco.keys()) - catalogados)
    pdfs_pendentes = _listar_pdfs_sem_md(diretorio)

    return novos_md + pdfs_pendentes


_LOCK_REINDEXACAO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".reindexando.lock"
)


def _adquirir_lock_reindexacao() -> bool:
    """Tenta criar arquivo de lock para evitar reindexacoes simultaneas.

    Returns:
        True se o lock foi adquirido, False se ja existe outra execucao.
    """
    import time as _time

    if os.path.exists(_LOCK_REINDEXACAO):
        try:
            mtime = os.path.getmtime(_LOCK_REINDEXACAO)
            idade_segundos = _time.time() - mtime
            if idade_segundos > 3600:
                logger.warning(
                    "Lock de reindexacao encontrado mas com mais de 1h "
                    "(possivelmente orfao). Removendo."
                )
                os.remove(_LOCK_REINDEXACAO)
            else:
                return False
        except OSError:
            return False

    try:
        with open(_LOCK_REINDEXACAO, "w") as f:
            f.write(str(os.getpid()))
        return True
    except OSError:
        return False


def _liberar_lock_reindexacao():
    """Remove o arquivo de lock."""
    try:
        os.remove(_LOCK_REINDEXACAO)
    except OSError:
        pass


def reindexar_base_completa(embedding_provider: str = "local") -> tuple:
    """
    Pipeline completo de reindexacao:
    1. Converte PDFs sem .md correspondente em markdown
    2. Regenera relatorio_documentos.json varrendo dados_antt/
    3. Remove vectorstore antigo
    4. Recria vectorstore com embeddings + OCR + deduplicacao

    Usa lock de arquivo para impedir execucoes simultaneas (comum
    quando o Streamlit dispara multiplos reruns).

    Args:
        embedding_provider: Provedor de embeddings a utilizar.

    Returns:
        tuple (sucesso: bool, mensagem: str)
    """
    import shutil

    if not _adquirir_lock_reindexacao():
        logger.warning("Reindexacao ja em andamento (lock ativo). Ignorando.")
        return False, "Reindexação já em andamento. Aguarde a conclusão."

    try:
        return _reindexar_base_impl(embedding_provider)
    finally:
        _liberar_lock_reindexacao()


def _reindexar_base_impl(embedding_provider: str) -> tuple:
    """Implementacao interna do pipeline de reindexacao (protegida por lock)."""
    import shutil

    n_pdfs = 0

    # 1) Converter PDFs que ainda nao tem .md
    try:
        n_pdfs = converter_pdfs_para_md()
        if n_pdfs > 0:
            logger.info(f"{n_pdfs} PDF(s) convertido(s) para markdown")
    except Exception as exc:
        logger.warning(f"Conversao de PDFs falhou (continuando): {exc}")

    # 2) Regenerar catalogo (agora inclui os .md recem-criados)
    try:
        from gerar_relatorio import gerar_relatorio_documentos
        docs = gerar_relatorio_documentos()
        logger.info(f"Catalogo regenerado: {len(docs)} documentos")
    except Exception as exc:
        msg = f"Erro ao regenerar catalogo: {exc}"
        logger.error(msg)
        return False, msg

    # 3) Remover vectorstore antigo
    vpath = "vectorstore_local"
    if os.path.isdir(vpath):
        shutil.rmtree(vpath)
        logger.info(f"Vectorstore antigo removido: {vpath}")

    # 4) Criar embeddings e reconstruir vectorstore
    try:
        embeddings = _criar_embeddings_local()
        if embeddings is None:
            return (
                False,
                "Erro ao reconstruir vectorstore: falha ao carregar embeddings locais. "
                "Reinicie o Streamlit e tente novamente. Se persistir, execute "
                "'pip install -U sentence-transformers accelerate' e use "
                "'Limpar Cache OCR e Reindexar'."
            )
        sucesso = criar_vectorstore_local(embeddings)
        if sucesso:
            msg_final = f"Reindexação concluída: {len(docs)} documentos catalogados"
            if n_pdfs > 0:
                msg_final += f" ({n_pdfs} PDF(s) convertido(s))"
            return True, msg_final
        return False, "Falha ao criar vectorstore"
    except Exception as exc:
        msg = f"Erro ao reconstruir vectorstore: {exc}"
        logger.error(msg)
        return False, msg


def criar_vectorstore_local(embeddings):
    """Cria vectorstore local usando embeddings locais"""
    try:
        import json
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document
        
        logger.info("🚀 Iniciando criação do vectorstore local...")
        
        # Carregar dados do relatório
        with open("relatorio_documentos.json", 'r', encoding='utf-8') as f:
            dados_documentos = json.load(f)
        
        logger.info(f"📄 Carregados {len(dados_documentos)} documentos do relatório")
        
        # Criar documentos para o vectorstore (com deduplicacao por nome de arquivo)
        documentos = []
        nomes_processados: set = set()
        duplicados_ignorados = 0

        for doc_info in dados_documentos:
            arquivo_md = doc_info.get("arquivo_md", "")
            if not arquivo_md or not os.path.exists(arquivo_md):
                continue

            # Deduplicar: usar nome do arquivo (sem caminho) como chave
            nome_arquivo = os.path.basename(arquivo_md)
            if nome_arquivo in nomes_processados:
                duplicados_ignorados += 1
                logger.debug(f"Duplicado ignorado: {arquivo_md}")
                continue
            nomes_processados.add(nome_arquivo)

            try:
                with open(arquivo_md, "r", encoding="utf-8") as f:
                    conteudo = f.read()

                conteudo = _enriquecer_imagens_documento(conteudo)

                doc = Document(
                    page_content=conteudo,
                    metadata={
                        "tipo_documento": doc_info.get("tipo", ""),
                        "nome_tipo": doc_info.get("tipo", ""),
                        "numero": doc_info.get("numero", ""),
                        "ano": doc_info.get("ano", ""),
                        "caminho": arquivo_md,
                        "titulo": doc_info.get("titulo", ""),
                        "ementa": doc_info.get("ementa", ""),
                        "orgao": doc_info.get("orgao", ""),
                    },
                )
                documentos.append(doc)

            except Exception as e:
                logger.warning(f"Erro ao ler {arquivo_md}: {e}")

        if duplicados_ignorados > 0:
            logger.info(
                f"Deduplicacao: {duplicados_ignorados} documentos duplicados ignorados"
            )
        
        logger.info(f"📚 Preparados {len(documentos)} documentos para indexação")
        
        # Dividir documentos em chunks por fronteiras estruturais (artigos/secoes)
        logger.info("Dividindo documentos em chunks por estrutura (artigos/secoes)...")
        splits = []

        for doc in documentos:
            texto = doc.page_content
            meta_base = doc.metadata.copy()

            chunks_texto = _dividir_por_estrutura(
                texto,
                chunk_max=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )

            total = len(chunks_texto)
            for idx, chunk_txt in enumerate(chunks_texto):
                meta = meta_base.copy()
                meta["chunk"] = idx + 1
                meta["total_chunks"] = total
                splits.append(Document(page_content=chunk_txt, metadata=meta))

        logger.info(f"Criados {len(splits)} chunks estruturais")
        
        # Criar vectorstore
        logger.info("🔍 Criando vectorstore com embeddings locais...")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        
        # Salvar vectorstore
        vectorstore_path = "vectorstore_local"
        os.makedirs(vectorstore_path, exist_ok=True)
        vectorstore.save_local(vectorstore_path)
        
        logger.info(f"✅ Vectorstore local salvo em {vectorstore_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao criar vectorstore local: {e}")
        return False

def criar_filtro_metadados(tipo_documento=None, ano=None, numero=None):
    """Cria um filtro para busca por metadados."""
    filtro = {}
    
    if tipo_documento:
        filtro["tipo_documento"] = tipo_documento
    
    if ano:
        filtro["ano"] = ano
    
    if numero:
        filtro["numero"] = numero
    
    return filtro if filtro else None

def extrair_keywords(query):
    """Extrai palavras-chave relevantes da consulta."""
    stop_words = set([
        "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as", "até",
        "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles", "depois",
        "do", "dos", "e", "ela", "elas", "ele", "eles", "em", "entre", "era",
        "eram", "essa", "essas", "esse", "esses", "esta", "estas", "este", "estes", "eu",
        "foi", "fomos", "for", "foram", "fosse", "fossem", "há", "isso", "isto", "já",
        "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu", "meus", "minha", "minhas",
        "muito", "na", "não", "nas", "nem", "no", "nos", "nós", "nossa", "nossas",
        "nosso", "nossos", "num", "numa", "o", "os", "ou", "para", "pela", "pelas",
        "pelo", "pelos", "por", "qual", "quando", "que", "quem", "se", "seja", "sejam",
        "sem", "seu", "seus", "só", "somos", "são", "sua", "suas", "também", "te",
        "tem", "temos", "tenho", "teu", "teus", "tu", "tua", "tuas", "um", "uma",
        "você", "vocês", "vos"
    ])
    
    palavras = query.split()
    keywords = []
    for w in palavras:
        wl = w.lower().strip(".,;:!?()")
        if not wl or wl in stop_words:
            continue
        if len(wl) <= 1:
            continue
        keywords.append(wl)
    return keywords

def simplificar_query(query):
    """Simplifica a consulta para busca mais geral."""
    keywords = extrair_keywords(query)
    
    if len(keywords) >= 2:
        keywords_sorted = sorted(keywords, key=len, reverse=True)[:3]
        return " ".join(keywords_sorted)
    
    return query

# ---------------------------------------------------------------------------
# Mapeamento de tipos de documentos regulatorios para siglas de arquivo
# ---------------------------------------------------------------------------
_INSTRUCOES_COMPLETUDE = """
REGRAS DE FOCO, FUNDAMENTACAO E COMPLETUDE (OBRIGATORIAS - TODOS OS DOCUMENTOS):

1) RESPOSTA DIRETA PRIMEIRO
- Comece afirmando objetivamente o que a norma/documento estabelece para a pergunta.
- Se o contexto contiver limites, faixas, tabelas ou valores numericos relevantes, apresente-os
  logo no inicio (tabela ou lista), antes de qualquer contextualizacao.
- NUNCA abra dizendo que "nao ha valores", "nao estao explicitamente mencionados" ou que ha
  "lacuna" se o contexto contiver esses dados (mesmo em tabela markdown, texto OCR ou anexo).

2) EQUIVALENCIA TERMINOLOGICA (GERAL)
- A pergunta pode citar equipamento (ex: FWD, perfilometro), metodo de ensaio, sigla ou nome
  coloquial. Identifique o PARAMETRO NORMATIVO correspondente no contexto (ex: Dadm, IRI, IFI)
  e responda com os LIMITES/VALORES desse parametro.
- Deixe claro a relacao quando util: "O ensaio FWD verifica a deflexao; os limites sao dados
  pela Deflexao Admissivel (Dadm), conforme tabela abaixo."

3) LACUNAS PONTUAIS (NAO GLOBAIS)
- Declare lacuna SOMENTE para campos especificos ausentes no contexto (ex: "periodicidade nao
  consta nos trechos fornecidos").
- Se parte da resposta existir no contexto, entregue essa parte com confianca. Nao invalide
  a resposta inteira por campos secundarios faltantes.
- NUNCA contradiga a propria resposta (ex: dizer que nao ha dados e em seguida listar tabela).

4) FIDELIDADE AO CONTEXTO
- Preencha SOMENTE campos com evidencia textual explicita nos documentos fornecidos.
- NAO invente detalhes operacionais (espacamento, exclusoes, equipamentos auxiliares,
  frequencias, tolerancias) sem trecho correspondente no contexto.
- Omita secoes do template quando nao houver evidencia; nao preencha com suposicoes.

5) TABELAS E VALORES NUMERICOS
- Tabelas markdown (linhas com |) e blocos tabulares em texto: extraia TODOS os valores,
  limites, faixas, unidades e criterios; preserve dimensoes (fase, pista, VDM, categoria, etc.).
- Quando a norma define valores por fase, pista, faixa, classe ou periodo, apresente CADA
  combinacao separadamente.
- Nao generalize multiplos valores em um unico limite (ex: nao dizer "3,5 para todos" se o
  contexto traz 2,7, 3,0 e 3,5 em contextos distintos).

6) ESCOPO DA PERGUNTA
- Responda exatamente o que foi perguntado; contexto complementar so depois da resposta direta.
- Nao inclua parametros, processos ou assuntos nao solicitados.
- Demandas ou processos distintos nos trechos: apresente separadamente; nao funda em narrativa unica.
"""

_TIPO_DOCUMENTO_MAP = {
    "resolucao": "RES",
    "res": "RES",
    "instrucao normativa": "INM",
    "in": "INM",
    "inm": "INM",
    "deliberacao": "DLB",
    "dlb": "DLB",
    "portaria": "POR",
    "por": "POR",
    "voto": "VTO",
    "vto": "VTO",
    "incidente": "INC",
    "inc": "INC",
    "lei": "LEI",
    "decreto": "DEC",
    "dec": "DEC",
}


def _dividir_por_estrutura(texto, chunk_max=1500, chunk_overlap=200):
    """
    Divide texto de documento regulatorio por fronteiras estruturais
    (artigos, secoes, capitulos), preservando contexto.

    Usa fronteiras de artigo/secao como pontos de corte primarios.
    Se um bloco individual exceder *chunk_max*, faz subdivisao com
    RecursiveCharacterTextSplitter.

    Args:
        texto (str): Conteudo completo do documento markdown.
        chunk_max (int): Tamanho maximo de cada chunk em caracteres.
        chunk_overlap (int): Sobreposicao em caracteres para subdivisoes internas.

    Returns:
        List[str]: Lista de chunks textuais.
    """
    padrao_fronteira = re.compile(
        r"(?=\n\s*(?:"
        r"Art\.\s*\d"
        r"|CAPITULO\s"
        r"|CAP[IiIi]TULO\s"
        r"|SE[CcCc][AaAa]O\s"
        r"|T[IiIi]TULO\s"
        r"|ANEXO\s"
        r"|##\s"
        r"|###\s"
        r"))",
        re.IGNORECASE,
    )

    blocos = padrao_fronteira.split(texto)
    blocos = [b.strip() for b in blocos if b.strip()]

    if len(blocos) <= 1:
        blocos = [texto]

    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_max,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " "],
    )

    resultado = []
    for bloco in blocos:
        tem_tabela_md = "| --- |" in bloco or (
            sum(1 for ln in bloco.split("\n") if ln.strip().startswith("|")) >= 2
        )
        if tem_tabela_md and len(bloco) <= chunk_max * 3:
            resultado.append(bloco)
        elif len(bloco) <= chunk_max:
            resultado.append(bloco)
        else:
            sub_chunks = fallback_splitter.split_text(bloco)
            resultado.extend(sub_chunks)

    return resultado


def _aplicar_limite_k_preservando_prioritarios(documentos, k):
    """
    Limita resultados a k chunks, mas mantem TODOS os chunks prioritarios.

    Chunks prioritarios sao os de documentos explicitamente referenciados
    na pergunta (ex: INM 34/2024). Nunca sao descartados por causa do limite k.

    Args:
        documentos: Lista de Document ja rerankeada.
        k: Limite maximo desejado para chunks nao prioritarios.

    Returns:
        Lista de Document com prioritarios preservados.
    """
    if not documentos:
        return []

    prioritarios = []
    outros = []
    vistos_prior = set()

    for doc in documentos:
        if doc.metadata.get("prioritario"):
            chave = (
                doc.metadata.get("caminho", "")
                + "|"
                + str(doc.metadata.get("chunk", ""))
            )
            if chave not in vistos_prior:
                vistos_prior.add(chave)
                prioritarios.append(doc)
        else:
            outros.append(doc)

    restante = max(0, k - len(prioritarios))
    final = prioritarios + outros[:restante]

    if len(prioritarios) > 0:
        logger.info(
            f"Limite k={k}: mantidos {len(prioritarios)} chunk(s) prioritario(s) "
            f"+ {min(restante, len(outros))} outros = {len(final)} total"
        )

    return final


def _buscar_chunks_por_caminho_no_vectorstore(vectorstore, caminho, limite=50):
    """
    Recupera chunks indexados de um documento especifico pelo caminho do arquivo.

    Complementa o carregamento direto do .md quando o vectorstore ja contem
    o documento enriquecido com OCR.

    Args:
        vectorstore: Vectorstore FAISS carregado.
        caminho: Caminho absoluto do arquivo .md.
        limite: Maximo de chunks a retornar.

    Returns:
        Lista de Document do vectorstore para o caminho informado.
    """
    if vectorstore is None or not caminho:
        return []

    try:
        docs = vectorstore.similarity_search(
            "conteudo documento",
            k=limite,
            filter={"caminho": caminho},
        )
        for doc in docs:
            doc.metadata["prioritario"] = True
        if docs:
            logger.info(
                f"Vectorstore: {len(docs)} chunk(s) recuperados para {caminho}"
            )
        return docs
    except Exception as exc:
        logger.debug(f"Busca por caminho no vectorstore falhou: {exc}")
        return []


def _detectar_referencia_documento(query):
    """
    Detecta referencias a documentos regulatorios na query do usuario.

    Exemplos detectados:
        "Resolucao 6053/2024"  -> [{"tipo": "RES", "numero": "6053", "ano": "2024"}]
        "IN 34/2024"           -> [{"tipo": "INM", "numero": "34", "ano": "2024"}]
        "RES 6053 de 2024"     -> [{"tipo": "RES", "numero": "6053", "ano": "2024"}]

    Args:
        query (str): Texto da pergunta do usuario.

    Returns:
        List[dict]: Lista de dicts com chaves tipo, numero, ano.
    """
    resultados = []

    tipos_regex = (
        r"(?:instru[cç][aã]o\s+normativa|instrucao\s+normativa|"
        r"resolu[cç][aã]o|delibera[cç][aã]o|portaria|decreto|lei|voto|"
        r"INM|RES|DLB|POR|DEC|INC|VTO|IN)"
    )

    padrao = re.compile(
        rf"({tipos_regex})\s*(?:n[o.]\s*)?(\d[\d.]*)\s*(?:/|,?\s*de\s+)(\d{{4}})",
        re.IGNORECASE,
    )

    for match in padrao.finditer(query):
        tipo_raw = re.sub(r"\s+", " ", match.group(1).strip().lower())
        tipo_raw = (
            tipo_raw
            .replace("ç", "c")
            .replace("ã", "a")
            .replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
        )
        numero_raw = match.group(2).replace(".", "")
        ano = match.group(3)

        sigla = _TIPO_DOCUMENTO_MAP.get(tipo_raw)
        if sigla is None:
            logger.debug(f"Tipo de documento nao mapeado: '{tipo_raw}'")
            continue

        resultados.append({
            "tipo": sigla,
            "numero": numero_raw,
            "ano": ano,
        })

    logger.info(f"Referencias detectadas na query: {resultados}")
    return resultados


def _resolver_caminho_documento(tipo, numero, ano):
    """
    Resolve tipo/numero/ano para o(s) caminho(s) fisico(s) do arquivo .md.

    Args:
        tipo (str): Sigla do documento (ex: "RES").
        numero (str): Numero sem zeros (ex: "6053").
        ano (str): Ano (ex: "2024").

    Returns:
        Tuple[List[str], str]: (caminhos_candidatos, nome_tipo_amigavel)
    """
    nomes = {
        "RES": "Resolucao",
        "INM": "Instrucao Normativa",
        "DLB": "Deliberacao",
        "POR": "Portaria",
        "VTO": "Voto",
        "INC": "Incidente",
        "LEI": "Lei",
        "DEC": "Decreto",
    }
    nome_tipo = nomes.get(tipo, tipo)

    numero_padded = numero.zfill(8)
    nome_arquivo = f"{tipo}-{numero_padded}-{ano}.md"

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dados_antt")

    candidatos = [
        os.path.join(base_dir, tipo, ano, nome_arquivo),
        os.path.join(base_dir, tipo, "dados_antt", tipo, ano, nome_arquivo),
    ]

    return candidatos, nome_tipo


def _carregar_documento_markdown(caminho, tipo, nome_tipo, numero, ano):
    """
    Carrega um documento markdown e retorna uma lista de Document chunks
    divididos por estrutura (artigo/secao).

    Args:
        caminho (str): Caminho completo do arquivo .md.
        tipo (str): Sigla (ex: "RES").
        nome_tipo (str): Nome por extenso (ex: "Resolucao").
        numero (str): Numero do documento.
        ano (str): Ano.

    Returns:
        List[Document]: Lista de chunks como Document objects.
    """
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            conteudo = f.read()
    except Exception as exc:
        logger.warning(f"Falha ao ler documento {caminho}: {exc}")
        return []

    if not conteudo.strip():
        return []

    conteudo = _enriquecer_imagens_documento(conteudo)
    chunks = _dividir_por_estrutura(conteudo)

    documentos = []
    for idx, texto_chunk in enumerate(chunks):
        doc = Document(
            page_content=texto_chunk,
            metadata={
                "tipo_documento": tipo,
                "nome_tipo": nome_tipo,
                "numero": numero.zfill(8),
                "ano": ano,
                "caminho": caminho,
                "chunk": idx + 1,
                "total_chunks": len(chunks),
                "prioritario": True,
            },
        )
        documentos.append(doc)

    return documentos


# ---------------------------------------------------------------------------
# Pipeline de conversao PDF -> Markdown
# ---------------------------------------------------------------------------


def _tabela_pdfplumber_para_markdown(tabela: list) -> str:
    """
    Converte uma tabela extraida por pdfplumber (lista de listas) em markdown.

    Aplica limpeza de OCR em cada celula. Pula tabelas vazias.

    Args:
        tabela: Lista de listas (linhas x colunas) do pdfplumber.

    Returns:
        str: Tabela em formato markdown, ou string vazia se nao tiver conteudo.
    """
    if not tabela or len(tabela) < 2:
        return ""

    headers = [_limpar_celula_ocr(c or "") for c in tabela[0]]
    if all(h == "" for h in headers):
        return ""

    header_line = "| " + " | ".join(h or " " for h in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"

    rows = []
    for linha in tabela[1:]:
        cells = [_limpar_celula_ocr(c or "").replace("|", "/") for c in linha]
        if all(c == "" for c in cells):
            continue
        rows.append("| " + " | ".join(cells) + " |")

    if not rows:
        return ""
    return "\n".join([header_line, separator] + rows)


def _converter_pdf_para_md(caminho_pdf: str) -> str:
    """
    Converte um arquivo PDF em texto markdown usando pdfplumber.

    Estrategia por pagina:
    1. Extrai tabelas estruturadas e converte para markdown
    2. Extrai texto restante da pagina
    3. Se nao houver texto (PDF escaneado), faz fallback OCR via pytesseract

    Args:
        caminho_pdf: Caminho do arquivo PDF.

    Returns:
        str: Conteudo completo em formato markdown.
    """
    import pdfplumber

    partes: list = []

    try:
        with pdfplumber.open(caminho_pdf) as pdf:
            total_paginas = len(pdf.pages)
            logger.info(f"Convertendo PDF: {caminho_pdf} ({total_paginas} paginas)")

            for i, pagina in enumerate(pdf.pages):
                conteudo_pagina: list = []

                # 1) Extrair tabelas
                tabelas = pagina.extract_tables() or []
                tabelas_md = []
                for tab in tabelas:
                    md = _tabela_pdfplumber_para_markdown(tab)
                    if md:
                        tabelas_md.append(md)

                # 2) Extrair texto
                texto = pagina.extract_text() or ""
                texto = texto.strip()

                # 3) Fallback OCR se pagina sem texto (escaneada)
                if not texto and not tabelas_md:
                    try:
                        img = pagina.to_image(resolution=300)
                        pil_img = img.original
                        texto_ocr = _limpar_texto_ocr(
                            __import__("pytesseract").image_to_string(
                                pil_img, lang="por"
                            )
                        )
                        if texto_ocr:
                            conteudo_pagina.append(texto_ocr)
                            logger.debug(
                                f"PDF pagina {i+1}: OCR fallback "
                                f"({len(texto_ocr)} chars)"
                            )
                    except Exception as exc:
                        logger.debug(f"OCR fallback falhou pagina {i+1}: {exc}")

                if texto:
                    conteudo_pagina.append(texto)

                for tmd in tabelas_md:
                    conteudo_pagina.append(f"\n{tmd}\n")

                if conteudo_pagina:
                    partes.append("\n\n".join(conteudo_pagina))

    except Exception as exc:
        logger.error(f"Erro ao converter PDF {caminho_pdf}: {exc}")
        return ""

    resultado = "\n\n---\n\n".join(partes)
    logger.info(
        f"PDF convertido: {caminho_pdf} -> {len(resultado)} chars, "
        f"{total_paginas} paginas"
    )
    return resultado


def converter_pdfs_para_md(diretorio: str = "dados_antt") -> int:
    """
    Varre dados_antt/ e converte todos os PDFs que nao tem .md correspondente.

    Para cada arquivo .pdf encontrado, verifica se existe um .md com o mesmo
    nome base. Se nao existir, converte via _converter_pdf_para_md e salva
    o .md ao lado do PDF original.

    Args:
        diretorio: Pasta raiz dos documentos.

    Returns:
        int: Quantidade de PDFs convertidos.
    """
    convertidos = 0

    for dirpath, _dirs, filenames in os.walk(diretorio):
        for fname in filenames:
            if not fname.lower().endswith(".pdf"):
                continue

            caminho_pdf = os.path.join(dirpath, fname)
            nome_base = os.path.splitext(fname)[0]
            caminho_md = os.path.join(dirpath, f"{nome_base}.md")

            if os.path.exists(caminho_md):
                logger.debug(f"PDF ja tem .md: {fname}")
                continue

            logger.info(f"Convertendo PDF -> MD: {fname}")
            conteudo_md = _converter_pdf_para_md(caminho_pdf)

            if not conteudo_md or len(conteudo_md.strip()) < 50:
                logger.warning(f"PDF sem conteudo extraivel: {fname}")
                continue

            try:
                with open(caminho_md, "w", encoding="utf-8") as f:
                    f.write(conteudo_md)
                convertidos += 1
                logger.info(f"Salvo: {caminho_md} ({len(conteudo_md)} chars)")
            except Exception as exc:
                logger.error(f"Erro ao salvar {caminho_md}: {exc}")

    if convertidos > 0:
        logger.info(f"Conversao PDF concluida: {convertidos} arquivo(s) convertido(s)")
    return convertidos


# ---------------------------------------------------------------------------
# Pipeline de enriquecimento OCR para imagens em documentos markdown
# ---------------------------------------------------------------------------

_OCR_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dados_antt",
    ".ocr_cache",
)


def _corrigir_tabela_markdown_cache(texto: str) -> str:
    """
    Corrige tabelas markdown em cache OCR onde celulas multi-linha
    quebraram a estrutura da tabela.

    Em markdown valido, cada linha de tabela deve comecar e terminar com |.
    Quando o OCR extrai celulas com quebras de linha, o resultado fica:
        | celula1 | celula2 parcial
        resto | celula3 |
    Esta funcao junta essas linhas quebradas em uma unica linha.

    Args:
        texto: Conteudo em cache (pode conter tabelas e/ou texto livre).

    Returns:
        str: Texto com tabelas markdown corrigidas.
    """
    linhas = texto.split("\n")
    resultado: list = []
    buffer = ""

    for linha in linhas:
        stripped = linha.strip()
        if buffer:
            buffer = buffer + " " + stripped
            if stripped.endswith("|"):
                resultado.append(buffer)
                buffer = ""
        elif stripped.startswith("|"):
            if stripped.endswith("|"):
                resultado.append(stripped)
            else:
                buffer = stripped
        else:
            resultado.append(linha)

    if buffer:
        resultado.append(buffer)

    return "\n".join(resultado)


def _obter_cache_ocr(url_hash):
    """Retorna texto OCR em cache para o hash da URL, ou None."""
    cache_path = os.path.join(_OCR_CACHE_DIR, f"{url_hash}.txt")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                texto_cache = f.read()

            if _cache_ocr_deve_ser_invalidado(texto_cache):
                logger.info(
                    f"Cache OCR invalido (versao/qualidade) para {url_hash}, "
                    "sera reprocessado"
                )
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
                return None

            _, conteudo = _parse_meta_cache_ocr(texto_cache)
            return _pos_processar_tabela_ocr(conteudo)
        except Exception:
            return None
    return None


def _salvar_cache_ocr(url_hash, texto):
    """Persiste resultado OCR em cache local com versao e score de qualidade."""
    os.makedirs(_OCR_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_OCR_CACHE_DIR, f"{url_hash}.txt")
    try:
        texto_proc = _pos_processar_tabela_ocr(texto)
        score = _calcular_score_qualidade_ocr(texto_proc)
        payload = _formatar_cache_ocr(texto_proc, score)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(payload)
        if score < _OCR_QUALIDADE_MINIMA:
            logger.warning(
                f"Cache OCR salvo com baixa qualidade ({score:.2f}): {url_hash}"
            )
    except Exception as exc:
        logger.warning(f"Falha ao salvar cache OCR {cache_path}: {exc}")


def _baixar_imagem(url, timeout=30):
    """
    Baixa uma imagem de uma URL e retorna como objeto PIL.Image.

    Args:
        url (str): URL da imagem.
        timeout (int): Timeout em segundos.

    Returns:
        PIL.Image | None: Imagem ou None em caso de falha.
    """
    import requests as _requests
    from io import BytesIO

    try:
        resp = _requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content))
    except Exception as exc:
        logger.warning(f"Falha ao baixar imagem {url}: {exc}")
        return None


_OCR_LIXO_PATTERNS = re.compile(
    r"^(?:None|\(?\s*[A-Z]\s*\)?|[\d]+[-\s]*[A-Z]{1,4}[a-z]{0,2}[A-Z]*|\W+)$"
)


def _limpar_celula_ocr(valor: str) -> str:
    """
    Limpa o valor de uma celula extraida por OCR/img2table.

    Remove:
    - Valores literais "None" (celulas vazias do img2table)
    - Fragmentos de OCR sem valor semantico: "(A)", "1- EXProD", etc.
    - Celulas contendo apenas pontuacao/espacos
    - Quebras de linha internas (substituidas por espaco) para manter
      a integridade da tabela markdown

    Args:
        valor: Conteudo textual da celula.

    Returns:
        str: Celula limpa ou string vazia se for lixo.
    """
    texto = str(valor).strip()
    if not texto or texto.lower() == "none":
        return ""
    texto = re.sub(r"\s*\n\s*", " ", texto)
    texto = re.sub(r"\s{2,}", " ", texto).strip()
    if not texto:
        return ""
    if len(texto) <= 2 and not texto.isdigit():
        return ""
    if _OCR_LIXO_PATTERNS.match(texto):
        return ""
    return texto


def _limpar_texto_ocr(texto: str) -> str:
    """
    Remove linhas de lixo de texto extraido por OCR textual (pytesseract).

    Filtra:
    - Linhas muito curtas sem conteudo alfanumerico relevante
    - Linhas que sao apenas "None" (artefato do img2table)
    - Linhas que casam com padroes conhecidos de lixo OCR
    - Colapsa espacos em branco excessivos

    Args:
        texto: Texto bruto do OCR.

    Returns:
        str: Texto limpo.
    """
    linhas_limpas = []
    for linha in texto.splitlines():
        linha_strip = linha.strip()
        if not linha_strip:
            continue
        if linha_strip.lower() == "none":
            continue
        conteudo_alfa = re.sub(r"[^a-zA-Z0-9]", "", linha_strip)
        if len(conteudo_alfa) < 2:
            continue
        if _OCR_LIXO_PATTERNS.match(linha_strip):
            continue
        linhas_limpas.append(linha_strip)
    return "\n".join(linhas_limpas)


def _dataframe_para_markdown(df):
    """
    Converte um pandas DataFrame para tabela markdown sem depender de tabulate.

    Aplica limpeza de OCR em cada celula antes da conversao.
    Quando o img2table gera headers numericos (0, 1, 2...), promove a
    primeira linha de dados como header real da tabela.

    Args:
        df: pandas.DataFrame

    Returns:
        str: Tabela em formato markdown.
    """
    # Detectar se os headers sao numericos auto-gerados pelo img2table
    headers_originais = [str(h).strip() for h in df.columns]
    headers_sao_numericos = all(h.isdigit() for h in headers_originais)

    if headers_sao_numericos and len(df) > 1:
        # Promover primeira linha como header e remover do dataframe
        primeira_linha = df.iloc[0]
        headers = [_limpar_celula_ocr(v) or f"Col{i}" for i, v in enumerate(primeira_linha)]
        df_corpo = df.iloc[1:]
    else:
        headers = [_limpar_celula_ocr(h) or f"Col{i}" for i, h in enumerate(df.columns)]
        df_corpo = df

    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    rows = []
    for _, row in df_corpo.iterrows():
        cells = [_limpar_celula_ocr(v).replace("|", "/") for v in row]
        if all(c == "" for c in cells):
            continue
        rows.append("| " + " | ".join(cells) + " |")
    if not rows:
        return ""
    return "\n".join([header_line, separator] + rows)


def _preprocessar_imagem_ocr(imagem_pil):
    """
    Pre-processa imagem para melhorar a qualidade do OCR.

    Aplica:
    - Upscale 2x (bilinear) para melhorar leitura de caracteres pequenos
      como virgulas decimais
    - Conversao para escala de cinza
    - Aumento de contraste para nitidez de texto

    Args:
        imagem_pil (PIL.Image): Imagem original.

    Returns:
        PIL.Image: Imagem pre-processada.
    """
    try:
        from PIL import ImageEnhance, ImageFilter

        w, h = imagem_pil.size
        if w < 1500:
            fator = max(2, 1500 // w)
            imagem_pil = imagem_pil.resize(
                (w * fator, h * fator), Image.LANCZOS
            )

        if imagem_pil.mode != "L":
            imagem_pil = imagem_pil.convert("L")

        imagem_pil = ImageEnhance.Contrast(imagem_pil).enhance(1.8)
        imagem_pil = ImageEnhance.Sharpness(imagem_pil).enhance(2.0)

        return imagem_pil
    except Exception as exc:
        logger.debug(f"Pre-processamento de imagem falhou (usando original): {exc}")
        return imagem_pil


_RE_NUMERO_UNIDADE = re.compile(
    r"\b(\d{2,})(\s*(?:m/km|mm))\b"
)

_RE_COMPARADOR_DECIMAL = re.compile(
    r"([><]=?\s*)0(\d)(?=\s*[|\s,;)]|$)"
)


def _corrigir_decimais_ocr(texto: str) -> str:
    """
    Corrige numeros decimais cujas virgulas foram perdidas pelo OCR.

    Duas heuristicas:
    1. Numeros seguidos de unidades de medida (m/km, mm): se o valor
       inteiro for improvavel (ex: 27 m/km deveria ser 2,7 m/km),
       insere virgula na posicao mais provavel.
    2. Valores com comparadores (>, <, >=, <=) seguidos de "0X":
       padroes como ">02" provavelmente sao ">0,2" (IFI, indices
       fracionarios).

    Args:
        texto: Texto OCR com possiveis erros de decimal.

    Returns:
        str: Texto corrigido.
    """
    limites_por_unidade = {
        "m/km": 10.0,
        "mm": 100.0,
    }

    def _corrigir_match_unidade(m):
        num_str = m.group(1)
        unidade = m.group(2).strip()

        unidade_lower = unidade.lower()
        limite = None
        for u, lim in limites_por_unidade.items():
            if unidade_lower.startswith(u):
                limite = lim
                break

        if limite is None:
            return m.group(0)

        try:
            valor = float(num_str)
        except ValueError:
            return m.group(0)

        if valor <= limite:
            return m.group(0)

        corrigido = num_str[:-1] + "," + num_str[-1]
        try:
            novo_valor = float(corrigido.replace(",", "."))
        except ValueError:
            return m.group(0)

        if novo_valor <= limite:
            return corrigido + m.group(2)

        return m.group(0)

    def _corrigir_match_comparador(m):
        """Converte '>02' em '>0,2', '<05' em '<0,5', etc."""
        return f"{m.group(1)}0,{m.group(2)}"

    resultado = _RE_NUMERO_UNIDADE.sub(_corrigir_match_unidade, texto)
    resultado = _RE_COMPARADOR_DECIMAL.sub(_corrigir_match_comparador, resultado)
    return resultado


_OCR_PIPELINE_VERSION = "3"
_OCR_QUALIDADE_MINIMA = 0.55
_OCR_QUALIDADE_ALTA = 0.85

_RE_GARBAGE_OCR = re.compile(
    r"(?:\b[a-z]{2,}\s*/\s*\]|\btooo\b|\bas\s*/\]|[^\w\s,;.\-+Ee\d/]{4,})",
    re.IGNORECASE,
)
_RE_NOTACAO_CIENTIFICA = re.compile(
    r"^\d+[,.]?\d*E[+\-]?\d+$",
    re.IGNORECASE,
)
_RE_CELULA_NUMERICA_CORROMPIDA = re.compile(
    r"^\d{5,8}$"
)


def _preprocessar_imagem_ocr_binarizada(imagem_pil):
    """
    Pre-processamento alternativo com binarizacao adaptativa.

    Melhora leitura de tabelas com linhas finas e texto pequeno,
    comum em anexos normativos escaneados ou gerados como imagem.
    """
    from PIL import ImageEnhance, ImageOps

    imagem = imagem_pil.copy()
    largura, altura = imagem.size
    if largura < 1500:
        fator = max(2, 1500 // largura)
        imagem = imagem.resize((largura * fator, altura * fator), Image.LANCZOS)

    if imagem.mode != "L":
        imagem = imagem.convert("L")

    imagem = ImageEnhance.Contrast(imagem).enhance(2.0)
    imagem = ImageOps.autocontrast(imagem)
    arr = np.array(imagem, dtype=np.float32)
    limiar = float(np.mean(arr)) * 0.85
    binario = (arr > limiar).astype(np.uint8) * 255
    return Image.fromarray(binario)


def _preprocessar_imagem_ocr_agressivo(imagem_pil):
    """Upscale mais agressivo para imagens pequenas de tabelas normativas."""
    from PIL import ImageEnhance

    imagem = imagem_pil.copy()
    largura, altura = imagem.size
    fator = max(3, 2400 // max(largura, 1))
    imagem = imagem.resize((largura * fator, altura * fator), Image.LANCZOS)

    if imagem.mode != "L":
        imagem = imagem.convert("L")

    imagem = ImageEnhance.Contrast(imagem).enhance(2.2)
    imagem = ImageEnhance.Sharpness(imagem).enhance(2.5)
    return imagem


def _gerar_variantes_preprocessamento_ocr(imagem_pil):
    """
    Gera variantes de pre-processamento para tentativas multiplas de OCR.

    Returns:
        list: Tuplas (nome_variante, imagem_pil_processada).
    """
    variantes = [("padrao", _preprocessar_imagem_ocr(imagem_pil))]

    try:
        variantes.append(("binarizada", _preprocessar_imagem_ocr_binarizada(imagem_pil)))
    except Exception as exc:
        logger.debug(f"Variante binarizada indisponivel: {exc}")

    largura, _ = imagem_pil.size
    if largura < 2000:
        try:
            variantes.append(
                ("upscale", _preprocessar_imagem_ocr_agressivo(imagem_pil))
            )
        except Exception as exc:
            logger.debug(f"Variante upscale indisponivel: {exc}")

    return variantes


def _celula_parece_corrompida(valor: str) -> bool:
    """
    Detecta celulas de tabela com conteudo provavelmente corrompido pelo OCR.

    Heuristicas gerais (nao especificas de um documento):
    - Caracteres especiais isolados (/], etc.)
    - Mistura de letras e numeros em celulas curtas
    - Sequencias longas de digitos sem notacao cientifica (ex.: 600807)
    """
    celula = str(valor).strip()
    if not celula:
        return False

    if _RE_GARBAGE_OCR.search(celula):
        return True

    if re.search(r"[/\[\]{}|\\^~`]", celula):
        return True

    if re.match(r"^[a-zA-Z]{2,}$", celula) and celula.lower() not in {
        "principal",
        "marginal",
        "quinquenal",
    }:
        if re.search(r"\d", celula):
            return True
        if len(celula) <= 5 and not celula.isupper():
            return True

    if _RE_CELULA_NUMERICA_CORROMPIDA.match(celula):
        if not _RE_NOTACAO_CIENTIFICA.match(celula):
            return True

    return False


def _calcular_score_qualidade_ocr(texto: str) -> float:
    """
    Calcula score de qualidade (0.0 a 1.0) do texto OCR extraido.

    Penaliza celulas corrompidas em tabelas markdown e padroes de lixo
    conhecidos. Usado para escolher a melhor estrategia de extracao e
    invalidar cache de baixa qualidade.
    """
    if not texto or not texto.strip():
        return 0.0

    score = 1.0
    lixo = len(_RE_GARBAGE_OCR.findall(texto))
    score -= min(0.5, lixo * 0.12)

    linhas_tabela = [
        linha.strip()
        for linha in texto.splitlines()
        if linha.strip().startswith("|") and "| ---" not in linha
    ]

    if not linhas_tabela:
        conteudo_util = re.sub(r"[^a-zA-Z0-9]", "", texto)
        if len(conteudo_util) < 20:
            return max(0.0, score - 0.4)
        return max(0.0, min(1.0, score))

    total_celulas = 0
    celulas_ruins = 0
    for linha in linhas_tabela:
        if re.match(r"^\|\s*[-:\s|]+\s*\|$", linha):
            continue
        celulas = [c.strip() for c in linha.split("|")[1:-1]]
        for celula in celulas:
            total_celulas += 1
            if _celula_parece_corrompida(celula):
                celulas_ruins += 1

    if total_celulas > 0:
        ratio_ruim = celulas_ruins / total_celulas
        score -= ratio_ruim * 0.75

    return max(0.0, min(1.0, score))


def _tentar_corrigir_notacao_cientifica_celula(celula: str) -> str:
    """
    Tenta reconstruir notacao cientifica corrompida em celulas numericas.

    Exemplo generico: '600807' -> '6,00E+07' (E e virgula perdidos pelo OCR).
    """
    valor = str(celula).strip()
    if not valor or _RE_NOTACAO_CIENTIFICA.match(valor):
        return valor

    if not _RE_CELULA_NUMERICA_CORROMPIDA.match(valor):
        return valor

    for tam_int in (1, 2):
        for tam_dec in (2,):
            if len(valor) <= tam_int + tam_dec + 2:
                continue
            parte_int = valor[:tam_int]
            parte_dec = valor[tam_int : tam_int + tam_dec]
            resto = valor[tam_int + tam_dec :]
            if not parte_int.isdigit() or not parte_dec.isdigit():
                continue
            if not resto.isdigit() or len(resto) < 2:
                continue
            expoente = resto[-2:]
            if expoente not in {"06", "07", "08", "09"}:
                continue
            candidato = f"{parte_int},{parte_dec}E+{expoente}"
            return candidato

    return valor


def _corrigir_celulas_tabela_markdown(linha: str) -> str:
    """Aplica correcoes celula a celula em uma linha de tabela markdown."""
    if not linha.strip().startswith("|") or "| ---" in linha:
        return linha

    partes = linha.split("|")
    if len(partes) < 3:
        return linha

    celulas_corrigidas = []
    for idx, celula in enumerate(partes):
        if idx == 0 or idx == len(partes) - 1:
            celulas_corrigidas.append(celula)
            continue
        limpa = _limpar_celula_ocr(celula)
        limpa = _tentar_corrigir_notacao_cientifica_celula(limpa)
        celulas_corrigidas.append(limpa)

    return "|".join(celulas_corrigidas)


def _pos_processar_tabela_ocr(texto: str) -> str:
    """
    Pos-processamento geral de texto/tabelas OCR.

    Aplica correcao de linhas quebradas, notacao cientifica, decimais
    e limpeza de celulas antes de cachear ou enviar ao vectorstore.
    """
    if not texto:
        return texto

    texto = _corrigir_tabela_markdown_cache(texto)
    linhas = [_corrigir_celulas_tabela_markdown(l) for l in texto.splitlines()]
    texto = "\n".join(linhas)
    texto = _corrigir_decimais_ocr(texto)
    return texto


def _parse_meta_cache_ocr(texto: str) -> tuple:
    """
    Separa metadados do cabecalho do cache OCR do conteudo util.

    Returns:
        tuple: (meta_dict, conteudo_sem_meta)
    """
    meta = {}
    linhas = texto.splitlines()
    idx_conteudo = 0

    for idx, linha in enumerate(linhas):
        if not linha.startswith("#"):
            idx_conteudo = idx
            break
        if "=" in linha:
            chave, _, val = linha[1:].strip().partition("=")
            meta[chave.strip()] = val.strip()
        idx_conteudo = idx + 1
    else:
        idx_conteudo = len(linhas)

    conteudo = "\n".join(linhas[idx_conteudo:]).lstrip("\n")
    return meta, conteudo


def _formatar_cache_ocr(texto: str, score: float) -> str:
    """Formata texto OCR com cabecalho de versao e score de qualidade."""
    cabecalho = (
        f"# ocr_pipeline_v={_OCR_PIPELINE_VERSION}\n"
        f"# quality={score:.3f}\n"
    )
    return cabecalho + texto


def _cache_ocr_deve_ser_invalidado(texto_cache: str) -> bool:
    """
    Decide se uma entrada de cache OCR deve ser descartada e reprocessada.

    Invalida caches de versoes antigas do pipeline ou com qualidade
    abaixo do minimo aceitavel apos pos-processamento.
    """
    meta, conteudo = _parse_meta_cache_ocr(texto_cache)
    versao = meta.get("ocr_pipeline_v", "1")
    if versao != _OCR_PIPELINE_VERSION:
        return True

    conteudo_proc = _pos_processar_tabela_ocr(conteudo)
    score = _calcular_score_qualidade_ocr(conteudo_proc)
    return score < _OCR_QUALIDADE_MINIMA


def _tentar_extrair_com_img2table(
    imagem_processada,
    borderless: bool = False,
    implicit_rows: bool = False,
) -> str:
    """Extrai tabelas via img2table e converte para markdown."""
    import tempfile

    from img2table.document import Image as Img2TableImage
    from img2table.ocr import TesseractOCR as Img2TableOCR

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        imagem_processada.save(tmp_path)

    try:
        ocr_engine = Img2TableOCR(lang="por+eng", psm=6)
        img_doc = Img2TableImage(src=tmp_path)
        tabelas = img_doc.extract_tables(
            ocr=ocr_engine,
            implicit_rows=implicit_rows,
            borderless_tables=borderless,
            min_confidence=50,
        )

        if not tabelas:
            return ""

        partes = []
        for tabela in tabelas:
            df = tabela.df
            if df is not None and not df.empty:
                md_table = _dataframe_para_markdown(df)
                if md_table.strip():
                    partes.append(md_table)

        return "\n\n".join(partes)
    except Exception as exc:
        logger.debug(f"img2table falhou (borderless={borderless}): {exc}")
        return ""
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _tentar_extrair_com_tesseract(imagem_processada, psm: int = 6) -> str:
    """Extrai texto livre via pytesseract com PSM configuravel."""
    import pytesseract

    config = f"--psm {psm} --oem 3"
    texto_ocr = pytesseract.image_to_string(
        imagem_processada,
        lang="por+eng",
        config=config,
    )
    return _limpar_texto_ocr(texto_ocr)


def _extrair_texto_imagem(imagem_pil, url=""):
    """
    Extrai texto de imagem com pipeline OCR multi-estrategia.

    Pipeline:
    1. Gera variantes de pre-processamento (padrao, binarizada, upscale)
    2. Para cada variante, tenta img2table (borderless e padrao) e Tesseract
    3. Pos-processa e pontua cada candidato
    4. Retorna o candidato com maior score de qualidade

    Args:
        imagem_pil (PIL.Image): Imagem carregada.
        url (str): URL original (para logging).

    Returns:
        str: Texto extraido em formato markdown.
    """
    variantes = _gerar_variantes_preprocessamento_ocr(imagem_pil)
    candidatos = []

    estrategias = [
        ("img2table_borderless", lambda img: _tentar_extrair_com_img2table(
            img, borderless=True, implicit_rows=True
        )),
        ("img2table", lambda img: _tentar_extrair_com_img2table(img)),
        ("tesseract_psm6", lambda img: _tentar_extrair_com_tesseract(img, psm=6)),
        ("tesseract_psm4", lambda img: _tentar_extrair_com_tesseract(img, psm=4)),
    ]

    for nome_variante, imagem_proc in variantes:
        for nome_estrategia, extrator in estrategias:
            try:
                bruto = extrator(imagem_proc)
            except Exception as exc:
                logger.debug(
                    f"Estrategia {nome_estrategia}/{nome_variante} falhou "
                    f"para {url}: {exc}"
                )
                continue

            if not bruto or not bruto.strip():
                continue

            processado = _pos_processar_tabela_ocr(bruto)
            score = _calcular_score_qualidade_ocr(processado)
            candidatos.append(
                (score, processado, nome_variante, nome_estrategia)
            )

            if score >= _OCR_QUALIDADE_ALTA:
                logger.info(
                    f"OCR alta qualidade ({score:.2f}) via "
                    f"{nome_estrategia}/{nome_variante} para {url}"
                )
                return processado

    if not candidatos:
        logger.warning(f"OCR nao extraiu texto utilizavel de {url}")
        return ""

    candidatos.sort(key=lambda item: item[0], reverse=True)
    melhor_score, melhor_texto, melhor_var, melhor_est = candidatos[0]

    if melhor_score < _OCR_QUALIDADE_MINIMA:
        logger.warning(
            f"OCR baixa qualidade ({melhor_score:.2f}) via "
            f"{melhor_est}/{melhor_var} para {url}"
        )
    else:
        logger.info(
            f"OCR qualidade {melhor_score:.2f} via "
            f"{melhor_est}/{melhor_var} para {url}"
        )

    return melhor_texto


_OCR_MAX_WORKERS = min(4, (os.cpu_count() or 2))


def _processar_imagem_ocr(url_e_hash: tuple) -> tuple:
    """Processa uma unica imagem: verifica cache, baixa e faz OCR.

    Funcao auxiliar projetada para execucao em ThreadPoolExecutor.
    O Tesseract roda como subprocesso externo, portanto nao e
    bloqueado pelo GIL do Python.

    Args:
        url_e_hash: Tupla (url, url_hash).

    Returns:
        Tupla (url_hash, texto_extraido) ou (url_hash, None) em caso de falha.
    """
    url, url_hash = url_e_hash

    texto_cache = _obter_cache_ocr(url_hash)
    if texto_cache is not None:
        logger.info(f"Cache OCR encontrado para {url_hash}")
        return (url_hash, texto_cache)

    imagem = _baixar_imagem(url)
    if imagem is None:
        return (url_hash, None)

    texto_extraido = _extrair_texto_imagem(imagem, url)
    _salvar_cache_ocr(url_hash, texto_extraido)
    return (url_hash, texto_extraido)


def _enriquecer_imagens_documento(conteudo_md):
    """
    Substitui referencias de imagem (![...](url)) por texto extraido via OCR.

    Percorre o markdown, identifica todas as tags de imagem com URLs externas,
    baixa cada imagem, executa OCR/extracao de tabelas e substitui a tag
    pelo conteudo textual extraido.

    Usa cache local (dados_antt/.ocr_cache/) para evitar re-download e
    re-processamento em re-indexacoes.

    Imagens sem cache sao processadas em paralelo usando ThreadPoolExecutor
    (ate _OCR_MAX_WORKERS threads). O Tesseract roda como subprocesso
    nativo, portanto o paralelismo e real mesmo com o GIL.

    Args:
        conteudo_md (str): Conteudo markdown do documento.

    Returns:
        str: Conteudo markdown enriquecido (imagens substituidas por texto).
    """
    import hashlib
    from concurrent.futures import ThreadPoolExecutor, as_completed

    padrao_imagem = re.compile(
        r"!\[([^\]]*)\]\((https?://[^\s)]+\.(?:png|jpg|jpeg|gif|svg))\)",
        re.IGNORECASE,
    )

    matches = list(padrao_imagem.finditer(conteudo_md))
    if not matches:
        return conteudo_md

    logger.info(f"Encontradas {len(matches)} imagens para enriquecimento OCR")

    urls_unicas: dict = {}
    for match in matches:
        url = match.group(2)
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        if url_hash not in urls_unicas:
            urls_unicas[url_hash] = url

    resultados_ocr: dict = {}

    tarefas = [(url, h) for h, url in urls_unicas.items()]
    n_workers = min(_OCR_MAX_WORKERS, len(tarefas))

    if n_workers <= 1:
        for url, h in tarefas:
            h_out, texto = _processar_imagem_ocr((url, h))
            resultados_ocr[h_out] = texto
    else:
        logger.info(
            f"Processando {len(tarefas)} imagens em paralelo "
            f"({n_workers} workers)"
        )
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futuros = {
                executor.submit(_processar_imagem_ocr, (url, h)): h
                for url, h in tarefas
            }
            for futuro in as_completed(futuros):
                try:
                    h_out, texto = futuro.result()
                    resultados_ocr[h_out] = texto
                except Exception as exc:
                    h_falha = futuros[futuro]
                    logger.warning(f"OCR falhou para {h_falha}: {exc}")
                    resultados_ocr[h_falha] = None

    resultado = conteudo_md
    substituicoes = 0

    for match in reversed(matches):
        url = match.group(2)
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        texto_extraido = resultados_ocr.get(url_hash)

        if texto_extraido and texto_extraido.strip():
            bloco = f"\n{texto_extraido}\n"
            resultado = resultado[:match.start()] + bloco + resultado[match.end():]
            substituicoes += 1

    logger.info(
        f"Enriquecimento OCR concluido: {substituicoes}/{len(matches)} imagens convertidas"
    )
    return resultado


def reranking_documentos(query, documentos):
    """Reordena os documentos com base na relevancia para a consulta."""
    if not documentos:
        return []
    
    keywords = set(extrair_keywords(query))
    # Detectar e priorizar caminhos explicitamente citados na pergunta
    caminhos_prioritarios = set()
    try:
        for ref in _detectar_referencia_documento(query):
            candidatos, _ = _resolver_caminho_documento(ref["tipo"], ref["numero"], ref["ano"])
            for c in candidatos:
                if os.path.exists(c):
                    caminhos_prioritarios.add(c)
    except Exception:
        pass

    # Pre-compilar padroes de artigo/secao para boost estrutural
    _padrao_artigo = re.compile(r"\bArt\.\s*\d+", re.IGNORECASE)

    def calcular_score_documento(doc):
        score = 0.0
        conteudo = doc.page_content.lower()
        metadados = doc.metadata

        # Score por keywords encontradas no conteudo
        hits = 0
        for keyword in keywords:
            if keyword in conteudo:
                hits += 1
                score += 1.0
            elif any(keyword in palavra for palavra in conteudo.split()):
                hits += 0.5
                score += 0.5

        # Densidade de keywords (proporcao de hits sobre total de keywords)
        if keywords:
            score += (hits / len(keywords)) * 3.0

        if metadados.get("relevancia_tecnica") == "Alta":
            score += 2.0
        if metadados.get("contem_tabelas") == "Sim":
            score += 1.0

        # Boost por conteudo estruturado (artigos regulatorios)
        artigos_encontrados = _padrao_artigo.findall(doc.page_content)
        if artigos_encontrados:
            score += min(len(artigos_encontrados) * 0.3, 2.0)

        # Boost para chunks com tabelas markdown (dados numericos extraidos por OCR)
        if "| --- |" in doc.page_content:
            score += 3.0

        # Boost forte para documentos priorizados (referencia explicita na query)
        caminho = metadados.get("caminho", "")
        if caminho in caminhos_prioritarios or metadados.get("prioritario"):
            score += 10.0

        return score
    
    docs_com_score = [(doc, calcular_score_documento(doc)) for doc in documentos]
    docs_ordenados = [doc for doc, score in sorted(docs_com_score, key=lambda x: x[1], reverse=True)]
    return docs_ordenados

def pesquisar_documentos(query, vectorstore, k=16, tipo_documento=None, ano=None, numero=None, embedding_provider="free"):
    """Pesquisa documentos com base em uma query, podendo filtrar por metadados."""
    resultados = []
    resultados_finais = []
    query_original = query

    logger.info(f"\n===== NOVA CONSULTA =====")
    logger.info(f"Pesquisando: '{query}'")
    if tipo_documento or ano or numero:
        logger.info(f"Filtros: Tipo={tipo_documento}, Ano={ano}, Numero={numero}")

    filtro = criar_filtro_metadados(tipo_documento, ano, numero)

    # 1) Priorizar documentos explicitamente citados na pergunta (ex.: INM 18/2023)
    try:
        refs = _detectar_referencia_documento(query)
        caminhos_prioritarios = set()
        for ref in refs:
            candidatos, nome_tipo = _resolver_caminho_documento(ref["tipo"], ref["numero"], ref["ano"])
            for caminho in candidatos:
                if not os.path.exists(caminho):
                    continue
                docs_carregados = _carregar_documento_markdown(
                    caminho, ref["tipo"], nome_tipo, ref["numero"], ref["ano"]
                )
                if docs_carregados:
                    resultados.extend(docs_carregados)
                    caminhos_prioritarios.add(caminho)
                    logger.info(
                        f"Documento priorizado carregado: {caminho} "
                        f"({len(docs_carregados)} chunks)"
                    )
                docs_vs = _buscar_chunks_por_caminho_no_vectorstore(
                    vectorstore, caminho
                )
                if docs_vs:
                    resultados.extend(docs_vs)
                    caminhos_prioritarios.add(caminho)
        if caminhos_prioritarios:
            for doc in resultados:
                if doc.metadata.get("caminho", "") in caminhos_prioritarios:
                    doc.metadata["prioritario"] = True
    except Exception as e:
        logger.warning(f"Deteccao de referencia de documento falhou: {e}")

    try:
        keywords = extrair_keywords(query)
        logger.info(f"Executando busca semantica com MMR...")
        docs_semantic = vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=k*3,
            lambda_mult=0.7,
            filter=filtro
        )
        resultados.extend(docs_semantic)
        logger.info(f"Busca semantica: {len(docs_semantic)} resultados")
        if keywords and len(keywords) > 1:
            keyword_query = " ".join(keywords)
            logger.info(f"Executando busca adicional por keywords: '{keyword_query}'")
            docs_keywords = vectorstore.similarity_search(
                keyword_query,
                k=max(3, k//2),
                filter=filtro
            )
            docs_ids = set(doc.metadata.get('caminho', '') + str(doc.metadata.get('chunk', '')) 
                           for doc in resultados)
            for doc in docs_keywords:
                doc_id = doc.metadata.get('caminho', '') + str(doc.metadata.get('chunk', ''))
                if doc_id not in docs_ids:
                    resultados.append(doc)
                    docs_ids.add(doc_id)
            logger.info(f"Apos busca por keywords: {len(resultados)} resultados")
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ["insufficient_quota", "429", "quota", "exceeded"]):
            logger.warning(f"Erro de cota detectado durante busca: {e}")
            if embedding_provider != "free":
                logger.info("Recarregando vectorstore com embeddings gratuitos...")
                try:
                    vectorstore_free = carregar_vectorstore_com_provider("free")
                    return pesquisar_documentos(query, vectorstore_free, k, tipo_documento, ano, numero, "free")
                except Exception as reload_error:
                    logger.error(f"Falha ao recarregar vectorstore: {reload_error}")
            logger.warning("Tentando busca sem embeddings (busca por texto)...")
            return busca_fallback_sem_embeddings(query, k, tipo_documento, ano, numero)
        else:
            logger.error(f"Erro na busca hibrida: {str(e)}")

    if len(resultados) < 2:
        logger.info("Resultados insuficientes. Tentando busca direta...")
        try:
            resultados = vectorstore.similarity_search(query_original, k=k, filter=filtro)
            logger.info(f"Busca direta: {len(resultados)} resultados")
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["insufficient_quota", "429", "quota", "exceeded"]):
                logger.warning(f"Erro de cota na busca direta: {e}")
                return busca_fallback_sem_embeddings(query, k, tipo_documento, ano, numero)
            else:
                logger.error(f"Erro na busca direta: {str(e)}")

    if len(resultados) < 2:
        logger.info("Resultados ainda insuficientes. Tentando busca com termos amplos...")
        try:
            termos_gerais = simplificar_query(query_original)
            resultados = vectorstore.similarity_search(termos_gerais, k=k)
            logger.info(f"Busca com termos amplos: {len(resultados)} resultados")
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["insufficient_quota", "429", "quota", "exceeded"]):
                logger.warning(f"Erro de cota na busca ampla: {e}")
                return busca_fallback_sem_embeddings(query, k, tipo_documento, ano, numero)
            else:
                logger.error(f"Erro na busca com termos amplos: {str(e)}")

    # Deduplicar resultados antes do reranking
    if resultados:
        vistos = set()
        resultados_unicos = []
        for doc in resultados:
            chave = (
                doc.metadata.get("caminho", "")
                + "|"
                + str(doc.metadata.get("chunk", ""))
            )
            if chave not in vistos:
                vistos.add(chave)
                resultados_unicos.append(doc)
        if len(resultados_unicos) < len(resultados):
            logger.info(
                f"Deduplicacao: {len(resultados)} -> {len(resultados_unicos)} chunks unicos"
            )
        resultados = resultados_unicos

    if resultados:
        resultados_finais = reranking_documentos(query_original, resultados)
        if len(resultados_finais) > k:
            logger.info(
                f"Reranking produziu {len(resultados_finais)} docs, "
                f"aplicando limite k={k} (preservando prioritarios)"
            )
        resultados_finais = _aplicar_limite_k_preservando_prioritarios(
            resultados_finais, k
        )
        logger.info(f"Apos reranking: {len(resultados_finais)} documentos retornados")

    return resultados_finais

def busca_fallback_sem_embeddings(query, k=12, tipo_documento=None, ano=None, numero=None):
    """Busca de fallback que não usa embeddings - busca por texto simples."""
    logger.info("🔍 Executando busca de emergência sem embeddings...")
    
    try:
        # Implementar busca básica por palavras-chave nos metadados e conteúdo
        # Esta é uma implementação simplificada que pode ser expandida
        
        # Tentar carregar dados do relatório de documentos se disponível
        relatorio_path = "relatorio_documentos.json"
        if os.path.exists(relatorio_path):
            logger.info("📄 Carregando dados do relatório de documentos...")
            
            with open(relatorio_path, 'r', encoding='utf-8') as f:
                dados_documentos = json.load(f)
            
            # Extrair palavras-chave da query
            keywords = extrair_keywords(query.lower())
            logger.info(f"🔍 Palavras-chave extraídas: {keywords}")
            
            # Buscar documentos que contenham as palavras-chave
            documentos_encontrados = []
            
            # O arquivo JSON é uma lista direta de documentos
            for doc_info in dados_documentos:
                score = 0
                
                # Buscar nas informações do documento
                texto_busca = f"{doc_info.get('titulo', '')} {doc_info.get('tipo', '')} {doc_info.get('ementa', '')}".lower()
                
                # Calcular score baseado nas palavras-chave encontradas
                for keyword in keywords:
                    if keyword in texto_busca:
                        score += 1
                
                # Aplicar filtros se especificados
                if tipo_documento and tipo_documento.lower() not in doc_info.get('tipo', '').lower():
                    continue
                
                if ano and str(ano) not in str(doc_info.get('ano', '')):
                    continue
                
                if numero and str(numero) not in str(doc_info.get('numero', '')):
                    continue
                
                if score > 0:
                    documentos_encontrados.append((doc_info, score))
            
            # Ordenar por score e retornar os melhores
            documentos_encontrados.sort(key=lambda x: x[1], reverse=True)
            
            # Converter para formato compatível (simulando Document objects)
            resultados = []
            for doc_info, score in documentos_encontrados[:k]:
                # Criar um objeto simples que simula um Document
                doc_simulado = SimpleNamespace()
                doc_simulado.page_content = f"""
DOCUMENTO: {doc_info.get('titulo', 'Título não disponível')}

EMENTA: {doc_info.get('ementa', 'Ementa não disponível')}

TIPO: {doc_info.get('tipo', 'N/A')}
NÚMERO: {doc_info.get('numero', 'N/A')}
ANO: {doc_info.get('ano', 'N/A')}
ÓRGÃO: {doc_info.get('orgao', 'N/A')}

OBSERVAÇÃO: Este documento foi encontrado através de busca de emergência sem embeddings. 
Para informações mais detalhadas, consulte o documento completo.
"""
                doc_simulado.metadata = {
                    'nome_tipo': doc_info.get('tipo', 'Documento'),
                    'numero': doc_info.get('numero', 'N/A'),
                    'ano': doc_info.get('ano', 'N/A'),
                    'caminho': doc_info.get('arquivo_md', 'N/A'),
                    'chunk': 1,
                    'total_chunks': 1,
                    'score_emergencia': score,
                    'modo_emergencia': True
                }
                resultados.append(doc_simulado)
            
            logger.info(f"✅ Busca de emergência encontrou {len(resultados)} documentos")
            return resultados
        
        else:
            logger.warning("⚠️ Arquivo de relatório não encontrado para busca de emergência")
            
            # Fallback ainda mais básico: criar uma resposta explicativa
            doc_explicativo = SimpleNamespace()
            doc_explicativo.page_content = f"""
            SISTEMA EM MODO DE EMERGÊNCIA
            
            Devido a limitações temporárias da API de embeddings, o sistema está operando em modo de emergência.
            
            Sua consulta: "{query}"
            
            Para obter respostas completas, tente:
            1. Aguardar alguns minutos e tentar novamente
            2. Usar termos mais específicos na busca
            3. Verificar se há documentos específicos que você gostaria de consultar
            
            O sistema tentará responder com base nas informações disponíveis, mas a qualidade pode ser limitada.
            """
            
            doc_explicativo.metadata = {
                'nome_tipo': 'Sistema',
                'numero': 'EMERGENCIA',
                'ano': '2025',
                'caminho': 'sistema_emergencia',
                'chunk': 1,
                'total_chunks': 1,
                'modo_emergencia': True
            }
            
            return [doc_explicativo]
        
    except Exception as e:
        logger.error(f"❌ Erro na busca de fallback: {e}")
        
        # Último recurso: documento explicativo de erro
        doc_erro = SimpleNamespace()
        doc_erro.page_content = f"""
        ERRO NO SISTEMA DE BUSCA
        
        O sistema encontrou dificuldades técnicas temporárias.
        
        Erro: {str(e)}
        
        Por favor, tente novamente em alguns minutos ou entre em contato com o suporte técnico.
        """
        
        doc_erro.metadata = {
            'nome_tipo': 'Sistema',
            'numero': 'ERRO',
            'ano': '2025',
            'caminho': 'sistema_erro',
            'chunk': 1,
            'total_chunks': 1,
            'modo_emergencia': True
        }
        
        return [doc_erro]

def _normalize_text_ascii_lower(value: str) -> str:
    """
    Normaliza texto para comparacao: remove diacriticos (NFD -> ASCII) e converte para minusculas.
    Retorna string segura para buscas de palavras-chave.
    """
    try:
        import unicodedata
        normalized = unicodedata.normalize("NFD", value)
        without_diacritics = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
        return without_diacritics.lower()
    except Exception:
        return value.lower()

_MAX_CONTEXT_CHARS = 50000
_MAX_CHUNKS_LLM = 30

_TEMPLATE_REESCRITA = """Reescreva a ULTIMA PERGUNTA como uma pergunta COMPLETA e AUTOCONTIDA
usando o contexto do historico. Se ja for autocontida, retorne-a sem alteracao.

REGRAS OBRIGATORIAS:
- Retorne SOMENTE a pergunta reescrita, nada mais.
- NAO inclua explicacoes, notas, comentarios ou parenteses.
- NAO comece com "Pergunta:" ou similar.
- A saida deve conter UMA UNICA frase interrogativa.

HISTORICO:
{historico}

ULTIMA PERGUNTA: "{pergunta}"

PERGUNTA REESCRITA:"""


def _reescrever_query_com_historico(pergunta, historico, llm):
    """
    Reescreve a pergunta do usuario incorporando contexto do historico de conversa.

    Usa o LLM para transformar perguntas ambiguas ou de aprofundamento em
    queries autocontidas que funcionam bem para busca semantica.

    Args:
        pergunta (str): Pergunta original do usuario.
        historico (list): Lista de dicts com chaves 'pergunta' e 'resposta'.
        llm: Instancia do LLM configurado.

    Returns:
        str: Pergunta reescrita (autocontida) ou a original se nao houver historico.
    """
    if not historico:
        return pergunta

    # Montar texto do historico (ultimas 3 interacoes)
    turnos = historico[-3:]
    linhas = []
    for turno in turnos:
        linhas.append(f"Usuario: {turno['pergunta']}")
        resumo_resp = turno["resposta"][:300]
        linhas.append(f"Assistente: {resumo_resp}...")
    texto_historico = "\n".join(linhas)

    prompt = PromptTemplate(
        template=_TEMPLATE_REESCRITA,
        input_variables=["historico", "pergunta"],
    )

    try:
        chain = prompt | llm
        resultado = chain.invoke({
            "historico": texto_historico,
            "pergunta": pergunta,
        })
        reescrita = resultado.content.strip() if hasattr(resultado, "content") else str(resultado).strip()

        # Limpar lixo que o LLM pode adicionar
        reescrita = reescrita.strip('"').strip("'")
        # Remover notas em parenteses no final (ex: "(Nota: ...)")
        reescrita = re.sub(r"\s*\((?:Nota|Obs|Observa).*\)\s*$", "", reescrita, flags=re.IGNORECASE | re.DOTALL)
        # Pegar apenas a primeira linha se o LLM retornou multiplas
        reescrita = reescrita.split("\n")[0].strip()

        if len(reescrita) < 10:
            return pergunta

        logger.info(f"Query reescrita: '{pergunta}' -> '{reescrita}'")
        return reescrita

    except Exception as exc:
        logger.warning(f"Falha na reescrita de query: {exc}. Usando original.")
        return pergunta


def gerar_resposta(pergunta, documentos, llm, modelo_usado="gpt-4"):
    """Gera uma resposta baseada nos documentos recuperados usando templates adaptativos."""
    if not documentos:
        return "Nao encontrei documentos relevantes para esta pergunta. Por favor, reformule sua consulta ou forneca mais detalhes.", modelo_usado

    if len(documentos) > _MAX_CHUNKS_LLM:
        logger.warning(
            f"Recebidos {len(documentos)} chunks, truncando para "
            f"{_MAX_CHUNKS_LLM} (preservando prioritarios)"
        )
        documentos = _aplicar_limite_k_preservando_prioritarios(
            documentos, _MAX_CHUNKS_LLM
        )

    contextos = []
    documentos_info = {}

    tipos_documentos = set()
    contagem_por_tipo = {}

    for i, doc in enumerate(documentos):
        metadados = doc.metadata
        tipo = metadados.get('nome_tipo', 'Documento')
        tipos_documentos.add(tipo)

        contagem_por_tipo[tipo] = contagem_por_tipo.get(tipo, 0) + 1

        numero = metadados.get('numero', 'N/A')
        ano = metadados.get('ano', 'N/A')
        doc_id = f"{tipo} {numero}/{ano}"

        if doc_id not in documentos_info:
            documentos_info[doc_id] = {
                'tipo': tipo,
                'numero': numero,
                'ano': ano,
                'caminho': metadados.get('caminho', 'Nao especificado'),
                'trechos': []
            }

        documentos_info[doc_id]['trechos'].append({
            'chunk': metadados.get('chunk', 'N/A'),
            'total_chunks': metadados.get('total_chunks', 'N/A'),
            'conteudo': doc.page_content
        })

        contexto = f"""
[Documento: {doc_id} - Parte {metadados.get('chunk', 'N/A')}/{metadados.get('total_chunks', 'N/A')}]
Fonte: {metadados.get('caminho', 'Nao especificado')}
Conteudo:
{doc.page_content}
"""
        contextos.append(contexto)

    contexto_completo = _INSTRUCOES_COMPLETUDE + "\n\n" + "\n\n".join(contextos)

    # Guarda de tamanho: truncar contexto se exceder limite de chars
    if len(contexto_completo) > _MAX_CONTEXT_CHARS:
        logger.warning(
            f"Contexto com {len(contexto_completo)} chars excede limite de "
            f"{_MAX_CONTEXT_CHARS}. Truncando."
        )
        contexto_completo = contexto_completo[:_MAX_CONTEXT_CHARS]

    # Log detalhado dos chunks enviados ao LLM para diagnostico
    logger.info("=" * 60)
    logger.info("CHUNKS ENVIADOS AO LLM")
    logger.info(f"Total de chunks: {len(documentos)}")
    logger.info(f"Tamanho total do contexto: {len(contexto_completo)} chars")
    logger.info(f"Modelo usado: {modelo_usado}")
    for tipo, contagem in contagem_por_tipo.items():
        logger.info(f"  Tipo '{tipo}': {contagem} chunks")
    for i, doc in enumerate(documentos):
        m = doc.metadata
        preview = doc.page_content[:120].replace("\n", " ")
        logger.info(
            f"  chunk[{i}] {m.get('nome_tipo','?')} {m.get('numero','?')}/{m.get('ano','?')} "
            f"parte {m.get('chunk','?')}/{m.get('total_chunks','?')} "
            f"({len(doc.page_content)} chars) >> {preview}..."
        )
    logger.info("=" * 60)
    
    # Determinar o tipo de consulta (normalizando para evitar problemas com acentos/maiusculas)
    normalized_question = _normalize_text_ascii_lower(pergunta)
    normalized_types = [_normalize_text_ascii_lower(t) for t in list(tipos_documentos)]
    
    keywords_technical = [
        "parametro", "tecnico", "valor", "valores", "limite", "limites", "medida",
        "metodologia", "pavimento", "deflexao", "dadm", "iri", "ifi", "atrito",
        "indice", "fwd", "ensaio", "equipamento", "tolerancia", "faixa", "vdm",
        "maximo", "minimo", "conformidade", "desempenho",
    ]
    
    keywords_normative = [
        'resolucao', 'instrucao normativa', 'deliberacao', 'portaria', 'regulamento',
        'normativo', 'legal', 'direito', 'obrigacao', 'dever', 'prazo', 'penalidade',
        'lei', 'decreto', 'in'
    ]
    
    # Selecionar tipo de template baseado na consulta normalizada
    if any(k in normalized_question for k in keywords_technical) or \
       ('instrucao normativa' in " ".join(normalized_types) and 'parametro' in normalized_question):
        logger.info("Detectada consulta sobre parametros tecnicos")
        template_tipo = 'parametros'
    elif any(k in normalized_question for k in keywords_normative) or \
         any(t in ['resolucao', 'deliberacao', 'portaria', 'instrucao normativa', 'lei', 'decreto'] for t in normalized_types):
        logger.info("Detectada consulta sobre aspectos normativos/juridicos")
        template_tipo = 'normativa'
    else:
        logger.info("Usando template padrao de resposta")
        template_tipo = 'resposta'

    tem_tabela_md = any("| --- |" in doc.page_content for doc in documentos)
    tem_bloco_numerico = any(
        re.search(r"\d+[,.]?\d*E[+\-]?\d+", doc.page_content)
        or re.search(r"\d+\s+\d+\s+\d+", doc.page_content)
        for doc in documentos
    )
    pergunta_sobre_limites = any(
        k in normalized_question
        for k in ("limite", "limites", "valor", "valores", "maximo", "minimo", "tolerancia")
    )
    if (tem_tabela_md or tem_bloco_numerico) and (
        any(k in normalized_question for k in keywords_technical) or pergunta_sobre_limites
    ):
        if template_tipo == 'resposta':
            template_tipo = 'parametros'
            logger.info(
                "Dados tabulares no contexto + pergunta sobre limites: template parametros"
            )
    
    # Selecionar template adaptativo baseado no modelo
    template_escolhido = selecionar_template_adaptativo(template_tipo, modelo_usado)
    logger.info(f"Template selecionado: {template_tipo} para modelo {modelo_usado}")
    
    prompt = PromptTemplate(
        template=template_escolhido,
        input_variables=["context", "question"]
    )
    
    chain = prompt | llm
    
    try:
        resposta = chain.invoke({
            "context": contexto_completo,
            "question": pergunta
        })
        
        logger.info(f"🔍 DEBUG: Resposta recebida do LLM")
        logger.info(f"🔍 DEBUG: Tipo da resposta: {type(resposta)}")
        logger.info(f"🔍 DEBUG: Resposta tem 'content'? {hasattr(resposta, 'content')}")
        
        if hasattr(resposta, 'content'):
            conteudo_resposta = resposta.content
            logger.info(f"🔍 DEBUG: Tamanho do conteúdo: {len(conteudo_resposta)} caracteres")
            logger.info(f"🔍 DEBUG: Primeiros 200 chars: {conteudo_resposta[:200]}")
        else:
            logger.error(f"🔍 DEBUG: Resposta sem content: {resposta}")
            conteudo_resposta = str(resposta)
        
        if any(frase in conteudo_resposta.lower() for frase in 
              ["não encontrei informações", "não há informações", "não foi possível encontrar", 
               "não foram encontradas", "não disponível nos documentos"]):
            logger.info("Resposta inicial insatisfatória. Tentando extração agressiva de informações...")
            
            # Usar template de extração agressiva adaptativo
            template_extracao = selecionar_template_adaptativo('extracao', modelo_usado)
            logger.info(f"Usando template de extração agressiva para modelo {modelo_usado}")
            
            prompt_extracao = PromptTemplate(
                template=template_extracao,
                input_variables=["context", "question"]
            )
            
            chain_extracao = prompt_extracao | llm
            nova_resposta = chain_extracao.invoke({
                "context": contexto_completo,
                "question": pergunta
            })
            
            if len(nova_resposta.content) > 100 and not any(frase in nova_resposta.content.lower() 
                                                          for frase in ["não encontrei", "não foi possível"]):
                conteudo_resposta = nova_resposta.content
                logger.info("Usado resultado da extração agressiva")
            else:
                logger.info("Mantida resposta original")
        
        logger.info(f"🔍 DEBUG: Retornando resposta final com {len(conteudo_resposta)} caracteres")
        return conteudo_resposta, modelo_usado
        
    except Exception as e:
        error_msg = str(e).lower()
        logger.error(f"Erro ao gerar resposta: {str(e)}")
        
        # Verificar se é erro de cota OpenAI e tentar fallback para DeepSeek
        if any(keyword in error_msg for keyword in ["insufficient_quota", "429", "quota", "exceeded"]) and "openai" in modelo_usado.lower():
            logger.warning(f"🚨 OpenAI com cota excedida durante geração de resposta. Tentando DeepSeek...")
            
            try:
                # Criar LLM DeepSeek para fallback
                from llm_providers import create_llm_manager
                deepseek_manager = create_llm_manager("deepseek")
                deepseek_llm = deepseek_manager.get_llm(temperature=0.1, max_tokens=2048)
                
                logger.info("✅ DeepSeek configurado com sucesso para fallback")
                
                # Tentar com DeepSeek usando o MESMO tipo de template da consulta original
                template_deepseek = selecionar_template_adaptativo(template_tipo, 'deepseek')
                prompt_deepseek = PromptTemplate(
                    template=template_deepseek,
                    input_variables=["context", "question"]
                )
                
                chain_deepseek = prompt_deepseek | deepseek_llm
                resposta_deepseek = chain_deepseek.invoke({
                    "context": contexto_completo,
                    "question": pergunta
                })
                
                logger.info("✅ Resposta gerada com DeepSeek (fallback)")
                return f"🔄 **Resposta gerada com DeepSeek (OpenAI indisponível por limite de cota)**\n\n{resposta_deepseek.content}", "deepseek"
                
            except Exception as e_deepseek:
                logger.error(f"❌ Fallback DeepSeek também falhou: {str(e_deepseek)}")
                # Continuar para o fallback original abaixo
        
        try:
            # Fallback com template de extração agressiva adaptativo
            template_fallback = selecionar_template_adaptativo('extracao', modelo_usado)
            prompt_fallback = PromptTemplate(
                template=template_fallback,
                input_variables=["context", "question"]
            )
            
            chain_fallback = prompt_fallback | llm
            resposta_fallback = chain_fallback.invoke({
                "context": contexto_completo,
                "question": pergunta
            })
            
            return resposta_fallback.content, modelo_usado
        except Exception as e2:
            error_msg2 = str(e2).lower()
            logger.error(f"Erro no fallback: {str(e2)}")
            
            if any(keyword in error_msg2 for keyword in ["insufficient_quota", "429", "quota", "exceeded"]):
                try:
                    from llm_providers import create_llm_manager
                    deepseek_manager = create_llm_manager("deepseek")
                    deepseek_llm = deepseek_manager.get_llm(temperature=0.1, max_tokens=2048)

                    # Usar template adaptativo correto com contexto truncado
                    _EMERGENCIA_MAX_CHARS = 8000
                    contexto_truncado = contexto_completo[:_EMERGENCIA_MAX_CHARS]
                    template_emergencia = selecionar_template_adaptativo(
                        template_tipo, "deepseek"
                    )

                    prompt_emergencia = PromptTemplate(
                        template=template_emergencia,
                        input_variables=["context", "question"],
                    )

                    chain_emergencia = prompt_emergencia | deepseek_llm
                    resposta_emergencia = chain_emergencia.invoke({
                        "context": contexto_truncado,
                        "question": pergunta,
                    })

                    logger.info("Resposta de emergencia gerada com DeepSeek (template adaptativo + contexto truncado)")
                    return (
                        f"Resposta de emergencia (OpenAI indisponivel, contexto reduzido)\n\n"
                        f"{resposta_emergencia.content}"
                    ), "deepseek"

                except Exception as e3:
                    logger.error(f"Resposta de emergencia falhou: {e3}")
            
            return f"❌ Não foi possível gerar uma resposta devido a limitações temporárias da API. Tente novamente em alguns minutos ou use o modo de embeddings locais.", modelo_usado

def _preparar_contexto_resposta(pergunta, documentos, modelo_usado="gpt-4"):
    """
    Prepara contexto, seleciona template e monta o prompt para geracao de resposta.

    Compartilha a logica de construcao de contexto entre gerar_resposta e
    gerar_resposta_streaming para evitar duplicacao.

    Args:
        pergunta: Pergunta do usuario.
        documentos: Lista de Document recuperados.
        modelo_usado: Identificador do provedor (gpt-4, deepseek, etc.).

    Returns:
        Tuple (prompt_formatado: str, template_tipo: str) prontos para envio ao LLM.
        Retorna (None, None) se documentos estiver vazio.
    """
    if not documentos:
        return None, None

    if len(documentos) > _MAX_CHUNKS_LLM:
        logger.warning(
            f"Recebidos {len(documentos)} chunks, truncando para "
            f"{_MAX_CHUNKS_LLM} (preservando prioritarios)"
        )
        documentos = _aplicar_limite_k_preservando_prioritarios(
            documentos, _MAX_CHUNKS_LLM
        )

    contextos = []
    tipos_documentos = set()
    contagem_por_tipo = {}

    for i, doc in enumerate(documentos):
        metadados = doc.metadata
        tipo = metadados.get("nome_tipo", "Documento")
        tipos_documentos.add(tipo)
        contagem_por_tipo[tipo] = contagem_por_tipo.get(tipo, 0) + 1
        numero = metadados.get("numero", "N/A")
        ano = metadados.get("ano", "N/A")
        doc_id = f"{tipo} {numero}/{ano}"
        contexto = (
            f"\n[Documento: {doc_id} - Parte "
            f"{metadados.get('chunk', 'N/A')}/{metadados.get('total_chunks', 'N/A')}]\n"
            f"Fonte: {metadados.get('caminho', 'Nao especificado')}\n"
            f"Conteudo:\n{doc.page_content}\n"
        )
        contextos.append(contexto)

    contexto_completo = _INSTRUCOES_COMPLETUDE + "\n\n" + "\n\n".join(contextos)

    if len(contexto_completo) > _MAX_CONTEXT_CHARS:
        logger.warning(
            f"Contexto com {len(contexto_completo)} chars excede limite de "
            f"{_MAX_CONTEXT_CHARS}. Truncando."
        )
        contexto_completo = contexto_completo[:_MAX_CONTEXT_CHARS]

    logger.info("=" * 60)
    logger.info("CHUNKS ENVIADOS AO LLM")
    logger.info(f"Total de chunks: {len(documentos)}")
    logger.info(f"Tamanho total do contexto: {len(contexto_completo)} chars")
    logger.info(f"Modelo usado: {modelo_usado}")
    for tipo, contagem in contagem_por_tipo.items():
        logger.info(f"  Tipo '{tipo}': {contagem} chunks")
    for i, doc in enumerate(documentos):
        m = doc.metadata
        preview = doc.page_content[:120].replace("\n", " ")
        logger.info(
            f"  chunk[{i}] {m.get('nome_tipo','?')} {m.get('numero','?')}/{m.get('ano','?')} "
            f"parte {m.get('chunk','?')}/{m.get('total_chunks','?')} "
            f"({len(doc.page_content)} chars) >> {preview}..."
        )
    logger.info("=" * 60)

    normalized_question = _normalize_text_ascii_lower(pergunta)
    normalized_types = [_normalize_text_ascii_lower(t) for t in list(tipos_documentos)]

    keywords_technical = [
        "parametro", "tecnico", "valor", "valores", "limite", "limites", "medida",
        "metodologia", "pavimento", "deflexao", "dadm", "iri", "ifi", "atrito",
        "indice", "fwd", "ensaio", "equipamento", "tolerancia", "faixa", "vdm",
        "maximo", "minimo", "conformidade", "desempenho",
    ]
    keywords_normative = [
        "resolucao", "instrucao normativa", "deliberacao", "portaria", "regulamento",
        "normativo", "legal", "direito", "obrigacao", "dever", "prazo", "penalidade",
        "lei", "decreto", "in",
    ]

    if any(k in normalized_question for k in keywords_technical) or \
       ("instrucao normativa" in " ".join(normalized_types) and "parametro" in normalized_question):
        logger.info("Detectada consulta sobre parametros tecnicos")
        template_tipo = "parametros"
    elif any(k in normalized_question for k in keywords_normative) or \
         any(t in ["resolucao", "deliberacao", "portaria", "instrucao normativa", "lei", "decreto"] for t in normalized_types):
        logger.info("Detectada consulta sobre aspectos normativos/juridicos")
        template_tipo = "normativa"
    else:
        logger.info("Usando template padrao de resposta")
        template_tipo = "resposta"

    tem_tabela_md = any("| --- |" in doc.page_content for doc in documentos)
    tem_bloco_numerico = any(
        re.search(r"\d+[,.]?\d*E[+\-]?\d+", doc.page_content)
        or re.search(r"\d+\s+\d+\s+\d+", doc.page_content)
        for doc in documentos
    )
    pergunta_sobre_limites = any(
        k in normalized_question
        for k in ("limite", "limites", "valor", "valores", "maximo", "minimo", "tolerancia")
    )
    if (tem_tabela_md or tem_bloco_numerico) and (
        any(k in normalized_question for k in keywords_technical) or pergunta_sobre_limites
    ):
        if template_tipo == "resposta":
            template_tipo = "parametros"
            logger.info(
                "Dados tabulares no contexto + pergunta sobre limites: template parametros"
            )

    template_escolhido = selecionar_template_adaptativo(template_tipo, modelo_usado)
    logger.info(f"Template selecionado: {template_tipo} para modelo {modelo_usado}")

    prompt = PromptTemplate(
        template=template_escolhido,
        input_variables=["context", "question"],
    )
    prompt_formatado = prompt.format(context=contexto_completo, question=pergunta)
    return prompt_formatado, template_tipo


def gerar_resposta_streaming(pergunta, documentos, llm, modelo_usado="gpt-4"):
    """
    Gera resposta via streaming (yield de tokens).

    Retorna um generator de strings para uso com st.write_stream.
    Ao final, o texto completo pode ser acessado via atributo .texto_completo
    do generator (apos consumo total).

    Args:
        pergunta: Pergunta do usuario.
        documentos: Lista de Document.
        llm: Instancia do LLM LangChain.
        modelo_usado: Identificador do provedor.

    Yields:
        str: Tokens incrementais da resposta.
    """
    if not documentos:
        yield "Nao encontrei documentos relevantes para esta pergunta."
        return

    prompt_formatado, template_tipo = _preparar_contexto_resposta(
        pergunta, documentos, modelo_usado
    )
    if prompt_formatado is None:
        yield "Nao encontrei documentos relevantes para esta pergunta."
        return

    from langchain_core.messages import HumanMessage

    try:
        for chunk in llm.stream([HumanMessage(content=prompt_formatado)]):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                yield token
    except Exception as exc:
        logger.error(f"Erro durante streaming: {exc}")
        yield f"\n\n[Erro durante geracao: {exc}]"


def extrair_citacoes_da_resposta(resposta):
    """Extrai as citações de documentos da resposta gerada."""
    padroes = [
        r"Documento:\s*([A-Za-zçÇáàâãéèêíìîóòôõúùûÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛ]+\s+\d+\/\d{4})",
        r"Documento:\s*([A-Za-zçÇáàâãéèêíìîóòôõúùûÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛ]+\s+\d+\s+de\s+\d{4})",
        r"TRECHOS DOS DOCUMENTOS CITADOS:[\s\S]*?(?:\*\*)?Documento:\s*([^\n\"]*\d{4})(?:\*\*)?",
        r"TRECHOS[\s\S]+?(?:\*\*)?([A-Za-zçÇáàâãéèêíìîóòôõúùûÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛ]+\s+\d+\/\d{4})(?:\*\*)?",
        r"(?:^|\s)(?:a\s+)?([Ii]nstrução\s+[Nn]ormativa\s+(?:n[°º\.]?\s*)?\d+\/?\d*\s*(?:de\s*)?\d{4})",
        r"(?:^|\s)(?:a\s+)?([Rr]esolução\s+(?:n[°º\.]?\s*)?\d+\/?\d*\s*(?:de\s*)?\d{4})",
        r"(?:^|\s)(?:o\s+)?([Vv]oto\s+(?:n[°º\.]?\s*)?\d+\/?\d*\s*(?:de\s*)?\d{4})",
        r"(?:^|\s)(?:a\s+)?([Dd]eliberação\s+(?:n[°º\.]?\s*)?\d+\/?\d*\s*(?:de\s*)?\d{4})",
        r"(?:^|\s)(?:a\s+)?([Pp]ortaria\s+(?:n[°º\.]?\s*)?\d+\/?\d*\s*(?:de\s*)?\d{4})",
        r"([A-Za-zçÇáàâãéèêíìîóòôõúùûÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛ]+\s+0+(\d+)\/\d{4})",
        r"FONTES CONSULTADAS[\s\S]*?(?:\*\*)?([^\n\"]*\d{4})(?:\*\*)?",
    ]
    
    secoes_trechos = [
        r"TRECHOS DOS DOCUMENTOS CITADOS:([\s\S]+?)(?:$|(?:###|\*\*\*))",
        r"FONTES CONSULTADAS:([\s\S]+?)(?:$|(?:###|\*\*\*))"
    ]
    
    citacoes = []
    
    for padrao_secao in secoes_trechos:
        match_secao = re.search(padrao_secao, resposta, re.IGNORECASE)
        if match_secao:
            secao_trechos = match_secao.group(1)
            for padrao in padroes:
                matches = re.findall(padrao, secao_trechos, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    citacao = match.strip()
                    if citacao and citacao not in citacoes:
                        citacoes.append(citacao)
    
    if not citacoes:
        for padrao in padroes:
            matches = re.findall(padrao, resposta, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                citacao = match.strip()
                if citacao and citacao not in citacoes:
                    citacoes.append(citacao)
    
    citacoes_normalizadas = []
    for citacao in citacoes:
        citacao = re.sub(r'[*_`]', '', citacao)
        citacao = re.sub(r'\s+', ' ', citacao).strip()
        if citacao and citacao not in citacoes_normalizadas:
            citacoes_normalizadas.append(citacao)
    
    logger.info(f"Citações encontradas: {citacoes_normalizadas}")
    return citacoes_normalizadas 

def interface_usuario_unificada():
    """Interface principal do sistema RAG unificado."""
    st.set_page_config(
        page_title=STREAMLIT_PAGE_TITLE,
        page_icon=STREAMLIT_PAGE_ICON,
        layout=STREAMLIT_LAYOUT,
        initial_sidebar_state="expanded"
    )

    # Inicializar historico de conversa no session_state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Historico visual do chat (pares pergunta/resposta para exibicao)
    if "mensagens_chat" not in st.session_state:
        st.session_state.mensagens_chat = []
    
    # CSS personalizado
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .provider-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .status-ok { background-color: #d4edda; color: #155724; }
    .status-error { background-color: #f8d7da; color: #721c24; }
    .status-warning { background-color: #fff3cd; color: #856404; }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .citation-box {
        background: #e9ecef;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>🚛 Sistema RAG Unificado - ANTT</h1>
        <p>Consulte regulamentações da ANTT com múltiplos provedores de IA</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar para configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Seleção do provedor
        st.subheader("🤖 Provedor de IA")
        providers = get_available_providers()
        provider_names = {
            "openai": "⚠️ OpenAI (GPT-4) - Limitado por Cota",
            "deepseek": "✅ DeepSeek (Recomendado)"
        }
        
        # Tornar DeepSeek padrão
        default_index = 0
        if "deepseek" in providers:
            default_index = list(providers.keys()).index("deepseek")
        
        selected_provider = st.selectbox(
            "Escolha o provedor:",
            options=list(providers.keys()),
            format_func=lambda x: provider_names.get(x, x),
            index=default_index,
            help="DeepSeek é recomendado por não ter limitações de cota"
        )
        
        # Seleção do modelo
        if selected_provider in providers:
            available_models = providers[selected_provider]["models"]
            selected_model = st.selectbox(
                "Modelo:",
                options=available_models,
                index=0
            )
        else:
            selected_model = "gpt-4"
        
        # Informação sobre fallback automático
        if selected_provider == "openai":
            st.warning("⚠️ **Fallback Automático Ativo**: Se OpenAI exceder cota, DeepSeek será usado automaticamente")
        elif selected_provider == "deepseek":
            st.success("✅ **Provedor Confiável**: DeepSeek não possui limitações de cota")
        
        # Status das APIs
        st.subheader("🔑 Status das APIs")
        
        # Verificar status do OpenAI
        try:
            openai_manager = create_llm_manager("openai")
            st.markdown('<div class="provider-status status-ok">✅ OpenAI: Conectado</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="provider-status status-error">❌ OpenAI: {str(e)[:50]}...</div>', unsafe_allow_html=True)
        
        # Verificar status do DeepSeek
        try:
            deepseek_manager = create_llm_manager("deepseek")
            st.markdown('<div class="provider-status status-ok">✅ DeepSeek: Conectado</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="provider-status status-error">❌ DeepSeek: {str(e)[:50]}...</div>', unsafe_allow_html=True)
        
        # Configurações avançadas
        st.subheader("🔧 Configurações Avançadas")
        
        temperatura = st.slider(
            "Temperatura:",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Controla a criatividade das respostas"
        )
        
        # Seleção do provedor de embeddings
        st.markdown("**🔤 Provedor de Embeddings**")
        embedding_providers = get_available_embedding_providers()
        
        # Criar lista de opções com descrições claras
        embedding_options = list(embedding_providers.keys())
        embedding_labels = [embedding_providers[key]["name"] for key in embedding_options]
        
        selected_embedding_provider = st.selectbox(
            "Escolha o provedor de embeddings:",
            options=embedding_options,
            format_func=lambda x: embedding_providers[x]["name"],
            index=0,  # Padrão é local (realmente gratuito)
            help="⚠️ IMPORTANTE: Local é 100% gratuito, OpenAI consome créditos pagos"
        )
        
        # Mostrar descrição detalhada da opção selecionada
        selected_info = embedding_providers[selected_embedding_provider]
        
        if selected_embedding_provider == "local":
            st.success(f"✅ **{selected_info['name']}**")
            st.info("🎉 **100% GRATUITO** - Usa sentence-transformers offline")
        elif selected_embedding_provider == "openai":
            st.warning(f"💰 **{selected_info['name']}**")
            st.warning("⚠️ **CONSOME CRÉDITOS** - Cada busca usa tokens pagos da OpenAI")
        elif selected_embedding_provider == "free":
            st.info(f"⚡ **{selected_info['name']}**")
            st.info("🔄 Tenta local primeiro (gratuito), fallback para OpenAI se necessário")
        
        # Mostrar status da dependência
        if selected_embedding_provider == "local":
            try:
                import sentence_transformers
                st.success("🔑 sentence-transformers instalado - Pronto para uso gratuito!")
            except ImportError:
                st.error("❌ sentence-transformers não encontrado - Execute: pip install sentence-transformers")
        elif selected_embedding_provider == "openai":
            try:
                if get_openai_api_key():
                    st.success("🔑 Chave OpenAI configurada")
                    st.warning("💸 Lembre-se: cada consulta consome créditos!")
                else:
                    st.error("❌ Chave OpenAI não encontrada")
            except:
                st.warning("⚠️ Verificação de chave OpenAI falhou")
        
        st.divider()

        btn_col_a, btn_col_b = st.columns(2)
        with btn_col_a:
            if st.button("Nova Conversa", use_container_width=True,
                          help="Limpa o histórico de conversa para iniciar um novo tema"):
                st.session_state.chat_history = []
                if "mensagens_chat" in st.session_state:
                    st.session_state.mensagens_chat = []
                st.rerun()

        with btn_col_b:
            if st.button("Reindexar Base", use_container_width=True,
                          help="Regenera o catálogo e reconstrói o vectorstore com todos os documentos de dados_antt/"):
                st.session_state["_reindexando"] = True
                st.rerun()

        if st.button(
            "Limpar Cache OCR e Reindexar",
            use_container_width=True,
            help=(
                "Remove todo o cache de OCR e forca a re-extracao "
                "das imagens com o pipeline v3 (multi-estrategia, "
                "validacao de qualidade, correcao de tabelas). "
                "Usar quando tabelas tiverem dados numericos incorretos."
            ),
        ):
            st.session_state["_limpar_ocr_e_reindexar"] = True
            st.rerun()

        # Executar limpeza OCR + reindexacao
        if st.session_state.get("_limpar_ocr_e_reindexar"):
            del st.session_state["_limpar_ocr_e_reindexar"]
            import glob as _glob_mod

            cache_dir = _OCR_CACHE_DIR
            cache_files = _glob_mod.glob(os.path.join(cache_dir, "*.txt"))
            for f in cache_files:
                try:
                    os.remove(f)
                except OSError:
                    pass
            st.info(
                f"Cache OCR limpo ({len(cache_files)} arquivo(s) removidos). "
                "Reindexando com re-extração..."
            )
            st.session_state["_reindexando"] = True

        # Executar reindexacao (precisa estar fora do button para manter o spinner)
        if st.session_state.get("_reindexando"):
            del st.session_state["_reindexando"]
            with st.spinner("Reindexando base de conhecimento... (pode levar alguns minutos, dependendo da quantidade de documentos)"):
                sucesso, msg = reindexar_base_completa(
                    embedding_provider=selected_embedding_provider
                )
            if sucesso:
                st.success(msg)
                # Limpar cache de verificacao para que o alerta desapareca
                st.session_state.pop("_docs_novos_checado", None)
                st.session_state.pop("_docs_novos_lista", None)
                st.session_state.pop("_vectorstore_desatualizado", None)
                st.balloons()
            else:
                st.error(msg)

        st.divider()

        max_tokens = st.number_input(
            "Maximo de tokens:",
            min_value=500,
            max_value=4096,
            value=2048,
            step=256,
            help="Limite de tokens para a resposta gerada pelo LLM. "
                 "Valores altos consomem mais creditos no OpenRouter. "
                 "Recomendado: 2048-3072 para respostas detalhadas."
        )
        
        num_documentos = st.slider(
            "Documentos para busca:",
            min_value=5,
            max_value=40,
            value=20,
            help="Numero de chunks recuperados da base. Valores maiores trazem "
                 "mais contexto, mas podem diluir a relevancia."
        )
        
        # Filtros de busca
        st.subheader("🔍 Filtros de Busca")
        
        tipo_documento = st.selectbox(
            "Tipo de documento:",
            options=["Todos", "RES", "INM", "DLB", "POR", "INC"],
            help="Filtrar por tipo de documento"
        )
        
        ano_filtro = st.selectbox(
            "Ano:",
            options=["Todos"] + [str(year) for year in range(2024, 2019, -1)],
            help="Filtrar por ano do documento"
        )
        
        numero_filtro = st.text_input(
            "Número do documento:",
            placeholder="Ex: 6057",
            help="Filtrar por número específico"
        )
    
    # Carregar vectorstore primeiro (fora dos containers)
    try:
        vectorstore = carregar_vectorstore_com_provider(selected_embedding_provider)
        vectorstore_status = "Sistema Pronto"
        vectorstore_loaded = True
    except Exception as e:
        logger.error(f"Erro ao carregar vectorstore: {str(e)}")
        vectorstore = None
        vectorstore_status = "Sistema Indisponivel"
        vectorstore_loaded = False

    # Verificacao automatica de documentos novos na inicializacao
    # Usa cache no session_state para nao repetir a varredura a cada rerun
    if "_docs_novos_checado" not in st.session_state:
        docs_novos = detectar_documentos_novos()

        # Verificar se o vectorstore esta desatualizado comparando
        # timestamps: se relatorio_documentos.json e mais recente que
        # o vectorstore, significa que novos docs foram catalogados
        # mas ainda nao indexados.
        vectorstore_desatualizado = False
        try:
            catalogo_path = "relatorio_documentos.json"
            vs_index_path = os.path.join("vectorstore_local", "index.faiss")
            if not vectorstore_loaded:
                vectorstore_desatualizado = True
            elif os.path.exists(catalogo_path) and os.path.exists(vs_index_path):
                ts_catalogo = os.path.getmtime(catalogo_path)
                ts_vectorstore = os.path.getmtime(vs_index_path)
                if ts_catalogo > ts_vectorstore:
                    vectorstore_desatualizado = True
                    logger.info(
                        "Vectorstore desatualizado: catalogo mais recente que o indice"
                    )
        except Exception:
            pass

        st.session_state["_docs_novos_checado"] = True
        st.session_state["_docs_novos_lista"] = docs_novos
        st.session_state["_vectorstore_desatualizado"] = vectorstore_desatualizado
    else:
        docs_novos = st.session_state.get("_docs_novos_lista", [])
        vectorstore_desatualizado = st.session_state.get("_vectorstore_desatualizado", False)

    if docs_novos:
        st.warning(
            f"Detectados **{len(docs_novos)}** documento(s) novo(s) em `dados_antt/` "
            f"que ainda não foram indexados na base de conhecimento. "
            f"Clique em **Reindexar Base** na barra lateral para atualizar."
        )
        with st.expander(f"Ver {len(docs_novos)} documento(s) pendente(s)"):
            for nome in docs_novos:
                st.text(nome)
    elif vectorstore_desatualizado:
        st.warning(
            "O vectorstore parece estar desatualizado em relação ao catálogo de documentos. "
            "Clique em **Reindexar Base** na barra lateral para reconstruir."
        )

    # Layout principal em duas colunas
    col_main, col_info = st.columns([2, 1])
    
    with col_main:
        # Exibir historico visual do chat
        if st.session_state.mensagens_chat:
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.mensagens_chat:
                    with st.chat_message("user"):
                        st.markdown(msg["pergunta"])
                    with st.chat_message("assistant"):
                        st.markdown(msg["resposta"])
                        if msg.get("provider"):
                            st.caption(f"Resposta via {msg['provider']}")
            st.divider()

        # Verificar se ha um exemplo selecionado no session_state
        valor_inicial = ""
        if "pergunta_exemplo" in st.session_state:
            valor_inicial = st.session_state.pergunta_exemplo

        # Campo de pergunta principal
        pergunta = st.text_area(
            "Digite sua pergunta sobre documentos da ANTT:",
            value=valor_inicial,
            height=100,
            help="Digite sua consulta sobre regulamentações, normas ou procedimentos da ANTT"
        )
        
        # Botões de ação
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([2, 2, 2, 2])
        
        with btn_col1:
            consultar = st.button("🔍 Consultar", type="primary", use_container_width=True)
        
        with btn_col2:
            limpar = st.button("🗑️ Limpar", use_container_width=True)
        
        with btn_col3:
            exemplos = st.button("💡 Exemplos", use_container_width=True)
        
        with btn_col4:
            if vectorstore_loaded:
                st.success(vectorstore_status)
            else:
                st.error(vectorstore_status)

    with col_info:
        st.subheader("📊 Informações")
        
        # Mostrar aviso se usando embeddings gratuitos
        if selected_embedding_provider == "local":
            st.success("🎉 **Modo 100% Gratuito Ativo**: Sistema usando embeddings locais offline - sem custos!")
        elif selected_embedding_provider == "free":
            st.info("⚡ **Modo Automático**: Sistema tentará usar embeddings locais primeiro, com fallback para OpenAI se necessário")
        elif selected_embedding_provider == "openai":
            st.warning("💰 **Modo Pago Ativo**: Sistema usando embeddings OpenAI - consome créditos a cada consulta")
        
        # Estatísticas básicas
        if vectorstore_loaded:
            # Verificar qual vectorstore foi carregado
            vectorstore_info = "📚 Base de Conhecimento Carregada"
            if hasattr(vectorstore, '_embedding_provider'):
                provider = vectorstore._embedding_provider
                path = getattr(vectorstore, '_vectorstore_path', 'N/A')
                
                if provider == "local":
                    vectorstore_info = "🆓 **Vectorstore Local** (Gratuito)"
                    st.success("✅ Usando base de conhecimento com embeddings locais")
                elif provider == "openai":
                    vectorstore_info = "💰 **Vectorstore OpenAI** (Pago)"
                    st.warning("⚠️ Usando base de conhecimento com embeddings pagos")
                else:
                    vectorstore_info = f"📚 **Vectorstore {provider.title()}**"
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>{vectorstore_info}</h4>
                <p>✅ Sistema operacional</p>
                <p>🔍 Busca semântica ativa</p>
                <p>🤖 IA: """ + provider_names.get(selected_provider, selected_provider) + """</p>
                <p>🔤 Embeddings: """ + embedding_providers.get(selected_embedding_provider, {}).get("name", selected_embedding_provider) + """</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("❌ Erro ao carregar base de conhecimento")
    
    # Verificar se deve processar automaticamente um exemplo
    processar_exemplo_automatico = False
    if "processar_automatico" in st.session_state and st.session_state.processar_automatico:
        processar_exemplo_automatico = True
        # Limpar o flag após usar
        del st.session_state.processar_automatico

    # Processamento da consulta
    if (consultar and pergunta and vectorstore_loaded) or (processar_exemplo_automatico and vectorstore_loaded):
        # Para processamento automático, usar a pergunta do session_state se disponível
        pergunta_para_processar = pergunta
        if processar_exemplo_automatico and "pergunta_exemplo" in st.session_state:
            pergunta_para_processar = st.session_state.pergunta_exemplo
        
        # Verificar se temos uma pergunta válida
        if not pergunta_para_processar or pergunta_para_processar.strip() == "":
            st.error("❌ Nenhuma pergunta fornecida para processamento")
        else:
            # Mostrar indicação se é processamento automático
            if processar_exemplo_automatico:
                st.info(f"🚀 **Processamento Automático**: Executando exemplo selecionado")
                st.markdown(f"**Pergunta:** {pergunta_para_processar}")
            
            # Limpar a pergunta do exemplo após usar
            if "pergunta_exemplo" in st.session_state:
                del st.session_state.pergunta_exemplo
            
            with st.spinner("🔍 Processando consulta..."):
                try:
                    # Tentar criar LLM manager com o provedor selecionado
                    llm_manager = None
                    llm = None
                    provider_usado = selected_provider
                    
                    try:
                        llm_manager = create_llm_manager(selected_provider, selected_model)
                        llm = llm_manager.get_llm(temperature=temperatura, max_tokens=max_tokens)
                        st.info(f"✅ Usando {provider_names.get(selected_provider, selected_provider)}")
                    except Exception as e:
                        error_msg = str(e).lower()
                        if any(keyword in error_msg for keyword in ["insufficient_quota", "429", "quota", "exceeded"]):
                            st.warning(f"⚠️ {provider_names.get(selected_provider, selected_provider)} com problema de cota. Tentando DeepSeek...")
                            
                            # Fallback para DeepSeek
                            try:
                                llm_manager = create_llm_manager("deepseek")
                                llm = llm_manager.get_llm(temperature=temperatura, max_tokens=max_tokens)
                                provider_usado = "deepseek"
                                st.success("✅ Usando DeepSeek como alternativa")
                            except Exception as e2:
                                st.error(f"❌ Erro ao usar DeepSeek: {str(e2)}")
                                raise e2
                        else:
                            st.error(f"❌ Erro ao configurar {provider_names.get(selected_provider, selected_provider)}: {str(e)}")
                            raise e
                    
                    if llm is None:
                        st.error("❌ Não foi possível configurar nenhum provedor de IA")
                        return
                    
                    # Aplicar filtros
                    filtro_tipo = None if tipo_documento == "Todos" else tipo_documento
                    filtro_ano = None if ano_filtro == "Todos" else ano_filtro
                    filtro_numero = numero_filtro if numero_filtro.strip() else None
                    
                    # Reescrever query se houver historico de conversa
                    pergunta_original = pergunta_para_processar
                    if st.session_state.chat_history:
                        pergunta_para_processar = _reescrever_query_com_historico(
                            pergunta_para_processar,
                            st.session_state.chat_history,
                            llm,
                        )
                        if pergunta_para_processar != pergunta_original:
                            st.info(
                                f"Pergunta contextualizada: **{pergunta_para_processar}**"
                            )

                    # Buscar documentos
                    documentos = pesquisar_documentos(
                        pergunta_para_processar,
                        vectorstore,
                        k=num_documentos,
                        tipo_documento=filtro_tipo,
                        ano=filtro_ano,
                        numero=filtro_numero,
                        embedding_provider=selected_embedding_provider
                    )
                    
                    if documentos:
                        logger.info(f"DEBUG: Gerando resposta com {len(documentos)} documentos")
                        modelo_usado_final = provider_usado
                        provider_label = provider_names.get(
                            modelo_usado_final, modelo_usado_final
                        )

                        with st.chat_message("user"):
                            st.markdown(pergunta_original)

                        with st.chat_message("assistant"):
                            stream_gen = gerar_resposta_streaming(
                                pergunta_para_processar, documentos, llm, provider_usado
                            )
                            resposta = st.write_stream(stream_gen)

                        if not resposta or not resposta.strip():
                            resposta, modelo_usado_final = gerar_resposta(
                                pergunta_para_processar, documentos, llm, provider_usado
                            )
                            provider_label = provider_names.get(
                                modelo_usado_final, modelo_usado_final
                            )
                            with st.chat_message("assistant"):
                                st.markdown(resposta)

                        logger.info(
                            f"DEBUG: Resposta gerada - Tamanho: "
                            f"{len(resposta) if resposta else 0} caracteres"
                        )

                        # Salvar no historico de conversa para reescrita (max 5 turnos)
                        st.session_state.chat_history.append({
                            "pergunta": pergunta_original,
                            "resposta": resposta if resposta else "",
                        })
                        if len(st.session_state.chat_history) > 5:
                            st.session_state.chat_history = st.session_state.chat_history[-5:]

                        # Salvar no historico visual do chat
                        st.session_state.mensagens_chat.append({
                            "pergunta": pergunta_original,
                            "resposta": resposta if resposta else "(sem resposta)",
                            "provider": provider_label,
                        })

                        # Mostrar qual provedor foi usado
                        st.success(f"Resposta gerada com {provider_label}")
                        
                        # Extrair e exibir citações
                        citacoes = extrair_citacoes_da_resposta(resposta)
                        if citacoes:
                            st.markdown("""
                            <div class="citation-box">
                                <h4>📚 Documentos Citados</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for citacao in citacoes:
                                st.markdown(f"• {citacao}")
                        
                        # Exibir trechos dos documentos citados
                        st.markdown("---")
                        st.markdown("### 📄 Trechos dos Documentos Citados")
                        
                        # Extrair citações de documentos da resposta
                        documentos_citados = extrair_citacoes_da_resposta(resposta)
                        
                        # Exibir trechos relevantes para os documentos citados
                        documentos_encontrados = False
                        
                        if documentos_citados:
                            # Criar mapeamento de documentos para facilitar a busca
                            docs_por_id = {}
                            for doc in documentos:
                                meta = doc.metadata
                                tipo = meta.get('nome_tipo', 'Documento')
                                numero = meta.get('numero', 'N/A')
                                ano = meta.get('ano', 'N/A')
                                doc_id = f"{tipo} {numero}/{ano}"
                                
                                # Normalizar o ID para comparação (tudo minúsculo, sem espaços extras)
                                doc_id_norm = re.sub(r'\s+', ' ', doc_id.lower()).strip()
                                
                                # Criar versões alternativas do ID para melhorar correspondências
                                alternativas = [
                                    doc_id_norm,
                                    f"{tipo.lower()} {numero}",
                                    f"{tipo.lower()} {numero.lstrip('0')}/{ano}",
                                    f"{tipo.lower()} {numero.lstrip('0')} de {ano}"
                                ]
                                
                                for alt_id in alternativas:
                                    if alt_id not in docs_por_id:
                                        docs_por_id[alt_id] = []
                                    docs_por_id[alt_id].append((doc, meta, doc_id))
                            
                            # Exibir documentos citados
                            documentos_exibidos = set()  # Para evitar duplicações
                            
                            for doc_citado in documentos_citados:
                                # Normalizar a citação
                                doc_citado_norm = re.sub(r'\s+', ' ', doc_citado.lower()).strip()
                                doc_encontrado = False
                                
                                # Tentar diferentes variações da citação para encontrar correspondência
                                for doc_key, doc_list in docs_por_id.items():
                                    # Verificar se a citação corresponde a alguma parte da chave
                                    if (doc_citado_norm in doc_key or 
                                        doc_key in doc_citado_norm or
                                        any(term in doc_citado_norm for term in doc_key.split())):
                                        
                                        for doc, meta, doc_id in doc_list:
                                            doc_display_id = f"{doc_id}_{meta.get('chunk', '')}"
                                            
                                            # Verificar se já exibimos este documento
                                            if doc_display_id in documentos_exibidos:
                                                continue
                                            
                                            # Marcar como encontrado
                                            doc_encontrado = True
                                            documentos_encontrados = True
                                            documentos_exibidos.add(doc_display_id)
                                            
                                            with st.container(border=True):
                                                # Adicionar uma indicação visual da cor
                                                if "Instrução" in doc_id:
                                                    st.markdown("🔵 **Instrução Normativa**")
                                                elif "Resolução" in doc_id:
                                                    st.markdown("🟠 **Resolução**")
                                                elif "Voto" in doc_id:
                                                    st.markdown("🟣 **Voto**")
                                                elif "Deliberação" in doc_id:
                                                    st.markdown("🟡 **Deliberação**")
                                                else:
                                                    st.markdown("🟢 **Documento**")
                                                    
                                                st.markdown(f"#### {doc_id}")
                                                st.caption(f"Parte {meta.get('chunk', 'N/A')}/{meta.get('total_chunks', 'N/A')} • Fonte: `{meta.get('caminho', 'Não especificado')}`")
                                                st.text_area(
                                                    "Trecho do documento:",
                                                    doc.page_content,
                                                    height=150,
                                                    key=f"citacao_{doc_display_id}",
                                                    disabled=True
                                                )
                                
                                # Se não encontrou o documento, registrar isso
                                if not doc_encontrado:
                                    logger.info(f"Documento citado não encontrado: {doc_citado}")
                        
                        # Se não encontramos citações explícitas ou os documentos citados
                        if not documentos_citados or not documentos_encontrados:
                            st.info("⚠️ Não foram encontradas citações de documentos específicos na resposta, ou os documentos citados não estão entre os recuperados. Exibindo os documentos mais relevantes:")
                            
                            # Mostrar os 3 documentos mais relevantes
                            with st.container(border=True):
                                for i, doc in enumerate(documentos[:3]):
                                    meta = doc.metadata
                                    tipo = meta.get('nome_tipo', 'Documento')
                                    numero = meta.get('numero', 'N/A')
                                    ano = meta.get('ano', 'N/A')
                                    doc_id = f"{tipo} {numero}/{ano}"
                                    
                                    # Adicionar indicação visual do tipo de documento
                                    if "Instrução" in tipo:
                                        st.markdown("🔵 **Instrução Normativa**")
                                    elif "Resolução" in tipo:
                                        st.markdown("🟠 **Resolução**")
                                    elif "Voto" in tipo:
                                        st.markdown("🟣 **Voto**")
                                    elif "Deliberação" in tipo:
                                        st.markdown("🟡 **Deliberação**")
                                    else:
                                        st.markdown("🟢 **Documento**")
                                    
                                    st.markdown(f"**{doc_id}** - Parte {meta.get('chunk', 'N/A')}/{meta.get('total_chunks', 'N/A')}")
                                    st.caption(f"Fonte: `{meta.get('caminho', 'Não especificado')}`")
                                    st.text_area(
                                        "Conteúdo do documento:",
                                        doc.page_content,
                                        height=130,
                                        key=f"relevante_{i}",
                                        disabled=True
                                    )
                                    
                                    if i < 2:  # Não adicionar separador após o último
                                        st.divider()
                        
                        # Informações sobre a busca
                        with st.expander("🔍 Detalhes da Busca"):
                            st.write(f"**Documentos encontrados:** {len(documentos)}")
                            
                            # Mostrar provedor que foi realmente usado
                            if modelo_usado_final != provider_usado:
                                st.write(f"**Provedor selecionado:** {provider_names.get(selected_provider, selected_provider)}")
                                st.write(f"**Provedor usado (fallback):** {provider_names.get(modelo_usado_final, modelo_usado_final)}")
                            else:
                                st.write(f"**Provedor usado:** {provider_names.get(modelo_usado_final, modelo_usado_final)}")
                                
                            st.write(f"**Modelo:** {selected_model}")
                            st.write(f"**Temperatura:** {temperatura}")
                            
                            # Mostrar informações sobre template adaptativo
                            if "gpt-4" in selected_model.lower() or modelo_usado_final == "openai":
                                template_info = "🧠 **Template GPT-4/OpenAI:** Estruturado e detalhado"
                            elif "deepseek" in selected_model.lower() or modelo_usado_final == "deepseek":
                                template_info = "⚡ **Template DeepSeek:** Direto e conciso"
                            else:
                                template_info = "📝 **Template Padrão:** Balanceado"
                            
                            st.markdown(template_info)
                            
                            # Mostrar documentos encontrados
                            for i, doc in enumerate(documentos[:5]):
                                metadata = doc.metadata
                                st.write(f"**Doc {i+1}:** {metadata.get('nome_tipo', 'N/A')} {metadata.get('numero', 'N/A')}/{metadata.get('ano', 'N/A')}")
                        
                        # Seção para mostrar todas as fontes consultadas (FORA do expander anterior)
                        with st.expander("📚 Todas as Fontes Consultadas"):
                            # Agrupar documentos por tipo/número/ano
                            documentos_agrupados = {}
                            for doc in documentos:
                                meta = doc.metadata
                                tipo = meta.get('nome_tipo', 'Documento')
                                numero = meta.get('numero', 'N/A')
                                ano = meta.get('ano', 'N/A')
                                
                                doc_id = f"{tipo} {numero}/{ano}"
                                if doc_id not in documentos_agrupados:
                                    documentos_agrupados[doc_id] = {
                                        'tipo': tipo,
                                        'numero': numero,
                                        'ano': ano,
                                        'caminho': meta.get('caminho', 'Não especificado'),
                                        'trechos': []
                                    }
                                
                                documentos_agrupados[doc_id]['trechos'].append({
                                    'chunk': meta.get('chunk', 'N/A'),
                                    'total_chunks': meta.get('total_chunks', 'N/A'),
                                    'conteudo': doc.page_content
                                })
                            
                            # Exibir documentos agrupados
                            for i, (doc_id, info) in enumerate(documentos_agrupados.items()):
                                # Criar um ícone baseado no tipo de documento
                                icone = "📄"
                                if "Resolução" in info['tipo']:
                                    icone = "📜"
                                elif "Instrução" in info['tipo']:
                                    icone = "📝"
                                elif "Deliberação" in info['tipo']:
                                    icone = "📋"
                                elif "Voto" in info['tipo']:
                                    icone = "✅"
                                
                                # Destacar documentos mais relevantes
                                destaque = " 🌟" if i < 3 else ""
                                
                                # Usar container em vez de expander aninhado
                                st.markdown(f"### {icone} {doc_id}{destaque}")
                                
                                with st.container(border=True):
                                    # Mostrar metadados do documento
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.markdown(f"**Tipo:** {info['tipo']}")
                                    with col2:
                                        st.markdown(f"**Número:** {info['numero']}")
                                    with col3:
                                        st.markdown(f"**Ano:** {info['ano']}")
                                    
                                    st.markdown(f"**Caminho:** `{info['caminho']}`")
                                    
                                    # Exibir trechos em tabs se houver múltiplos
                                    if len(info['trechos']) > 1:
                                        trechos_tabs = st.tabs([f"Trecho {t['chunk']}/{t['total_chunks']}" for t in info['trechos']])
                                        for j, tab in enumerate(trechos_tabs):
                                            with tab:
                                                trecho = info['trechos'][j]
                                                st.text_area(
                                                    f"Conteúdo do trecho {trecho['chunk']}:",
                                                    trecho['conteudo'],
                                                    height=200,
                                                    key=f"fonte_{doc_id}_{j}",
                                                    disabled=True
                                                )
                                    else:
                                        # Se há apenas um trecho, exibir diretamente
                                        trecho = info['trechos'][0]
                                        st.text_area(
                                            f"Conteúdo:",
                                            trecho['conteudo'],
                                            height=200,
                                            key=f"fonte_unico_{doc_id}",
                                            disabled=True
                                        )
                    
                    else:
                        st.warning("Nenhum documento relevante encontrado. Tente reformular sua pergunta.")
                        
                except Exception as e:
                    st.error(f"Erro ao processar consulta: {str(e)}")
                    logger.error(f"Erro na consulta: {str(e)}")
        
    # Limpar campos
    if limpar:
        # Limpar session_state
        for key in list(st.session_state.keys()):
            if key.startswith(('pergunta_exemplo', 'processar_automatico')):
                del st.session_state[key]
        st.rerun()
    
    # Separador visual
    st.markdown("---")
    
    # Mostrar exemplos
    if exemplos:
        st.subheader("💡 Exemplos de Consultas")
        
        st.markdown("**Clique diretamente em um dos exemplos abaixo:**")
        
        exemplos_consultas = [
            "Quais são os parâmetros técnicos para pavimentos rodoviários?",
            "Como funciona o processo de fiscalização da ANTT?",
            "Quais são as penalidades por descumprimento das normas?",
            "Resolução 6057 de 2024 - principais pontos",
            "Instrução Normativa 34 de 2024 sobre parâmetros de desempenho",
            "Critérios de segurança para transporte rodoviário",
            "Procedimentos para licenciamento de transportadoras",
            "Normas sobre tempo de direção e descanso"
        ]
        
        # Opção para processamento automático
        processar_automatico_exemplos = st.checkbox(
            "🚀 Processar automaticamente ao selecionar exemplo",
            value=True,
            help="Quando ativado, o exemplo será processado automaticamente após seleção",
            key="processar_auto_exemplos"
        )
        
        # Organizar exemplos em duas colunas
        col_ex1, col_ex2 = st.columns(2)
        
        for i, exemplo in enumerate(exemplos_consultas):
            # Alternar entre as colunas
            col = col_ex1 if i % 2 == 0 else col_ex2
            
            with col:
                # Criar um botão para cada exemplo
                if st.button(
                    f"📋 {exemplo[:60]}{'...' if len(exemplo) > 60 else ''}", 
                    key=f"btn_exemplo_{i}",
                    use_container_width=True,
                    help=exemplo  # Tooltip com o texto completo
                ):
                    # Salvar o exemplo selecionado
                    st.session_state.pergunta_exemplo = exemplo
                    
                    if processar_automatico_exemplos and vectorstore_loaded:
                        # Marcar para processamento automático
                        st.session_state.processar_automatico = True
                        st.success(f"✅ Processando: {exemplo[:50]}...")
                        # Forçar rerun para processar imediatamente
                        st.rerun()
                    else:
                        st.success(f"✅ Exemplo selecionado: {exemplo[:50]}...")
                        st.info("👆 Role para cima e clique em 'Consultar' para processar")
        
        # Botão para limpar seleção
        if st.button("🔄 Limpar Seleção de Exemplo", use_container_width=True):
            if "pergunta_exemplo" in st.session_state:
                del st.session_state.pergunta_exemplo
            if "processar_automatico" in st.session_state:
                del st.session_state.processar_automatico
            st.success("✅ Seleção limpa")
    
    # Upload de PDF
    st.subheader("📄 Processar Novo Documento")
    
    upload_col1, upload_col2 = st.columns([3, 1])
    
    with upload_col1:
        uploaded_file = st.file_uploader(
            "Envie um PDF para adicionar à base de conhecimento:",
            type=['pdf'],
            help="O documento será processado e adicionado ao vectorstore"
        )
    
    with upload_col2:
        if uploaded_file:
            if st.button("🔄 Processar PDF", use_container_width=True):
                try:
                    vectorstore_novo, tabelas = process_pdf(uploaded_file)
                    if vectorstore_novo:
                        st.success(f"PDF processado com sucesso! {len(tabelas)} tabelas analisadas.")
                        st.rerun()
                    else:
                        st.warning("Nenhuma tabela relevante encontrada no PDF.")
                except Exception as e:
                    st.error(f"Erro ao processar PDF: {str(e)}")

def main():
    """Função principal do sistema."""
    try:
        interface_usuario_unificada()
    except Exception as e:
        st.error(f"Erro fatal na aplicação: {str(e)}")
        logger.error(f"Erro fatal: {str(e)}")

if __name__ == "__main__":
    main() 