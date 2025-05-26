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

# Importar configurações e gerenciador de LLM
from config import (
    get_openai_api_key,
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

from llm_providers import LLMManager, get_available_providers, create_llm_manager

# Inicializa o logger globalmente
logger = setup_logging()

# Templates de prompts otimizados (copiados do chat-RAG.py)
TEMPLATE_RESPOSTA_COM_CITACOES = """
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
6. Quando houver dados numéricos ou técnicos, apresente-os de forma estruturada
7. NÃO INVENTE INFORMAÇÕES! Se algo não estiver nos documentos, indique claramente a limitação
8. Após sua resposta, SEMPRE adicione uma seção "TRECHOS DOS DOCUMENTOS CITADOS" estruturada assim:

### TRECHOS DOS DOCUMENTOS CITADOS:

**Documento: [TIPO] [NÚMERO]/[ANO]**
"[Trecho exato usado]"

**Documento: [TIPO] [NÚMERO]/[ANO]**
"[Trecho exato usado]"

Contextos:
{context}

Sua resposta em português:
"""

TEMPLATE_EXTRACAO_AGRESSIVA = """
Você é um especialista em análise documental com foco em regulamentações da ANTT.
Nos documentos abaixo, ANALISE MINUCIOSAMENTE todas as informações relevantes para: "{question}"

INSTRUÇÕES DE EXTRAÇÃO:
1. EXAMINE cada documento com atenção - informações importantes podem estar em qualquer parte
2. CONECTE informações fragmentadas de diferentes documentos para formar uma resposta coesa
3. EXTRAIA todos os detalhes relevantes: datas, números, requisitos, procedimentos, exceções
4. ORGANIZE as informações de forma lógica e estruturada, usando:
   - Listas para sequências ou itens relacionados
   - Seções com subtítulos para diferentes aspectos da resposta
5. NUNCA diga que não encontrou informações se houver QUALQUER conteúdo útil
6. Se os documentos apresentarem informações contraditórias ou ambíguas, EXPLIQUE as diferentes interpretações
7. Para cada informação, CITE A FONTE EXATA de onde a extraiu
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

TEMPLATE_PARAMETROS_TECNICOS = """
Você é um especialista técnico em regulamentações da ANTT.
Para a consulta sobre parâmetros ou especificações técnicas: "{question}"

INSTRUÇÕES DE ANÁLISE TÉCNICA:
1. IDENTIFIQUE todos os parâmetros técnicos, especificações, limites ou critérios mencionados
2. Para cada parâmetro técnico, DETALHE:
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
7. SEMPRE adicione uma seção "TRECHOS DOS DOCUMENTOS CITADOS" estruturada assim:

### TRECHOS DOS DOCUMENTOS CITADOS:

**Documento: [TIPO] [NÚMERO]/[ANO]**
"[Trecho exato usado]"

**Documento: [TIPO] [NÚMERO]/[ANO]**
"[Trecho exato usado]"

Contextos técnicos analisados:
{context}

Sua resposta técnica detalhada em português:
"""

TEMPLATE_ANALISE_NORMATIVA = """
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
6. SEMPRE adicione uma seção "TRECHOS DOS DOCUMENTOS CITADOS" estruturada assim:

### TRECHOS DOS DOCUMENTOS CITADOS:

**Documento: [TIPO] [NÚMERO]/[ANO]**
"[Trecho exato usado]"

**Documento: [TIPO] [NÚMERO]/[ANO]**
"[Trecho exato usado]"

Documentos normativos analisados:
{context}

Sua análise jurídica em português:
"""

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
    
    # Usar o LLMManager para obter embeddings
    try:
        llm_manager = create_llm_manager("openai")  # Sempre usar OpenAI para embeddings
        embeddings = llm_manager.get_embeddings()
    except Exception as e:
        logger.error(f"Erro ao criar embeddings: {e}")
        # Fallback para método tradicional
        embeddings = OpenAIEmbeddings(openai_api_key=get_openai_api_key())
    
    vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Atualizar com documentos importantes
    vectorstore = atualizar_vectorstore_com_documentos_importantes(vectorstore, embeddings)
    
    logger.info("Vectorstore carregado com sucesso")
    return vectorstore

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
    
    keywords = [w.lower() for w in query.split() 
                if w.lower() not in stop_words and len(w) > 3]
    return keywords

def simplificar_query(query):
    """Simplifica a consulta para busca mais geral."""
    keywords = extrair_keywords(query)
    
    if len(keywords) >= 2:
        keywords_sorted = sorted(keywords, key=len, reverse=True)[:3]
        return " ".join(keywords_sorted)
    
    return query

def reranking_documentos(query, documentos):
    """Reordena os documentos com base na relevância para a consulta."""
    if not documentos:
        return []
    
    keywords = set(extrair_keywords(query))
    
    def calcular_score_documento(doc):
        score = 0
        conteudo = doc.page_content.lower()
        metadados = doc.metadata
        
        for keyword in keywords:
            if keyword in conteudo:
                score += 1
            elif any(keyword in palavra for palavra in conteudo.split()):
                score += 0.5
        
        palavras_total = len(conteudo.split())
        if palavras_total > 0:
            score += (score / palavras_total) * 5
        
        if metadados.get("relevancia_tecnica") == "Alta":
            score += 2
        
        if metadados.get("contem_tabelas") == "Sim":
            score += 1
        
        if metadados.get("chunk") == 1:
            score += 0.5
        
        return score
    
    docs_com_score = [(doc, calcular_score_documento(doc)) for doc in documentos]
    docs_ordenados = [doc for doc, score in sorted(docs_com_score, key=lambda x: x[1], reverse=True)]
    
    return docs_ordenados

def pesquisar_documentos(query, vectorstore, k=12, tipo_documento=None, ano=None, numero=None):
    """Pesquisa documentos com base em uma query, podendo filtrar por metadados."""
    resultados = []
    resultados_finais = []
    query_original = query
    
    logger.info(f"\n===== NOVA CONSULTA =====")
    logger.info(f"Pesquisando: '{query}'")
    if tipo_documento or ano or numero:
        logger.info(f"Filtros: Tipo={tipo_documento}, Ano={ano}, Número={numero}")
    
    filtro = criar_filtro_metadados(tipo_documento, ano, numero)
    
    try:
        keywords = extrair_keywords(query)
        
        logger.info(f"Executando busca semântica com MMR...")
        docs_semantic = vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=k*3,
            lambda_mult=0.7,
            filter=filtro
        )
        
        resultados.extend(docs_semantic)
        logger.info(f"Busca semântica: {len(docs_semantic)} resultados")
        
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
            
            logger.info(f"Após busca por keywords: {len(resultados)} resultados")
    except Exception as e:
        logger.error(f"Erro na busca híbrida: {str(e)}")
    
    if len(resultados) < 2:
        logger.info("Resultados insuficientes. Tentando busca direta...")
        try:
            resultados = vectorstore.similarity_search(query_original, k=k, filter=filtro)
            logger.info(f"Busca direta: {len(resultados)} resultados")
        except Exception as e:
            logger.error(f"Erro na busca direta: {str(e)}")
    
    if len(resultados) < 2:
        logger.info("Resultados ainda insuficientes. Tentando busca com termos amplos...")
        try:
            termos_gerais = simplificar_query(query_original)
            resultados = vectorstore.similarity_search(termos_gerais, k=k)
            logger.info(f"Busca com termos amplos: {len(resultados)} resultados")
        except Exception as e:
            logger.error(f"Erro na busca com termos amplos: {str(e)}")
    
    if resultados:
        resultados_finais = reranking_documentos(query_original, resultados)
        logger.info(f"Após reranking: {len(resultados_finais)} documentos retornados")
    
    return resultados_finais 

def gerar_resposta(pergunta, documentos, llm):
    """Gera uma resposta baseada nos documentos recuperados."""
    if not documentos:
        return "Não encontrei documentos relevantes para esta pergunta. Por favor, reformule sua consulta ou forneça mais detalhes."
    
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
                'caminho': metadados.get('caminho', 'Não especificado'),
                'trechos': []
            }
        
        documentos_info[doc_id]['trechos'].append({
            'chunk': metadados.get('chunk', 'N/A'),
            'total_chunks': metadados.get('total_chunks', 'N/A'),
            'conteudo': doc.page_content
        })
        
        contexto = f"""
[Documento: {doc_id} - Parte {metadados.get('chunk', 'N/A')}/{metadados.get('total_chunks', 'N/A')}]
Fonte: {metadados.get('caminho', 'Não especificado')}
Conteúdo:
{doc.page_content}
"""
        contextos.append(contexto)
    
    contexto_completo = "\n\n".join(contextos)
    
    logger.info(f"Documentos encontrados: {len(documentos)}")
    for tipo, contagem in contagem_por_tipo.items():
        logger.info(f"- {tipo}: {contagem} documentos")
    
    # Determinar o tipo de consulta
    palavras_parametros_tecnicos = ['parâmetro', 'técnico', 'valor', 'limite', 'medida', 'metodologia', 
                                   'pavimento', 'deflexão', 'iri', 'atrito', 'índice']
    
    palavras_normativas = ['resolução', 'instrução normativa', 'deliberação', 'portaria', 'regulamento', 
                           'normativo', 'legal', 'direito', 'obrigação', 'dever', 'prazo', 'penalidade']
    
    pergunta_lower = pergunta.lower()
    
    if any(palavra in pergunta_lower for palavra in palavras_parametros_tecnicos) or \
       'INSTRUÇÃO NORMATIVA' in ' '.join(tipos_documentos) and 'parâmetro' in pergunta_lower:
        logger.info("Detectada consulta sobre parâmetros técnicos")
        template_escolhido = TEMPLATE_PARAMETROS_TECNICOS
    elif any(palavra in pergunta_lower for palavra in palavras_normativas) or \
         any(tipo in ['Resolução', 'Deliberação', 'Portaria'] for tipo in tipos_documentos):
        logger.info("Detectada consulta sobre aspectos normativos/jurídicos")
        template_escolhido = TEMPLATE_ANALISE_NORMATIVA
    else:
        logger.info("Usando template padrão de resposta")
        template_escolhido = TEMPLATE_RESPOSTA_COM_CITACOES
    
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
        
        conteudo_resposta = resposta.content
        if any(frase in conteudo_resposta.lower() for frase in 
              ["não encontrei informações", "não há informações", "não foi possível encontrar", 
               "não foram encontradas", "não disponível nos documentos"]):
            logger.info("Resposta inicial insatisfatória. Tentando extração agressiva de informações...")
            
            prompt_extracao = PromptTemplate(
                template=TEMPLATE_EXTRACAO_AGRESSIVA,
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
        
        return conteudo_resposta
        
    except Exception as e:
        logger.error(f"Erro ao gerar resposta: {str(e)}")
        try:
            prompt_fallback = PromptTemplate(
                template=TEMPLATE_EXTRACAO_AGRESSIVA,
                input_variables=["context", "question"]
            )
            
            chain_fallback = prompt_fallback | llm
            resposta_fallback = chain_fallback.invoke({
                "context": contexto_completo,
                "question": pergunta
            })
            
            return resposta_fallback.content
        except Exception as e2:
            logger.error(f"Erro no fallback: {str(e2)}")
            return f"Não foi possível gerar uma resposta devido a um erro interno. Tente reformular sua consulta de forma mais específica."

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
            "openai": "OpenAI (GPT-4)",
            "deepseek": "DeepSeek (via OpenRouter)"
        }
        
        selected_provider = st.selectbox(
            "Escolha o provedor:",
            options=list(providers.keys()),
            format_func=lambda x: provider_names.get(x, x),
            index=0 if "deepseek" not in providers else list(providers.keys()).index("deepseek")
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
        
        max_tokens = st.number_input(
            "Máximo de tokens:",
            min_value=100,
            max_value=4000,
            value=2000,
            step=100,
            help="Limite de tokens para a resposta"
        )
        
        num_documentos = st.slider(
            "Documentos para busca:",
            min_value=5,
            max_value=20,
            value=12,
            help="Número de documentos a recuperar"
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
    
    # Área principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Consulta")
        
        # Campo de pergunta
        pergunta = st.text_area(
            "Faça sua pergunta sobre regulamentações da ANTT:",
            height=100,
            placeholder="Ex: Quais são os parâmetros técnicos para pavimentos rodoviários?"
        )
        
        # Botões de ação
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            consultar = st.button("🔍 Consultar", type="primary", use_container_width=True)
        
        with col_btn2:
            limpar = st.button("🗑️ Limpar", use_container_width=True)
        
        with col_btn3:
            exemplos = st.button("💡 Exemplos", use_container_width=True)
    
    with col2:
        st.subheader("📊 Informações")
        
        # Carregar vectorstore e mostrar estatísticas
        try:
            vectorstore = carregar_vectorstore()
            
            # Estatísticas básicas
            st.markdown("""
            <div class="metric-card">
                <h4>📚 Base de Conhecimento</h4>
                <p>✅ Vectorstore carregado</p>
                <p>🔍 Busca semântica ativa</p>
                <p>🤖 IA: """ + provider_names.get(selected_provider, selected_provider) + """</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Erro ao carregar vectorstore: {str(e)}")
            vectorstore = None
    
    # Processamento da consulta
    if consultar and pergunta and vectorstore:
        with st.spinner("🔍 Processando consulta..."):
            try:
                # Criar LLM manager
                llm_manager = create_llm_manager(selected_provider, selected_model)
                llm = llm_manager.get_llm(temperature=temperatura, max_tokens=max_tokens)
                
                # Aplicar filtros
                filtro_tipo = None if tipo_documento == "Todos" else tipo_documento
                filtro_ano = None if ano_filtro == "Todos" else ano_filtro
                filtro_numero = numero_filtro if numero_filtro.strip() else None
                
                # Buscar documentos
                documentos = pesquisar_documentos(
                    pergunta, 
                    vectorstore, 
                    k=num_documentos,
                    tipo_documento=filtro_tipo,
                    ano=filtro_ano,
                    numero=filtro_numero
                )
                
                if documentos:
                    # Gerar resposta
                    resposta = gerar_resposta(pergunta, documentos, llm)
                    
                    # Exibir resposta
                    st.subheader("📝 Resposta")
                    st.markdown(resposta)
                    
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
                    
                    # Informações sobre a busca
                    with st.expander("🔍 Detalhes da Busca"):
                        st.write(f"**Documentos encontrados:** {len(documentos)}")
                        st.write(f"**Provedor usado:** {provider_names.get(selected_provider, selected_provider)}")
                        st.write(f"**Modelo:** {selected_model}")
                        st.write(f"**Temperatura:** {temperatura}")
                        
                        # Mostrar documentos encontrados
                        for i, doc in enumerate(documentos[:5]):
                            metadata = doc.metadata
                            st.write(f"**Doc {i+1}:** {metadata.get('nome_tipo', 'N/A')} {metadata.get('numero', 'N/A')}/{metadata.get('ano', 'N/A')}")
                
                else:
                    st.warning("Nenhum documento relevante encontrado. Tente reformular sua pergunta.")
                    
            except Exception as e:
                st.error(f"Erro ao processar consulta: {str(e)}")
                logger.error(f"Erro na consulta: {str(e)}")
    
    # Limpar campos
    if limpar:
        st.rerun()
    
    # Mostrar exemplos
    if exemplos:
        st.subheader("💡 Exemplos de Consultas")
        
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
        
        for exemplo in exemplos_consultas:
            if st.button(f"📋 {exemplo}", key=f"exemplo_{exemplo[:20]}"):
                st.session_state.pergunta_exemplo = exemplo
                st.rerun()
    
    # Upload de PDF
    st.subheader("📄 Processar Novo Documento")
    
    uploaded_file = st.file_uploader(
        "Envie um PDF para adicionar à base de conhecimento:",
        type=['pdf'],
        help="O documento será processado e adicionado ao vectorstore"
    )
    
    if uploaded_file:
        if st.button("🔄 Processar PDF"):
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