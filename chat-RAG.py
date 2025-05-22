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

# Importar configurações do config.py
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

# Inicializa o logger globalmente
logger = setup_logging()

# Templates de prompts otimizados
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

# Template especializado para parâmetros técnicos
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

# Template para análise de normativas
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
def call_openai_with_retry(llm, prompt, context):
    """
    Chama a API OpenAI com retry em caso de rate limit.
    """
    try:
        return llm.invoke(prompt.format(context=context))
    except RateLimitError as e:
        logger.warning(f"Rate limit atingido, tentando novamente: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Erro na chamada OpenAI: {str(e)}")
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
    
    Args:
        metrics: Dicionário com os elementos da interface
        status: Status atual do processamento
        progress: Valor do progresso (0-100)
        info: Mensagem informativa
        error: Mensagem de erro
    """
    try:
        if status:
            metrics['status'].text(f"Status: {status}")
        if progress is not None:
            # Garante que o progresso esteja entre 0 e 100
            progress = max(0, min(100, progress))
            # Normaliza para valor entre 0 e 1
            normalized_progress = progress / 100.0
            metrics['progress'].progress(normalized_progress)
        if info:
            metrics['info'].info(info)
        if error:
            metrics['error'].error(error)
        # Atualiza timestamp
        metrics['time'].text(f"Última atualização: {datetime.now().strftime('%H:%M:%S')}")
    except Exception as e:
        logger.error(f"Erro ao atualizar métricas: {str(e)}")

def get_cropped_table(table, page, pdf):
    """
    Extrai uma imagem recortada da tabela do PDF.
    
    Args:
        table: Objeto tabela com as coordenadas bbox
        page: Número da página
        pdf: Objeto PDF
    
    Returns:
        Image: Imagem recortada da tabela
    """
    bbox = table.bbox
    # Converte a região da tabela em um array numpy
    table_array = np.array(pdf.images[page][bbox.y1:bbox.y2, bbox.x1:bbox.x2])
    # Converte o array numpy em uma imagem PIL
    return Image.fromarray(table_array.astype('uint8'))

def process_single_table(table_info):
    """
    Processa uma única tabela com foco em descumprimentos.
    """
    try:
        table = table_info['table']
        page_num = table_info['page_num']
        table_num = table_info['table_num']
        pdf_doc = table_info['pdf_doc']
        llm = table_info['llm']
        
        table_text = table.html_repr()
        table_image = get_cropped_table(table, page_num, pdf_doc)
        inferred_context = infer_table_context(table_text, llm)
        
        # Só inclui a tabela se houver descumprimentos identificados
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
    """
    Extrai tabelas e infere seu contexto usando LLM e OCR com processamento paralelo.
    """
    tables_with_context = []
    start_time = time.time()
    
    if metrics:
        update_metrics(metrics, 
                      "Iniciando processamento do PDF", 
                      progress=0,
                      info="Configurando ambiente...")
    
    try:
        llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0,
            api_key=get_openai_api_key()
        )
        
        if metrics:
            update_metrics(metrics, 
                          "Configurando OCR", 
                          progress=10,
                          info="Inicializando OCR...")
        
        # Otimiza o OCR para usar múltiplos threads
        num_cores = multiprocessing.cpu_count()
        ocr = TesseractOCR(n_threads=min(num_cores, 4), lang="por")
        
        if metrics:
            update_metrics(metrics, "Carregando PDF", progress=0.2)
        
        pdf_doc = PDF(
            src=pdf_path,
            detect_rotation=False,
            pdf_text_extraction=True
        )
        
        if metrics:
            update_metrics(metrics, "Extraindo tabelas", progress=0.3)
        
        # Otimiza a extração de tabelas
        extracted_tables = pdf_doc.extract_tables(
            ocr=ocr,
            implicit_rows=True,
            borderless_tables=True,
            min_confidence=50  # Ajuste conforme necessário
        )
        
        # Prepara as tabelas para processamento paralelo
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
            update_metrics(metrics, 
                          f"Encontradas {total_tables} tabelas", 
                          progress=40,
                          info=f"Iniciando processamento paralelo...")
        
        # Processa as tabelas em paralelo
        processed_tables = 0
        max_workers = min(num_cores, 4)  # Limita o número de workers
        
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
                            update_metrics(metrics,
                                        f"Processando tabelas",
                                        progress=progress,
                                        info=f"Processadas {processed_tables}/{total_tables} tabelas")
                except Exception as e:
                    if metrics:
                        update_metrics(metrics,
                                     "Erro no processamento",
                                     error=f"Erro ao processar tabela: {str(e)}")
        
        # Ordena as tabelas por página e número
        tables_with_context.sort(key=lambda x: (x['page_number'], x['table_number']))
        
        processing_time = time.time() - start_time
        if metrics:
             update_metrics(metrics,
                          "Processamento concluído",
                          progress=100,
                          info=f"Processamento concluído em {time.time() - start_time:.2f} segundos")
        
        return tables_with_context
        
    except Exception as e:
        if metrics:
            update_metrics(metrics,
                          "Erro no processamento",
                          progress=0,
                          error=f"Erro durante o processamento: {str(e)}")
        raise

def infer_table_context(table_text, llm):
    """
    Analisa especificamente tabelas de parâmetros de desempenho da ANTT.
    """
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
        response = call_openai_with_retry(chain, prompt, {"table_content": table_text})
        
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
            update_metrics(metrics, 
                          status="Salvando arquivo temporário", 
                          progress=0.05)
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

            try:
                # Extrai tabelas com contexto
                tables_data = extract_tables_with_context(tmp_file_path, metrics)
                
                if not tables_data:
                    update_metrics(metrics,
                                 status="Nenhuma tabela encontrada",
                                 progress=1.0,
                                 info="O documento não contém tabelas para processar")
                    return None, []

                update_metrics(metrics, 
                             status="Preparando documentos para indexação", 
                             progress=0.9)
                
                # Prepara os documentos para o FAISS
                documents = []
                for table_data in tables_data:
                    documents.append(table_data['full_text'])

                # Divide o texto em chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP
                )
                splits = text_splitter.create_documents(documents)

                update_metrics(metrics, 
                             status="Criando embeddings", 
                             progress=0.95)
                
                # Cria embeddings com retry
                try:
                    embeddings = OpenAIEmbeddings(
                        api_key=get_openai_api_key(),
                        max_retries=3,
                        timeout=30
                    )
                    vectorstore = FAISS.from_documents(
                        documents=splits,
                        embedding=embeddings
                    )
                except RateLimitError:
                    update_metrics(metrics,
                                 status="Limite de requisições atingido",
                                 progress=0.95,
                                 error="Aguarde alguns segundos e tente novamente")
                    raise
                
                update_metrics(metrics, 
                             status="Salvando base de conhecimento", 
                             progress=0.98)
                
                # Salva a base de conhecimento
                os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
                vectorstore.save_local(DB_FAISS_PATH)
                
                update_metrics(metrics, 
                             status="Processamento finalizado", 
                             progress=1.0,
                             info=f"Processadas {len(tables_data)} tabelas com sucesso")
                
                os.remove(tmp_file_path)
                return vectorstore, tables_data
                
            except Exception as e:
                update_metrics(metrics, 
                             status="Erro no processamento",
                             progress=0.0,
                             error=f"Erro durante o processamento: {str(e)}")
                raise
    except Exception as e:
        update_metrics(metrics, 
                      status="Erro fatal",
                      progress=0.0,
                      error=f"Erro ao processar arquivo: {str(e)}")
        raise

def format_docs(docs):
    """Formata os documentos para o contexto."""
    return "\n\n".join(doc.page_content for doc in docs)

def verificar_documentos_importantes():
    """Verifica se documentos importantes estão no vectorstore e os adiciona se necessário."""
    documentos_importantes = [
        "dados_antt/INM/2024/INM-00000034-2024.md",  # Instrução Normativa 34 de 2024 - Parâmetros de Desempenho de Pavimento
        "dados_antt/RES/2024/RES-00006057-2024.md",  # Resolução 6057 de 2024 - Programa de Sustentabilidade
        "dados_antt/DLB/2024/DLB-00000092-2024.md"   # Deliberação 92 de 2024 - Limites de Peso
    ]
    
    documentos_indexados = []
    
    # Verificar quais documentos existem
    for caminho in documentos_importantes:
        if os.path.exists(caminho):
            with open(caminho, 'r', encoding='utf-8') as f:
                conteudo = f.read()
            
            # Extrair metadados básicos do caminho
            partes_caminho = caminho.split('/')
            tipo_documento = partes_caminho[1] if len(partes_caminho) > 1 else None
            ano = partes_caminho[2] if len(partes_caminho) > 2 else None
            numero = None
            
            # Extrair número do nome do arquivo
            nome_arquivo = os.path.basename(caminho)
            if '-' in nome_arquivo:
                partes = nome_arquivo.split('-')
                if len(partes) >= 2:
                    numero = partes[1]
            
            # Mapear tipo para nome completo
            tipo_nome = {
                "RES": "Resolução",
                "POR": "Portaria",
                "INM": "Instrução Normativa",
                "DLB": "Deliberação",
                "INC": "Instrução Normativa Complementar"
            }.get(tipo_documento, tipo_documento)
            
            # Dividir em chunks menores para capturar informações específicas
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # Chunks menores para capturar detalhes específicos
                chunk_overlap=200,
                length_function=len,
            )
            
            # Criar documentos
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
        print(f"Atualizando vectorstore com {len(documentos)} documentos importantes...")
        vectorstore.add_documents(documentos)
        # Salvar o vectorstore atualizado
        vectorstore.save_local(DB_FAISS_PATH)
        print("Vectorstore atualizado e salvo.")
    return vectorstore

def carregar_vectorstore(api_key, caminho_vectorstore="vectorstore"):
    """Carrega o vectorstore ANTT."""
    print(f"Carregando vectorstore de {caminho_vectorstore}...")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.load_local(caminho_vectorstore, embeddings, allow_dangerous_deserialization=True)
    
    # Atualizar com documentos importantes
    vectorstore = atualizar_vectorstore_com_documentos_importantes(vectorstore, embeddings)
    
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

def pesquisar_documentos(query, vectorstore, k=12, tipo_documento=None, ano=None, numero=None):
    """Pesquisa documentos com base em uma query, podendo filtrar por metadados."""
    resultados = []
    resultados_finais = []
    query_original = query
    
    # Log da consulta
    print(f"\n===== NOVA CONSULTA =====")
    print(f"Pesquisando: '{query}'")
    if tipo_documento or ano or numero:
        print(f"Filtros: Tipo={tipo_documento}, Ano={ano}, Número={numero}")
    
    # Criar filtro de metadados
    filtro = criar_filtro_metadados(tipo_documento, ano, numero)
    
    # ESTRATÉGIA 1: Hybrid search - combinar busca semântica com keywords
    try:
        # Extrair palavras-chave da consulta (remover stop words)
        keywords = extrair_keywords(query)
        
        # Executar busca semântica com MMR (para diversidade)
        print(f"Executando busca semântica com MMR...")
        docs_semantic = vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=k*3,
            lambda_mult=0.7,
            filter=filtro
        )
        
        # Adicionar resultados da busca semântica
        resultados.extend(docs_semantic)
        print(f"Busca semântica: {len(docs_semantic)} resultados")
        
        # Se temos keywords relevantes, fazer busca por keywords também
        if keywords and len(keywords) > 1:
            keyword_query = " ".join(keywords)
            print(f"Executando busca adicional por keywords: '{keyword_query}'")
            docs_keywords = vectorstore.similarity_search(
                keyword_query,
                k=max(3, k//2),
                filter=filtro
            )
            
            # Adicionar apenas documentos não duplicados
            docs_ids = set(doc.metadata.get('caminho', '') + str(doc.metadata.get('chunk', '')) 
                           for doc in resultados)
            
            for doc in docs_keywords:
                doc_id = doc.metadata.get('caminho', '') + str(doc.metadata.get('chunk', ''))
                if doc_id not in docs_ids:
                    resultados.append(doc)
                    docs_ids.add(doc_id)
            
            print(f"Após busca por keywords: {len(resultados)} resultados")
    except Exception as e:
        print(f"Erro na busca híbrida: {str(e)}")
    
    # ESTRATÉGIA 2: Se não obtivemos resultados suficientes, fazer busca normal
    if len(resultados) < 2:
        print("Resultados insuficientes. Tentando busca direta...")
        try:
            resultados = vectorstore.similarity_search(query_original, k=k, filter=filtro)
            print(f"Busca direta: {len(resultados)} resultados")
        except Exception as e:
            print(f"Erro na busca direta: {str(e)}")
    
    # ESTRATÉGIA 3: Busca por termos mais amplos como último recurso
    if len(resultados) < 2:
        print("Resultados ainda insuficientes. Tentando busca com termos amplos...")
        try:
            # Tentar com uma versão simplificada da query
            termos_gerais = simplificar_query(query_original)
            resultados = vectorstore.similarity_search(termos_gerais, k=k)
            print(f"Busca com termos amplos: {len(resultados)} resultados")
        except Exception as e:
            print(f"Erro na busca com termos amplos: {str(e)}")
    
    # Reranking dos documentos recuperados
    if resultados:
        resultados_finais = reranking_documentos(query_original, resultados)
        print(f"Após reranking: {len(resultados_finais)} documentos retornados")
    
    return resultados_finais

def extrair_keywords(query):
    """Extrai palavras-chave relevantes da consulta."""
    # Lista de stop words em português
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
    
    # Extrair palavras da consulta, filtrar stop words e palavras curtas
    keywords = [w.lower() for w in query.split() 
                if w.lower() not in stop_words and len(w) > 3]
    return keywords

def simplificar_query(query):
    """Simplifica a consulta para busca mais geral."""
    # Extrair substantivos e palavras-chave mais importantes
    keywords = extrair_keywords(query)
    
    # Se temos pelo menos 2 keywords, usar as mais longas (provavelmente mais específicas)
    if len(keywords) >= 2:
        # Ordenar por tamanho decrescente e pegar até 3 palavras-chave mais longas
        keywords_sorted = sorted(keywords, key=len, reverse=True)[:3]
        return " ".join(keywords_sorted)
    
    # Se não temos keywords suficientes, usar a query original
    return query

def reranking_documentos(query, documentos):
    """
    Reordena os documentos com base na relevância para a consulta.
    Implementa uma versão simplificada de reranking baseada em heurísticas.
    """
    if not documentos:
        return []
    
    # Extrair keywords da consulta
    keywords = set(extrair_keywords(query))
    
    # Função para calcular score de relevância
    def calcular_score_documento(doc):
        score = 0
        conteudo = doc.page_content.lower()
        metadados = doc.metadata
        
        # Fator 1: Presença de keywords no conteúdo (match lexical)
        for keyword in keywords:
            if keyword in conteudo:
                # Mais pontos para matches exatos de palavras completas
                score += 1
            # Pontos parciais para substrings
            elif any(keyword in palavra for palavra in conteudo.split()):
                score += 0.5
        
        # Fator 2: Densidade de keywords (quanto maior a densidade, melhor)
        palavras_total = len(conteudo.split())
        if palavras_total > 0:
            score += (score / palavras_total) * 5  # Multiplicador para dar mais peso
        
        # Fator 3: Metadados de relevância específica
        if metadados.get("relevancia_tecnica") == "Alta":
            score += 2
        
        # Fator 4: Conteúdo de tabelas (valorizar documentos com dados estruturados)
        if metadados.get("contem_tabelas") == "Sim":
            score += 1
        
        # Fator 5: Primeiros chunks têm precedência (geralmente mais contextuais)
        if metadados.get("chunk") == 1:
            score += 0.5
        
        return score
    
    # Calcular scores e ordenar documentos
    docs_com_score = [(doc, calcular_score_documento(doc)) for doc in documentos]
    docs_ordenados = [doc for doc, score in sorted(docs_com_score, key=lambda x: x[1], reverse=True)]
    
    return docs_ordenados

def criar_qa_chain(vectorstore, llm):
    """Cria uma chain QA usando o template especializado para ANTT."""
    prompt = PromptTemplate(
        template=TEMPLATE_RESPOSTA_COM_CITACOES,
        input_variables=["context", "question"]
    )
    
    # Configurar o retriever para usar MMR (Maximum Marginal Relevance)
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance
        search_kwargs={
            "k": 12,        # Número de documentos a recuperar
            "fetch_k": 20,  # Número de documentos a buscar antes de aplicar MMR
            "lambda_mult": 0.7  # Equilíbrio entre relevância (1.0) e diversidade (0.0)
        }
    )
    
    # Criar chain RAG otimizada
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Combina todos os documentos em um único contexto
        retriever=retriever,
        chain_type_kwargs={
            "prompt": prompt,
            "verbose": False
        },
        return_source_documents=True  # Retorna os documentos fonte junto com a resposta
    )
    
    return chain

def gerar_resposta(pergunta, documentos, llm):
    """Gera uma resposta baseada nos documentos recuperados."""
    if not documentos:
        return "Não encontrei documentos relevantes para esta pergunta. Por favor, reformule sua consulta ou forneça mais detalhes."
    
    # Preparar o contexto a partir dos documentos
    contextos = []
    documentos_info = {}  # Para armazenar informações sobre cada documento
    
    # Categorizar documentos encontrados por tipo
    tipos_documentos = set()
    contagem_por_tipo = {}
    
    for i, doc in enumerate(documentos):
        metadados = doc.metadata
        tipo = metadados.get('nome_tipo', 'Documento')
        tipos_documentos.add(tipo)
        
        # Contar ocorrências por tipo
        contagem_por_tipo[tipo] = contagem_por_tipo.get(tipo, 0) + 1
        
        # Criar identificador único para o documento
        numero = metadados.get('numero', 'N/A')
        ano = metadados.get('ano', 'N/A')
        doc_id = f"{tipo} {numero}/{ano}"
        
        # Armazenar informações do documento para uso posterior
        if doc_id not in documentos_info:
            documentos_info[doc_id] = {
                'tipo': tipo,
                'numero': numero,
                'ano': ano,
                'caminho': metadados.get('caminho', 'Não especificado'),
                'trechos': []
            }
        
        # Adicionar o trecho atual à lista de trechos deste documento
        documentos_info[doc_id]['trechos'].append({
            'chunk': metadados.get('chunk', 'N/A'),
            'total_chunks': metadados.get('total_chunks', 'N/A'),
            'conteudo': doc.page_content
        })
        
        # Formatação para o contexto enviado ao LLM
        contexto = f"""
[Documento: {doc_id} - Parte {metadados.get('chunk', 'N/A')}/{metadados.get('total_chunks', 'N/A')}]
Fonte: {metadados.get('caminho', 'Não especificado')}
Conteúdo:
{doc.page_content}
"""
        contextos.append(contexto)
    
    # Limitar o tamanho total do contexto para não exceder limites de token
    contexto_completo = "\n\n".join(contextos)
    
    # Log para depuração
    print(f"Documentos encontrados: {len(documentos)}")
    for tipo, contagem in contagem_por_tipo.items():
        print(f"- {tipo}: {contagem} documentos")
    
    # Determinar o tipo de consulta com base em palavras-chave e documentos recuperados
    palavras_parametros_tecnicos = ['parâmetro', 'técnico', 'valor', 'limite', 'medida', 'metodologia', 
                                   'pavimento', 'deflexão', 'iri', 'atrito', 'índice']
    
    palavras_normativas = ['resolução', 'instrução normativa', 'deliberação', 'portaria', 'regulamento', 
                           'normativo', 'legal', 'direito', 'obrigação', 'dever', 'prazo', 'penalidade']
    
    # Verificar tipo de consulta
    pergunta_lower = pergunta.lower()
    
    # Verificar se é consulta técnica
    if any(palavra in pergunta_lower for palavra in palavras_parametros_tecnicos) or \
       'INSTRUÇÃO NORMATIVA' in ' '.join(tipos_documentos) and 'parâmetro' in pergunta_lower:
        print("Detectada consulta sobre parâmetros técnicos")
        template_escolhido = TEMPLATE_PARAMETROS_TECNICOS
    
    # Verificar se é consulta normativa/jurídica
    elif any(palavra in pergunta_lower for palavra in palavras_normativas) or \
         any(tipo in ['Resolução', 'Deliberação', 'Portaria'] for tipo in tipos_documentos):
        print("Detectada consulta sobre aspectos normativos/jurídicos")
        template_escolhido = TEMPLATE_ANALISE_NORMATIVA
    
    # Caso padrão - resposta geral
    else:
        print("Usando template padrão de resposta")
        template_escolhido = TEMPLATE_RESPOSTA_COM_CITACOES
    
    # Usar o template escolhido
    prompt = PromptTemplate(
        template=template_escolhido,
        input_variables=["context", "question"]
    )
    
    # Executar a chain de pergunta e resposta com o contexto
    chain = prompt | llm
    
    try:
        # Gerar resposta
        resposta = chain.invoke({
            "context": contexto_completo,
            "question": pergunta
        })
        
        # Se a resposta for muito genérica ou indicar falta de informações, tentar extração agressiva
        conteudo_resposta = resposta.content
        if any(frase in conteudo_resposta.lower() for frase in 
              ["não encontrei informações", "não há informações", "não foi possível encontrar", 
               "não foram encontradas", "não disponível nos documentos"]):
            print("Resposta inicial insatisfatória. Tentando extração agressiva de informações...")
            
            # Tentar extração agressiva
            prompt_extracao = PromptTemplate(
                template=TEMPLATE_EXTRACAO_AGRESSIVA,
                input_variables=["context", "question"]
            )
            
            chain_extracao = prompt_extracao | llm
            nova_resposta = chain_extracao.invoke({
                "context": contexto_completo,
                "question": pergunta
            })
            
            # Se a nova resposta for mais informativa, use-a
            if len(nova_resposta.content) > 100 and not any(frase in nova_resposta.content.lower() 
                                                          for frase in ["não encontrei", "não foi possível"]):
                conteudo_resposta = nova_resposta.content
                print("Usado resultado da extração agressiva")
            else:
                print("Mantida resposta original")
        
        return conteudo_resposta
        
    except Exception as e:
        print(f"Erro ao gerar resposta: {str(e)}")
        # Tentar com o template de extração como fallback em caso de erro
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
            print(f"Erro no fallback: {str(e2)}")
            return f"Não foi possível gerar uma resposta devido a um erro interno. Tente reformular sua consulta de forma mais específica."

def interface_usuario(vectorstore, llm):
    st.set_page_config(
        page_title=STREAMLIT_PAGE_TITLE,
        page_icon=STREAMLIT_PAGE_ICON,
        layout=STREAMLIT_LAYOUT
    )
    
    # Carregar logo e exibir cabeçalho
    try:
        logo = Image.open("antt-logo.png")
        col1, col2 = st.columns([1, 5])
        with col1:
            st.image(logo, width=100)
        with col2:
            st.title("Sistema RAG - Consulta a Documentos da ANTT")
    except:
        # Fallback se não encontrar a imagem
        st.title("Sistema RAG - Consulta a Documentos da ANTT")
    
    # Carregar os tipos de documentos e anos disponíveis do relatório
    try:
        with open("relatorio_documentos.json", "r") as f:
            relatorio = json.load(f)
        
        tipos_documento = sorted(list(set([doc["tipo"] for doc in relatorio if doc["tipo"]])))
        anos = sorted(list(set([doc["ano"] for doc in relatorio if doc["ano"]])))
    except Exception as e:
        st.warning(f"Não foi possível carregar o relatório de documentos. Erro: {str(e)}")
        tipos_documento = []
        anos = []
    
    # Filtros laterais
    with st.sidebar:
        st.header("Filtros de Pesquisa")
        
        tipo_selecionado = st.selectbox(
            "Tipo de Documento",
            ["Todos"] + tipos_documento
        )
        
        ano_selecionado = st.selectbox(
            "Ano",
            ["Todos"] + anos
        )
        
        numero_documento = st.text_input("Número do Documento (opcional)")
        
        st.markdown("---")
        st.markdown("### Configurações")
        
        col_a, col_b = st.columns(2)
        with col_a:
            mostrar_trechos = st.checkbox("Mostrar trechos", value=True)
        with col_b:
            modo_detalhado = st.checkbox("Exibição detalhada", value=False)
        
        num_documentos = st.slider("Documentos a recuperar", 3, 20, 10)
        
        st.markdown("---")
        st.markdown("### Estatísticas")
        
        # Mostrar estatísticas básicas
        if relatorio:
            total_docs = len(relatorio)
            st.metric("Total de Documentos", total_docs)
            
            # Top 3 tipos de documentos
            tipos_count = {}
            for doc in relatorio:
                tipo = doc.get("tipo")
                if tipo:
                    tipos_count[tipo] = tipos_count.get(tipo, 0) + 1
            
            # Mostrar top 3 tipos
            if tipos_count:
                st.write("**Tipos mais comuns:**")
                top_tipos = sorted(tipos_count.items(), key=lambda x: x[1], reverse=True)[:3]
                for tipo, count in top_tipos:
                    st.write(f"- {tipo}: {count}")
    
    # Converter seleções para valores para a busca
    tipo_filtro = None if tipo_selecionado == "Todos" else tipo_selecionado
    ano_filtro = None if ano_selecionado == "Todos" else ano_selecionado
    numero_filtro = None if not numero_documento else numero_documento
    
    # Título e descrição
    st.markdown("""
    ### Assistente de Consulta Inteligente
    
    Este sistema utiliza Inteligência Artificial para consultar documentos oficiais da ANTT. 
    Digite sua pergunta abaixo para obter informações detalhadas extraídas dos documentos.
    """)
    
    # Mostrar filtros ativos
    filtros_ativos = []
    if tipo_filtro:
        filtros_ativos.append(f"Tipo: {tipo_filtro}")
    if ano_filtro:
        filtros_ativos.append(f"Ano: {ano_filtro}")
    if numero_filtro:
        filtros_ativos.append(f"Número: {numero_filtro}")
    
    if filtros_ativos:
        st.info("📋 **Filtros ativos:** " + ", ".join(filtros_ativos))
    
    # Campo de pergunta
    pergunta = st.text_area("Digite sua pergunta sobre documentos da ANTT:", height=100)
    
    # Exemplos de perguntas
    with st.expander("📝 Exemplos de perguntas"):
        st.write("""
        - Quais são os parâmetros de desempenho de pavimento definidos pela ANTT?
        - Explique os principais pontos da Instrução Normativa nº 34 de 2024.
        - Quais são os valores limites de deflexão para pavimentos segundo a normativa mais recente?
        - Como são medidos os índices de irregularidade longitudinal nos contratos de concessão?
        - Quais são os prazos para conformidade com os parâmetros técnicos estabelecidos?
        """)
    
    col1, col2 = st.columns([1, 5])
    with col1:
        button = st.button("🔍 Pesquisar", type="primary", use_container_width=True)
    with col2:
        st.write("")  # Espaçamento
    
    if button and pergunta:
        with st.spinner("⌛ Analisando documentos..."):
            # Pesquisar documentos com os filtros aplicados
            start_time = time.time()
            documentos = pesquisar_documentos(
                pergunta, 
                vectorstore,
                k=num_documentos,
                tipo_documento=tipo_filtro, 
                ano=ano_filtro, 
                numero=numero_filtro
            )
            search_time = time.time() - start_time
            
            # Gerar resposta
            if documentos:
                with st.spinner("🧠 Gerando resposta..."):
                    start_time = time.time()
                    resposta = gerar_resposta(pergunta, documentos, llm)
                    response_time = time.time() - start_time
            else:
                resposta = "Não encontrei documentos relevantes para sua consulta. Por favor, tente reformular sua pergunta ou ajustar os filtros."
                response_time = 0
            
            # Mostrar tempos de processamento em formato discreto
            st.caption(f"⏱️ Pesquisa: {search_time:.2f}s | Resposta: {response_time:.2f}s")
            
            # Exibir resposta em um container destacado
            resposta_container = st.container(border=True)
            with resposta_container:
                st.markdown("### Resposta")
                st.markdown(resposta)
            
            # Sempre exibir os trechos mais relevantes
            # Independente da opção "mostrar_trechos" para exibir detalhes completos
            st.markdown("---")
            st.markdown("### Trechos dos Documentos Citados")
            
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
                st.subheader("Trechos dos documentos citados na resposta:", divider="blue")
                
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
                                
                                # Adicionar um marcador colorido para destacar
                                # Cor baseada no tipo de documento
                                cor_borda = "green"
                                if "Instrução" in doc_id:
                                    cor_borda = "blue"
                                elif "Resolução" in doc_id:
                                    cor_borda = "orange"
                                elif "Voto" in doc_id:
                                    cor_borda = "violet"
                                
                                with st.container(border=True):
                                    # Adicionar uma indicação visual da cor
                                    if "Instrução" in doc_id:
                                        st.markdown("🔵 **Instrução Normativa**")
                                    elif "Resolução" in doc_id:
                                        st.markdown("🟠 **Resolução**")
                                    elif "Voto" in doc_id:
                                        st.markdown("🟣 **Voto**")
                                    else:
                                        st.markdown("🟢 **Documento**")
                                        
                                    st.markdown(f"#### {doc_id}")
                                    st.caption(f"Parte {meta.get('chunk', 'N/A')}/{meta.get('total_chunks', 'N/A')} • Fonte: `{meta.get('caminho', 'Não especificado')}`")
                                    st.text_area(
                                        "",
                                        doc.page_content,
                                        height=150,
                                        key=f"citacao_{doc_display_id}",
                                        disabled=True
                                    )
                    
                    # Se não encontrou o documento, registrar isso
                    if not doc_encontrado:
                        print(f"Documento citado não encontrado: {doc_citado}")
            
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
                        else:
                            st.markdown("🟢 **Documento**")
                        
                        st.markdown(f"**{doc_id}** - Parte {meta.get('chunk', 'N/A')}/{meta.get('total_chunks', 'N/A')}")
                        st.caption(f"Fonte: `{meta.get('caminho', 'Não especificado')}`")
                        st.text_area(
                            "",
                            doc.page_content,
                            height=130,
                            key=f"relevante_{i}",
                            disabled=True
                        )
                        
                        if i < 2:  # Não adicionar separador após o último
                            st.divider()
            
            # Se a opção de mostrar trechos estiver ativada, exibir todos os documentos
            if mostrar_trechos and documentos:
                st.markdown("---")
                st.markdown("### Todas as Fontes Consultadas")
                
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
                    
                    with st.expander(f"{icone} {doc_id}{destaque}"):
                        
                        # Mostrar metadados do documento
                        if modo_detalhado:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"**Tipo:** {info['tipo']}")
                            with col2:
                                st.markdown(f"**Número:** {info['numero']}")
                            with col3:
                                st.markdown(f"**Ano:** {info['ano']}")
                            
                            st.markdown(f"**Caminho:** `{info['caminho']}`")
                        
                        # Exibir trechos em tabs
                        if len(info['trechos']) > 0:
                            trechos_tabs = st.tabs([f"Trecho {t['chunk']}/{t['total_chunks']}" for t in info['trechos']])
                            for i, tab in enumerate(trechos_tabs):
                                with tab:
                                    trecho = info['trechos'][i]
                                    st.text_area(
                                        "Conteúdo do trecho" if modo_detalhado else "", 
                                        trecho['conteudo'],
                                        height=150,
                                        key=f"trecho_{doc_id}_{i}"  # Chave única para cada text_area
                                    )
    
    # Adicionar rodapé com informações
    st.markdown("---")
    st.caption(
        "Sistema RAG (Retrieval Augmented Generation) desenvolvido para consulta aos documentos normativos da ANTT. "
        "Este sistema utiliza IA para extrair informações relevantes dos documentos oficiais."
    )

def extrair_citacoes_da_resposta(resposta):
    """Extrai as citações de documentos da resposta gerada."""
    # Lista de padrões para identificar citações
    padroes = [
        # Padrões de citação formal
        r"Documento:\s*([A-Za-zçÇáàâãéèêíìîóòôõúùûÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛ]+\s+\d+\/\d{4})",  # "Documento: TIPO NUMERO/ANO"
        r"Documento:\s*([A-Za-zçÇáàâãéèêíìîóòôõúùûÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛ]+\s+\d+\s+de\s+\d{4})",  # "Documento: TIPO NUMERO de ANO"
        
        # Padrões da seção de trechos
        r"TRECHOS DOS DOCUMENTOS CITADOS:[\s\S]*?(?:\*\*)?Documento:\s*([^\n\"]*\d{4})(?:\*\*)?",  # Após "TRECHOS DOS DOCUMENTOS CITADOS:"
        r"TRECHOS[\s\S]+?(?:\*\*)?([A-Za-zçÇáàâãéèêíìîóòôõúùûÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛ]+\s+\d+\/\d{4})(?:\*\*)?",  # Após palavra "TRECHOS"
        
        # Padrões específicos para tipos de documentos
        r"(?:^|\s)(?:a\s+)?([Ii]nstrução\s+[Nn]ormativa\s+(?:n[°º\.]?\s*)?\d+\/?\d*\s*(?:de\s*)?\d{4})",  # Instrução Normativa
        r"(?:^|\s)(?:a\s+)?([Rr]esolução\s+(?:n[°º\.]?\s*)?\d+\/?\d*\s*(?:de\s*)?\d{4})",  # Resolução
        r"(?:^|\s)(?:o\s+)?([Vv]oto\s+(?:n[°º\.]?\s*)?\d+\/?\d*\s*(?:de\s*)?\d{4})",  # Voto
        r"(?:^|\s)(?:a\s+)?([Dd]eliberação\s+(?:n[°º\.]?\s*)?\d+\/?\d*\s*(?:de\s*)?\d{4})",  # Deliberação
        r"(?:^|\s)(?:a\s+)?([Pp]ortaria\s+(?:n[°º\.]?\s*)?\d+\/?\d*\s*(?:de\s*)?\d{4})",  # Portaria

        # Padrões com números zerados
        r"([A-Za-zçÇáàâãéèêíìîóòôõúùûÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛ]+\s+0+(\d+)\/\d{4})",  # Tipo 00000NNN/ANO
        
        # Padrões da seção de fontes
        r"FONTES CONSULTADAS[\s\S]*?(?:\*\*)?([^\n\"]*\d{4})(?:\*\*)?",  # Após "FONTES CONSULTADAS:"
    ]
    
    # Verificar se há uma seção específica de trechos
    secoes_trechos = [
        r"TRECHOS DOS DOCUMENTOS CITADOS:([\s\S]+?)(?:$|(?:###|\*\*\*))",
        r"FONTES CONSULTADAS:([\s\S]+?)(?:$|(?:###|\*\*\*))"
    ]
    
    citacoes = []
    
    # Primeiro, tenta extrair da seção específica de trechos
    for padrao_secao in secoes_trechos:
        match_secao = re.search(padrao_secao, resposta, re.IGNORECASE)
        if match_secao:
            secao_trechos = match_secao.group(1)
            # Dentro da seção, busca pelas citações
            for padrao in padroes:
                matches = re.findall(padrao, secao_trechos, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    # Pode ser uma tupla ou string, dependendo do padrão
                    if isinstance(match, tuple):
                        match = match[0]  # Pegar o primeiro grupo capturado
                    citacao = match.strip()
                    if citacao and citacao not in citacoes:
                        citacoes.append(citacao)
    
    # Se não encontrou na seção específica, ou para complementar, busca no texto todo
    if not citacoes:
        for padrao in padroes:
            matches = re.findall(padrao, resposta, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                # Pode ser uma tupla ou string, dependendo do padrão
                if isinstance(match, tuple):
                    match = match[0]  # Pegar o primeiro grupo capturado
                citacao = match.strip()
                if citacao and citacao not in citacoes:
                    citacoes.append(citacao)
    
    # Limpar e normalizar citações
    citacoes_normalizadas = []
    for citacao in citacoes:
        # Remover marcações markdown
        citacao = re.sub(r'[*_`]', '', citacao)
        # Normalizar espaços
        citacao = re.sub(r'\s+', ' ', citacao).strip()
        if citacao and citacao not in citacoes_normalizadas:
            citacoes_normalizadas.append(citacao)
    
    print(f"Citações encontradas: {citacoes_normalizadas}")
    return citacoes_normalizadas

def main():
    # Obter a chave API usando a função segura em config.py
    api_key = get_openai_api_key()
    
    # Verifica se a chave API está disponível
    if not api_key or api_key.startswith("sk-") == False:
        # Tenta obter da variável de ambiente
        api_key = os.getenv("OPENAI_API_KEY", "")
        
        # Tenta obter da configuração de secrets, mas trata a exceção se não existir
        if not api_key:
            try:
                api_key = st.secrets.get("OPENAI_API_KEY", "")
            except:
                # Ignora o erro se não existir arquivo de secrets
                pass
        
        # Se ainda não tiver a chave, solicita ao usuário
        if not api_key:
            api_key = st.text_input("Insira sua chave da API OpenAI:", type="password")
            
    if not api_key or api_key.startswith("sk-") == False:
        st.error("Chave da API OpenAI não encontrada ou inválida. Por favor, insira uma chave válida.")
        return
    
    # Carregar vectorstore
    vectorstore = carregar_vectorstore(api_key)
    
    # Configurar LLM
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name=DEFAULT_LLM_MODEL, 
        temperature=0
    )
    
    # Criar chain QA
    qa_chain = criar_qa_chain(vectorstore, llm)
    
    # Executar interface - passar vectorstore e llm como parâmetros
    interface_usuario(vectorstore, llm)

if __name__ == "__main__":
    main()