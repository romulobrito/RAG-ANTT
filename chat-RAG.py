import logging
import time
from datetime import datetime

def setup_logging():
    """Configura o sistema de logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

# Inicializa o logger globalmente
logger = setup_logging()

# Resto das importações
import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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

# Constantes
OPENAI_API_KEY = 'sk-proj-SJzLGfezVCxJLft228F2T3BlbkFJ2lSCkYReBn53ZYbMfmKh'
# "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"
#"sk-proj-Ceq96zToY4PU7ezy732IT3BlbkFJL1cCp66BEjeN21b88XIf" #lenep
# 'sk-proj-SJzLGfezVCxJLft228F2T3BlbkFJ2lSCkYReBn53ZYbMfmKh'
DB_FAISS_PATH = "vectorstore/db_faiss"

from PIL import Image
import numpy as np
from img2table.document import PDF
from img2table.ocr import TesseractOCR


import logging
import time
from datetime import datetime


import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from openai import RateLimitError

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

def setup_logging():
    """Configura o sistema de logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

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




from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing



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
            api_key=OPENAI_API_KEY
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
                    chunk_size=1500,
                    chunk_overlap=300
                )
                splits = text_splitter.create_documents(documents)

                update_metrics(metrics, 
                             status="Criando embeddings", 
                             progress=0.95)
                
                # Cria embeddings com retry
                try:
                    embeddings = OpenAIEmbeddings(
                        api_key=OPENAI_API_KEY,
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




# def query_performance_parameters(question, tables_analysis, metrics=None, llm=None):
#     """
#     Responde a perguntas sobre parâmetros de desempenho com controle de tokens.
#     """
#     if metrics:
#         update_metrics(metrics, "Iniciando análise da consulta", progress=0)
        
#     if llm is None:
#         llm = ChatOpenAI(
#             model_name="gpt-4",
#             temperature=0,
#             api_key=OPENAI_API_KEY,
#             max_tokens=2000
#         )
#     time.sleep(3.5)
    
#     try:
#         if metrics:
#             update_metrics(metrics, "Filtrando tabelas relevantes", progress=20)
            
#         # Filtra tabelas relevantes
#         non_conforming_tables = []
#         conforming_tables = []
        
#         for table in tables_analysis:
#             if ("não atende" in table['descumprimentos'].lower() or 
#                 "não conforme" in table['descumprimentos'].lower()):
#                 non_conforming_tables.append(table)
#             else:
#                 conforming_tables.append(table)
        
#         if metrics:
#             update_metrics(metrics, "Preparando contexto", progress=40)
            
#         # Prepara contexto em chunks menores
#         def prepare_table_summary(table):
#             return f"""
#             Página {table['page_number']}, Tabela {table['table_number']}:
#             Análise: {table['descumprimentos'][:500]}
#             """
        
#         context = "RESUMO DAS NÃO CONFORMIDADES:\n"
#         if non_conforming_tables:
#             for table in non_conforming_tables[:5]:
#                 context += prepare_table_summary(table)
        
#         context += "\n\nRESUMO DAS CONFORMIDADES:\n"
#         if conforming_tables:
#             context += f"Total de {len(conforming_tables)} tabelas em conformidade.\n"
#             for table in conforming_tables[:3]:
#                 context += prepare_table_summary(table)
        
#         context = context[:4000]
        
#         if metrics:
#             update_metrics(metrics, "Gerando resposta", progress=60)
        
#         prompt = ChatPromptTemplate.from_template("""
#         Você é um especialista em análise de parâmetros de desempenho da ANTT.
#         Analise o contexto resumido e responda à pergunta do usuário.
        
#         Contexto:
#         {context}
        
#         Pergunta:
#         {question}
        
#         Forneça uma resposta objetiva focando em:
#         1. Não conformidades encontradas (se houver)
#         2. Localização das não conformidades
#         3. Motivos principais das não conformidades
#         4. Resumo dos parâmetros em conformidade
        
#         Limite sua resposta a 2000 caracteres.
#         """)
        
#         chain = prompt | llm | StrOutputParser()
        
#         if metrics:
#             update_metrics(metrics, "Processando resposta final", progress=80)
            
#         response = chain.invoke({
#             "context": context,
#             "question": question
#         })
        
#         if metrics:
#             update_metrics(metrics, "Análise concluída", progress=100)
            
#         return response, non_conforming_tables
        
#     except Exception as e:
#         if "rate_limit_exceeded" in str(e):
#             if metrics:
#                 update_metrics(metrics, "Limite de taxa atingido, aguardando...", progress=50)
#             logger.warning("Limite de taxa atingido, aguardando...")
#             time.sleep(5)
#             return query_performance_parameters(question, tables_analysis, metrics, llm)
#         else:
#             if metrics:
#                 update_metrics(metrics, f"Erro: {str(e)}", progress=0, error=str(e))
#             logger.error(f"Erro ao processar pergunta: {str(e)}")
#             return "Erro ao processar pergunta. Por favor, tente novamente em alguns segundos.", []

# def main():
#     """
#     Função principal do aplicativo Streamlit.
#     """
#     logger = setup_logging()
#     st.set_page_config(page_title="Chatbot ANTT", page_icon="🤖", layout="wide")
    
#     st.header("Análise de Descumprimentos de Parâmetros de Desempenho ANTT")
#     st.write("Este sistema é especializado em identificar e analisar descumprimentos de parâmetros de desempenho em documentos contratuais.")
    
#     # Métricas do sistema na barra lateral
#     system_metrics = st.sidebar.container()
#     with system_metrics:
#         st.subheader("Status do Sistema")
#         st.write("🟢 Sistema Online")
#         st.write(f"⏰ Atualizado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
#     uploaded_file = st.file_uploader("Carregue um documento PDF para análise 📄", type=["pdf"])
    
#     if uploaded_file:
#         try:
#             # Métricas para processamento do PDF
#             pdf_metrics = create_processing_metrics()
#             update_metrics(pdf_metrics, 
#                          status="Iniciando análise do documento",
#                          progress=0,
#                          info="Preparando para identificar descumprimentos")
            
#             vectorstore, tables_data = process_pdf(uploaded_file)
            
#             if tables_data:
#                 num_descumprimentos = len([t for t in tables_data if "Nenhum descumprimento identificado" not in t.get('descumprimentos', '')])
#                 update_metrics(pdf_metrics,
#                              status="Análise concluída",
#                              progress=100,
#                              info=f"Identificados {num_descumprimentos} casos de descumprimento em {len(tables_data)} tabelas analisadas")
                
#                 # Interface de consulta
#                 chat_container = st.container()
#                 with chat_container:
#                     st.markdown("### 🔍 Consulta de Descumprimentos")
                    
#                     # Campo de pergunta livre
#                     question = st.text_input(
#                         "Faça uma pergunta específica sobre os descumprimentos encontrados:",
#                         placeholder="Ex: Quais os principais descumprimentos identificados na página 5?"
#                     )
                    
#                     # Processamento da pergunta do usuário
#                     if question:
#                         chat_metrics = create_processing_metrics()
#                         with st.spinner('Analisando sua pergunta...'):
#                             try:
#                                 response, relevant_tables = query_performance_parameters(
#                                     question=question,
#                                     tables_analysis=tables_data,
#                                     metrics=chat_metrics,
#                                     llm=ChatOpenAI(
#                                         model_name="gpt-4",
#                                         temperature=0,
#                                         api_key=OPENAI_API_KEY
#                                     )
#                                 )
                                
#                                 # Exibe a resposta
#                                 st.markdown("### 📊 Análise")
#                                 st.write(response)
                                
#                                 # Exibe tabelas relevantes
#                                 if relevant_tables:
#                                     st.markdown("### ⚠️ Tabelas Relevantes")
#                                     for table_data in relevant_tables:
#                                         with st.expander(f"📑 Página {table_data['page_number']} - Tabela {table_data['table_number']}"):
#                                             if table_data.get('table_image'):
#                                                 st.image(table_data['table_image'],
#                                                        caption=f"Visualização da Tabela")
#                                             st.write(table_data['descumprimentos'])
                                            
#                             except Exception as e:
#                                 update_metrics(chat_metrics,
#                                              status="Erro na análise",
#                                              progress=0,
#                                              error=f"Erro ao processar consulta: {str(e)}")
                    
#                     # Sugestões de perguntas
#                     st.markdown("### 💡 Sugestões de Perguntas")
#                     col1, col2 = st.columns(2)
                    
#                     example_questions = [
#                         "Quais são os principais descumprimentos encontrados?",
#                         "Quais parâmetros estão fora dos limites estabelecidos?",
#                         "Em quais páginas há descumprimentos críticos?",
#                         "Qual o impacto dos descumprimentos identificados?",
#                         "Quais são os valores limite para cada parâmetro?"
#                     ]
                    
#                     # Distribui os botões em duas colunas
#                     for i, q in enumerate(example_questions):
#                         with col1 if i % 2 == 0 else col2:
#                             if st.button(q, key=f"btn_{i}"):
#                                 chat_metrics = create_processing_metrics()
#                                 with st.spinner('Analisando...'):
#                                     try:
#                                         response, relevant_tables = query_performance_parameters(
#                                             question=q,
#                                             tables_analysis=tables_data,
#                                             metrics=chat_metrics,
#                                             llm=ChatOpenAI(
#                                                 model_name="gpt-4",
#                                                 temperature=0,
#                                                 api_key=OPENAI_API_KEY
#                                             )
#                                         )
                                        
#                                         st.markdown("### 📊 Análise")
#                                         st.write(response)
                                        
#                                         if relevant_tables:
#                                             st.markdown("### ⚠️ Tabelas Relevantes")
#                                             for table_data in relevant_tables:
#                                                 with st.expander(f"📑 Página {table_data['page_number']} - Tabela {table_data['table_number']}"):
#                                                     if table_data.get('table_image'):
#                                                         st.image(table_data['table_image'],
#                                                                caption=f"Visualização da Tabela")
#                                                     st.write(table_data['descumprimentos'])
                                                    
#                                     except Exception as e:
#                                         update_metrics(chat_metrics,
#                                                      status="Erro na análise",
#                                                      progress=0,
#                                                      error=f"Erro ao processar consulta: {str(e)}")
#             else:
#                 st.warning("Nenhum descumprimento foi identificado no documento analisado.")
                
#         except Exception as e:
#             st.error(f"Erro ao processar o documento: {str(e)}")
#             logger.error(f"Erro no processamento: {str(e)}")

# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         st.error(f"Erro na inicialização: {str(e)}")





def main():
    """
    Função principal do aplicativo Streamlit.
    """
    logger = setup_logging()
    st.set_page_config(page_title="Chatbot ANTT", page_icon="🤖", layout="wide")
    
    st.header("Análise de Descumprimentos de Parâmetros de Desempenho ANTT")
    st.write("Este sistema é especializado em identificar e analisar descumprimentos de parâmetros de desempenho em documentos contratuais.")
    
    # Métricas do sistema na barra lateral
    system_metrics = st.sidebar.container()
    with system_metrics:
        st.subheader("Status do Sistema")
        st.write("🟢 Sistema Online")
        st.write(f"⏰ Atualizado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    uploaded_file = st.file_uploader("Carregue um documento PDF para análise 📄", type=["pdf"])
    
    if uploaded_file:
        try:
            pdf_metrics = create_processing_metrics()
            update_metrics(pdf_metrics, 
                         status="Iniciando análise do documento",
                         progress=0.0,
                         info="Preparando para identificar descumprimentos")
            
            vectorstore, tables_data = process_pdf(uploaded_file)
            
            if tables_data:
                num_descumprimentos = len([t for t in tables_data if "Nenhum descumprimento identificado" not in t.get('descumprimentos', '')])
                update_metrics(pdf_metrics,
                             status="Análise concluída",
                             progress=1.0,
                             info=f"Identificados {num_descumprimentos} casos de descumprimento em {len(tables_data)} tabelas analisadas")
                
                # Interface de consulta
                chat_container = st.container()
                with chat_container:
                    st.markdown("### 🔍 Consulta de Descumprimentos")
                    question = st.text_input(
                        "Faça uma pergunta específica sobre os descumprimentos encontrados:",
                        placeholder="Ex: Quais os principais descumprimentos identificados na página 5?"
                    )
                    
                    if question:
                        chat_metrics = create_processing_metrics()
                        update_metrics(chat_metrics, 
                                     status="Processando consulta",
                                     progress=0.0)
                        
                        try:
                            llm = ChatOpenAI(
                                model_name="gpt-4",
                                temperature=0,
                                api_key=OPENAI_API_KEY
                            )
                            
                            retriever = vectorstore.as_retriever()
                            
                            update_metrics(chat_metrics, 
                                         status="Analisando descumprimentos",
                                         progress=0.3,
                                         info="Buscando casos relevantes")
                            
                            prompt = ChatPromptTemplate.from_template("""
                            Você é um especialista em análise de descumprimentos de parâmetros de desempenho da ANTT.
                            Use o contexto fornecido para responder à pergunta, focando exclusivamente nos casos de descumprimento.

                            Contexto:
                            {context}

                            Pergunta:
                            {question}

                            Forneça uma resposta estruturada com:
                            1. Identificação dos parâmetros descumpridos
                            2. Localização exata (página e tabela)
                            3. Gravidade do descumprimento
                            4. Impacto potencial
                            5. Recomendações, se aplicável

                            Se não houver descumprimentos relevantes para a pergunta, indique claramente.
                            """)
                            
                            update_metrics(chat_metrics, 
                                         status="Gerando análise",
                                         progress=0.6,
                                         info="Elaborando resposta detalhada")
                            
                            rag_chain = (
                                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                                | prompt
                                | llm
                                | StrOutputParser()
                            )
                            
                            response = rag_chain.invoke(question)
                            
                            update_metrics(chat_metrics, 
                                         status="Preparando visualização",
                                         progress=0.8,
                                         info="Organizando resultados")
                            
                            # Exibe a resposta
                            st.markdown("### 📊 Análise de Descumprimentos")
                            st.write(response)
                            
                            # Identifica e exibe tabelas relevantes com descumprimentos
                            relevant_tables = []
                            for table_data in tables_data:
                                if (str(table_data['page_number']) in response or 
                                    str(table_data['table_number']) in response) and \
                                    "Nenhum descumprimento identificado" not in table_data.get('descumprimentos', ''):
                                    relevant_tables.append(table_data)
                            
                            if relevant_tables:
                                st.markdown("### ⚠️ Tabelas com Descumprimentos Identificados")
                                for table_data in relevant_tables:
                                    with st.expander(f"📑 Página {table_data['page_number']} - Tabela {table_data['table_number']}"):
                                        if table_data.get('table_image'):
                                            st.image(table_data['table_image'],
                                                   caption=f"Visualização da Tabela")
                                        
                                        # st.markdown("**🚫 Descumprimentos Identificados:**")
                                        # st.write(table_data['descumprimentos'])
                                        
                                        # st.markdown("**📋 Dados da Tabela:**")
                                        # st.write(table_data['table_content'])
                            
                            update_metrics(chat_metrics, 
                                         status="Análise concluída",
                                         progress=1.0,
                                         info=f"Encontrados {len(relevant_tables)} casos relevantes")
                            
                        except Exception as e:
                            update_metrics(chat_metrics, 
                                         status="Erro na análise",
                                         progress=0.0,
                                         error=f"Erro ao processar consulta: {str(e)}",
                                         info="Tente reformular sua pergunta")
                            logger.error(f"Erro na análise: {str(e)}")
            else:
                st.warning("Nenhum descumprimento foi identificado no documento analisado.")
                
        except Exception as e:
            st.error(f"Erro ao processar o documento: {str(e)}")
            logger.error(f"Erro no processamento: {str(e)}")

if __name__ == "__main__":
    main()



















# import os
# import tempfile
# import streamlit as st
# from img2table.document import PDF
# from img2table.ocr import TesseractOCR
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.vectorstores import FAISS
# import numpy as np
# from PIL import Image

# from langchain.text_splitter import RecursiveCharacterTextSplitter

# import logging
# import time

# from multiprocessing import Pool, cpu_count
# import concurrent.futures


# import concurrent.futures
# from functools import partial
# import multiprocessing

# # Configuração do logging
# logging.basicConfig(level=logging.INFO,
#                    format='%(asctime)s - %(levelname)s - %(message)s',
#                    datefmt='%Y-%m-%d %H:%M:%S')
# logger = logging.getLogger(__name__)

# OPENAI_API_KEY = "sk-proj-Ceq96zToY4PU7ezy732IT3BlbkFJL1cCp66BEjeN21b88XIf"
# #'sk-proj-SJzLGfezVCxJLft228F2T3BlbkFJ2lSCkYReBn53ZYbMfmKh'   # 7d
# DB_FAISS_PATH = "vectorstore/db_faiss"



# from multiprocessing import Pool, cpu_count
# import concurrent.futures
# import time

# def extract_and_analyze_tables(pdf_path, update_metrics_callback=None):
#     """
#     Extrai e analisa tabelas do PDF usando processamento paralelo otimizado.
#     """
#     logger.info(f"Iniciando processamento do PDF: {pdf_path}")
    
#     try:
#         # Configuração básica do OCR
#         ocr = TesseractOCR(
#             n_threads=min(4, cpu_count()),  # Limita a 4 threads
#             lang="por"
#         )
#         logger.info("OCR configurado com sucesso")
        
#         # Carrega o PDF
#         pdf = PDF(
#             src=pdf_path,
#             detect_rotation=False,
#             pdf_text_extraction=True
#         )
#         logger.info("PDF carregado com sucesso")
        
#         # Extrai tabelas com configurações mínimas
#         extracted_tables = pdf.extract_tables(
#             ocr=ocr,
#             implicit_rows=False,
#             borderless_tables=False
#         )
        
#         if not extracted_tables:
#             logger.warning("Nenhuma tabela encontrada")
#             return []
            
#         # Prepara as tabelas para processamento
#         tables_to_process = []
#         total_tables = 0
        
#         for page_num, tables in extracted_tables.items():
#             if not tables:
#                 continue
#             total_tables += len(tables)
#             for idx, table in enumerate(tables):
#                 if table and hasattr(table, 'html_repr'):
#                     tables_to_process.append({
#                         'page': page_num,
#                         'idx': idx,
#                         'table': table
#                     })
        
#         logger.info(f"Total de tabelas encontradas: {total_tables}")
        
#         # Processa as tabelas em paralelo usando ThreadPoolExecutor
#         tables_with_analysis = []
#         processed = 0
        
#         with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
#             futures = []
            
#             for table_info in tables_to_process:
#                 futures.append(
#                     executor.submit(
#                         process_single_table,
#                         table_info['table'],
#                         table_info['page'],
#                         table_info['idx'],
#                         pdf
#                     )
#                 )
            
#             for future in concurrent.futures.as_completed(futures):
#                 try:
#                     result = future.result()
#                     if result:
#                         tables_with_analysis.append(result)
#                         processed += 1
                        
#                         if update_metrics_callback:
#                             current_page = result['page_number']
#                             update_metrics_callback(
#                                 current_page,
#                                 processed,
#                                 total_tables,
#                                 f"Processada tabela {processed}/{total_tables}"
#                             )
#                 except Exception as e:
#                     logger.error(f"Erro ao processar tabela: {str(e)}")
        
#         logger.info(f"Processamento concluído. Tabelas processadas: {len(tables_with_analysis)}")
#         return tables_with_analysis
        
#     except Exception as e:
#         logger.error(f"Erro ao extrair tabelas: {str(e)}")
#         raise







# def process_single_table(table, page_num, idx, pdf_doc):
#     """
#     Processa uma única tabela com seu contexto e imagem.
#     """
#     try:
#         table_text = table.html_repr()
#         if not table_text or not table_text.strip():
#             return None
        
#         # Extrai o contexto e imagem da tabela
#         context = get_table_context(pdf_doc, page_num, table)
#         table_image = get_cropped_table(table, page_num, pdf_doc)
        
#         # Análise com texto, contexto e imagem
#         analysis = process_table_content(
#             table_text=table_text,
#             context=context,
#             table_image=table_image
#         )
        
#         return {
#             'page_number': page_num + 1,
#             'table_number': idx + 1,
#             'table_content': table_text,
#             'table_image': table_image,
#             'context': context,
#             'performance_analysis': analysis
#         }
#     except Exception as e:
#         logger.error(f"Erro ao processar tabela {idx + 1} da página {page_num + 1}: {str(e)}")
#         return None


# def get_cropped_table(table, page, pdf):
#     """
#     Extrai uma imagem recortada da tabela do PDF.
#     """
#     try:
#         bbox = table.bbox
#         if bbox and hasattr(pdf, 'images') and page in pdf.images:
#             table_array = np.array(pdf.images[page][bbox.y1:bbox.y2, bbox.x1:bbox.x2])
#             return Image.fromarray(table_array.astype('uint8'))
#     except Exception as e:
#         logger.error(f"Erro ao extrair imagem da tabela: {str(e)}")
#     return None


# # def process_tables_parallel(extracted_tables, pdf_doc, update_metrics_callback=None):
# #     """
# #     Processa as tabelas extraídas em paralelo.
# #     """
# #     logger.info("Iniciando processamento das tabelas...")
# #     start_time = time.time()
# #     tables_with_analysis = []
# #     processed_tables = 0
    
# #     # Verifica se extracted_tables é um dicionário
# #     if isinstance(extracted_tables, dict):
# #         total_tables = sum(len(tables) for tables in extracted_tables.values())
# #     else:
# #         total_tables = len(extracted_tables)
# #         # Converte para formato esperado
# #         extracted_tables = {0: extracted_tables}
    
# #     if total_tables == 0:
# #         logger.warning("Nenhuma tabela para processar")
# #         return []
    
# #     for page, tables in extracted_tables.items():
# #         if not tables:
# #             continue
            
# #         for idx, table in enumerate(tables):
# #             if table is None:
# #                 continue
                
# #             try:
# #                 # Verifica se a tabela tem conteúdo válido
# #                 if not hasattr(table, 'html_repr'):
# #                     continue
                    
# #                 table_text = table.html_repr()
# #                 if not table_text or not table_text.strip():
# #                     continue
                
# #                 # Processa a tabela
# #                 logger.info(f"Processando tabela {idx + 1} da página {page + 1}")
                
# #                 try:
# #                     table_image = get_cropped_table(table, page, pdf_doc)
# #                     analysis = process_table_content(table_text)
                    
# #                     tables_with_analysis.append({
# #                         'page_number': page + 1,
# #                         'table_number': idx + 1,
# #                         'table_content': table_text,
# #                         'table_image': table_image,
# #                         'performance_analysis': analysis
# #                     })
                    
# #                     processed_tables += 1
# #                     if update_metrics_callback:
# #                         update_metrics_callback(
# #                             page + 1,
# #                             processed_tables,
# #                             total_tables,
# #                             f"Processada tabela {idx + 1} da página {page + 1}"
# #                         )
                        
# #                 except Exception as e:
# #                     logger.error(f"Erro ao processar conteúdo da tabela {idx + 1} da página {page + 1}: {str(e)}")
                    
# #             except Exception as e:
# #                 logger.error(f"Erro ao processar tabela {idx + 1} da página {page + 1}: {str(e)}")
    
# #     logger.info(f"Processamento concluído. Tabelas processadas: {len(tables_with_analysis)}")
# #     return tables_with_analysis





# def analyze_performance_parameters(table_content, llm):
#     """
#     Analisa os parâmetros de desempenho na tabela com controle de tamanho.
#     """
#     # Dividir tabelas muito grandes
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=3000,
#         chunk_overlap=200,
#         length_function=len,
#     )
    
#     if len(table_content) > 3000:
#         chunks = text_splitter.split_text(table_content)
#         analyses = []
        
#         for chunk in chunks:
#             prompt = ChatPromptTemplate.from_template("""
#             Você é um especialista em análise de parâmetros de desempenho da ANTT.
#             Analise o trecho da tabela abaixo com foco em não conformidades.
            
#             Trecho da Tabela:
#             {table_content}
            
#             Identifique:
#             1. Verifique qualquer conteúdo no contexto como "Não atende", "Não cumpre" ou termos similares. 
#             Em caso, de existencia classifique como estando em Não Conformidade.
#             1. Parâmetros que NÃO atendem aos requisitos
#             2. Valores encontrados vs valores esperados
#             3. Percentual de não conformidade (quando aplicável)
            
#             Análise:
#             """)
            
#             chain = prompt | llm | StrOutputParser()
#             analysis = chain.invoke({"table_content": chunk})
#             analyses.append(analysis)
        
#         # Consolidar análises
#         final_prompt = ChatPromptTemplate.from_template("""
#         Consolide as análises parciais abaixo em uma única análise coerente.
        
#         Análises Parciais:
#         {analyses}
        
#         Forneça uma análise consolidada que destaque as não conformidades encontradas.
#         """)
        
#         chain = final_prompt | llm | StrOutputParser()
#         return chain.invoke({"analyses": "\n\n".join(analyses)})
    
#     else:
#         # Para tabelas pequenas, usar a análise original
#         prompt = ChatPromptTemplate.from_template("""
#         Você é um especialista em análise de parâmetros de desempenho da ANTT.
#         Analise a tabela abaixo com foco em não conformidades.
        
#         Tabela:
#         {table_content}
        
#         Por favor, forneça:
#         1. Identificação clara dos parâmetros que NÃO atendem aos requisitos
#         2. Valores encontrados vs valores esperados
#         3. Percentual de não conformidade (quando aplicável)
#         4. Impacto potencial dessas não conformidades
        
#         Se todos os parâmetros estiverem conformes, indique explicitamente.
#         """)
        
#         chain = prompt | llm | StrOutputParser()
#         return chain.invoke({"table_content": table_content})




# @st.cache_data(ttl=3600, show_spinner=False)
# def get_cropped_table(table, page, pdf):
#     """
#     Extrai uma imagem recortada da tabela do PDF com cache.
#     """
#     bbox = table.bbox
#     table_array = np.array(pdf.images[page][bbox.y1:bbox.y2, bbox.x1:bbox.x2])
#     return Image.fromarray(table_array.astype('uint8'))



# def process_table_content(table_text, context=None, table_image=None):
#     """
#     Processa o conteúdo da tabela com foco específico em não conformidades.
#     """
#     try:
#         # Inicializa o modelo
#         llm = ChatOpenAI(
#             model_name="gpt-4",
#             temperature=0,
#             api_key=OPENAI_API_KEY,
#             max_tokens=1000
#         )
#         time.sleep(3.5)
#         # Lista de termos que indicam não conformidade
#         non_conformity_terms = [
#             "não atende", "não cumpre", "não conforme", "não conformidade",
#             "abaixo do esperado", "inferior ao limite", "acima do limite",
#             "fora do padrão", "inadequado", "insatisfatório", "reprovado",
#             "em desacordo", "divergente", "não alcançou", "não atingiu"
#         ]
        
#         # Prepara o contexto
#         content = [
#             {
#                 "type": "text",
#                 "text": f"""
#                 Analise rigorosamente a tabela e seu contexto para identificar não conformidades.
                
#                 Contexto:
#                 {context if context else ''}
                
#                 Texto da Tabela:
#                 {table_text}
                
#                 INSTRUÇÕES ESPECÍFICAS:
#                 1. BUSCA DE NÃO CONFORMIDADES:
#                    - Procure especificamente por termos como: {', '.join(non_conformity_terms)}
#                    - Identifique valores numéricos fora dos limites estabelecidos
#                    - Compare valores encontrados com valores de referência
                
#                 2. ANÁLISE DE PARÂMETROS:
#                    - Verifique CADA parâmetro individualmente
#                    - Compare com os requisitos/limites estabelecidos
#                    - Indique explicitamente se está conforme ou não conforme
                
#                 3. CLASSIFICAÇÃO:
#                    - Se QUALQUER parâmetro estiver não conforme, a tabela deve ser classificada como NÃO CONFORME
#                    - Apenas classifique como CONFORME se TODOS os parâmetros atenderem aos requisitos
                
#                 4. DETALHAMENTO OBRIGATÓRIO:
#                    - Liste cada parâmetro não conforme encontrado
#                    - Indique o valor encontrado vs. valor esperado
#                    - Calcule o percentual de desvio quando aplicável
                
#                 IMPORTANTE: Se houver QUALQUER indicação de não conformidade, mesmo que em apenas um parâmetro,
#                 você DEVE classificar a tabela como NÃO CONFORME e detalhar o motivo.
#                 """
#             }
#         ]
        
#         # Adiciona a imagem se disponível
#         if table_image:
#             try:
#                 import base64
#                 from io import BytesIO
                
#                 buffered = BytesIO()
#                 table_image.save(buffered, format="PNG")
#                 img_str = base64.b64encode(buffered.getvalue()).decode()
                
#                 content.append({
#                     "type": "image",
#                     "image_url": {
#                         "url": f"data:image/png;base64,{img_str}",
#                         "detail": "high"
#                     }
#                 })
#             except Exception as e:
#                 logger.warning(f"Erro ao processar imagem: {str(e)}")
        
#         messages = [
#             {
#                 "role": "system",
#                 "content": """Você é um especialista rigoroso em análise de conformidade.
#                 Sua função é identificar QUALQUER não conformidade, por menor que seja.
#                 NUNCA classifique como conforme se houver QUALQUER indicação de não conformidade."""
#             },
#             {
#                 "role": "user",
#                 "content": content
#             }
#         ]
        
#         response = llm.invoke(messages)
#         return response.content
        
#     except Exception as e:
#         logger.error(f"Erro ao analisar tabela: {str(e)}")
#         return "Não foi possível analisar esta tabela."



# # def query_performance_parameters(question, tables_analysis, llm=None):
# #     """
# #     Responde a perguntas sobre parâmetros de desempenho baseado nas análises das tabelas.
    
# #     Args:
# #         question (str): Pergunta do usuário
# #         tables_analysis (list): Lista de dicionários com análises das tabelas
# #         llm (ChatOpenAI, optional): Instância do modelo LLM
    
# #     Returns:
# #         str: Resposta elaborada pelo modelo
# #     """
# #     if llm is None:
# #         llm = ChatOpenAI(
# #             model_name="gpt-4",
# #             temperature=0,
# #             api_key=OPENAI_API_KEY
# #         )
    
# #     # Prepara o contexto com todas as análises relevantes
# #     context = "\n\n".join([
# #         f"""
# #         Página {table['page_number']}, Tabela {table['table_number']}:
        
# #         Análise de Parâmetros:
# #         {table['performance_analysis']}
        
# #         Conteúdo da Tabela:
# #         {table['table_content']}
# #         """
# #         for table in tables_analysis
# #     ])
    
# #     prompt = ChatPromptTemplate.from_template("""
# #     Você é um especialista em análise de parâmetros de desempenho da ANTT.
# #     Use as informações fornecidas para responder à pergunta do usuário.
    
# #     Contexto das Tabelas e Análises:
# #     {context}
    
# #     Pergunta do Usuário:
# #     {question}
    
# #     Por favor, forneça uma resposta detalhada que:
# #     1. Identifique as tabelas relevantes para a pergunta
# #     2. Explique os parâmetros de desempenho relacionados
# #     3. Forneça valores específicos e análises quando aplicável
# #     4. Mencione metas ou limites estabelecidos, se houver
    
# #     Resposta:
# #     """)
    
# #     chain = prompt | llm | StrOutputParser()
# #     return chain.invoke({
# #         "context": context,
# #         "question": question
# #     })

# def get_table_context(pdf, page_num, table):
#     """
#     Extrai o contexto textual ao redor da tabela com melhor tratamento de erros.
#     """
#     try:
#         page = pdf.pages[page_num]
#         page_text = page.extract_text()
        
#         # Verifica se temos coordenadas válidas da tabela
#         if not hasattr(table, 'bbox') or not table.bbox:
#             # Tenta extrair coordenadas da tabela de forma alternativa
#             table_coords = {
#                 'top': min(cell['top'] for row in table.cells for cell in row if cell),
#                 'bottom': max(cell['bottom'] for row in table.cells for cell in row if cell),
#                 'x0': min(cell['x0'] for row in table.cells for cell in row if cell),
#                 'x1': max(cell['x1'] for row in table.cells for cell in row if cell)
#             }
#         else:
#             table_coords = {
#                 'top': table.bbox[1],
#                 'bottom': table.bbox[3],
#                 'x0': table.bbox[0],
#                 'x1': table.bbox[2]
#             }
        
#         # Extrai palavras da página
#         words = page.extract_words()
        
#         # Separa texto antes e depois da tabela
#         text_before = []
#         text_after = []
        
#         for word in words:
#             try:
#                 if word['top'] < table_coords['top']:
#                     text_before.append(word['text'])
#                 elif word['top'] > table_coords['bottom']:
#                     text_after.append(word['text'])
#             except (KeyError, TypeError):
#                 continue
        
#         # Limita o tamanho do contexto
#         text_before = ' '.join(text_before[-100:])  # últimas 100 palavras
#         text_after = ' '.join(text_after[:100])    # primeiras 100 palavras
        
#         return {
#             'before': text_before.strip(),
#             'after': text_after.strip(),
#             'page_text': page_text  # Inclui todo o texto da página como fallback
#         }
        
#     except Exception as e:
#         logger.warning(f"Erro ao extrair contexto específico da tabela: {str(e)}")
#         # Fallback: retorna todo o texto da página
#         try:
#             page_text = pdf.pages[page_num].extract_text()
#             return {
#                 'before': '',
#                 'after': '',
#                 'page_text': page_text
#             }
#         except:
#             logger.error(f"Erro ao extrair texto da página: {str(e)}")
#             return {
#                 'before': '',
#                 'after': '',
#                 'page_text': ''
#             }


# def query_performance_parameters(question, tables_analysis, llm=None):
#     """
#     Responde a perguntas sobre parâmetros de desempenho com controle de tokens.
#     """
#     if llm is None:
#         llm = ChatOpenAI(
#             model_name="gpt-4",
#             temperature=0,
#             api_key=OPENAI_API_KEY,
#             max_tokens=2000  # Limita tokens da resposta
#         )
#     time.sleep(3.5)
#     # Filtra tabelas relevantes
#     non_conforming_tables = []
#     conforming_tables = []
    
#     for table in tables_analysis:
#         if ("não atende" in table['performance_analysis'].lower() or 
#             "não conforme" in table['performance_analysis'].lower()):
#             non_conforming_tables.append(table)
#         else:
#             conforming_tables.append(table)
    
#     # Prepara contexto em chunks menores
#     def prepare_table_summary(table):
#         return f"""
#         Página {table['page_number']}, Tabela {table['table_number']}:
#         Análise: {table['performance_analysis'][:500]}  # Limita tamanho da análise
#         """
    
#     context = "RESUMO DAS NÃO CONFORMIDADES:\n"
#     if non_conforming_tables:
#         for table in non_conforming_tables[:5]:  # Limita a 5 tabelas não conformes
#             context += prepare_table_summary(table)
    
#     context += "\n\nRESUMO DAS CONFORMIDADES:\n"
#     if conforming_tables:
#         context += f"Total de {len(conforming_tables)} tabelas em conformidade.\n"
#         for table in conforming_tables[:3]:  # Limita a 3 tabelas conformes
#             context += prepare_table_summary(table)

    
#     context = context[:4000]  # Limita a 4000 caracteres
    
#     prompt = ChatPromptTemplate.from_template("""
#     Você é um especialista em análise de parâmetros de desempenho da ANTT.
#     Analise o contexto resumido e responda à pergunta do usuário.
    
#     Contexto:
#     {context}
    
#     Pergunta:
#     {question}
    
#     Forneça uma resposta objetiva focando em:
#     1. Não conformidades encontradas (se houver)
#     2. Localização das não conformidades
#     3. Motivos principais das não conformidades
#     4. Resumo dos parâmetros em conformidade
    
#     Limite sua resposta a 2000 caracteres.
#     """)
    
#     try:
#         chain = prompt | llm | StrOutputParser()
#         response = chain.invoke({
#             "context": context,
#             "question": question
#         })
        
#         return response, non_conforming_tables
        
#     except Exception as e:
#         if "rate_limit_exceeded" in str(e):
#             logger.warning("Limite de taxa atingido, aguardando...")
#             time.sleep(5)  # Espera mais tempo em caso de erro de limite
#             return query_performance_parameters(question, tables_analysis, llm)
#         else:
#             logger.error(f"Erro ao processar pergunta: {str(e)}")
#             return "Erro ao processar pergunta. Por favor, tente novamente em alguns segundos.", []





# # def main():
# #     logger.info("Iniciando aplicação...")
# #     st.set_page_config(page_title="Análise de Parâmetros de Desempenho ANTT", page_icon="📊", layout="wide")
    
# #     # Inicializa o estado da sessão para logs e métricas
# #     if "log_output" not in st.session_state:
# #         st.session_state.log_output = []
# #     if "processing_metrics" not in st.session_state:
# #         st.session_state.processing_metrics = {
# #             "total_tables": 0,
# #             "processed_tables": 0,
# #             "current_page": 0,
# #             "processing_time": 0,
# #             "status": "Aguardando arquivo..."
# #         }
    
# #     st.header("Análise de Parâmetros de Desempenho em Relatórios ANTT")
# #     st.write("Esta ferramenta extrai e analisa parâmetros de desempenho de relatórios contratuais.")
    
# #     # Área de métricas
# #     col1, col2, col3, col4 = st.columns(4)
# #     with col1:
# #         st.metric("Tabelas Encontradas", st.session_state.processing_metrics["total_tables"])
# #     with col2:
# #         st.metric("Tabelas Processadas", st.session_state.processing_metrics["processed_tables"])
# #     with col3:
# #         st.metric("Página Atual", st.session_state.processing_metrics["current_page"])
# #     with col4:
# #         st.metric("Tempo de Processamento", f"{st.session_state.processing_metrics['processing_time']:.2f}s")
    
# #     # Status atual
# #     st.info(f"Status: {st.session_state.processing_metrics['status']}")
    
# #     # Área de logs
# #     with st.expander("Logs de Processamento", expanded=True):
# #         log_placeholder = st.empty()
# #         log_text = "\n".join(st.session_state.log_output[-50:])  # Mantém os últimos 50 logs
# #         log_placeholder.text_area("", log_text, height=200)
    
# #     uploaded_file = st.file_uploader("Carregue um relatório PDF 📝", type=["pdf"])
    
# #     if uploaded_file:
# #         logger.info(f"Arquivo carregado: {uploaded_file.name}")
# #         st.session_state.processing_metrics["status"] = "Processando arquivo..."
        
# #         with st.spinner('Processando relatório e analisando parâmetros...'):
# #             try:
# #                 start_time = time.time()
# #                 with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
# #                     tmp_file.write(uploaded_file.getvalue())
# #                     tmp_file_path = tmp_file.name
# #                     log_message = f"Arquivo temporário criado: {tmp_file_path}"
# #                     logger.info(log_message)
# #                     st.session_state.log_output.append(log_message)
                
# #                 # Barra de progresso
# #                 progress_bar = st.progress(0)
                
# #                 def update_metrics(page, processed, total, status):
# #                     st.session_state.processing_metrics.update({
# #                         "current_page": page,
# #                         "processed_tables": processed,
# #                         "total_tables": total,
# #                         "processing_time": time.time() - start_time,
# #                         "status": status
# #                     })
# #                     if total > 0:
# #                         progress_bar.progress(processed / total)
                
# #                 tables_analysis = extract_and_analyze_tables(tmp_file_path, update_metrics)
                
# #                 processing_time = time.time() - start_time
# #                 st.session_state.processing_metrics["processing_time"] = processing_time
# #                 st.session_state.processing_metrics["status"] = "Processamento concluído"
                
# #                 log_message = f"Processamento concluído em {processing_time:.2f} segundos"
# #                 logger.info(log_message)
# #                 st.session_state.log_output.append(log_message)
                
# #                 st.success(f"Relatório processado com sucesso! Encontradas {len(tables_analysis)} tabelas")
                
# #                 st.subheader("📊 Resumo de Não Conformidades")
# #                 non_conformities = []
# #                 for table in tables_analysis:
# #                     if "não atende" in table['performance_analysis'].lower() or \
# #                     "não conforme" in table['performance_analysis'].lower():
# #                         non_conformities.append({
# #                             'page': table['page_number'],
# #                             'table': table['table_number'],
# #                             'analysis': table['performance_analysis']
# #                         })
                
# #                 if non_conformities:
# #                     for nc in non_conformities:
# #                         with st.expander(f"⚠️ Não Conformidade - Página {nc['page']}, Tabela {nc['table']}"):
# #                             st.write(nc['analysis'])
# #                 else:
# #                     st.success("✅ Todos os parâmetros analisados estão em conformidade!")
                
# #                 # Interface de perguntas e respostas
# #                 st.subheader("💬 Pergunte sobre o documento")
# #                 question = st.text_input("Digite sua pergunta sobre os parâmetros de desempenho:")
                
# #                 if question:
# #                     with st.spinner('Analisando sua pergunta...'):
# #                         response, non_conforming_tables = query_performance_parameters(question, tables_analysis)
# #                         st.write("Resposta:", response)
                        
# #                         # Mostra imagens das tabelas com não conformidades
# #                         if non_conforming_tables:
# #                             st.subheader("📊 Tabelas com Não Conformidades")
# #                             for table in non_conforming_tables:
# #                                 with st.expander(f"Tabela da Página {table['page_number']}"):
# #                                     if table.get('table_image'):
# #                                         st.image(table['table_image'], 
# #                                             caption=f"Tabela {table['table_number']} - Página {table['page_number']}")
# #                                     st.write("**Análise:**", table['performance_analysis'])
                
# #                 # Sugestões de perguntas
# #                 st.write("Sugestões de perguntas:")
# #                 example_questions = [
# #                     "Quais parâmetros não atendem aos requisitos?",
# #                     "Qual o percentual de conformidade para o parâmetro X?",
# #                     "Quais são os valores limite para cada parâmetro?",
# #                     "Onde estão as principais não conformidades?",
# #                     "Compare os valores encontrados com os valores esperados"
# #                 ]
# #                 for q in example_questions:
# #                     if st.button(q):
# #                         with st.spinner('Analisando...'):
# #                             response = query_performance_parameters(q, tables_analysis)
# #                             st.write("Resposta:", response)

                
# #             except Exception as e:
# #                 error_message = f"Erro ao processar o relatório: {str(e)}"
# #                 logger.error(error_message)
# #                 st.session_state.log_output.append(error_message)
# #                 st.session_state.processing_metrics["status"] = "Erro no processamento"
# #                 st.error(error_message)


# def main():
#     logger.info("Iniciando aplicação...")
#     st.set_page_config(page_title="Análise de Parâmetros de Desempenho ANTT", page_icon="📊", layout="wide")
    
#     # Inicialização do estado da sessão (mantido como está)
#     if "log_output" not in st.session_state:
#         st.session_state.log_output = []
#     if "processing_metrics" not in st.session_state:
#         st.session_state.processing_metrics = {
#             "total_tables": 0,
#             "processed_tables": 0,
#             "current_page": 0,
#             "processing_time": 0,
#             "status": "Aguardando arquivo..."
#         }
    
#     # Interface principal (mantida como está até o upload do arquivo)
#     st.header("Análise de Parâmetros de Desempenho em Relatórios ANTT")
#     st.write("Esta ferramenta extrai e analisa parâmetros de desempenho de relatórios contratuais.")
    
#     # Métricas e logs (mantidos como estão)
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Tabelas Encontradas", st.session_state.processing_metrics["total_tables"])
#     with col2:
#         st.metric("Tabelas Processadas", st.session_state.processing_metrics["processed_tables"])
#     with col3:
#         st.metric("Página Atual", st.session_state.processing_metrics["current_page"])
#     with col4:
#         st.metric("Tempo de Processamento", f"{st.session_state.processing_metrics['processing_time']:.2f}s")
    
#     st.info(f"Status: {st.session_state.processing_metrics['status']}")
    
#     with st.expander("Logs de Processamento", expanded=True):
#         log_placeholder = st.empty()
#         log_text = "\n".join(st.session_state.log_output[-50:])
#         log_placeholder.text_area("", log_text, height=200)
    
#     uploaded_file = st.file_uploader("Carregue um relatório PDF 📝", type=["pdf"])
    
#     if uploaded_file:
#         logger.info(f"Arquivo carregado: {uploaded_file.name}")
#         st.session_state.processing_metrics["status"] = "Processando arquivo..."
        
#         with st.spinner('Processando relatório e analisando parâmetros...'):
#             try:
#                 # Processamento inicial do arquivo
#                 start_time = time.time()
#                 with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
#                     tmp_file.write(uploaded_file.getvalue())
#                     tmp_file_path = tmp_file.name
#                     logger.info(f"Arquivo temporário criado: {tmp_file_path}")
                
#                 progress_bar = st.progress(0)
                
#                 def update_metrics(page, processed, total, status):
#                     st.session_state.processing_metrics.update({
#                         "current_page": page,
#                         "processed_tables": processed,
#                         "total_tables": total,
#                         "processing_time": time.time() - start_time,
#                         "status": status
#                     })
#                     if total > 0:
#                         progress_bar.progress(processed / total)
                
#                 # Extração e análise das tabelas
#                 tables_analysis = extract_and_analyze_tables(tmp_file_path, update_metrics)
                
#                 # Atualização das métricas finais
#                 processing_time = time.time() - start_time
#                 st.session_state.processing_metrics.update({
#                     "processing_time": processing_time,
#                     "status": "Processamento concluído"
#                 })
                
#                 st.success(f"Relatório processado com sucesso! Encontradas {len(tables_analysis)} tabelas")
                
#                 # Seção de não conformidades
#                 st.subheader("📊 Resumo de Não Conformidades")
#                 non_conformities = []
#                 for table in tables_analysis:
#                     if ("não atende" in table['performance_analysis'].lower() or 
#                         "não conforme" in table['performance_analysis'].lower()):
#                         non_conformities.append({
#                             'page': table['page_number'],
#                             'table': table['table_number'],
#                             'analysis': table['performance_analysis'],
#                             'image': table.get('table_image')
#                         })
                
#                 if non_conformities:
#                     for nc in non_conformities:
#                         with st.expander(f"⚠️ Não Conformidade - Página {nc['page']}, Tabela {nc['table']}"):
#                             if nc.get('image'):
#                                 st.image(nc['image'], 
#                                     caption=f"Tabela {nc['table']} - Página {nc['page']}")
#                             st.write(nc['analysis'])
#                 else:
#                     st.success("✅ Todos os parâmetros analisados estão em conformidade!")
                
#                 # Interface de perguntas
#                 st.subheader("💬 Pergunte sobre o documento")
                
#                 def process_question(q):
#                     with st.spinner('Analisando...'):
#                         response, non_conforming_tables = query_performance_parameters(q, tables_analysis)
#                         st.write("Resposta:", response)
                        
#                         if non_conforming_tables:
#                             st.subheader("📊 Tabelas Relevantes")
#                             for table in non_conforming_tables:
#                                 with st.expander(f"Tabela da Página {table['page_number']}"):
#                                     if table.get('table_image'):
#                                         st.image(table['table_image'], 
#                                             caption=f"Tabela {table['table_number']} - Página {table['page_number']}")
#                                     st.write("**Análise:**", table['performance_analysis'])
                
#                 # Campo de pergunta livre
#                 question = st.text_input("Digite sua pergunta sobre os parâmetros de desempenho:")
#                 if question:
#                     process_question(question)
                
#                 # Sugestões de perguntas
#                 st.write("Sugestões de perguntas:")
#                 example_questions = [
#                     "Quais parâmetros não atendem aos requisitos?",
#                     "Qual o percentual de conformidade para o parâmetro X?",
#                     "Quais são os valores limite para cada parâmetro?",
#                     "Onde estão as principais não conformidades?",
#                     "Compare os valores encontrados com os valores esperados"
#                 ]
                
#                 for q in example_questions:
#                     if st.button(q):
#                         process_question(q)
                
#             except Exception as e:
#                 error_message = f"Erro ao processar o relatório: {str(e)}"
#                 logger.error(error_message)
#                 st.session_state.log_output.append(error_message)
#                 st.session_state.processing_metrics["status"] = "Erro no processamento"
#                 st.error(error_message)

# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         st.error(f"Erro na inicialização: {str(e)}")