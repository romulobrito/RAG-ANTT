import re
import unicodedata
import os
import tempfile
import json

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from PIL import Image
from deep_translator import GoogleTranslator
import numpy as np
import pandas as pd
import pdfplumber

# nlp = spacy.load("en_core_web_sm")

DEFAULT_PDF_PATH = r"/home/romulobrito/projetos/test_RAG/chatbot7D-RAG/7d-qna-documentation/single_file_v3.pdf"

def translate_to_english(text):
    translator = GoogleTranslator(target='en')
    translated_text = translator.translate(text)
    return translated_text

# def preprocess_query(query):
#     query = query.lower()
#     query = re.sub(r'[^\w\s]', ' ', query)  
#     query = re.sub(r'\s+', ' ', query).strip()  
#     doc = nlp(query)
#     words = [token.lemma_ for token in doc if not token.is_stop]
#     processed_query = ' '.join(words)
#     return processed_query

def extract_tables_from_pdf(pdf_path, output_format='xlsx', start_page=None, end_page=None):
    """
    Extrai tabelas de um arquivo PDF e as salva no formato especificado.
    
    :param pdf_path: Caminho para o arquivo PDF
    :param output_format: Formato de saída ('xlsx' ou 'json')
    :param start_page: Página inicial para extração
    :param end_page: Página final para extração
    :return: Lista de caminhos para os arquivos de saída
    """
    output_files = []
    with pdfplumber.open(pdf_path) as pdf:
        # Ajusta os índices das páginas
        start_page = max(1, start_page or 1) - 1  # Converte para índice base-0
        end_page = min(len(pdf.pages), end_page or len(pdf.pages))
        
        # Itera apenas sobre as páginas selecionadas
        for i in range(start_page, end_page):
            page = pdf.pages[i]
            tables = page.extract_tables()
            for j, table in enumerate(tables):
                df = pd.DataFrame(table[1:], columns=table[0])
                if output_format == 'xlsx':
                    output_path = f"tabela_pagina{i+1}_tabela{j+1}.xlsx"
                    df.to_excel(output_path, index=False)
                elif output_format == 'json':
                    output_path = f"tabela_pagina{i+1}_tabela{j+1}.json"
                    df.to_json(output_path, orient='records')
                output_files.append(output_path)
    
    return output_files

# def identify_table_name(page_text, table_content, llm):
#     """
#     Usa o LLM para identificar o nome da tabela baseado no contexto.
#     """
#     prompt = ChatPromptTemplate.from_template("""
#     Analise o texto e o conteúdo da tabela fornecidos abaixo e identifique o nome ou título oficial da tabela.
#     Retorne APENAS o nome da tabela, sem explicações adicionais.
#     Se não encontrar um nome específico, retorne 'Não identificado'.

#     Texto da página:
#     {page_text}

#     Conteúdo da tabela:
#     {table_content}
#     """)

#     chain = prompt | llm | StrOutputParser()
    
#     return chain.invoke({
#         "page_text": page_text,
#         "table_content": str(table_content)
#     })


# def extract_tables_from_pdf(pdf_path, output_format='xlsx'):
#     """
#     Extrai tabelas de um arquivo PDF e as salva no formato especificado,
#     usando o nome identificado pelo LLM para nomear os arquivos.
#     """
#     output_files = []
#     llm = ChatOpenAI(
#         model_name="gpt-4",
#         temperature=0,
#         api_key="sk-proj-SJzLGfezVCxJLft228F2T3BlbkFJ2lSCkYReBn53ZYbMfmKh"
#     )

#     with pdfplumber.open(pdf_path) as pdf:
#         for i, page in enumerate(pdf.pages):
#             # Extrai todo o texto da página
#             page_text = page.extract_text()
            
#             # Extrai as tabelas
#             tables = page.extract_tables()
#             for j, table in enumerate(tables):
#                 # Usa o LLM para identificar o nome da tabela
#                 table_name = identify_table_name(page_text, table, llm)
                
#                 # Se o LLM não identificou um nome específico, usa o padrão
#                 if table_name == 'Não identificado':
#                     table_name = f"Tabela_{i+1}_{j+1}"
                
#                 # Limpa o nome da tabela para usar como nome de arquivo
#                 safe_table_name = re.sub(r'[^\w\s-]', '_', table_name)
#                 safe_table_name = re.sub(r'\s+', '_', safe_table_name)
                
#                 # Cria o DataFrame
#                 df = pd.DataFrame(table[1:], columns=table[0])
                
#                 if output_format == 'xlsx':
#                     output_path = f"{safe_table_name}.xlsx"
#                     with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
#                         # Aba de informações
#                         info_df = pd.DataFrame({
#                             'Informação': ['Nome da Tabela'],
#                             'Valor': [table_name]
#                         })
#                         info_df.to_excel(writer, sheet_name='Informações', index=False)
                        
#                         # Adiciona o nome da tabela como primeira coluna nos dados
#                         df.insert(0, 'Nome da Tabela', table_name)
#                         df.to_excel(writer, sheet_name='Dados', index=False)
                        
#                 elif output_format == 'json':
#                     output_path = f"{safe_table_name}.json"
#                     df.insert(0, 'Nome da Tabela', table_name)
#                     json_data = {
#                         "metadata": {
#                             "nome_tabela": table_name
#                         },
#                         "dados": df.to_dict(orient='records')
#                     }
#                     with open(output_path, 'w', encoding='utf-8') as f:
#                         json.dump(json_data, f, ensure_ascii=False, indent=4)
                
#                 output_files.append({
#                     "nome_tabela": table_name,
#                     "caminho_arquivo": output_path
#                 })
    
#     return output_files


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    return text

def load_knowledgeBase():
    embeddings = OpenAIEmbeddings(
        api_key="sk-proj-SJzLGfezVCxJLft228F2T3BlbkFJ2lSCkYReBn53ZYbMfmKh"
    )
    DB_FAISS_PATH = "vectorstore/db_faiss"
    if os.path.exists(DB_FAISS_PATH):
        db = FAISS.load_local(
            DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
        )
    else:
        db = None
    return db

def load_llm():
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.5,
        api_key="sk-proj-SJzLGfezVCxJLft228F2T3BlbkFJ2lSCkYReBn53ZYbMfmKh",
    )
    return llm

def load_prompt():
    prompt = """
    Você está prestes a analisar o conteúdo de um documento PDF. Sua tarefa é identificar e classificar as seguintes informações do documento:

    1. **Título do Documento**: Identifique o título principal do documento.
    2. **Assunto do Documento**: Resuma o assunto principal abordado no documento, mesmo quando o título não estiver explicitamente claro no documento fornecido.
    3. **Identifique os parâmetros de desempenho**: Traga de forma clara quais são os parâmetros de desempenho do documento.

    Utilize o conteúdo do documento PDF fornecido para responder a essas perguntas. O documento pode conter seções como introdução, resumo e tópicos principais que ajudarão na 
    identificação do título e assunto. Responda em português, garantindo clareza e precisão nas informações extraídas.

    ### Documento:
    {context}
    
    ### Perguntas:
    1. Qual é o título do documento?
    2. Qual é o assunto principal abordado no documento?

    Responda às perguntas acima com base no conteúdo do documento.
    """
    prompt = ChatPromptTemplate.from_template(prompt)
    return prompt


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def process_and_add_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()

    # Limpeza do texto dos documentos
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    splits = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(
        # api_key= "sk-proj-Ceq96zToY4PU7ezy732IT3BlbkFJL1cCp66BEjeN21b88XIf" 
        "sk-proj-SJzLGfezVCxJLft228F2T3BlbkFJ2lSCkYReBn53ZYbMfmKh"  # 7d    
    )
    
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local("vectorstore/db_faiss")
    
    os.remove(tmp_file_path)
        
    return vectorstore

if __name__ == "__main__":
    st.set_page_config(
        page_title="AI Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    logo_path = r'/home/romulobrito/projetos/classification_RAG_ANTT/antt-logo.png'
    logo = Image.open(logo_path)
    st.image(logo, width=700)

    st.header("Bem-vindo ao PDFto-xlsx-json!!!")
    st.write("🤖 Você pode carrega um arquivo PDF para a extração das tabelas em formato xlsx ou json.")
    st.write("🤖 Utilize o botão para carregar os arquivos e, em seguida, clique no botão de classificação.")

    # uploaded_files = st.file_uploader("Carregue o arquivo PDF 📝", type=["pdf"], accept_multiple_files=True)
    uploaded_files = st.file_uploader("Carregue o arquivo PDF  📝 :page_facing_up:", type=["pdf"], accept_multiple_files=True)
    
    output_format = st.selectbox("Selecione o formato de saída para as tabelas:", ['xlsx', 'json'])

    # Adiciona campos para seleção de páginas
    col1, col2 = st.columns(2)
    with col1:
        start_page = st.number_input("Página inicial", min_value=1, value=1, step=1)
    with col2:
        end_page = st.number_input("Página final", min_value=1, value=1, step=1)

        if st.button('Processar'):
            if uploaded_files:
                results = []
                file_pages = {}
                
                # Barra de progresso principal
                progress_text = "Operação em andamento. Por favor, aguarde."
                progress_bar = st.progress(0, text=progress_text)
                
                # Status para acompanhamento detalhado
                status = st.status("Processando arquivos...", expanded=True)
                
                # Calcula o total de arquivos para o progresso
                total_files = len(uploaded_files)
                
                # Primeiro loop: coleta informações sobre páginas
                with status.container():
                    st.write("Analisando arquivos...")
                    for idx, uploaded_file in enumerate(uploaded_files):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            with pdfplumber.open(tmp_file.name) as pdf:
                                total_pages = len(pdf.pages)
                                file_pages[uploaded_file.name] = total_pages
                                if end_page == 1:
                                    end_page = total_pages
                            os.remove(tmp_file.name)
                        # Atualiza a barra de progresso
                        progress = (idx + 1) / (total_files * 2)
                        progress_bar.progress(progress, text=f"Analisando arquivo {idx + 1} de {total_files}")
                
                # Validação das páginas selecionadas
                if start_page > end_page:
                    status.update(label="Erro!", state="error")
                    st.error("A página inicial não pode ser maior que a página final!")
                else:
                    # Mostrar informações sobre os arquivos carregados
                    with status.container():
                        st.write("Arquivos carregados:")
                    
                        # Segundo loop: processa cada arquivo
                        for idx, uploaded_file in enumerate(uploaded_files):
                            st.write(f"{uploaded_file.name} - {file_pages[uploaded_file.name]} páginas")
                            if end_page > file_pages[uploaded_file.name]:
                                st.warning(f"O arquivo {uploaded_file.name} tem apenas {file_pages[uploaded_file.name]} páginas. A extração será limitada até a última página.")
                            
                            with st.spinner(f"Extraindo tabelas do arquivo {uploaded_file.name}..."):
                                # Salva o arquivo temporariamente
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                                    tmp_file.write(uploaded_file.getvalue())
                                    tmp_file_path = tmp_file.name
                                
                                # Extrai tabelas com as páginas especificadas
                                table_files = extract_tables_from_pdf(
                                    tmp_file_path, 
                                    output_format,
                                    start_page=start_page,
                                    end_page=min(end_page, file_pages[uploaded_file.name])
                                )
                                
                                # Remove arquivo temporário
                                os.remove(tmp_file_path)
                            
                            # Atualiza a barra de progresso
                            progress = 0.5 + ((idx + 1) / (total_files * 2))
                            progress_bar.progress(progress, text=f"Processando arquivo {idx + 1} de {total_files}")
                    
                    # Atualiza status final
                    progress_bar.progress(1.0, text="Processamento concluído!")
                    status.update(label="Processamento concluído com sucesso!", state="complete")
                    
                    # Mostra resultados
                    with status.container():
                        st.write("Tabelas extraídas:")
                        for table_file in table_files:
                            st.write(f"{table_file}")
        else:
            st.info("Nenhum arquivo PDF carregado.")