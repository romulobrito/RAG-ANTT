import re
import unicodedata
import os
import tempfile

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
import spacy
import numpy as np

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
    Você está prestes a analisar o conteúdo de um documento PDF em inglês. Sua tarefa é identificar e classificar as seguintes informações do documento:

    1. **Título do Documento**: Identifique o título principal do documento.
    2. **Assunto do Documento**: Resuma o assunto principal abordado no documento, mesmo quando o título não estiver explicitamente claro no documento fornecido.

    Utilize o conteúdo do documento PDF fornecido para responder a essas perguntas. O documento pode conter seções como introdução, resumo e tópicos principais que ajudarão na identificação do título e assunto. Responda em português, garantindo clareza e precisão nas informações extraídas.

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
        api_key="sk-proj-SJzLGfezVCxJLft228F2T3BlbkFJ2lSCkYReBn53ZYbMfmKh"
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

    logo_path = r'/home/romulobrito/projetos/classification/logo-antt.jpg'
    logo = Image.open(logo_path)
    st.image(logo, width=700)

    st.header("Bem-vindo ao DocClassificador-ANTT!!!")
    st.write("🤖 Você pode carregar múltiplos arquivos PDF e, após o upload, classificar cada um deles.")
    st.write("🤖 Utilize o botão para carregar os arquivos e, em seguida, clique no botão de classificação.")

    uploaded_files = st.file_uploader("Carregue arquivos PDF 📝", type=["pdf"], accept_multiple_files=True)

    if st.button('Classificar'):
        if uploaded_files:
            results = []
            for uploaded_file in uploaded_files:
                st.write(f"Processando arquivo: {uploaded_file.name}")
                knowledgeBase = process_and_add_pdf(uploaded_file)
                llm = load_llm()
                prompt = load_prompt()

                # Extraindo e classificando o conteúdo
                similar_embeddings = knowledgeBase.similarity_search("Qual é o título e o assunto do documento?")
                retriever = FAISS.from_documents(
                    documents=similar_embeddings,
                    embedding=OpenAIEmbeddings(
                        api_key="sk-proj-SJzLGfezVCxJLft228F2T3BlbkFJ2lSCkYReBn53ZYbMfmKh"
                    ),
                ).as_retriever()

                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                response = rag_chain.invoke("Qual é o título e o assunto do documento?")
                results.append((uploaded_file.name, response))

            st.write("Classificação dos documentos:")
            for file_name, result in results:
                st.write(f"**{file_name}**: {result}")

        else:
            st.info("Nenhum arquivo PDF carregado.")
