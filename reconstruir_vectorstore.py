import os
import json
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

# Importar configurações do config.py
from config import (
    get_openai_api_key,
    DB_FAISS_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    logger
)

# Configurações específicas para esse script
DIR_DADOS = "dados_antt"
CAMINHO_INM34 = "dados_antt/INM/2024/INM-00000034-2024.md"

def ler_arquivo(caminho: str) -> str:
    """Lê o conteúdo de um arquivo."""
    try:
        with open(caminho, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Erro ao ler arquivo {caminho}: {str(e)}")
        return ""

def extrair_metadados(caminho: str) -> dict:
    """Extrai metadados do caminho do arquivo."""
    partes_caminho = caminho.split('/')
    tipo_documento = partes_caminho[1] if len(partes_caminho) > 1 else "DESCONHECIDO"
    ano = partes_caminho[2] if len(partes_caminho) > 2 else "0000"
    
    # Tentar extrair número do arquivo
    nome_arquivo = os.path.basename(caminho)
    numero = None
    if '-' in nome_arquivo:
        partes_nome = nome_arquivo.split('-')
        if len(partes_nome) >= 2:
            numero_parte = partes_nome[1]
            numero = numero_parte
    
    return {
        "tipo_documento": tipo_documento,
        "nome_tipo": mapear_tipo_documento(tipo_documento),
        "ano": ano,
        "numero": numero,
        "caminho": caminho
    }

def mapear_tipo_documento(tipo: str) -> str:
    """Mapeia a sigla do tipo de documento para seu nome completo."""
    mapeamento = {
        "RES": "Resolução",
        "POR": "Portaria",
        "INC": "Instrução Normativa Complementar",
        "DLB": "Deliberação",
        "INM": "Instrução Normativa"
    }
    return mapeamento.get(tipo, tipo)

def criar_documentos(texto: str, metadados: dict) -> List[Document]:
    """Divide o texto em chunks e cria documentos."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    
    # Dividir o texto em chunks
    textos_divididos = text_splitter.split_text(texto)
    
    # Criar documentos LangChain
    documentos = []
    for i, texto_chunk in enumerate(textos_divididos):
        # Adicionar informação de chunk aos metadados
        meta_chunk = metadados.copy()
        meta_chunk["chunk"] = i + 1
        meta_chunk["total_chunks"] = len(textos_divididos)
        
        # Criar documento LangChain
        doc = Document(page_content=texto_chunk, metadata=meta_chunk)
        documentos.append(doc)
    
    return documentos

def documentos_extras(caminho_inm34: str) -> List[Document]:
    """Processa o arquivo da Instrução Normativa específica."""
    documentos = []
    
    # Lista de documentos importantes a serem processados
    documentos_importantes = [
        caminho_inm34,  # Instrução Normativa 34/2024
        "dados_antt/INM/dados_antt/VTO/2024/VTO-00000097-2024.md",  # Voto que contém os parâmetros técnicos
    ]
    
    for caminho in documentos_importantes:
        # Ler conteúdo do arquivo
        conteudo = ler_arquivo(caminho)
        if not conteudo:
            logger.warning(f"Não foi possível ler o arquivo {caminho}")
            continue
        
        # Extrair metadados
        metadados = extrair_metadados(caminho)
        
        # Para o documento do voto, adicionar informações extras nos metadados
        if "VTO-00000097-2024" in caminho:
            metadados["relacionado_a"] = "INM-00000034-2024"
            metadados["contem_tabelas"] = "Sim"
            metadados["descricao"] = "Voto que contém parâmetros técnicos detalhados de pavimento"
        
        # Criar documentos com chunks menores para maior precisão
        docs = criar_documentos(conteudo, metadados)
        documentos.extend(docs)
        
        logger.info(f"Processado {caminho}: {len(docs)} chunks gerados")
    
    return documentos

def carregar_vectorstore_existente(embeddings):
    """Tenta carregar um vectorstore existente."""
    try:
        if os.path.exists(DB_FAISS_PATH):
            logger.info(f"Carregando vectorstore existente de {DB_FAISS_PATH}...")
            return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        logger.error(f"Erro ao carregar vectorstore existente: {str(e)}")
    return None

def main():
    logger.info(f"Reconstruindo vectorstore com foco na INM-00000034-2024...")
    
    # Inicializar embeddings com chave segura
    embeddings = OpenAIEmbeddings(openai_api_key=get_openai_api_key())
    
    # Processar a Instrução Normativa nº 34 de 2024
    docs_inm34 = documentos_extras(CAMINHO_INM34)
    
    if not docs_inm34:
        logger.error("Não foi possível processar a Instrução Normativa nº 34. Verifique o caminho e o arquivo.")
        return
    
    # Carregar vectorstore existente ou criar um novo
    vectorstore = carregar_vectorstore_existente(embeddings)
    
    if vectorstore:
        # Adicionar documentos ao vectorstore existente
        logger.info(f"Adicionando {len(docs_inm34)} documentos ao vectorstore existente...")
        vectorstore.add_documents(docs_inm34)
    else:
        # Criar novo vectorstore
        logger.info(f"Criando novo vectorstore com {len(docs_inm34)} documentos...")
        vectorstore = FAISS.from_documents(docs_inm34, embeddings)
    
    # Salvar vectorstore
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    vectorstore.save_local(DB_FAISS_PATH)
    logger.info(f"Vectorstore salvo em {DB_FAISS_PATH}")

if __name__ == "__main__":
    main() 