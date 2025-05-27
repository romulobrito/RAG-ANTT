#!/usr/bin/env python3
"""
Script para recriar o vectorstore usando embeddings DeepSeek
"""

import os
import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from llm_providers import create_llm_manager
from config import setup_logging, logger

def recriar_vectorstore_deepseek():
    """Recria o vectorstore usando embeddings DeepSeek"""
    
    logger.info("🚀 Iniciando recriação do vectorstore com embeddings DeepSeek...")
    
    # Carregar dados do relatório
    relatorio_path = "relatorio_documentos.json"
    if not os.path.exists(relatorio_path):
        logger.error("❌ Arquivo relatorio_documentos.json não encontrado")
        return False
    
    with open(relatorio_path, 'r', encoding='utf-8') as f:
        dados_documentos = json.load(f)
    
    logger.info(f"📄 Carregados {len(dados_documentos)} documentos do relatório")
    
    # Criar documentos para o vectorstore
    documentos = []
    
    for doc_info in dados_documentos:
        # Ler o arquivo markdown se existir
        arquivo_md = doc_info.get('arquivo_md', '')
        if arquivo_md and os.path.exists(arquivo_md):
            try:
                with open(arquivo_md, 'r', encoding='utf-8') as f:
                    conteudo = f.read()
                
                # Criar documento com metadados
                doc = Document(
                    page_content=conteudo,
                    metadata={
                        'tipo_documento': doc_info.get('tipo', ''),
                        'nome_tipo': doc_info.get('tipo', ''),
                        'numero': doc_info.get('numero', ''),
                        'ano': doc_info.get('ano', ''),
                        'caminho': arquivo_md,
                        'titulo': doc_info.get('titulo', ''),
                        'ementa': doc_info.get('ementa', ''),
                        'orgao': doc_info.get('orgao', '')
                    }
                )
                documentos.append(doc)
                
            except Exception as e:
                logger.warning(f"⚠️ Erro ao ler {arquivo_md}: {e}")
    
    logger.info(f"📚 Preparados {len(documentos)} documentos para indexação")
    
    # Dividir documentos em chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        length_function=len,
    )
    
    logger.info("✂️ Dividindo documentos em chunks...")
    splits = text_splitter.split_documents(documentos)
    logger.info(f"📝 Criados {len(splits)} chunks")
    
    # Criar embeddings DeepSeek
    try:
        logger.info("🤖 Configurando embeddings DeepSeek...")
        llm_manager = create_llm_manager("deepseek")
        embeddings = llm_manager.get_embeddings()
        logger.info("✅ Embeddings DeepSeek configurados")
        
    except Exception as e:
        logger.error(f"❌ Erro ao configurar embeddings DeepSeek: {e}")
        return False
    
    # Criar vectorstore
    try:
        logger.info("🔍 Criando vectorstore com embeddings DeepSeek...")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        
        # Salvar vectorstore
        vectorstore_path = "vectorstore_deepseek"
        os.makedirs(vectorstore_path, exist_ok=True)
        vectorstore.save_local(vectorstore_path)
        
        logger.info(f"✅ Vectorstore DeepSeek salvo em {vectorstore_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao criar vectorstore: {e}")
        return False

if __name__ == "__main__":
    setup_logging()
    sucesso = recriar_vectorstore_deepseek()
    if sucesso:
        print("🎉 Vectorstore DeepSeek criado com sucesso!")
    else:
        print("❌ Falha ao criar vectorstore DeepSeek") 