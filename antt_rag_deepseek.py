"""
Sistema RAG ANTT com suporte a múltiplos provedores de LLM.
Suporta OpenAI e DeepSeek via OpenRouter.
"""

import streamlit as st
import os
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

# Imports do LangChain
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Imports locais
from config import (
    DB_FAISS_PATH, 
    STREAMLIT_PAGE_TITLE, 
    STREAMLIT_PAGE_ICON, 
    STREAMLIT_LAYOUT,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_MODEL,
    logger
)
from llm_providers import create_llm_manager, get_available_providers

# Configuração da página
st.set_page_config(
    page_title=STREAMLIT_PAGE_TITLE,
    page_icon=STREAMLIT_PAGE_ICON,
    layout=STREAMLIT_LAYOUT
)

def load_vectorstore():
    """Carrega o vectorstore FAISS"""
    try:
        # Cria um LLMManager temporário para obter embeddings
        llm_manager = create_llm_manager("openai")  # Sempre usa OpenAI para embeddings
        embeddings = llm_manager.get_embeddings()
        
        vectorstore = FAISS.load_local(
            DB_FAISS_PATH, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("Vectorstore carregado com sucesso")
        return vectorstore
    except Exception as e:
        logger.error(f"Erro ao carregar vectorstore: {e}")
        return None

def load_document_stats():
    """Carrega estatísticas dos documentos do relatório"""
    try:
        with open('relatorio_documentos.json', 'r', encoding='utf-8') as f:
            relatorio = json.load(f)
        
        stats = {
            'total_documentos': relatorio.get('total_documentos', 0),
            'total_chunks': relatorio.get('total_chunks', 0),
            'tipos_documento': relatorio.get('tipos_documento', {}),
            'anos_disponiveis': relatorio.get('anos_disponiveis', [])
        }
        
        return stats
    except Exception as e:
        logger.error(f"Erro ao carregar estatísticas: {e}")
        return None

def create_rag_chain(vectorstore, llm_manager, filters=None):
    """Cria a cadeia RAG com filtros opcionais"""
    
    # Template de prompt especializado para ANTT
    template = """Você é um assistente especializado em regulamentação da ANTT (Agência Nacional de Transportes Terrestres).

Contexto dos documentos consultados:
{context}

Pergunta: {question}

Instruções:
1. Responda baseando-se EXCLUSIVAMENTE nas informações fornecidas no contexto
2. Se a informação não estiver no contexto, diga claramente que não encontrou a informação nos documentos consultados
3. Cite sempre o tipo de documento (resolução, portaria, etc.) e o número quando disponível
4. Seja preciso e objetivo, focando na regulamentação específica da ANTT
5. Se houver múltiplas fontes relevantes, organize a resposta de forma clara

Resposta:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Configurar retriever com filtros se fornecidos
    retriever_kwargs = {"k": 5}
    if filters:
        retriever_kwargs["filter"] = filters
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=retriever_kwargs
    )
    
    # Obter LLM do manager
    llm = llm_manager.get_llm(temperature=0.1)
    
    # Criar cadeia RAG
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return qa_chain

def format_sources(source_docs):
    """Formata os documentos fonte para exibição"""
    sources = []
    for i, doc in enumerate(source_docs, 1):
        metadata = doc.metadata
        
        source_info = f"**Fonte {i}:**\n"
        source_info += f"- **Tipo:** {metadata.get('tipo_documento', 'N/A')}\n"
        source_info += f"- **Número:** {metadata.get('numero_documento', 'N/A')}\n"
        source_info += f"- **Ano:** {metadata.get('ano', 'N/A')}\n"
        source_info += f"- **Arquivo:** {metadata.get('source', 'N/A')}\n"
        
        # Adicionar trecho do conteúdo
        content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        source_info += f"- **Trecho:** {content_preview}\n"
        
        sources.append(source_info)
    
    return sources

def main():
    """Função principal da aplicação"""
    
    # Título e descrição
    st.title("🚆 RAG ANTT - Sistema de Consulta a Documentos")
    st.markdown("Sistema de busca semântica em documentos regulatórios da ANTT")
    
    # Sidebar para configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Seleção do provedor de LLM
        providers = get_available_providers()
        provider_names = {k: v["name"] for k, v in providers.items()}
        
        selected_provider = st.selectbox(
            "Provedor de LLM:",
            options=list(provider_names.keys()),
            format_func=lambda x: provider_names[x],
            index=list(provider_names.keys()).index(DEFAULT_LLM_PROVIDER) if DEFAULT_LLM_PROVIDER in provider_names else 0
        )
        
        # Seleção do modelo
        available_models = providers[selected_provider]["models"]
        selected_model = st.selectbox(
            "Modelo:",
            options=available_models,
            index=available_models.index(DEFAULT_LLM_MODEL) if DEFAULT_LLM_MODEL in available_models else 0
        )
        
        st.divider()
        
        # Verificar chaves de API
        if selected_provider == "deepseek":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if api_key:
                st.success("✅ Chave OpenRouter configurada")
            else:
                st.error("❌ Chave OpenRouter não encontrada")
                st.info("Configure a variável OPENROUTER_API_KEY")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                st.success("✅ Chave OpenAI configurada")
            else:
                st.error("❌ Chave OpenAI não encontrada")
                st.info("Configure a variável OPENAI_API_KEY")
    
    # Carregar vectorstore
    with st.spinner("Carregando base de conhecimento..."):
        vectorstore = load_vectorstore()
    
    if not vectorstore:
        st.error("❌ Erro ao carregar a base de conhecimento. Verifique se o vectorstore foi criado.")
        return
    
    # Carregar estatísticas
    stats = load_document_stats()
    
    if stats:
        # Exibir estatísticas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de Documentos", stats['total_documentos'])
        
        with col2:
            st.metric("Total de Chunks", stats['total_chunks'])
        
        with col3:
            st.metric("Tipos de Documento", len(stats['tipos_documento']))
        
        with col4:
            st.metric("Anos Disponíveis", len(stats['anos_disponiveis']))
        
        # Filtros
        st.subheader("🔍 Filtros de Busca")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tipos_disponiveis = ["Todos"] + list(stats['tipos_documento'].keys())
            tipo_selecionado = st.selectbox("Tipo de Documento:", tipos_disponiveis)
        
        with col2:
            anos_disponiveis = ["Todos"] + sorted(stats['anos_disponiveis'], reverse=True)
            ano_selecionado = st.selectbox("Ano:", anos_disponiveis)
        
        with col3:
            numero_documento = st.text_input("Número do Documento (opcional):")
    
    # Interface de chat
    st.subheader("💬 Consulta aos Documentos")
    
    # Área de pergunta
    pergunta = st.text_area(
        "Digite sua pergunta sobre a regulamentação da ANTT:",
        height=100,
        placeholder="Ex: Quais são os requisitos para transporte de cargas perigosas?"
    )
    
    if st.button("🔍 Consultar", type="primary"):
        if not pergunta.strip():
            st.warning("Por favor, digite uma pergunta.")
            return
        
        try:
            # Criar LLM Manager
            with st.spinner("Inicializando modelo de linguagem..."):
                llm_manager = create_llm_manager(selected_provider, selected_model)
                
                # Exibir informações do provedor
                provider_info = llm_manager.get_provider_info()
                st.info(f"🤖 Usando: {provider_info['name']} - {provider_info['model_name']}")
            
            # Construir filtros
            filters = {}
            if stats:
                if tipo_selecionado != "Todos":
                    filters["tipo_documento"] = tipo_selecionado
                
                if ano_selecionado != "Todos":
                    filters["ano"] = ano_selecionado
                
                if numero_documento.strip():
                    filters["numero_documento"] = numero_documento.strip()
            
            # Criar cadeia RAG
            with st.spinner("Processando consulta..."):
                qa_chain = create_rag_chain(vectorstore, llm_manager, filters if filters else None)
                
                # Executar consulta
                resultado = qa_chain({"query": pergunta})
            
            # Exibir resposta
            st.subheader("📋 Resposta")
            st.write(resultado["result"])
            
            # Exibir fontes
            if resultado.get("source_documents"):
                st.subheader("📚 Fontes Consultadas")
                sources = format_sources(resultado["source_documents"])
                
                for source in sources:
                    with st.expander(f"Ver detalhes da fonte"):
                        st.markdown(source)
            
        except Exception as e:
            st.error(f"❌ Erro ao processar consulta: {str(e)}")
            logger.error(f"Erro na consulta: {e}")
    
    # Informações adicionais
    with st.expander("ℹ️ Sobre o Sistema"):
        st.markdown("""
        ### Como usar:
        1. **Selecione o provedor de LLM** na barra lateral (OpenAI ou DeepSeek)
        2. **Configure os filtros** para refinar sua busca (opcional)
        3. **Digite sua pergunta** sobre regulamentação da ANTT
        4. **Clique em Consultar** para obter a resposta
        
        ### Tipos de documentos disponíveis:
        - Resoluções
        - Portarias
        - Instruções Normativas
        - Deliberações
        - E outros documentos regulatórios
        
        ### Configuração de API:
        - **OpenAI**: Configure a variável `OPENAI_API_KEY`
        - **DeepSeek**: Configure a variável `OPENROUTER_API_KEY`
        """)

if __name__ == "__main__":
    main() 