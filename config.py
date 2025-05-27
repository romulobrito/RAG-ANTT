"""
Módulo de configuração para o sistema RAG-ANTT.
Gerencia variáveis de ambiente e configurações do aplicativo.
"""

import os
from dotenv import load_dotenv
import logging

# Carregar variáveis do arquivo .env, se existir
load_dotenv()

# Configuração de logging
def setup_logging():
    """Configura o sistema de logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

# Inicializar logger
logger = setup_logging()

# Função para obter a chave da API da OpenAI de maneira segura
def get_openai_api_key():
    """
    Obtém a chave da API OpenAI de fontes seguras na seguinte ordem:
    1. Variável de ambiente OPENAI_API_KEY
    2. Arquivo .env (carregado via python-dotenv)
    
    Em produção, sempre use variáveis de ambiente ou .env
    """
    # Tenta obter de variável de ambiente
    api_key = os.environ.get("OPENAI_API_KEY", "")
    
    if not api_key:
        logger.warning("ATENÇÃO: Chave da API OpenAI não encontrada. Configure a variável de ambiente OPENAI_API_KEY.")
        return ""  # Retorna string vazia em vez de chave padrão
        
    return api_key

# Função para obter a chave da API do OpenRouter (DeepSeek)
def get_openrouter_api_key():
    """
    Obtém a chave da API OpenRouter de fontes seguras na seguinte ordem:
    1. Variável de ambiente OPENROUTER_API_KEY
    2. Arquivo .env (carregado via python-dotenv)
    
    Em produção, sempre use variáveis de ambiente ou .env
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    
    if not api_key:
        logger.warning("ATENÇÃO: Chave da API OpenRouter não encontrada. Configure a variável de ambiente OPENROUTER_API_KEY.")
        
    return api_key

# Configurações dos provedores de LLM
LLM_PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "base_url": None,  # URL padrão da OpenAI
        "models": {
            "gpt-4o": "gpt-4o",
            "gpt-4": "gpt-4",
            "gpt-3.5-turbo": "gpt-3.5-turbo"
        },
        "embedding_model": "text-embedding-ada-002",
        "get_api_key": get_openai_api_key
    },
    "deepseek": {
        "name": "DeepSeek (via OpenRouter)",
        "base_url": "https://openrouter.ai/api/v1",
        "models": {
            "deepseek-chat": "deepseek/deepseek-chat",
            "deepseek-r1": "deepseek/deepseek-r1:free"
        },
        "embedding_model": "text-embedding-ada-002",  # Ainda usa OpenAI para embeddings
        "get_api_key": get_openrouter_api_key,
        "extra_headers": {
            "HTTP-Referer": "https://rag-antt.streamlit.app",
            "X-Title": "RAG-ANTT"
        }
    }
}

# Constantes e configurações do sistema
DB_FAISS_PATH = "vectorstore/db_faiss"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150

# Configurações padrão (pode ser alterado via interface)
DEFAULT_LLM_PROVIDER = "deepseek"  # Mudando para DeepSeek como padrão
DEFAULT_LLM_MODEL = "deepseek-chat"  # Usar deepseek-chat que funciona
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"

# Configurações do Streamlit
STREAMLIT_PAGE_TITLE = "RAG ANTT - Sistema de Consulta a Documentos"
STREAMLIT_PAGE_ICON = "🚆"
STREAMLIT_LAYOUT = "wide"

# Exportar constantes para uso na aplicação
__all__ = [
    'get_openai_api_key',
    'get_openrouter_api_key',
    'LLM_PROVIDERS',
    'DB_FAISS_PATH',
    'CHUNK_SIZE',
    'CHUNK_OVERLAP',
    'DEFAULT_LLM_PROVIDER',
    'DEFAULT_LLM_MODEL',
    'DEFAULT_EMBEDDING_MODEL',
    'STREAMLIT_PAGE_TITLE',
    'STREAMLIT_PAGE_ICON',
    'STREAMLIT_LAYOUT',
    'setup_logging',
    'logger'
] 