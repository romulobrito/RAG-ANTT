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
    3. Valor padrão (somente para desenvolvimento)
    
    Em produção, sempre use variáveis de ambiente ou .env
    """
    # Tenta obter de variável de ambiente
    api_key = os.environ.get("OPENAI_API_KEY", "")
    
    if not api_key:
        # Valor padrão apenas para desenvolvimento (não usar em produção)
        default_key = 'sk-proj-SJzLGfezVCxJLft228F2T3BlbkFJ2lSCkYReBn53ZYbMfmKh'
        logger.warning("ATENÇÃO: Usando chave de API padrão. Em produção, configure a variável de ambiente OPENAI_API_KEY.")
        return default_key
        
    return api_key

# Constantes e configurações do sistema
DB_FAISS_PATH = "vectorstore/db_faiss"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"
DEFAULT_LLM_MODEL = "gpt-4o"

# Configurações do Streamlit
STREAMLIT_PAGE_TITLE = "RAG ANTT - Sistema de Consulta a Documentos"
STREAMLIT_PAGE_ICON = "🚆"
STREAMLIT_LAYOUT = "wide"

# Exportar constantes para uso na aplicação
__all__ = [
    'get_openai_api_key',
    'DB_FAISS_PATH',
    'CHUNK_SIZE',
    'CHUNK_OVERLAP',
    'DEFAULT_EMBEDDING_MODEL',
    'DEFAULT_LLM_MODEL',
    'STREAMLIT_PAGE_TITLE',
    'STREAMLIT_PAGE_ICON',
    'STREAMLIT_LAYOUT',
    'setup_logging',
    'logger'
] 