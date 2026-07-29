"""
Módulo de configuração para o sistema RAG-ANTT.
Gerencia variáveis de ambiente e configurações do aplicativo.
"""

import os
from typing import List, Optional
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


def get_ollama_base_url() -> str:
    """
    Retorna a URL base da API OpenAI-compatible do Ollama.

    Returns:
        URL terminando em /v1 (padrao: http://localhost:11434/v1).
    """
    raw = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1").strip()
    if not raw:
        return "http://localhost:11434/v1"
    return raw.rstrip("/")


def get_ollama_api_key() -> str:
    """
    Placeholder exigido pelo cliente ChatOpenAI; Ollama nao valida a chave.

    Returns:
        String nao vazia (padrao: ollama).
    """
    return os.environ.get("OLLAMA_API_KEY", "ollama").strip() or "ollama"


def get_deploy_profile() -> str:
    """
    Perfil de deploy: dev, homolog ou antt_prod.

    Returns:
        Nome do perfil em minusculas.
    """
    return os.environ.get("RAG_DEPLOY_PROFILE", "dev").strip().lower() or "dev"


def get_allowed_llm_providers() -> Optional[List[str]]:
    """
    Lista de provedores de chat permitidos via env.

    Returns:
        Lista de ids ou None para permitir todos os registrados.
    """
    raw = os.environ.get("RAG_LLM_ALLOWED_PROVIDERS", "").strip()
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def cloud_fallback_enabled() -> bool:
    """
    Se True, falha do Ollama pode cair em DeepSeek (perfil dev).

    Em producao ANTT use RAG_LLM_CLOUD_FALLBACK=false.

    Returns:
        True quando o fallback cloud esta habilitado.
    """
    raw = os.environ.get("RAG_LLM_CLOUD_FALLBACK", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    # Padrao: desligado em antt_prod; ligado nos demais.
    return get_deploy_profile() != "antt_prod"


# Configurações dos provedores de LLM
LLM_PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "base_url": None,  # URL padrão da OpenAI
        "requires_api_key": True,
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
        "requires_api_key": True,
        # A primeira chave e usada como padrao quando nenhum modelo e
        # informado. Slugs verificados no catalogo do OpenRouter.
        "models": {
            # V4 Flash: 1M de contexto e custo menor que o V3 antigo.
            "deepseek-v4-flash": "deepseek/deepseek-v4-flash",
            # V4 Pro: maior capacidade, para consultas mais exigentes.
            "deepseek-v4-pro": "deepseek/deepseek-v4-pro",
            # V3.2: geracao intermediaria, mantida para comparacao.
            "deepseek-v3.2": "deepseek/deepseek-v3.2",
            # V3: modelo usado ate entao, mantido para regressao.
            "deepseek-chat": "deepseek/deepseek-chat",
            # R1: raciocinio explicito. O slug ":free" foi descontinuado
            # pelo OpenRouter; usar a variante paga.
            "deepseek-r1": "deepseek/deepseek-r1"
        },
        "embedding_model": "text-embedding-ada-002",  # Ainda usa OpenAI para embeddings
        "get_api_key": get_openrouter_api_key,
        "extra_headers": {
            "HTTP-Referer": "https://rag-antt.streamlit.app",
            "X-Title": "RAG-ANTT"
        }
    },
    # Provedor local (SUTEC/GETIC: CPU, sem API externa). Aditivo.
    "ollama": {
        "name": "Local (Ollama)",
        "base_url": None,  # Resolvido em runtime via get_ollama_base_url()
        "requires_api_key": False,
        "models": {
            # Padrao CPU no notebook / GETIC.
            "llama3.2:3b": "llama3.2:3b",
            # Homologacao de qualidade (mais RAM/latencia).
            "qwen2.5:7b": "qwen2.5:7b",
            # Alternativa leve.
            "phi3:mini": "phi3:mini",
        },
        "embedding_model": "text-embedding-ada-002",
        "get_api_key": get_ollama_api_key,
        "get_base_url": get_ollama_base_url,
        "request_timeout": 300,
    },
}

# Constantes e configurações do sistema
DB_FAISS_PATH = "vectorstore/db_faiss"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# Configurações padrão (pode ser alterado via interface)
DEFAULT_LLM_PROVIDER = "deepseek"
# V4 Flash e o padrao: contexto de 1M de tokens e custo menor que o V3.
DEFAULT_LLM_MODEL = "deepseek-v4-flash"
# Embedding OpenAI (quando o provedor de embeddings e "openai").
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"
# Embedding local open source. Escolhido pelo A/B em CPU
# (comparar_embeddings.py): ganho de ~50 p.p. vs MiniLM nas perguntas
# de IRI/prazos. Trocar este valor exige reindexacao completa.
LOCAL_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# Aliases curtos opcionais: so entram se a sigla existir na base.
# A existencia dos tipos vem dos arquivos em dados_antt/, nao desta lista.
TIPOS_DOCUMENTO_ALIASES_CURTOS = {
    "INM": ("in",),
}

# Se True, tenta LLM quando o cabecalho nao entrega nome (custo/latencia).
TIPOS_DOCUMENTO_USAR_LLM_ALIASES = False

# Pastas sob dados_antt/ que nao sao tipos documentais.
TIPOS_DOCUMENTO_IGNORAR_DIRS = (
    "tabelas_auxiliares",
    ".ocr_cache",
)

# Configurações do Streamlit
STREAMLIT_PAGE_TITLE = "Sistema de Consulta Normativa - ANTT"
# Caminho para o favicon institucional. None usa o icone padrao do Streamlit.
# Substituir por "static/favicon.png" quando o logo autorizado estiver disponivel.
STREAMLIT_PAGE_ICON: Optional[str] = None
STREAMLIT_LAYOUT = "wide"

# Exportar constantes para uso na aplicação
__all__ = [
    "get_openai_api_key",
    "get_openrouter_api_key",
    "get_ollama_base_url",
    "get_ollama_api_key",
    "get_deploy_profile",
    "get_allowed_llm_providers",
    "cloud_fallback_enabled",
    "LLM_PROVIDERS",
    "DB_FAISS_PATH",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "DEFAULT_LLM_PROVIDER",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "LOCAL_EMBEDDING_MODEL",
    "TIPOS_DOCUMENTO_ALIASES_CURTOS",
    "TIPOS_DOCUMENTO_USAR_LLM_ALIASES",
    "TIPOS_DOCUMENTO_IGNORAR_DIRS",
    "STREAMLIT_PAGE_TITLE",
    "STREAMLIT_PAGE_ICON",
    "STREAMLIT_LAYOUT",
    "setup_logging",
    "logger",
] 