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


def get_vagas_geracao_local() -> int:
    """
    Quantas geracoes Ollama este processo mantem ao mesmo tempo.

    RAG_VAGAS_GERACAO_LOCAL padrao 1 (piloto em CPU). Valor maior so
    cabe com nucleo e RAM para mais de uma inferencia, e com o
    processo Ollama aceitando esse paralelismo. Provedor externo
    ignora este limite: a concorrencia fica no servico remoto.
    Teto interno: 8.

    Returns:
        Inteiro entre 1 e 8.
    """
    raw = os.environ.get("RAG_VAGAS_GERACAO_LOCAL", "1").strip()
    if not raw:
        return 1
    try:
        valor = int(raw)
    except ValueError:
        logger.warning(
            "RAG_VAGAS_GERACAO_LOCAL invalido (%s); usando 1",
            raw,
        )
        return 1
    if valor < 1:
        return 1
    if valor > 8:
        logger.warning(
            "RAG_VAGAS_GERACAO_LOCAL %s acima do teto; usando 8",
            valor,
        )
        return 8
    return valor


def _inteiro_ambiente(nome: str, padrao: int, minimo: int, maximo: int) -> int:
    """
    Le um inteiro de ambiente, com piso, teto e valor padrao.

    Args:
        nome: Nome da variavel.
        padrao: Valor usado quando a variavel falta ou e invalida.
        minimo: Menor valor aceito.
        maximo: Maior valor aceito.

    Returns:
        Inteiro dentro do intervalo.
    """
    raw = os.environ.get(nome, str(padrao)).strip()
    if not raw:
        return padrao
    try:
        valor = int(raw)
    except ValueError:
        logger.warning("%s invalido (%s); usando %s", nome, raw, padrao)
        return padrao
    if valor < minimo:
        return minimo
    if valor > maximo:
        logger.warning("%s %s acima do teto; usando %s", nome, valor, maximo)
        return maximo
    return valor


def get_historico_turnos_guardados() -> int:
    """
    Quantas trocas da conversa ficam guardadas na sessao.

    RAG_HISTORICO_TURNOS padrao 10. Teto 20.

    Returns:
        Inteiro entre 1 e 20.
    """
    return _inteiro_ambiente("RAG_HISTORICO_TURNOS", 10, 1, 20)


def get_historico_turnos_reescrita() -> int:
    """
    Quantas trocas recentes entram na reescrita da pergunta seguinte.

    RAG_HISTORICO_REESCRITA padrao 8. Teto 10. Se for maior que o
    que esta guardado, a reescrita usa so o que existe.

    Returns:
        Inteiro entre 1 e 10.
    """
    return _inteiro_ambiente("RAG_HISTORICO_REESCRITA", 8, 1, 10)


def get_llm_model() -> str:
    """
    Modelo de geracao quando o chamador nao escolhe um.

    RAG_LLM_MODEL padrao qwen2.5:7b.

    Returns:
        Nome do modelo no Ollama ou no provedor configurado.
    """
    raw = os.environ.get("RAG_LLM_MODEL", "qwen2.5:7b").strip()
    return raw or "qwen2.5:7b"


def get_llm_temperature() -> float:
    """
    Liberdade de redacao quando o chamador omite o valor.

    RAG_LLM_TEMPERATURE padrao 0.1. Fora de 0.0 a 1.0 volta ao padrao.

    Returns:
        Temperatura entre 0.0 e 1.0.
    """
    raw = os.environ.get("RAG_LLM_TEMPERATURE", "0.1").strip()
    if not raw:
        return 0.1
    try:
        valor = float(raw)
    except ValueError:
        logger.warning(
            "RAG_LLM_TEMPERATURE invalido (%s); usando 0.1",
            raw,
        )
        return 0.1
    if valor < 0.0 or valor > 1.0:
        logger.warning(
            "RAG_LLM_TEMPERATURE %s fora de 0.0 a 1.0; usando 0.1",
            valor,
        )
        return 0.1
    return valor


def get_llm_max_tokens() -> int:
    """
    Tamanho maximo da resposta quando o chamador omite o valor.

    RAG_LLM_MAX_TOKENS padrao 4096. Teto 4096.

    Returns:
        Inteiro entre 1 e 4096.
    """
    return _inteiro_ambiente("RAG_LLM_MAX_TOKENS", 4096, 1, 4096)


def get_max_documentos() -> int:
    """
    Quantos trechos entram na consulta quando o chamador omite o valor.

    RAG_MAX_DOCUMENTOS padrao 30. Teto do contrato: 40.

    Returns:
        Inteiro entre 1 e 40.
    """
    return _inteiro_ambiente("RAG_MAX_DOCUMENTOS", 30, 1, 40)


def get_embedding_allowed() -> List[str]:
    """
    Embeddings que a reindexacao e a consulta podem usar.

    RAG_EMBEDDING_ALLOWED padrao local. Lista separada por virgula.
    Trocar de local para outro provedor exige reindexar a base.

    Returns:
        Lista nao vazia de identificadores.
    """
    raw = os.environ.get("RAG_EMBEDDING_ALLOWED", "local").strip()
    if not raw:
        return ["local"]
    itens = [item.strip() for item in raw.split(",") if item.strip()]
    return itens or ["local"]


def get_llm_provider_padrao() -> str:
    """
    Provedor de geracao quando o chamador nao envia um.

    Se RAG_LLM_ALLOWED_PROVIDERS estiver definido, usa o primeiro
    da lista. Senao, ollama.

    Returns:
        Identificador do provedor.
    """
    permitidos = get_allowed_llm_providers()
    if permitidos:
        return permitidos[0]
    return "ollama"


def get_historico_chars_resposta() -> int:
    """
    Quantos caracteres da resposta anterior entram na reescrita.

    RAG_HISTORICO_CHARS_RESPOSTA padrao 1000. Teto 2000.

    Returns:
        Inteiro entre 50 e 2000.
    """
    return _inteiro_ambiente("RAG_HISTORICO_CHARS_RESPOSTA", 1000, 50, 2000)


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

# Grupo dos arquivos da entrada que nao casam com uma sigla conhecida.
TIPO_DOCUMENTO_OUTROS = "OUTROS"

# Pastas sob dados_antt/ que nao sao tipos documentais.
TIPOS_DOCUMENTO_IGNORAR_DIRS = (
    "tabelas_auxiliares",
    ".ocr_cache",
    "entrada",
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
    "get_historico_turnos_guardados",
    "get_historico_turnos_reescrita",
    "get_historico_chars_resposta",
    "get_llm_model",
    "get_llm_temperature",
    "get_llm_max_tokens",
    "get_max_documentos",
    "get_embedding_allowed",
    "get_llm_provider_padrao",
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
    "TIPO_DOCUMENTO_OUTROS",
    "TIPOS_DOCUMENTO_IGNORAR_DIRS",
    "STREAMLIT_PAGE_TITLE",
    "STREAMLIT_PAGE_ICON",
    "STREAMLIT_LAYOUT",
    "setup_logging",
    "logger",
] 