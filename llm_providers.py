"""
Modulo para gerenciar diferentes provedores de LLM.
Suporta OpenAI e embeddings locais gratuitos.
"""

import gc
import os
import threading
import time

from typing import Dict, Optional, Tuple

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings as CommunityOpenAIEmbeddings
from config import (
    LLM_PROVIDERS,
    LOCAL_EMBEDDING_MODEL,
    cloud_fallback_enabled,
    get_allowed_llm_providers,
    get_ollama_base_url,
    logger,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_LOCAL_EMBEDDINGS_LOCK = threading.Lock()
_LOADED_SENTENCE_MODELS: dict = {}

# Modelos E5 exigem prefixos distintos para consulta e documento.
# Sem eles a qualidade de recuperacao cai fortemente.
_PREFIXOS_POR_MODELO: dict = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
}


def _preparar_ambiente_cpu_embeddings() -> None:
    """Reduz conflitos CUDA/meta tensor ao carregar sentence-transformers."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        import torch

        if hasattr(torch, "set_default_device"):
            torch.set_default_device("cpu")
    except Exception:
        pass


def _repo_ids_candidatos(model_name: str) -> list:
    """
    Monta a lista de repo_ids HuggingFace a tentar para um modelo.

    Aceita tanto o id curto legado (paraphrase-multilingual-MiniLM-L12-v2)
    quanto o id completo (intfloat/multilingual-e5-small).
    """
    nome = (model_name or "").strip()
    if not nome:
        return []
    if "/" in nome:
        return [nome]
    return [f"sentence-transformers/{nome}", nome]


def _resolver_snapshot_local(model_name: str):
    """
    Resolve caminho local do snapshot HuggingFace do modelo, se existir.

    Returns:
        str | None: Caminho do snapshot ou None.
    """
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        snapshot_download = None

    hub_root = os.path.expanduser("~/.cache/huggingface/hub")

    for repo_id in _repo_ids_candidatos(model_name):
        if snapshot_download is not None:
            try:
                return snapshot_download(repo_id, local_files_only=True)
            except Exception:
                pass

        slug = f"models--{repo_id.replace('/', '--')}"
        snapshots_dir = os.path.join(hub_root, slug, "snapshots")
        if not os.path.isdir(snapshots_dir):
            continue

        candidatos = []
        for entry in os.listdir(snapshots_dir):
            path = os.path.join(snapshots_dir, entry)
            if os.path.isdir(path):
                candidatos.append(path)

        if candidatos:
            candidatos.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidatos[0]

    return None


def _kwargs_seguros_sentence_transformer():
    """Kwargs que evitam meta tensors em transformers/sentence-transformers recentes."""
    import torch

    seguros = {
        "low_cpu_mem_usage": False,
        "device_map": None,
        "torch_dtype": torch.float32,
    }
    return seguros, dict(seguros)


class LocalEmbeddings:
    """
    Embeddings locais via sentence-transformers.

    Padrao: intfloat/multilingual-e5-small (escolhido no A/B em CPU).
    Modelos da familia E5 aplicam automaticamente os prefixos query:/passage:
    e normalizam os vetores (compativel com FAISS por distancia L2).
    """

    def __init__(self, model_name=None):
        """
        Args:
            model_name: Id HuggingFace do modelo. None usa LOCAL_EMBEDDING_MODEL.
        """
        self.model_name = model_name or LOCAL_EMBEDDING_MODEL
        self.model = None
        prefixos = _PREFIXOS_POR_MODELO.get(self.model_name, ("", ""))
        self.query_prefix = prefixos[0]
        self.doc_prefix = prefixos[1]
        logger.info(f"Carregando modelo local de embeddings: {self.model_name}")

        try:
            self._load_model()
            logger.info("Modelo de embeddings local carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de embeddings: {e}")
            raise

    def _load_model(self):
        """Carrega sentence-transformers com estrategias robustas anti meta tensor."""
        with _LOCAL_EMBEDDINGS_LOCK:
            cached = _LOADED_SENTENCE_MODELS.get(self.model_name)
            if cached is not None:
                self.model = cached
                logger.info("Modelo local reutilizado do cache em memoria")
                return

            self.model = self._load_model_impl()
            _LOADED_SENTENCE_MODELS[self.model_name] = self.model

    def _load_model_impl(self):
        """Implementacao interna de carregamento (protegida por lock)."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers nao esta instalado. "
                "Execute: pip install sentence-transformers"
            ) from exc

        _preparar_ambiente_cpu_embeddings()
        gc.collect()

        cache_dir = os.path.expanduser("~/.cache/sentence_transformers")
        os.makedirs(cache_dir, exist_ok=True)
        snapshot_path = _resolver_snapshot_local(self.model_name)
        model_kwargs_safe, config_kwargs_safe = _kwargs_seguros_sentence_transformer()

        tentativas = []

        tentativas.append(
            (
                "cache local",
                {
                    "model_name_or_path": self.model_name,
                    "local_files_only": True,
                    "model_kwargs": {},
                    "config_kwargs": {},
                },
            )
        )
        tentativas.append(
            (
                "cache local (anti meta tensor)",
                {
                    "model_name_or_path": self.model_name,
                    "local_files_only": True,
                    "model_kwargs": model_kwargs_safe,
                    "config_kwargs": config_kwargs_safe,
                },
            )
        )

        if snapshot_path:
            tentativas.append(
                (
                    f"snapshot local ({os.path.basename(snapshot_path)})",
                    {
                        "model_name_or_path": snapshot_path,
                        "local_files_only": True,
                        "model_kwargs": model_kwargs_safe,
                        "config_kwargs": config_kwargs_safe,
                    },
                )
            )

        tentativas.append(
            (
                "download (anti meta tensor)",
                {
                    "model_name_or_path": self.model_name,
                    "local_files_only": False,
                    "model_kwargs": model_kwargs_safe,
                    "config_kwargs": config_kwargs_safe,
                },
            )
        )

        last_err = None
        for desc, params in tentativas:
            retries = 1 if params.get("local_files_only") else 3
            backoff = 2
            for attempt in range(1, retries + 1):
                try:
                    model = SentenceTransformer(
                        params["model_name_or_path"],
                        device="cpu",
                        cache_folder=cache_dir,
                        local_files_only=params["local_files_only"],
                        model_kwargs=params["model_kwargs"],
                        config_kwargs=params["config_kwargs"],
                        trust_remote_code=False,
                        backend="torch",
                    )
                    _ = model.encode(["validacao"], show_progress_bar=False)
                    logger.info(f"Modelo carregado via {desc}")
                    return model
                except Exception as exc:
                    last_err = exc
                    msg = str(exc).lower()
                    if params.get("local_files_only") and (
                        "not found" in msg
                        or "no such" in msg
                        or "does not appear" in msg
                        or "offline" in msg
                    ):
                        break
                    if "429" in msg or "rate" in msg or "too many" in msg:
                        logger.warning(
                            f"HTTP 429 em {desc} (tentativa {attempt}/{retries}). "
                            f"Aguardando {backoff}s..."
                        )
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    if "meta tensor" in msg:
                        logger.warning(
                            f"Meta tensor em {desc}; tentando proxima estrategia"
                        )
                        break
                    logger.warning(f"Falha em {desc}: {exc}")
                    break

        model_manual = self._load_model_manual_transformer(snapshot_path)
        if model_manual is not None:
            logger.info("Modelo carregado via fallback manual Transformer+Pooling")
            return model_manual

        raise RuntimeError(
            f"Falha ao carregar modelo local '{self.model_name}': {last_err}"
        )

    def _load_model_manual_transformer(self, snapshot_path):
        """
        Fallback manual quando SentenceTransformer falha por meta tensor.

        Monta o modelo a partir de Transformer + Pooling com carregamento
        explicito em CPU e sem device_map.
        """
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            from sentence_transformers.models import Pooling, Transformer
        except ImportError:
            return None

        origem = snapshot_path or self.model_name
        model_kwargs_safe, _ = _kwargs_seguros_sentence_transformer()

        try:
            transformer = Transformer(
                origem,
                max_seq_length=128,
                model_args=model_kwargs_safe,
                config_args=model_kwargs_safe,
                cache_dir=os.path.expanduser("~/.cache/sentence_transformers"),
            )
            dim = transformer.get_word_embedding_dimension()
            pooling = Pooling(dim)
            model = SentenceTransformer(
                modules=[transformer, pooling],
                device="cpu",
                backend="torch",
            )
            _ = model.encode(["validacao manual"], show_progress_bar=False)
            return model
        except Exception as exc:
            logger.warning(f"Fallback manual de embeddings falhou: {exc}")
            try:
                import torch

                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return None
    
    def embed_query(self, text):
        """
        Gera embedding para uma unica consulta.

        Args:
            text (str): Texto da consulta.

        Returns:
            List[float]: Vetor de embedding.
        """
        if self.model is None:
            self._load_model()

        preparado = f"{self.query_prefix}{text}" if self.query_prefix else text
        try:
            embedding = self.model.encode(
                [preparado],
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Erro ao gerar embedding para query: {e}")
            raise

    def embed_documents(self, texts):
        """
        Gera embeddings para multiplos documentos.

        Args:
            texts (List[str]): Textos a indexar.

        Returns:
            List[List[float]]: Vetores de embedding.
        """
        if self.model is None:
            self._load_model()

        try:
            logger.info(
                f"Processando {len(texts)} documentos com embeddings locais "
                f"({self.model_name})..."
            )
            batch_size = 32
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                preparados = [
                    f"{self.doc_prefix}{item}" if self.doc_prefix else item
                    for item in batch
                ]
                batch_embeddings = self.model.encode(
                    preparados,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                all_embeddings.extend([emb.tolist() for emb in batch_embeddings])
                processed = min(i + batch_size, len(texts))
                logger.info(f"Processados {processed}/{len(texts)} documentos")
            logger.info("Documentos processados com embeddings locais")
            return all_embeddings
        except Exception as e:
            logger.error(f"Erro ao gerar embeddings para documentos: {e}")
            raise

    def __call__(self, text):
        """Compatibilidade com FAISS: trata chamada como consulta."""
        return self.embed_query(text)

def verificar_ollama_disponivel(base_url: Optional[str] = None) -> Tuple[bool, str]:
    """
    Verifica se o servico Ollama responde em /api/tags.

    Args:
        base_url: URL .../v1 do ChatOpenAI; a checagem usa o host sem /v1.

    Returns:
        (ok, mensagem_diagnostico).
    """
    url = (base_url or get_ollama_base_url()).rstrip("/")
    if url.endswith("/v1"):
        root = url[:-3]
    else:
        root = url
    tags_url = f"{root.rstrip('/')}/api/tags"
    try:
        import urllib.request

        with urllib.request.urlopen(tags_url, timeout=3) as response:
            if int(response.status) >= 400:
                return False, f"Ollama respondeu HTTP {response.status}"
        return True, f"Ollama acessivel em {root}"
    except Exception as exc:  # noqa: BLE001
        return False, f"Ollama indisponivel ({exc})"


class LLMManager:
    """Gerenciador unificado para diferentes provedores de LLM"""
    
    def __init__(self, provider="deepseek", model=None, embedding_provider="local"):
        """
        Inicializa o gerenciador de LLM
        
        Args:
            provider (str): Provedor do LLM ('openai', 'deepseek' ou 'ollama')
            model (str): Modelo específico a ser usado
            embedding_provider (str): Provedor para embeddings ('openai', 'local', 'free')
        """
        self.provider = provider
        self.embedding_provider = embedding_provider
        self.config = LLM_PROVIDERS.get(provider)
        
        if not self.config:
            raise ValueError(f"Provedor '{provider}' não suportado. Use: {list(LLM_PROVIDERS.keys())}")
        
        # Define o modelo padrão se não especificado
        if model is None:
            self.model = list(self.config["models"].keys())[0]
        else:
            self.model = model
            
        # Verifica se o modelo é suportado
        if self.model not in self.config["models"]:
            raise ValueError(f"Modelo '{model}' não suportado para {provider}. Use: {list(self.config['models'].keys())}")
        
        self.api_key = self.config["get_api_key"]()
        requires_key = bool(self.config.get("requires_api_key", True))
        
        if requires_key and not self.api_key:
            raise ValueError(f"Chave de API não encontrada para {provider}")

        if self.provider == "ollama":
            ok, detalhe = verificar_ollama_disponivel()
            if not ok:
                raise ValueError(
                    f"Ollama nao esta acessivel. {detalhe}. "
                    "Instale o Ollama e execute: ollama pull llama3.2:3b"
                )
    
    def get_llm(self, temperature=0.1, max_tokens=None):
        """
        Retorna uma instância do LLM configurado
        
        Args:
            temperature (float): Temperatura para geração de texto
            max_tokens (int): Número máximo de tokens
            
        Returns:
            ChatOpenAI: Instância do LLM
        """
        model_name = self.config["models"][self.model]
        
        llm_kwargs = {
            "model": model_name,
            "temperature": temperature,
        }
        
        # Adiciona max_tokens se especificado
        if max_tokens:
            llm_kwargs["max_tokens"] = max_tokens
        
        # Configurações específicas para DeepSeek via OpenRouter
        if self.provider == "deepseek":
            llm_kwargs["base_url"] = self.config["base_url"]
            llm_kwargs["openai_api_key"] = self.api_key  # OpenRouter usa este parâmetro
            
            # Headers específicos para OpenRouter
            extra_headers = self.config.get("extra_headers", {})
            llm_kwargs["default_headers"] = extra_headers
        elif self.provider == "ollama":
            get_base = self.config.get("get_base_url", get_ollama_base_url)
            llm_kwargs["base_url"] = get_base()
            llm_kwargs["api_key"] = self.api_key or "ollama"
            timeout = int(self.config.get("request_timeout", 300))
            llm_kwargs["timeout"] = timeout
            llm_kwargs["max_retries"] = 1
        else:
            # Para OpenAI, usa o parâmetro padrão
            llm_kwargs["api_key"] = self.api_key
        
        logger.info(f"Inicializando LLM: {self.config['name']} - {model_name}")
        
        return ChatOpenAI(**llm_kwargs)
    
    def get_embeddings(self):
        """
        Retorna uma instância do modelo de embeddings baseado no provedor escolhido
        
        Returns:
            Embeddings: Instância do modelo de embeddings
        """
        if self.embedding_provider == "openai":
            # Usar OpenAI para embeddings
            openai_key = LLM_PROVIDERS["openai"]["get_api_key"]()
            
            if not openai_key:
                raise ValueError("Chave de API OpenAI necessária para embeddings OpenAI")
            
            logger.info(f"Inicializando embeddings OpenAI: {self.config['embedding_model']}")
            
            return OpenAIEmbeddings(
                model=self.config["embedding_model"],
                openai_api_key=openai_key,
                max_retries=1,  # Reduzir tentativas para falhar mais rápido
                timeout=10      # Timeout menor
            )
            
        elif self.embedding_provider == "local":
            logger.info(
                f"Inicializando embeddings locais: {LOCAL_EMBEDDING_MODEL}"
            )
            try:
                return LocalEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
            except Exception as e:
                logger.error(f"Erro ao inicializar embeddings locais: {e}")
                raise ValueError(f"Falha ao configurar embeddings locais: {str(e)}")

        elif self.embedding_provider == "free":
            # Tentar embeddings locais primeiro, fallback para OpenAI
            try:
                logger.info(
                    f"Tentando embeddings locais ({LOCAL_EMBEDDING_MODEL})..."
                )
                return LocalEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)

            except Exception as e:
                logger.warning(f"Embeddings locais falharam: {e}")
                
                # Fallback para OpenAI com configuração muito conservadora
                logger.info("🔄 Tentando fallback para OpenAI...")
                try:
                    openai_key = LLM_PROVIDERS["openai"]["get_api_key"]()
                    if openai_key:
                        return OpenAIEmbeddings(
                            model="text-embedding-ada-002",
                            openai_api_key=openai_key,
                            max_retries=0,  # Não tentar novamente
                            timeout=5       # Timeout muito baixo
                        )
                    else:
                        raise ValueError("Nenhuma chave de API disponível para embeddings")
                except Exception as e2:
                    logger.error(f"❌ Fallback OpenAI também falhou: {e2}")
                    raise ValueError("❌ Não foi possível configurar embeddings. Instale sentence-transformers ou configure chave OpenAI.")
                
        else:
            raise ValueError(f"Provedor de embeddings '{self.embedding_provider}' não suportado")
    
    def get_embeddings_fallback(self):
        """
        Fallback para embeddings quando o provedor principal falha
        """
        try:
            openai_key = LLM_PROVIDERS["openai"]["get_api_key"]()
            if openai_key:
                logger.info("Usando fallback: embeddings OpenAI com configuração mínima")
                return OpenAIEmbeddings(
                    model="text-embedding-ada-002",
                    openai_api_key=openai_key,
                    max_retries=0,
                    timeout=5
                )
            else:
                raise ValueError("Nenhuma chave de API disponível para fallback")
        except Exception as e:
            logger.error(f"Fallback de embeddings falhou: {e}")
            raise ValueError("Todos os provedores de embeddings falharam")
    
    def get_provider_info(self):
        """
        Retorna informações sobre o provedor atual
        
        Returns:
            dict: Informações do provedor
        """
        return {
            "provider": self.provider,
            "name": self.config["name"],
            "model": self.model,
            "model_name": self.config["models"][self.model],
            "embedding_model": self.config["embedding_model"],
            "embedding_provider": self.embedding_provider
        }

def get_available_providers() -> Dict[str, Dict[str, object]]:
    """
    Retorna lista de provedores disponiveis (respeita RAG_LLM_ALLOWED_PROVIDERS).

    Returns:
        Dicionario com provedores e seus modelos.
    """
    allowed = get_allowed_llm_providers()
    resultado: Dict[str, Dict[str, object]] = {}
    for provider, config in LLM_PROVIDERS.items():
        if allowed is not None and provider not in allowed:
            continue
        resultado[provider] = {
            "name": config["name"],
            "models": list(config["models"].keys()),
        }
    return resultado

def get_available_embedding_providers():
    """
    Retorna lista de provedores de embeddings disponíveis
    
    Returns:
        dict: Dicionário com provedores de embeddings
    """
    # Verificar se sentence-transformers está disponível
    local_available = True
    try:
        import sentence_transformers
    except ImportError:
        local_available = False
    
    # Os rotulos em "name" sao exibidos diretamente na interface, portanto
    # seguem o padrao institucional: sem icones e sem referencia a custo.
    providers = {
        "local": {
            "name": "Local (offline)",
            "description": "Busca processada na própria máquina, sem envio de dados para fora.",
            "status": "available" if local_available else "needs_install"
        },
        "openai": {
            "name": "OpenAI (serviço externo)",
            "description": "Busca processada pelo serviço da OpenAI.",
            "status": "available" if LLM_PROVIDERS["openai"]["get_api_key"]() else "no_key"
        },
        "free": {
            "name": "Automático (local primeiro)",
            "description": "Tenta a busca local e recorre à OpenAI apenas se necessário.",
            "status": "available"
        }
    }
    
    return providers

def create_llm_manager(provider="deepseek", model=None, embedding_provider="local"):
    """
    Função de conveniência para criar um LLMManager
    
    Args:
        provider (str): Provedor do LLM
        model (str): Modelo específico
        embedding_provider (str): Provedor para embeddings
        
    Returns:
        LLMManager: Instância do gerenciador
    """
    try:
        return LLMManager(provider=provider, model=model, embedding_provider=embedding_provider)
    except Exception as e:
        logger.error(f"Erro ao criar LLMManager: {e}")
        # Ollama: fallback cloud so se habilitado (dev). Em antt_prod falha clara.
        if provider == "ollama":
            if cloud_fallback_enabled():
                logger.warning("Ollama indisponivel; fallback para DeepSeek (dev).")
                return LLMManager(
                    provider="deepseek",
                    model=None,
                    embedding_provider=embedding_provider,
                )
            raise
        # Fallback historico: DeepSeek/outros -> OpenAI
        if provider != "openai":
            logger.warning("Tentando fallback para OpenAI...")
            return LLMManager(provider="openai", model="gpt-4o", embedding_provider=embedding_provider)
        raise 