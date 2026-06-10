"""
Módulo para gerenciar diferentes provedores de LLM.
Suporta OpenAI e embeddings locais gratuitos.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings as CommunityOpenAIEmbeddings
from config import LLM_PROVIDERS, logger
import os

class LocalEmbeddings:
    """Classe para embeddings locais usando sentence-transformers - 100% GRATUITO.
    Modelo padrao: paraphrase-multilingual-MiniLM-L12-v2 (50+ idiomas, 384 dims).
    """
    
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = None
        logger.info(f"🔄 Carregando modelo local de embeddings: {model_name}")
        
        try:
            self._load_model()
            logger.info("✅ Modelo de embeddings local carregado com sucesso")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo de embeddings: {e}")
            raise
    
    def _load_model(self):
        """Carrega o modelo sentence-transformers com tentativas e modo offline."""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            import os
            import time
            
            # Definir pasta de cache local
            cache_dir = os.path.expanduser("~/.cache/sentence_transformers")
            os.makedirs(cache_dir, exist_ok=True)
            
            # 1) Tentar carregar somente de arquivos locais (sem rede)
            try:
                self.model = SentenceTransformer(self.model_name, device="cpu", cache_folder=cache_dir, local_files_only=True)
                logger.info("📦 Modelo carregado do cache local (offline)")
                return
            except Exception as e_local_only:
                logger.warning(f"⚠️ Modelo nao encontrado no cache local: {e_local_only}")
            
            # 2) Tentar baixar com algumas tentativas graduais (tratar 429)
            retries = 3
            backoff = 2
            last_err = None
            for attempt in range(1, retries + 1):
                try:
                    self.model = SentenceTransformer(self.model_name, device="cpu", cache_folder=cache_dir)
                    logger.info("⬇️ Download do modelo concluido e carregado com sucesso")
                    return
                except Exception as e_dl:
                    last_err = e_dl
                    msg = str(e_dl).lower()
                    if "429" in msg or "rate" in msg or "too many" in msg:
                        logger.warning(f"HTTP 429/Rate limit ao baixar modelo (tentativa {attempt}/{retries}). Aguardando {backoff}s...")
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    if "meta tensor" in msg:
                        logger.warning("Erro de 'meta tensor' ao mover modelo; tentando recarregar em CPU pura")
                        time.sleep(1)
                        continue
                    # Outros erros: nao insistir
                    break
            
            # Se chegou aqui, falhou
            raise RuntimeError(f"Falha ao carregar modelo local '{self.model_name}': {last_err}")
        except ImportError:
            raise ImportError(
                "sentence-transformers nao esta instalado. Execute: pip install sentence-transformers"
            )
    
    def embed_query(self, text):
        """
        Gera embedding para uma única query
        
        Args:
            text (str): Texto para gerar embedding
            
        Returns:
            List[float]: Lista de embeddings
        """
        if self.model is None:
            self._load_model()
        
        try:
            embedding = self.model.encode([text])
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Erro ao gerar embedding para query: {e}")
            raise
    
    def embed_documents(self, texts):
        """
        Gera embeddings para múltiplos documentos
        
        Args:
            texts (List[str]): Lista de textos para gerar embeddings
            
        Returns:
            List[List[float]]: Lista de embeddings
        """
        if self.model is None:
            self._load_model()
        
        try:
            logger.info(f"🔄 Processando {len(texts)} documentos com embeddings locais...")
            batch_size = 32
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(batch, show_progress_bar=False)
                all_embeddings.extend([emb.tolist() for emb in batch_embeddings])
                processed = min(i + batch_size, len(texts))
                logger.info(f"📊 Processados {processed}/{len(texts)} documentos")
            logger.info("✅ Todos os documentos processados com embeddings locais")
            return all_embeddings
        except Exception as e:
            logger.error(f"Erro ao gerar embeddings para documentos: {e}")
            raise
    
    def __call__(self, text):
        """
        Torna a classe callable para compatibilidade com FAISS
        
        Args:
            text (str): Texto para gerar embedding
            
        Returns:
            List[float]: Lista de embeddings
        """
        return self.embed_query(text)

class LLMManager:
    """Gerenciador unificado para diferentes provedores de LLM"""
    
    def __init__(self, provider="deepseek", model=None, embedding_provider="local"):
        """
        Inicializa o gerenciador de LLM
        
        Args:
            provider (str): Provedor do LLM ('openai' ou 'deepseek')
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
        
        if not self.api_key:
            raise ValueError(f"Chave de API não encontrada para {provider}")
    
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
            # Usar embeddings locais REALMENTE GRATUITOS
            logger.info("🆓 Inicializando embeddings locais (100% GRATUITO)")
            
            try:
                return LocalEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
            except Exception as e:
                logger.error(f"Erro ao inicializar embeddings locais: {e}")
                raise ValueError(f"Falha ao configurar embeddings locais: {str(e)}")
            
        elif self.embedding_provider == "free":
            # Tentar embeddings locais primeiro, fallback para OpenAI
            try:
                logger.info("🔄 Tentando embeddings locais gratuitos...")
                return LocalEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
                
            except Exception as e:
                logger.warning(f"⚠️ Embeddings locais falharam: {e}")
                
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

def get_available_providers():
    """
    Retorna lista de provedores disponíveis
    
    Returns:
        dict: Dicionário com provedores e seus modelos
    """
    return {
        provider: {
            "name": config["name"],
            "models": list(config["models"].keys())
        }
        for provider, config in LLM_PROVIDERS.items()
    }

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
    
    providers = {
        "local": {
            "name": "🆓 Local (100% Gratuito)",
            "description": "Embeddings locais com sentence-transformers - totalmente offline e gratuito",
            "status": "available" if local_available else "needs_install"
        },
        "openai": {
            "name": "💰 OpenAI (Pago - Alta Qualidade)",
            "description": "Embeddings de alta qualidade da OpenAI",
            "status": "available" if LLM_PROVIDERS["openai"]["get_api_key"]() else "no_key"
        },
        "free": {
            "name": "⚡ Automático (Local Primeiro)",
            "description": "Tenta embeddings locais primeiro, fallback para OpenAI se necessário",
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
        # Fallback para OpenAI se DeepSeek falhar
        if provider != "openai":
            logger.warning("Tentando fallback para OpenAI...")
            return LLMManager(provider="openai", model="gpt-4o", embedding_provider=embedding_provider)
        raise 