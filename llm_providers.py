"""
Módulo para gerenciar diferentes provedores de LLM.
Suporta OpenAI e DeepSeek via OpenRouter.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings as CommunityOpenAIEmbeddings
from config import LLM_PROVIDERS, logger
import os

class LLMManager:
    """Gerenciador unificado para diferentes provedores de LLM"""
    
    def __init__(self, provider="deepseek", model=None, embedding_provider="openai"):
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
            
        elif self.embedding_provider == "free":
            # Usar embeddings gratuitos via OpenRouter (se disponível)
            try:
                openrouter_key = LLM_PROVIDERS["deepseek"]["get_api_key"]()
                if openrouter_key:
                    logger.info("Tentando usar embeddings gratuitos via OpenRouter...")
                    # Por enquanto, ainda usa OpenAI mas com configurações mais tolerantes
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
                else:
                    raise ValueError("Chave OpenRouter não disponível")
            except Exception as e:
                logger.warning(f"Embeddings gratuitos falharam: {e}")
                raise ValueError("Embeddings gratuitos não disponíveis no momento")
                
        elif self.embedding_provider == "local":
            # Placeholder para embeddings locais (futuro)
            logger.warning("Embeddings locais ainda não implementados, usando OpenAI como fallback")
            return self.get_embeddings_fallback()
            
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
    return {
        "openai": {
            "name": "OpenAI (Pago)",
            "description": "Embeddings de alta qualidade da OpenAI",
            "status": "available" if LLM_PROVIDERS["openai"]["get_api_key"]() else "no_key"
        },
        "free": {
            "name": "Gratuito (Limitado)",
            "description": "Embeddings com configuração otimizada para uso limitado",
            "status": "available"
        },
        "local": {
            "name": "Local (Em desenvolvimento)",
            "description": "Embeddings locais - em desenvolvimento",
            "status": "development"
        }
    }

def create_llm_manager(provider="deepseek", model=None, embedding_provider="free"):
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