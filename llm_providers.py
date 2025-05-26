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
    
    def __init__(self, provider="deepseek", model=None):
        """
        Inicializa o gerenciador de LLM
        
        Args:
            provider (str): Provedor do LLM ('openai' ou 'deepseek')
            model (str): Modelo específico a ser usado
        """
        self.provider = provider
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
        Retorna uma instância do modelo de embeddings
        
        Note: Por enquanto, sempre usa OpenAI para embeddings
        
        Returns:
            OpenAIEmbeddings: Instância do modelo de embeddings
        """
        # Para embeddings, sempre usa OpenAI por enquanto
        openai_key = LLM_PROVIDERS["openai"]["get_api_key"]()
        
        if not openai_key:
            raise ValueError("Chave de API OpenAI necessária para embeddings")
        
        logger.info(f"Inicializando embeddings: {self.config['embedding_model']}")
        
        return OpenAIEmbeddings(
            model=self.config["embedding_model"],
            openai_api_key=openai_key
        )
    
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
            "embedding_model": self.config["embedding_model"]
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

def create_llm_manager(provider="deepseek", model=None):
    """
    Função de conveniência para criar um LLMManager
    
    Args:
        provider (str): Provedor do LLM
        model (str): Modelo específico
        
    Returns:
        LLMManager: Instância do gerenciador
    """
    try:
        return LLMManager(provider=provider, model=model)
    except Exception as e:
        logger.error(f"Erro ao criar LLMManager: {e}")
        # Fallback para OpenAI se DeepSeek falhar
        if provider != "openai":
            logger.warning("Tentando fallback para OpenAI...")
            return LLMManager(provider="openai", model="gpt-4o")
        raise 