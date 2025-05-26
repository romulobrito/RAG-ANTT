#!/usr/bin/env python3
"""
Script de teste para verificar a configuração dos provedores de LLM.
"""

import os
from dotenv import load_dotenv
from llm_providers import create_llm_manager, get_available_providers
from config import logger

# Carregar variáveis de ambiente
load_dotenv()

def test_api_keys():
    """Testa se as chaves de API estão configuradas"""
    print("🔑 Testando chaves de API...")
    
    # Testar OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print(f"✅ OpenAI API Key: Configurada (termina em ...{openai_key[-4:]})")
    else:
        print("❌ OpenAI API Key: Não encontrada")
    
    # Testar OpenRouter
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        print(f"✅ OpenRouter API Key: Configurada (termina em ...{openrouter_key[-4:]})")
    else:
        print("❌ OpenRouter API Key: Não encontrada")
    
    print()

def test_providers():
    """Testa os provedores disponíveis"""
    print("🤖 Testando provedores disponíveis...")
    
    providers = get_available_providers()
    for provider_id, provider_info in providers.items():
        print(f"📋 {provider_info['name']}:")
        for model in provider_info['models']:
            print(f"   - {model}")
    
    print()

def test_llm_creation():
    """Testa a criação de instâncias de LLM"""
    print("🧪 Testando criação de LLMs...")
    
    # Testar DeepSeek
    try:
        print("Testando DeepSeek...")
        deepseek_manager = create_llm_manager("deepseek", "deepseek-r1")
        llm = deepseek_manager.get_llm()
        print("✅ DeepSeek: LLM criado com sucesso")
        
        # Teste simples
        response = llm.invoke("Olá! Responda apenas 'Funcionando' se você conseguir me entender.")
        print(f"📝 Resposta DeepSeek: {response.content[:50]}...")
        
    except Exception as e:
        print(f"❌ DeepSeek: Erro - {e}")
    
    print()
    
    # Testar OpenAI
    try:
        print("Testando OpenAI...")
        openai_manager = create_llm_manager("openai", "gpt-3.5-turbo")
        llm = openai_manager.get_llm()
        print("✅ OpenAI: LLM criado com sucesso")
        
        # Teste simples
        response = llm.invoke("Olá! Responda apenas 'Funcionando' se você conseguir me entender.")
        print(f"📝 Resposta OpenAI: {response.content[:50]}...")
        
    except Exception as e:
        print(f"❌ OpenAI: Erro - {e}")
    
    print()

def test_embeddings():
    """Testa a criação de embeddings"""
    print("🔤 Testando embeddings...")
    
    try:
        # Embeddings sempre usam OpenAI
        manager = create_llm_manager("openai")
        embeddings = manager.get_embeddings()
        
        # Teste simples
        test_text = "Este é um teste de embedding"
        embedding_vector = embeddings.embed_query(test_text)
        
        print(f"✅ Embeddings: Criado com sucesso")
        print(f"📊 Dimensões do vetor: {len(embedding_vector)}")
        
    except Exception as e:
        print(f"❌ Embeddings: Erro - {e}")
    
    print()

def main():
    """Função principal do teste"""
    print("🚀 Iniciando testes dos provedores de LLM...\n")
    
    test_api_keys()
    test_providers()
    test_llm_creation()
    test_embeddings()
    
    print("✨ Testes concluídos!")
    print("\n💡 Dicas:")
    print("- Se algum teste falhou, verifique suas chaves de API no arquivo .env")
    print("- Para DeepSeek: Crie uma conta gratuita em https://openrouter.ai")
    print("- Para OpenAI: Obtenha sua chave em https://platform.openai.com")

if __name__ == "__main__":
    main() 