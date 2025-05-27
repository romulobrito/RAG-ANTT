#!/usr/bin/env python3
"""
Script de teste para verificar se o sistema RAG com embeddings locais está funcionando
"""

import os
import sys

def teste_embeddings_locais():
    """Testa se os embeddings locais estão funcionando"""
    print("🔍 Testando embeddings locais...")
    
    try:
        from llm_providers import LocalEmbeddings
        
        print("✅ Importação do LocalEmbeddings bem-sucedida")
        
        # Criar instância
        embeddings = LocalEmbeddings()
        print("✅ Instância criada com sucesso")
        
        # Testar embedding de uma query
        test_query = "regulamentação ANTT"
        embedding = embeddings.embed_query(test_query)
        print(f"✅ Embedding gerado - Dimensão: {len(embedding)}")
        
        # Testar embedding de múltiplos documentos
        test_docs = ["documento 1", "documento 2"]
        doc_embeddings = embeddings.embed_documents(test_docs)
        print(f"✅ Embeddings de documentos gerados - Quantidade: {len(doc_embeddings)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro nos embeddings locais: {e}")
        return False

def teste_provedores():
    """Testa se os provedores estão configurados corretamente"""
    print("\n🔍 Testando provedores disponíveis...")
    
    try:
        from llm_providers import get_available_embedding_providers
        
        providers = get_available_embedding_providers()
        print("✅ Provedores de embeddings:")
        
        for key, info in providers.items():
            status_emoji = "✅" if info["status"] == "available" else "⚠️"
            print(f"  {status_emoji} {key}: {info['name']} - Status: {info['status']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao verificar provedores: {e}")
        return False

def teste_vectorstore():
    """Testa se o vectorstore pode ser carregado ou criado"""
    print("\n🔍 Testando vectorstore...")
    
    try:
        from antt_rag_unified import carregar_vectorstore_com_provider
        
        # Verificar se existe vectorstore local
        vectorstore_path = "vectorstore_local"
        if os.path.exists(vectorstore_path):
            print(f"✅ Encontrado vectorstore local em: {vectorstore_path}")
        else:
            print(f"⚠️ Vectorstore local não encontrado em: {vectorstore_path}")
            print("   (Será criado automaticamente na primeira execução)")
        
        # Tentar carregar vectorstore
        print("🔄 Tentando carregar vectorstore local...")
        vectorstore = carregar_vectorstore_com_provider("local")
        print("✅ Vectorstore carregado com sucesso!")
        
        # Testar busca simples
        docs = vectorstore.similarity_search("teste", k=1)
        print(f"✅ Busca no vectorstore funcionando - {len(docs)} documento(s) encontrado(s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no vectorstore: {e}")
        return False

def teste_llm_manager():
    """Testa se o LLM Manager está funcionando"""
    print("\n🔍 Testando LLM Manager...")
    
    try:
        from llm_providers import create_llm_manager
        
        # Testar criação do manager
        llm_manager = create_llm_manager("deepseek", embedding_provider="local")
        print("✅ LLM Manager criado com sucesso")
        
        # Testar embeddings através do manager
        embeddings = llm_manager.get_embeddings()
        print("✅ Embeddings obtidos através do manager")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no LLM Manager: {e}")
        return False

def main():
    """Executa todos os testes"""
    print("=" * 60)
    print("🚀 TESTE DO SISTEMA RAG COM EMBEDDINGS LOCAIS")
    print("=" * 60)
    
    testes = [
        ("Embeddings Locais", teste_embeddings_locais),
        ("Provedores", teste_provedores),
        ("LLM Manager", teste_llm_manager),
        ("Vectorstore", teste_vectorstore),
    ]
    
    resultados = []
    
    for nome, teste_func in testes:
        try:
            sucesso = teste_func()
            resultados.append((nome, sucesso))
        except Exception as e:
            print(f"❌ Erro crítico no teste {nome}: {e}")
            resultados.append((nome, False))
    
    # Resumo final
    print("\n" + "=" * 60)
    print("📊 RESUMO DOS TESTES")
    print("=" * 60)
    
    sucessos = 0
    for nome, sucesso in resultados:
        status = "✅ PASSOU" if sucesso else "❌ FALHOU"
        print(f"{status}: {nome}")
        if sucesso:
            sucessos += 1
    
    print(f"\n🎯 Resultado: {sucessos}/{len(resultados)} testes passaram")
    
    if sucessos == len(resultados):
        print("🎉 TODOS OS TESTES PASSARAM! Sistema pronto para uso.")
        print("\n📝 Para iniciar o sistema:")
        print("   streamlit run antt_rag_unified.py")
    else:
        print("⚠️ Alguns testes falharam. Verifique os erros acima.")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 