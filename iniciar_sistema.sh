#!/bin/bash

# Script de Inicialização do Sistema RAG-ANTT
# Autor: Sistema RAG Unificado
# Data: $(date)

echo "🚛 Sistema RAG-ANTT - Inicializador"
echo "=================================="
echo ""

# Verificar se estamos no diretório correto
if [ ! -f "antt_rag_unified.py" ]; then
    echo "❌ Erro: Execute este script no diretório do projeto RAG-ANTT"
    exit 1
fi

# Verificar se o ambiente virtual existe
if [ ! -d "venv" ]; then
    echo "❌ Erro: Ambiente virtual não encontrado. Execute 'python -m venv venv' primeiro."
    exit 1
fi

# Verificar se o arquivo .env existe
if [ ! -f ".env" ]; then
    echo "❌ Erro: Arquivo .env não encontrado. Configure suas chaves de API primeiro."
    echo "📝 Crie um arquivo .env com:"
    echo "OPENAI_API_KEY=sua-chave-openai"
    echo "OPENROUTER_API_KEY=sua-chave-openrouter"
    exit 1
fi

# Ativar ambiente virtual
source venv/bin/activate
source .env

echo "✅ Ambiente configurado com sucesso!"
echo ""

# Menu de opções
echo "Escolha o sistema para executar:"
echo ""
echo "1) 🚀 Sistema Unificado (Recomendado)"
echo "   - Todas as funcionalidades avançadas"
echo "   - Templates especializados"
echo "   - Busca híbrida inteligente"
echo "   - Interface completa"
echo ""
echo "2) 📱 Sistema Original (Simples)"
echo "   - Interface básica"
echo "   - Funcionalidades essenciais"
echo "   - Mais leve"
echo ""
echo "3) 🧪 Testar Configurações"
echo "   - Verificar APIs"
echo "   - Testar conectividade"
echo ""
echo "4) 📊 Ambos os Sistemas"
echo "   - Executar em paralelo"
echo "   - Comparar funcionalidades"
echo ""

read -p "Digite sua escolha (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Iniciando Sistema Unificado..."
        echo "📍 URL: http://localhost:8502"
        echo "⏹️  Para parar: Ctrl+C"
        echo ""
        streamlit run antt_rag_unified.py --server.port 8502
        ;;
    2)
        echo ""
        echo "📱 Iniciando Sistema Original..."
        echo "📍 URL: http://localhost:8501"
        echo "⏹️  Para parar: Ctrl+C"
        echo ""
        streamlit run antt_rag_deepseek.py --server.port 8501
        ;;
    3)
        echo ""
        echo "🧪 Testando configurações..."
        python test_llm_providers.py
        echo ""
        echo "✅ Teste concluído!"
        ;;
    4)
        echo ""
        echo "📊 Iniciando ambos os sistemas..."
        echo "🚀 Sistema Unificado: http://localhost:8502"
        echo "📱 Sistema Original: http://localhost:8501"
        echo "⏹️  Para parar: Ctrl+C"
        echo ""
        
        # Iniciar sistema original em background
        streamlit run antt_rag_deepseek.py --server.port 8501 &
        ORIGINAL_PID=$!
        
        # Aguardar um pouco
        sleep 3
        
        # Iniciar sistema unificado em foreground
        streamlit run antt_rag_unified.py --server.port 8502
        
        # Quando o unificado for interrompido, parar o original também
        kill $ORIGINAL_PID 2>/dev/null
        ;;
    *)
        echo "❌ Opção inválida. Execute o script novamente."
        exit 1
        ;;
esac

echo ""
echo "👋 Sistema encerrado. Até logo!" 