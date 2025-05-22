# import os
# import json
# import streamlit as st
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA

# # Constante para API key da OpenAI
# OPENAI_API_KEY = 'sk-proj-SJzLGfezVCxJLft228F2T3BlbkFJ2lSCkYReBn53ZYbMfmKh'

# # Configuração da página Streamlit - esta deve ser a primeira chamada Streamlit
# st.set_page_config(
#     page_title="RAG ANTT",
#     page_icon="🚆",
#     layout="wide"
# )

# def carregar_vectorstore(api_key, caminho_vectorstore="vectorstore"):
#     """Carrega o vectorstore ANTT."""
#     with st.spinner(f"Carregando base de conhecimento de {caminho_vectorstore}..."):
#         embeddings = OpenAIEmbeddings(openai_api_key=api_key)
#         vectorstore = FAISS.load_local(caminho_vectorstore, embeddings, allow_dangerous_deserialization=True)
#         return vectorstore

# def criar_filtro_metadados(tipo_documento=None, ano=None, numero=None):
#     """Cria um filtro para busca por metadados."""
#     filtro = {}
    
#     if tipo_documento:
#         filtro["tipo_documento"] = tipo_documento
    
#     if ano:
#         filtro["ano"] = ano
    
#     if numero:
#         filtro["numero"] = numero
    
#     return filtro if filtro else None

# def pesquisar_documentos(query, vectorstore, k=12, tipo_documento=None, ano=None, numero=None):
#     """Pesquisa documentos com base em uma query, podendo filtrar por metadados."""
#     resultados = []
    
#     # Estratégia 1: Busca com filtros específicos se fornecidos
#     filtro = criar_filtro_metadados(tipo_documento, ano, numero)
#     if filtro:
#         try:
#             resultados = vectorstore.similarity_search(
#                 query,
#                 k=k,
#                 filter=filtro
#             )
#         except Exception as e:
#             print(f"Erro na busca com filtro: {str(e)}")
    
#     # Estratégia 2: Se não houver resultados suficientes ou não houver filtros, faz busca geral
#     if len(resultados) < 2:
#         try:
#             resultados = vectorstore.similarity_search(query, k=k)
#         except Exception as e:
#             print(f"Erro na busca geral: {str(e)}")
    
#     # Estratégia 3: Busca semântica mais agressiva se ainda não encontrou resultados
#     if len(resultados) < 2:
#         try:
#             # Ajustar a query para ser mais genérica
#             termos_chave = ' '.join([palavra for palavra in query.split() if len(palavra) > 3])
#             if termos_chave:
#                 resultados = vectorstore.similarity_search(termos_chave, k=k)
#         except Exception as e:
#             print(f"Erro na busca com termos-chave: {str(e)}")
    
#     return resultados

# TEMPLATE_PROMPT_ANTT = """
# Você é um assistente especializado em regulamentação da ANTT (Agência Nacional de Transportes Terrestres).
# Responda à pergunta do usuário usando apenas os contextos fornecidos.

# Quando a pergunta for sobre parâmetros técnicos, normas ou especificações:
# 1. Extraia os valores, definições e requisitos técnicos precisos
# 2. Organize as informações em formato estruturado
# 3. Inclua detalhes como valores, unidades de medida e critérios de aplicação

# Se você não encontrar a resposta nos contextos, diga "Não encontrei informações específicas sobre isso nos documentos da ANTT."
# Ao citar informações, mencione sempre a fonte do documento (tipo, número e ano).

# Contextos:
# {context}

# Pergunta: {question}

# Resposta:
# """

# def gerar_resposta(pergunta, documentos, llm):
#     """Gera uma resposta baseada nos documentos recuperados."""
#     if not documentos:
#         return "Não encontrei documentos relevantes para esta pergunta."
    
#     # Preparar o contexto a partir dos documentos - formato mais detalhado
#     contextos = []
#     documentos_para_exibicao = []  # Lista para armazenar informações para exibição
    
#     for i, doc in enumerate(documentos):
#         metadados = doc.metadata
#         tipo = metadados.get('nome_tipo', 'Documento')
#         numero = metadados.get('numero', 'N/A')
#         ano = metadados.get('ano', 'N/A')
#         chunk = metadados.get('chunk', 'N/A')
#         total_chunks = metadados.get('total_chunks', 'N/A')
        
#         # Informações do documento para exibição
#         documentos_para_exibicao.append({
#             "indice": i+1,
#             "tipo": tipo,
#             "numero": numero,
#             "ano": ano,
#             "trecho": doc.page_content,
#             "caminho": metadados.get('caminho', 'Não especificado')
#         })
        
#         # Formatação para o contexto para o LLM
#         contexto = f"""
# [Documento {i+1}: {tipo} {numero}/{ano} - Parte {chunk}/{total_chunks}]
# Fonte: {metadados.get('caminho', 'Não especificado')}
# Conteúdo:
# {doc.page_content}
# """
#         contextos.append(contexto)
    
#     # Limitar o tamanho total do contexto para não exceder limites de token
#     contexto_completo = "\n\n".join(contextos)
    
#     # Template para gerar resposta com citações específicas
#     prompt_template = """
# Você é um assistente especializado em regulamentação da ANTT (Agência Nacional de Transportes Terrestres).
# Para a consulta: "{question}"

# INSTRUÇÕES IMPORTANTES:
# 1. Responda com base EXCLUSIVAMENTE nas informações dos documentos fornecidos
# 2. Para cada informação que você usar na resposta, CITE O TRECHO EXATO do documento
# 3. Depois da resposta principal, adicione uma seção "Trechos dos Documentos" onde você lista TODOS os trechos relevantes usados 
# 4. Use o formato: "Documento: [TIPO DO DOCUMENTO] [NÚMERO]/[ANO] - [TÍTULO]" seguido pelo trecho exato

# Contextos:
# {context}

# Sua resposta DEVE seguir este formato:
# 1. Resposta completa à pergunta
# 2. Seção "Trechos dos Documentos Citados:" com citações diretas
# """
    
#     # Usar o template de prompt
#     prompt = PromptTemplate(
#         template=prompt_template,
#         input_variables=["context", "question"]
#     )
    
#     # Executar a chain de pergunta e resposta com o contexto
#     chain = prompt | llm
    
#     # Gerar resposta
#     resposta = chain.invoke({
#         "context": contexto_completo,
#         "question": pergunta
#     })
    
#     return resposta.content, documentos_para_exibicao

# def carregar_relatorio_documentos():
#     """Carrega o relatório de documentos processados."""
#     try:
#         with open("relatorio_documentos.json", "r", encoding="utf-8") as f:
#             return json.load(f)
#     except Exception as e:
#         st.error(f"Erro ao carregar o relatório de documentos: {str(e)}")
#         return []

# def verificar_documentos_importantes():
#     """Verifica se documentos importantes estão no vectorstore e os adiciona se necessário."""
#     documentos_importantes = [
#         "dados_antt/INM/2024/INM-00000034-2024.md",  # Instrução Normativa 34 de 2024 - Parâmetros de Desempenho de Pavimento
#         "dados_antt/RES/2024/RES-00006057-2024.md",  # Resolução 6057 de 2024 - Programa de Sustentabilidade
#         "dados_antt/DLB/2024/DLB-00000092-2024.md"   # Deliberação 92 de 2024 - Limites de Peso
#     ]
    
#     documentos_indexados = []
    
#     # Verificar quais documentos existem
#     for caminho in documentos_importantes:
#         if os.path.exists(caminho):
#             with open(caminho, 'r', encoding='utf-8') as f:
#                 conteudo = f.read()
            
#             # Extrair metadados básicos do caminho
#             partes_caminho = caminho.split('/')
#             tipo_documento = partes_caminho[1] if len(partes_caminho) > 1 else None
#             ano = partes_caminho[2] if len(partes_caminho) > 2 else None
#             numero = None
            
#             # Extrair número do nome do arquivo
#             nome_arquivo = os.path.basename(caminho)
#             if '-' in nome_arquivo:
#                 partes = nome_arquivo.split('-')
#                 if len(partes) >= 2:
#                     numero = partes[1]
            
#             # Mapear tipo para nome completo
#             tipo_nome = {
#                 "RES": "Resolução",
#                 "POR": "Portaria",
#                 "INM": "Instrução Normativa",
#                 "DLB": "Deliberação",
#                 "INC": "Instrução Normativa Complementar"
#             }.get(tipo_documento, tipo_documento)
            
#             # Dividir em chunks menores para capturar informações específicas
#             from langchain_text_splitters import RecursiveCharacterTextSplitter
#             from langchain_core.documents import Document
            
#             text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=500,  # Chunks menores para capturar detalhes específicos
#                 chunk_overlap=200,
#                 length_function=len,
#             )
            
#             # Criar documentos
#             chunks = text_splitter.split_text(conteudo)
#             for i, chunk in enumerate(chunks):
#                 documento = Document(
#                     page_content=chunk,
#                     metadata={
#                         "tipo_documento": tipo_documento,
#                         "nome_tipo": tipo_nome,
#                         "ano": ano,
#                         "numero": numero,
#                         "caminho": caminho,
#                         "chunk": i + 1,
#                         "total_chunks": len(chunks)
#                     }
#                 )
#                 documentos_indexados.append(documento)
    
#     return documentos_indexados

# def atualizar_vectorstore_com_documentos_importantes(vectorstore, embeddings):
#     """Atualiza o vectorstore com documentos importantes se necessário."""
#     documentos = verificar_documentos_importantes()
#     if documentos:
#         st.info(f"Atualizando vectorstore com {len(documentos)} documentos importantes...")
#         vectorstore.add_documents(documentos)
#         # Salvar o vectorstore atualizado
#         vectorstore.save_local("vectorstore")
#         st.success("Vectorstore atualizado com sucesso!")
#     return vectorstore

# def main():
#     st.title("Sistema RAG - Documentos da ANTT")
#     st.write("Este sistema permite consultar a base de documentos regulatórios da ANTT.")
    
#     # Usar a chave API definida como constante
#     api_key = OPENAI_API_KEY
    
#     # Verifica se a chave API constante está vazia ou inválida
#     if not api_key or api_key.startswith("sk-") == False:
#         # Tenta obter da variável de ambiente
#         api_key = os.getenv("OPENAI_API_KEY", "")
        
#         # Tenta obter da configuração de secrets, mas trata a exceção se não existir
#         if not api_key:
#             try:
#                 api_key = st.secrets.get("OPENAI_API_KEY", "")
#             except:
#                 # Ignora o erro se não existir arquivo de secrets
#                 pass
        
#         # Se ainda não tiver a chave, solicita ao usuário
#         if not api_key:
#             api_key = st.text_input("Insira sua chave da API OpenAI:", type="password")
            
#     if not api_key or api_key.startswith("sk-") == False:
#         st.error("É necessário fornecer uma chave válida da API OpenAI que comece com 'sk-'.")
#         return
    
#     try:
#         # Carregar vectorstore
#         vectorstore = carregar_vectorstore(api_key)
        
#         # Configurar LLM
#         llm = ChatOpenAI(
#             openai_api_key=api_key,
#             model_name="gpt-4o",
#             temperature=0
#         )
        
#         # Atualizar o vectorstore com documentos importantes se necessário
#         embeddings = OpenAIEmbeddings(openai_api_key=api_key)
#         vectorstore = atualizar_vectorstore_com_documentos_importantes(vectorstore, embeddings)
        
#         # Carregar os tipos de documentos e anos disponíveis do relatório
#         relatorio = carregar_relatorio_documentos()
        
#         tipos_documento = sorted(list(set([doc["tipo"] for doc in relatorio if doc.get("tipo")])))
#         anos = sorted(list(set([doc["ano"] for doc in relatorio if doc.get("ano")])))
        
#         # Interface principal
#         col1, col2 = st.columns([2, 1])
        
#         with col2:
#             st.header("Filtros")
            
#             tipo_selecionado = st.selectbox(
#                 "Tipo de Documento",
#                 ["Todos"] + tipos_documento
#             )
            
#             ano_selecionado = st.selectbox(
#                 "Ano",
#                 ["Todos"] + anos
#             )
            
#             numero_documento = st.text_input("Número do Documento (opcional)")
            
#             # Estatísticas da base
#             st.header("Estatísticas")
#             st.metric("Total de Documentos", len(relatorio))
            
#             # Contagem por tipo
#             if tipos_documento:
#                 tipo_counts = {}
#                 for doc in relatorio:
#                     tipo = doc.get("tipo")
#                     if tipo:
#                         tipo_counts[tipo] = tipo_counts.get(tipo, 0) + 1
                
#                 st.subheader("Documentos por Tipo")
#                 for tipo, count in sorted(tipo_counts.items(), key=lambda x: x[1], reverse=True):
#                     st.text(f"{tipo}: {count}")
        
#         with col1:
#             # Converter seleções para valores para a busca
#             tipo_filtro = None if tipo_selecionado == "Todos" else tipo_selecionado
#             ano_filtro = None if ano_selecionado == "Todos" else ano_selecionado
#             numero_filtro = None if not numero_documento else numero_documento
            
#             # Mostrar filtros ativos
#             filtros_ativos = []
#             if tipo_filtro:
#                 filtros_ativos.append(f"Tipo: {tipo_filtro}")
#             if ano_filtro:
#                 filtros_ativos.append(f"Ano: {ano_filtro}")
#             if numero_filtro:
#                 filtros_ativos.append(f"Número: {numero_filtro}")
            
#             if filtros_ativos:
#                 st.info("Filtros ativos: " + ", ".join(filtros_ativos))
            
#             # Campo de pergunta
#             st.header("Consulta")
#             pergunta = st.text_area(
#                 "Digite sua pergunta sobre documentos da ANTT:",
#                 height=100
#             )
            
#             k_docs = st.slider(
#                 "Número de documentos a recuperar",
#                 min_value=1,
#                 max_value=10,
#                 value=4
#             )
            
#             if st.button("Buscar", type="primary"):
#                 if not pergunta:
#                     st.warning("Por favor, digite uma pergunta.")
#                 else:
#                     with st.spinner("Buscando resposta..."):
#                         # Pesquisar documentos com os filtros aplicados
#                         documentos = pesquisar_documentos(
#                             pergunta, 
#                             vectorstore, 
#                             k=k_docs,
#                             tipo_documento=tipo_filtro, 
#                             ano=ano_filtro, 
#                             numero=numero_filtro
#                         )
                        
#                         # Gerar resposta
#                         resposta, documentos_para_exibicao = gerar_resposta(pergunta, documentos, llm)
                        
#                         # Exibir resposta
#                         st.markdown("### Resposta")
#                         st.markdown(resposta)
                        
#                         # Exibir fontes
#                         st.markdown("### Fontes Consultadas")
#                         for i, doc in enumerate(documentos_para_exibicao):
#                             with st.expander(f"Documento {doc['indice']}"):
#                                 st.markdown(f"**Tipo:** {doc['tipo']}")
#                                 st.markdown(f"**Número:** {doc['numero']}")
#                                 st.markdown(f"**Ano:** {doc['ano']}")
#                                 st.markdown("**Trecho:**")
#                                 st.markdown(doc['trecho'])
    
#     except Exception as e:
#         st.error(f"Ocorreu um erro: {str(e)}")

# if __name__ == "__main__":
#     main() 