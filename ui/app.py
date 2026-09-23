"""
Pagina Streamlit de QA do RAG-ANTT.

O comando de subida continua sendo streamlit run antt_rag_unified.py,
que delega para esta pagina. Streaming da resposta fica aqui.
A busca sincrona da API passa por rag_service.consultar.
"""

import html
import os
import re
import unicodedata
from typing import Dict

import streamlit as st

from config import (
    LLM_PROVIDERS,
    STREAMLIT_LAYOUT,
    STREAMLIT_PAGE_ICON,
    STREAMLIT_PAGE_TITLE,
    get_historico_turnos_guardados,
    get_openai_api_key,
    logger,
)
from llm_providers import (
    create_llm_manager,
    get_available_embedding_providers,
    get_available_providers,
    verificar_ollama_disponivel,
)
from rag_service import (
    ProvedorNaoLiberadoError,
    disparar_reindexacao,
    incluir_documento,
    listar_anos,
    obter_status,
    recuperar_trechos,
)
from tipos_documento import listar_siglas_tipo
from ui.theme import (
    ALTURA_CAMPO_PADRAO,
    AZUL_ESCURO,
    AZUL_HOVER,
    AZUL_INSTITUCIONAL,
    AZUL_MEDIO,
    AZUL_PRIMARIO,
    CHAVE_ALTURA_CAMPO,
    CINZA_SECUNDARIO,
    aplicar_estilo_institucional,
    renderizar_cabecalho,
    resolver_altura_campo_pergunta,
    rotulos_altura_campo_pergunta,
)
from antt_rag_unified import (
    carregar_vectorstore_com_provider,
    detectar_documentos_novos,
    extrair_citacoes_da_resposta,
    gerar_resposta,
    gerar_resposta_streaming,
    resposta_indica_falha,
)


# Cores institucionais por tipo documental (tokens do Design System gov.br,
# definidos em ui/theme.py). Usadas apenas como faixa lateral do badge, nunca
# como unico portador de significado (requisito de acessibilidade WCAG).
_CORES_TIPO_DOCUMENTO: Dict[str, str] = {
    "instrucao normativa": AZUL_PRIMARIO,
    "resolucao": AZUL_HOVER,
    "voto": AZUL_MEDIO,
    "deliberacao": AZUL_INSTITUCIONAL,
    "portaria": AZUL_ESCURO,
}

# Tipos nao mapeados recebem o cinza neutro de texto secundario.
_COR_TIPO_PADRAO: str = CINZA_SECUNDARIO


def _normalizar_rotulo_tipo(tipo: str) -> str:
    """
    Normaliza o rotulo de tipo documental para busca na tabela de cores.

    Remove acentuacao e caixa, permitindo casar "Instrucao Normativa",
    "Instrucao normativa" e "INSTRUCAO NORMATIVA" na mesma chave.

    Args:
        tipo: Rotulo bruto vindo dos metadados do documento.

    Returns:
        Rotulo em minusculas e sem acentuacao.
    """
    if not isinstance(tipo, str):
        return ""
    sem_acento = unicodedata.normalize("NFKD", tipo)
    sem_acento = "".join(c for c in sem_acento if not unicodedata.combining(c))
    return sem_acento.strip().lower()


def badge_tipo_documento(tipo: str) -> str:
    """
    Gera o HTML de um badge institucional para o tipo de documento.

    Substitui os marcadores coloridos (bolinhas) por um rotulo textual com
    faixa lateral colorida, mantendo o significado legivel sem depender de
    cor ou icone.

    Args:
        tipo: Tipo documental (ex.: "Instrucao Normativa", "Resolucao").

    Returns:
        Fragmento HTML pronto para st.markdown(..., unsafe_allow_html=True).
    """
    rotulo = tipo.strip() if isinstance(tipo, str) and tipo.strip() else "Documento"
    cor = _CORES_TIPO_DOCUMENTO.get(
        _normalizar_rotulo_tipo(rotulo), _COR_TIPO_PADRAO
    )
    rotulo_seguro = html.escape(rotulo)
    return (
        f'<span class="badge-tipo" style="border-left:3px solid {cor};">'
        f"{rotulo_seguro}</span>"
    )


def interface_usuario_unificada():
    """Interface principal do sistema RAG unificado."""
    st.set_page_config(
        page_title=STREAMLIT_PAGE_TITLE,
        page_icon=STREAMLIT_PAGE_ICON,
        layout=STREAMLIT_LAYOUT,
        initial_sidebar_state="expanded"
    )

    # Inicializar historico de conversa no session_state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Historico visual do chat (pares pergunta/resposta para exibicao)
    if "mensagens_chat" not in st.session_state:
        st.session_state.mensagens_chat = []
    
    # Identidade visual gov.br centralizada em ui/theme.py. A altura do
    # campo de pergunta vem do preset escolhido na barra lateral.
    if CHAVE_ALTURA_CAMPO not in st.session_state:
        st.session_state[CHAVE_ALTURA_CAMPO] = ALTURA_CAMPO_PADRAO
    aplicar_estilo_institucional(
        altura_campo_rem=resolver_altura_campo_pergunta(
            st.session_state[CHAVE_ALTURA_CAMPO]
        )
    )

    renderizar_cabecalho(
        "Sistema de Consulta Normativa - ANTT",
        "Consulta assistida a resoluções, instruções normativas e "
        "deliberações",
    )
    
    # Sidebar para configurações
    with st.sidebar:
        st.header("Configurações")
        
        # Seleção do provedor
        st.subheader("Serviço de IA")
        providers = get_available_providers()
        provider_names = {
            "openai": "OpenAI (GPT-4)",
            "deepseek": "DeepSeek",
            "ollama": "Local (Ollama) - CPU",
        }
        
        # Tornar DeepSeek padrão quando disponivel no filtro
        provider_keys = list(providers.keys())
        default_index = 0
        if "deepseek" in providers:
            default_index = provider_keys.index("deepseek")
        elif "ollama" in providers:
            default_index = provider_keys.index("ollama")
        
        selected_provider = st.selectbox(
            "Serviço:",
            options=provider_keys,
            format_func=lambda x: provider_names.get(x, x),
            index=default_index if provider_keys else 0,
            help="Serviço de IA que redige a resposta. Local (Ollama) nao envia dados a APIs externas."
        )
        
        # Seleção do modelo
        if selected_provider in providers:
            available_models = providers[selected_provider]["models"]
            selected_model = st.selectbox(
                "Modelo:",
                options=available_models,
                index=0
            )
        else:
            selected_model = "gpt-4"
        
        # Informação sobre fallback automático
        if selected_provider == "openai":
            st.caption(
                "Se a OpenAI atingir o limite de uso, o DeepSeek assume "
                "automaticamente e a consulta continua."
            )
        elif selected_provider == "ollama":
            st.caption(
                "Inferencia local via Ollama (CPU). Ideal para ambiente ANTT "
                "sem API externa. Respostas podem demorar 20-60 s."
            )

        if st.button(
            "Atualizar base",
            use_container_width=True,
            help="Inclui na consulta os documentos novos ou "
                 "alterados. Leva alguns minutos.",
        ):
            st.session_state["_reindexando"] = True
            st.rerun()
        
        # Status das APIs
        st.subheader("Situação dos serviços")
        status_servico = obter_status()
        if status_servico.vectorstore_ok:
            st.caption(
                "Indice disponivel. Documentos catalogados: {0}.".format(
                    status_servico.n_docs
                )
            )
        else:
            st.caption(
                "Indice ainda nao encontrado. Documentos catalogados: {0}.".format(
                    status_servico.n_docs
                )
            )
        
        # Verificar status do OpenAI
        try:
            openai_manager = create_llm_manager("openai")
            st.markdown(
                '<div class="provider-status status-ok">OpenAI: conectado</div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.markdown(
                f'<div class="provider-status status-error">'
                f"OpenAI: indisponível ({str(e)[:50]})</div>",
                unsafe_allow_html=True,
            )
        
        # Verificar status do DeepSeek
        try:
            deepseek_manager = create_llm_manager("deepseek")
            st.markdown(
                '<div class="provider-status status-ok">DeepSeek: conectado</div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.markdown(
                f'<div class="provider-status status-error">'
                f"DeepSeek: indisponível ({str(e)[:50]})</div>",
                unsafe_allow_html=True,
            )

        # Verificar status do Ollama (sem criar LLMManager completo se filtrado)
        if "ollama" in providers or "ollama" in LLM_PROVIDERS:
            ollama_ok, ollama_msg = verificar_ollama_disponivel()
            if ollama_ok:
                st.markdown(
                    '<div class="provider-status status-ok">Ollama: conectado</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="provider-status status-error">'
                    f"Ollama: indisponível ({ollama_msg[:60]})</div>",
                    unsafe_allow_html=True,
                )
        
        # Configurações avançadas
        st.subheader("Configurações avançadas")
        
        temperatura = st.slider(
            "Liberdade de redação:",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Valores baixos deixam a resposta mais objetiva e fiel "
                 "ao texto das normas."
        )

        rotulos_altura = rotulos_altura_campo_pergunta()
        st.radio(
            "Tamanho do campo de pergunta:",
            options=list(rotulos_altura.keys()),
            format_func=lambda chave: rotulos_altura[chave],
            horizontal=True,
            key=CHAVE_ALTURA_CAMPO,
            help="Define a altura do campo no rodape. Depois, arraste a "
                 "borda inferior para refinar.",
        )
        
        # Seleção do provedor de embeddings
        st.markdown("**Localização dos documentos**")
        embedding_providers = get_available_embedding_providers()
        
        # Criar lista de opções com descrições claras
        embedding_options = list(embedding_providers.keys())
        embedding_labels = [embedding_providers[key]["name"] for key in embedding_options]
        
        selected_embedding_provider = st.selectbox(
            "Como localizar os trechos:",
            options=embedding_options,
            format_func=lambda x: embedding_providers[x]["name"],
            index=0,  # Padrão é processamento local
            help="Local funciona sem internet. OpenAI usa serviço externo."
        )
        
        # Mostrar descrição detalhada da opção selecionada
        selected_info = embedding_providers[selected_embedding_provider]
        
        if selected_embedding_provider == "local":
            st.caption(
                f"{selected_info['name']}: a busca é feita na própria "
                "máquina, sem enviar dados para fora."
            )
        elif selected_embedding_provider == "openai":
            st.caption(
                f"{selected_info['name']}: a busca utiliza o serviço "
                "externo da OpenAI."
            )
        elif selected_embedding_provider == "free":
            st.caption(
                f"{selected_info['name']}: tenta a busca local primeiro e "
                "recorre à OpenAI apenas se necessário."
            )
        
        # Mostrar status da dependência
        if selected_embedding_provider == "local":
            try:
                import sentence_transformers
                st.caption("Componente de busca local instalado.")
            except ImportError:
                st.error(
                    "Componente de busca local ausente. Peça à equipe "
                    "técnica para instalar o pacote sentence-transformers."
                )
        elif selected_embedding_provider == "openai":
            try:
                if get_openai_api_key():
                    st.caption("Credencial da OpenAI configurada.")
                else:
                    st.error(
                        "Credencial da OpenAI não encontrada. "
                        "Use a opção local ou solicite a configuração."
                    )
            except Exception:
                st.warning(
                    "Não foi possível verificar a credencial da OpenAI."
                )

        # Executar reindexacao (precisa estar fora do button para manter o spinner)
        if st.session_state.get("_reindexando"):
            del st.session_state["_reindexando"]
            with st.spinner(
                "Atualizando a base de documentos. "
                "Isso pode levar alguns minutos."
            ):
                try:
                    resultado_reindex = disparar_reindexacao(
                        embedding_provider=selected_embedding_provider
                    )
                    sucesso = resultado_reindex.sucesso
                    msg = resultado_reindex.mensagem
                except ProvedorNaoLiberadoError as exc_reindex:
                    sucesso = False
                    msg = str(exc_reindex)
            if sucesso:
                st.success(msg)
                # Limpar cache de verificacao para que o alerta desapareca
                st.session_state.pop("_docs_novos_checado", None)
                st.session_state.pop("_docs_novos_lista", None)
                st.session_state.pop("_vectorstore_desatualizado", None)
            else:
                st.error(msg)

        st.divider()

        max_tokens = st.number_input(
            "Tamanho máximo da resposta:",
            min_value=500,
            max_value=4096,
            value=4096,
            step=256,
            help="Limita o comprimento da resposta. Padrao: o maximo permitido (4096)."
        )
        
        num_documentos = st.slider(
            "Trechos consultados por pergunta:",
            min_value=5,
            max_value=40,
            value=20,
            help="Quantos trechos o sistema lê antes de responder. "
                 "Recomendado: 20."
        )
        
        # Filtros de busca
        st.subheader("Filtros de busca")
        
        tipo_documento = st.selectbox(
            "Tipo de documento:",
            options=["Todos"] + listar_siglas_tipo(),
            help="Restringe a busca a um tipo de ato normativo."
        )
        
        ano_filtro = st.selectbox(
            "Ano:",
            options=["Todos"] + listar_anos(),
            help="Restringe a busca ao ano escolhido. "
                 "A lista sai dos anos que ja estao na base."
        )
        
        numero_filtro = st.text_input(
            "Número do documento:",
            placeholder="Ex: 6057",
            help="Busca um documento pelo número. Exemplo: 6057."
        )

        st.divider()

        # Orientacao detalhada fica aqui, e nao nos tooltips: o balao de
        # ajuda do Streamlit e estreito e corta textos longos.
        with st.expander("Ajuda"):
            st.markdown(
                """
**Como perguntar**

Escreva a dúvida em linguagem natural, como faria a um colega.
Se souber o documento, cite-o na pergunta para melhorar a precisão.

**Quando usar Atualizar base**

- Documentos foram incluídos, substituídos ou removidos da pasta de dados
- Aparece o aviso de documentos pendentes no topo da tela
- Uma norma que você sabe que existe não é encontrada
- O modelo de localização dos trechos (embeddings) foi alterado

Reconstrói o índice a partir dos textos já convertidos. Tabelas que estão
em imagem são lidas de novo quando a leitura guardada está desatualizada
ou com qualidade baixa. Leva alguns minutos.

**Se a resposta parecer incompleta**

- Aumente Trechos consultados por pergunta nas configurações avançadas
- Verifique se algum filtro na barra lateral está restringindo a busca
- Reformule a pergunta com os termos usados na norma
                """
            )
    
    # Carregar vectorstore primeiro (fora dos containers)
    try:
        vectorstore = carregar_vectorstore_com_provider(selected_embedding_provider)
        vectorstore_loaded = True
    except Exception as e:
        logger.error(f"Erro ao carregar vectorstore: {str(e)}")
        vectorstore = None
        vectorstore_loaded = False

    # Verificacao automatica de documentos novos na inicializacao
    # Usa cache no session_state para nao repetir a varredura a cada rerun
    if "_docs_novos_checado" not in st.session_state:
        docs_novos = detectar_documentos_novos()

        # Verificar se o vectorstore esta desatualizado comparando
        # timestamps: se relatorio_documentos.json e mais recente que
        # o vectorstore, significa que novos docs foram catalogados
        # mas ainda nao indexados.
        vectorstore_desatualizado = False
        try:
            catalogo_path = "relatorio_documentos.json"
            vs_index_path = os.path.join("vectorstore_local", "index.faiss")
            if not vectorstore_loaded:
                vectorstore_desatualizado = True
            elif os.path.exists(catalogo_path) and os.path.exists(vs_index_path):
                ts_catalogo = os.path.getmtime(catalogo_path)
                ts_vectorstore = os.path.getmtime(vs_index_path)
                if ts_catalogo > ts_vectorstore:
                    vectorstore_desatualizado = True
                    logger.info(
                        "Vectorstore desatualizado: catalogo mais recente que o indice"
                    )
        except Exception:
            pass

        st.session_state["_docs_novos_checado"] = True
        st.session_state["_docs_novos_lista"] = docs_novos
        st.session_state["_vectorstore_desatualizado"] = vectorstore_desatualizado
    else:
        docs_novos = st.session_state.get("_docs_novos_lista", [])
        vectorstore_desatualizado = st.session_state.get("_vectorstore_desatualizado", False)

    if docs_novos:
        st.warning(
            f"Há **{len(docs_novos)}** documento(s) ainda não incluído(s) na "
            f"base de consulta. Enquanto isso, eles não aparecerão nas "
            f"respostas. Use **Atualizar base** na barra lateral para incluí-los."
        )
        with st.expander(f"Ver {len(docs_novos)} documento(s) pendente(s)"):
            for nome in docs_novos:
                st.text(nome)
    elif vectorstore_desatualizado:
        st.warning(
            "A base de consulta está desatualizada em relação aos documentos "
            "disponíveis. Use **Atualizar base** na barra lateral para atualizá-la."
        )

    # Situacao da consulta em linha unica, acima da conversa. Antes ocupava
    # uma coluna lateral de um terco da largura, que repetia dados da barra
    # lateral e estreitava a leitura da resposta.
    if vectorstore_loaded:
        origem_dos_trechos = "Base de documentos carregada"
        if hasattr(vectorstore, "_embedding_provider"):
            provider = vectorstore._embedding_provider
            if provider == "local":
                origem_dos_trechos = "Base de documentos (busca local)"
            elif provider == "openai":
                origem_dos_trechos = "Base de documentos (busca via OpenAI)"

        nome_provedor_ia = provider_names.get(
            selected_provider, selected_provider
        )
        nome_embeddings = embedding_providers.get(
            selected_embedding_provider, {}
        ).get("name", selected_embedding_provider)

        st.caption(
            f"{origem_dos_trechos}. Serviço de IA: {nome_provedor_ia}. "
            f"Localização dos trechos: {nome_embeddings}."
        )
    else:
        st.error(
            "A base de documentos não pôde ser carregada. "
            "Use Atualizar base na barra lateral ou contate a "
            "equipe técnica."
        )

    # A conversa ocupa a largura total e cresce para baixo; o campo de
    # pergunta fica fixo no rodape, logo abaixo da ultima resposta.
    for msg in st.session_state.mensagens_chat:
        with st.chat_message("user"):
            st.markdown(msg["pergunta"])
        with st.chat_message("assistant"):
            st.markdown(msg["resposta"])
            if msg.get("provider"):
                st.caption(f"Resposta via {msg['provider']}")

    # Acima da caixa de pergunta. O Streamlit fixa o chat_input no
    # rodape; um botao depois dele nao fica abaixo da caixa, e o script
    # que forcava essa posicao apagava a tela no segundo envio.
    if st.button(
        "Nova conversa",
        key="btn_nova_conversa",
        use_container_width=True,
        help="Apaga as perguntas e respostas desta sessão.",
    ):
        st.session_state.chat_history = []
        if "mensagens_chat" in st.session_state:
            st.session_state.mensagens_chat = []
        st.session_state.pop("pergunta_exemplo", None)
        st.session_state.pop("processar_automatico", None)
        st.rerun()

    # Campo de pergunta. O Streamlit fixa o st.chat_input no rodape da
    # pagina, de modo que ele acompanha o fim da conversa em vez de ficar
    # acima da ultima resposta. Altura e largura sao ajustaveis pelo
    # usuario (alca no canto inferior direito; ver ui/theme.py).
    pergunta = st.chat_input(
        "Digite sua pergunta sobre documentos da ANTT",
        disabled=not vectorstore_loaded,
    )

    # Sugestoes sempre montadas no script (mesmo apos a primeira resposta).
    # Se os botoes so existirem com a conversa vazia, o segundo clique se
    # perde: no rerun o chat ja tem mensagens, o widget nao e recriado e o
    # Streamlit descarta o evento antes de gravar processar_automatico.
    # Mistura normas indexadas (INM/RES) com notas tecnicas SEI da base.
    _EXEMPLOS_CONSULTA = (
        "Quais são os parâmetros técnicos para pavimentos rodoviários?",
        "Instrução Normativa 34 de 2024 sobre parâmetros de desempenho",
        "Resolução 6057 de 2024 - principais pontos",
        "Quais são as penalidades por descumprimento das normas?",
        "O que diz a Nota Técnica SEI sobre flexibilização da Frente de Serviços Operacionais?",
        "Qual a orientação da GEGIR sobre regularização e melhoria de acessos nas rodovias federais concedidas?",
        "O que tratam as notas técnicas SEI sobre disponibilização de veículos para policiamento rodoviário?",
        "Há análise SEI sobre reposicionamento de agulhas do sistema FreeFlow na RMSP?",
    )

    def _agendar_exemplo(texto_exemplo: str) -> None:
        """Agenda uma sugestao para processamento no proximo ciclo."""
        st.session_state.pergunta_exemplo = texto_exemplo
        st.session_state.processar_automatico = True

    conversa_vazia = not st.session_state.mensagens_chat
    with st.expander(
        "Sugestões de consulta",
        expanded=conversa_vazia,
    ):
        if conversa_vazia:
            st.caption("Escolha uma sugestão ou escreva sua pergunta abaixo.")
        else:
            st.caption(
                "Escolha outra sugestão para enviar uma nova pergunta "
                "nesta conversa."
            )
        col_ex1, col_ex2 = st.columns(2)
        for i, exemplo in enumerate(_EXEMPLOS_CONSULTA):
            col = col_ex1 if i % 2 == 0 else col_ex2
            with col:
                st.button(
                    f"{exemplo[:60]}{'...' if len(exemplo) > 60 else ''}",
                    key=f"btn_exemplo_{i}",
                    use_container_width=True,
                    help=exemplo,
                    on_click=_agendar_exemplo,
                    args=(exemplo,),
                    disabled=not vectorstore_loaded,
                )

    # Verificar se deve processar automaticamente um exemplo
    processar_exemplo_automatico = False
    if st.session_state.get("processar_automatico"):
        processar_exemplo_automatico = True
        st.session_state.processar_automatico = False

    # Processamento da consulta
    if vectorstore_loaded and (pergunta or processar_exemplo_automatico):
        # Para processamento automatico, usar a pergunta do session_state
        pergunta_para_processar = pergunta
        if processar_exemplo_automatico:
            pergunta_para_processar = st.session_state.get("pergunta_exemplo", "")
            st.session_state.pop("pergunta_exemplo", None)
        
        # Verificar se temos uma pergunta válida
        if not pergunta_para_processar or pergunta_para_processar.strip() == "":
            st.error("Nenhuma pergunta fornecida para processamento.")
        else:
            with st.spinner("Processando consulta..."):
                try:
                    # Tentar criar LLM manager com o provedor selecionado
                    llm_manager = None
                    llm = None
                    provider_usado = selected_provider
                    
                    try:
                        llm_manager = create_llm_manager(selected_provider, selected_model)
                        llm = llm_manager.get_llm(temperature=temperatura, max_tokens=max_tokens)
                        provider_usado = llm_manager.provider
                        st.caption(
                            f"Serviço em uso: "
                            f"{provider_names.get(provider_usado, provider_usado)}"
                        )
                        if selected_provider == "ollama" and provider_usado != "ollama":
                            st.warning(
                                "Ollama indisponivel; usando fallback DeepSeek "
                                "(desligue com RAG_LLM_CLOUD_FALLBACK=false)."
                            )
                    except Exception as e:
                        error_msg = str(e).lower()
                        if any(keyword in error_msg for keyword in ["insufficient_quota", "429", "quota", "exceeded"]):
                            st.warning(
                                f"{provider_names.get(selected_provider, selected_provider)} "
                                "atingiu o limite de uso. Acionando o DeepSeek."
                            )
                            
                            # Fallback para DeepSeek
                            try:
                                llm_manager = create_llm_manager("deepseek")
                                llm = llm_manager.get_llm(temperature=temperatura, max_tokens=max_tokens)
                                provider_usado = "deepseek"
                                st.caption(
                                    "Serviço em uso: DeepSeek "
                                    "(acionado automaticamente)."
                                )
                            except Exception as e2:
                                st.error(f"Falha ao acionar o DeepSeek: {str(e2)}")
                                raise e2
                        else:
                            st.error(
                                f"Falha ao configurar "
                                f"{provider_names.get(selected_provider, selected_provider)}: "
                                f"{str(e)}"
                            )
                            raise e
                    
                    if llm is None:
                        st.error(
                            "Nenhum serviço de inteligência artificial está "
                            "disponível no momento. Verifique a configuração "
                            "na barra lateral."
                        )
                        return
                    
                    # Mesma busca do consultar. O streaming fica so na tela.
                    pergunta_original = pergunta_para_processar
                    filtro_ano = None if ano_filtro == "Todos" else ano_filtro
                    pacote_busca = recuperar_trechos(
                        pergunta_para_processar,
                        filtros={
                            "tipo_documento": (
                                None if tipo_documento == "Todos" else tipo_documento
                            ),
                            "ano": filtro_ano,
                            "numero": (
                                numero_filtro if numero_filtro.strip() else None
                            ),
                        },
                        max_documentos=int(num_documentos),
                        embedding_provider=selected_embedding_provider,
                        historico=list(st.session_state.chat_history),
                        provider=provider_usado,
                        modelo=selected_model,
                        temperatura=float(temperatura),
                        max_tokens=int(max_tokens),
                    )
                    pergunta_para_processar = pacote_busca.pergunta_busca
                    documentos = pacote_busca.documentos
                    if pergunta_para_processar != pergunta_original:
                        st.info(
                            "Pergunta contextualizada: **{0}**".format(
                                pergunta_para_processar
                            )
                        )
                    
                    if documentos:
                        logger.info(f"DEBUG: Gerando resposta com {len(documentos)} documentos")
                        modelo_usado_final = provider_usado
                        provider_label = provider_names.get(
                            modelo_usado_final, modelo_usado_final
                        )

                        with st.chat_message("user"):
                            st.markdown(pergunta_original)

                        with st.chat_message("assistant"):
                            stream_gen = gerar_resposta_streaming(
                                pergunta_para_processar, documentos, llm, provider_usado
                            )
                            resposta = st.write_stream(stream_gen)

                        if not resposta or not resposta.strip():
                            resposta, modelo_usado_final = gerar_resposta(
                                pergunta_para_processar, documentos, llm, provider_usado
                            )
                            provider_label = provider_names.get(
                                modelo_usado_final, modelo_usado_final
                            )
                            with st.chat_message("assistant"):
                                st.markdown(resposta)

                        houve_falha = resposta_indica_falha(resposta)
                        logger.info(
                            f"DEBUG: Resposta {'nao ' if houve_falha else ''}"
                            f"gerada - Tamanho: "
                            f"{len(resposta) if resposta else 0} caracteres"
                        )

                        # Um aviso de falha nao pode entrar no historico de
                        # conversa: ele seria reaproveitado como contexto na
                        # reescrita da proxima pergunta.
                        if not houve_falha:
                            st.session_state.chat_history.append({
                                "pergunta": pergunta_original,
                                "resposta": resposta if resposta else "",
                            })
                            limite_historico = get_historico_turnos_guardados()
                            if len(st.session_state.chat_history) > limite_historico:
                                st.session_state.chat_history = (
                                    st.session_state.chat_history[-limite_historico:]
                                )

                        # Salvar no historico visual do chat
                        st.session_state.mensagens_chat.append({
                            "pergunta": pergunta_original,
                            "resposta": resposta if resposta else "(sem resposta)",
                            "provider": provider_label,
                        })

                        if houve_falha:
                            st.caption(
                                "A consulta não foi respondida. Os trechos "
                                "localizados na base continuam listados "
                                "abaixo."
                            )
                        else:
                            st.caption(f"Resposta gerada com {provider_label}")

                        # Extrair e exibir citações
                        citacoes = extrair_citacoes_da_resposta(resposta)
                        if citacoes:
                            st.markdown("""
                            <div class="citation-box">
                                <h4>Documentos citados</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for indice, citacao in enumerate(citacoes, start=1):
                                st.markdown(f"{indice}. {citacao}")
                        
                        # Exibir trechos dos documentos citados
                        st.markdown("---")
                        st.markdown("### Trechos dos documentos citados")
                        
                        # Extrair citações de documentos da resposta
                        documentos_citados = extrair_citacoes_da_resposta(resposta)
                        
                        # Exibir trechos relevantes para os documentos citados
                        documentos_encontrados = False
                        
                        if documentos_citados:
                            # Criar mapeamento de documentos para facilitar a busca
                            docs_por_id = {}
                            for doc in documentos:
                                meta = doc.metadata
                                tipo = meta.get('nome_tipo', 'Documento')
                                numero = meta.get('numero', 'N/A')
                                ano = meta.get('ano', 'N/A')
                                doc_id = f"{tipo} {numero}/{ano}"
                                
                                # Normalizar o ID para comparação (tudo minúsculo, sem espaços extras)
                                doc_id_norm = re.sub(r'\s+', ' ', doc_id.lower()).strip()
                                
                                # Criar versões alternativas do ID para melhorar correspondências
                                alternativas = [
                                    doc_id_norm,
                                    f"{tipo.lower()} {numero}",
                                    f"{tipo.lower()} {numero.lstrip('0')}/{ano}",
                                    f"{tipo.lower()} {numero.lstrip('0')} de {ano}"
                                ]
                                
                                for alt_id in alternativas:
                                    if alt_id not in docs_por_id:
                                        docs_por_id[alt_id] = []
                                    docs_por_id[alt_id].append((doc, meta, doc_id))
                            
                            # Exibir documentos citados
                            documentos_exibidos = set()  # Para evitar duplicações
                            
                            for doc_citado in documentos_citados:
                                # Normalizar a citação
                                doc_citado_norm = re.sub(r'\s+', ' ', doc_citado.lower()).strip()
                                doc_encontrado = False
                                
                                # Tentar diferentes variações da citação para encontrar correspondência
                                for doc_key, doc_list in docs_por_id.items():
                                    # Verificar se a citação corresponde a alguma parte da chave
                                    if (doc_citado_norm in doc_key or 
                                        doc_key in doc_citado_norm or
                                        any(term in doc_citado_norm for term in doc_key.split())):
                                        
                                        for doc, meta, doc_id in doc_list:
                                            doc_display_id = f"{doc_id}_{meta.get('chunk', '')}"
                                            
                                            # Verificar se já exibimos este documento
                                            if doc_display_id in documentos_exibidos:
                                                continue
                                            
                                            # Marcar como encontrado
                                            doc_encontrado = True
                                            documentos_encontrados = True
                                            documentos_exibidos.add(doc_display_id)
                                            
                                            with st.container(border=True):
                                                st.markdown(
                                                    badge_tipo_documento(
                                                        meta.get("nome_tipo", "Documento")
                                                    ),
                                                    unsafe_allow_html=True,
                                                )

                                                st.markdown(f"#### {doc_id}")
                                                st.caption(f"Parte {meta.get('chunk', 'N/A')}/{meta.get('total_chunks', 'N/A')} • Fonte: `{meta.get('caminho', 'Não especificado')}`")
                                                st.text_area(
                                                    "Trecho do documento:",
                                                    doc.page_content,
                                                    height=150,
                                                    key=f"citacao_{doc_display_id}",
                                                    disabled=True
                                                )
                                
                                # Se não encontrou o documento, registrar isso
                                if not doc_encontrado:
                                    logger.info(f"Documento citado não encontrado: {doc_citado}")
                        
                        # Se não encontramos citações explícitas ou os documentos citados
                        if not documentos_citados or not documentos_encontrados:
                            st.info(
                                "A resposta não citou documentos de forma explícita. "
                                "Veja abaixo os documentos mais relacionados à "
                                "sua pergunta."
                            )
                            
                            # Mostrar os 3 documentos mais relevantes
                            with st.container(border=True):
                                for i, doc in enumerate(documentos[:3]):
                                    meta = doc.metadata
                                    tipo = meta.get('nome_tipo', 'Documento')
                                    numero = meta.get('numero', 'N/A')
                                    ano = meta.get('ano', 'N/A')
                                    doc_id = f"{tipo} {numero}/{ano}"
                                    
                                    st.markdown(
                                        badge_tipo_documento(tipo),
                                        unsafe_allow_html=True,
                                    )

                                    st.markdown(f"**{doc_id}** - Parte {meta.get('chunk', 'N/A')}/{meta.get('total_chunks', 'N/A')}")
                                    st.caption(f"Fonte: `{meta.get('caminho', 'Não especificado')}`")
                                    st.text_area(
                                        "Conteúdo do documento:",
                                        doc.page_content,
                                        height=130,
                                        key=f"relevante_{i}",
                                        disabled=True
                                    )
                                    
                                    if i < 2:  # Não adicionar separador após o último
                                        st.divider()
                        
                        # Informações sobre a busca
                        with st.expander("Detalhes da busca"):
                            st.write(f"**Documentos encontrados:** {len(documentos)}")
                            
                            # Mostrar provedor que foi realmente usado
                            if modelo_usado_final != provider_usado:
                                st.write(f"**Serviço escolhido:** {provider_names.get(selected_provider, selected_provider)}")
                                st.write(
                                    f"**Serviço efetivamente usado:** "
                                    f"{provider_names.get(modelo_usado_final, modelo_usado_final)} "
                                    "(acionado automaticamente porque o escolhido "
                                    "estava indisponível)"
                                )
                            else:
                                st.write(f"**Serviço usado:** {provider_names.get(modelo_usado_final, modelo_usado_final)}")
                                
                            st.write(f"**Modelo:** {selected_model}")
                            st.write(f"**Liberdade de redação:** {temperatura}")
                            
                            # Mostrar informações sobre template adaptativo
                            if "gpt-4" in selected_model.lower() or modelo_usado_final == "openai":
                                template_info = "**Estilo da resposta:** estruturado e detalhado"
                            elif "deepseek" in selected_model.lower() or modelo_usado_final == "deepseek":
                                template_info = "**Estilo da resposta:** direto e conciso"
                            elif modelo_usado_final == "ollama" or "llama" in selected_model.lower():
                                template_info = "**Estilo da resposta:** local (Ollama/CPU)"
                            else:
                                template_info = "**Estilo da resposta:** balanceado"
                            
                            st.markdown(template_info)
                            
                            # Mostrar documentos encontrados
                            for i, doc in enumerate(documentos[:5]):
                                metadata = doc.metadata
                                st.write(f"**Doc {i+1}:** {metadata.get('nome_tipo', 'N/A')} {metadata.get('numero', 'N/A')}/{metadata.get('ano', 'N/A')}")
                        
                        # Seção para mostrar todas as fontes consultadas (FORA do expander anterior)
                        with st.expander("Todas as fontes consultadas"):
                            # Agrupar documentos por tipo/número/ano
                            documentos_agrupados = {}
                            for doc in documentos:
                                meta = doc.metadata
                                tipo = meta.get('nome_tipo', 'Documento')
                                numero = meta.get('numero', 'N/A')
                                ano = meta.get('ano', 'N/A')
                                
                                doc_id = f"{tipo} {numero}/{ano}"
                                if doc_id not in documentos_agrupados:
                                    documentos_agrupados[doc_id] = {
                                        'tipo': tipo,
                                        'numero': numero,
                                        'ano': ano,
                                        'caminho': meta.get('caminho', 'Não especificado'),
                                        'trechos': []
                                    }
                                
                                documentos_agrupados[doc_id]['trechos'].append({
                                    'chunk': meta.get('chunk', 'N/A'),
                                    'total_chunks': meta.get('total_chunks', 'N/A'),
                                    'conteudo': doc.page_content
                                })
                            
                            # Exibir documentos agrupados
                            for i, (doc_id, info) in enumerate(documentos_agrupados.items()):
                                # Destacar documentos mais relevantes por rotulo textual,
                                # nao por icone ou apenas cor (acessibilidade).
                                destaque = " (mais relevante)" if i < 3 else ""
                                
                                # Usar container em vez de expander aninhado
                                st.markdown(f"### {doc_id}{destaque}")
                                
                                with st.container(border=True):
                                    st.markdown(
                                        badge_tipo_documento(info['tipo']),
                                        unsafe_allow_html=True,
                                    )
                                    # Mostrar metadados do documento
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.markdown(f"**Tipo:** {info['tipo']}")
                                    with col2:
                                        st.markdown(f"**Número:** {info['numero']}")
                                    with col3:
                                        st.markdown(f"**Ano:** {info['ano']}")
                                    
                                    st.markdown(f"**Caminho:** `{info['caminho']}`")
                                    
                                    # Exibir trechos em tabs se houver múltiplos
                                    if len(info['trechos']) > 1:
                                        trechos_tabs = st.tabs([f"Trecho {t['chunk']}/{t['total_chunks']}" for t in info['trechos']])
                                        for j, tab in enumerate(trechos_tabs):
                                            with tab:
                                                trecho = info['trechos'][j]
                                                st.text_area(
                                                    f"Conteúdo do trecho {trecho['chunk']}:",
                                                    trecho['conteudo'],
                                                    height=200,
                                                    key=f"fonte_{doc_id}_{j}",
                                                    disabled=True
                                                )
                                    else:
                                        # Se há apenas um trecho, exibir diretamente
                                        trecho = info['trechos'][0]
                                        st.text_area(
                                            f"Conteúdo:",
                                            trecho['conteudo'],
                                            height=200,
                                            key=f"fonte_unico_{doc_id}",
                                            disabled=True
                                        )
                    
                    else:
                        st.warning(
                            "Nenhum documento relacionado à pergunta foi "
                            "encontrado. Tente reformulá-la com outros termos "
                            "ou remova os filtros da barra lateral."
                        )
                        
                except Exception as e:
                    st.error(
                        "Não foi possível concluir a consulta. "
                        "Tente novamente em alguns instantes."
                    )
                    st.caption(f"Detalhe técnico: {str(e)}")
                    logger.error(f"Erro na consulta: {str(e)}")
        
    # Inclusao de documento: acao de operacao, nao de consulta. Fica na barra
    # lateral para nao se interpor entre a ultima resposta e o campo de
    # pergunta, que o Streamlit fixa no rodape da area principal.
    with st.sidebar:
        with st.expander("Processar novo documento"):
            uploaded_file = st.file_uploader(
                "Envie um PDF para adicionar à base de conhecimento:",
                type=["pdf"],
                help="O arquivo passará a ser considerado nas próximas consultas."
            )

            if uploaded_file and st.button(
                "Processar PDF", use_container_width=True
            ):
                try:
                    caminho_pdf = incluir_documento(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                    )
                    st.caption(
                        "PDF incluido na base comum ({0}). "
                        "A consulta passa a ve-lo depois de Atualizar base.".format(
                            caminho_pdf
                        )
                    )
                except Exception as e:
                    st.error(f"Erro ao processar PDF: {str(e)}")

