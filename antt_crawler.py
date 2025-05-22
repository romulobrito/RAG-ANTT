import os
import re
import json
import glob
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from bs4 import BeautifulSoup
import markdown
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

# Importar configurações do config.py
from config import (
    get_openai_api_key,
    DB_FAISS_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    logger
)

@dataclass
class DocumentoANTT:
    """Classe para representar um documento da ANTT com seus metadados."""
    conteudo: str
    tipo_documento: str  # RES, POR, INC, etc.
    ano: str
    numero: Optional[str] = None
    caminho: Optional[str] = None
    formato: Optional[str] = None
    data_publicacao: Optional[str] = None

class ANTTCrawler:
    """Crawler para processar documentos da ANTT."""
    
    def __init__(self, diretorio_base: str, diretorio_saida: str = DB_FAISS_PATH):
        """
        Inicializa o crawler.
        
        Args:
            diretorio_base: Caminho para o diretório raiz dos documentos da ANTT
            diretorio_saida: Caminho onde o vectorstore será salvo
        """
        self.diretorio_base = diretorio_base
        self.diretorio_saida = diretorio_saida
        self.documentos_processados = []
        
        # Configuração do splitter para chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        
        # Mapeamento de tipos de documentos
        self.tipos_documentos = {
            "RES": "Resolução",
            "POR": "Portaria",
            "INC": "Instrução Normativa Complementar",
            "DLB": "Deliberação",
            "LEI": "Lei",
            "VTO": "Voto",
            "DEC": "Decreto",
            "INM": "Instrução Normativa"
        }
        
        logger.info(f"Inicializando crawler para diretório {diretorio_base}")
    
    def processar_diretorio(self) -> List[Document]:
        """Processa recursivamente o diretório de documentos ANTT."""
        documentos_langchain = []
        
        for root, dirs, files in os.walk(self.diretorio_base):
            for file in files:
                if file.endswith(('.html', '.json', '.md')):
                    caminho_completo = os.path.join(root, file)
                    
                    # Extrair metadados do caminho
                    metadados = self._extrair_metadados_do_caminho(caminho_completo)
                    formato = file.split('.')[-1]
                    
                    # Processar o arquivo de acordo com seu formato
                    try:
                        if formato == 'html':
                            documento = self._processar_html(caminho_completo, metadados)
                        elif formato == 'json':
                            documento = self._processar_json(caminho_completo, metadados)
                        elif formato == 'md':
                            documento = self._processar_markdown(caminho_completo, metadados)
                        
                        if documento:
                            # Realizar chunking e criar documentos no formato LangChain
                            chunks = self._criar_chunks(documento)
                            documentos_langchain.extend(chunks)
                            
                            # Adicionar à lista de documentos processados para controle
                            self.documentos_processados.append(documento)
                    except Exception as e:
                        print(f"Erro ao processar {caminho_completo}: {str(e)}")
        
        print(f"Total de documentos processados: {len(self.documentos_processados)}")
        print(f"Total de chunks gerados: {len(documentos_langchain)}")
        
        return documentos_langchain
    
    def _extrair_metadados_do_caminho(self, caminho: str) -> Dict[str, Any]:
        """Extrai metadados do caminho do arquivo."""
        # Remover o diretório base para trabalhar apenas com o caminho relativo
        caminho_relativo = os.path.relpath(caminho, self.diretorio_base)
        partes_caminho = caminho_relativo.split(os.sep)
        
        # Inicializar metadados com valores padrão
        metadados = {
            "tipo_documento": None,
            "ano": None,
            "numero": None
        }
        
        # Extrair tipo de documento
        for parte in partes_caminho:
            for tipo_sigla in self.tipos_documentos.keys():
                if parte.startswith(tipo_sigla):
                    metadados["tipo_documento"] = tipo_sigla
                    break
        
        # Extrair ano da estrutura de diretórios
        for parte in partes_caminho:
            if re.match(r'^(19|20)\d{2}$', parte):  # Padrão para anos (1900-2099)
                metadados["ano"] = parte
                break
        
        # Extrair número do arquivo
        nome_arquivo = os.path.basename(caminho)
        if '-' in nome_arquivo:
            partes_nome = nome_arquivo.split('-')
            # Assumindo formato como XXX-00000000-YYYY.ext
            if len(partes_nome) >= 2:
                numero_match = re.search(r'(\d+)', partes_nome[1])
                if numero_match:
                    metadados["numero"] = numero_match.group(1)
        
        return metadados
    
    def _processar_html(self, caminho: str, metadados: Dict[str, Any]) -> Optional[DocumentoANTT]:
        """Processa um arquivo HTML e extrai seu conteúdo."""
        with open(caminho, 'r', encoding='utf-8') as file:
            conteudo_html = file.read()
        
        soup = BeautifulSoup(conteudo_html, 'html.parser')
        
        # Remover scripts e estilos
        for script in soup(['script', 'style']):
            script.extract()
        
        # Extrair texto
        texto = soup.get_text(separator='\n', strip=True)
        
        # Tentar identificar data de publicação
        data_publicacao = None
        data_elem = soup.find(text=re.compile(r'publicad[oa]\s+em', re.IGNORECASE))
        if data_elem:
            data_match = re.search(r'\d{2}/\d{2}/\d{4}', data_elem)
            if data_match:
                data_publicacao = data_match.group(0)
        
        return DocumentoANTT(
            conteudo=texto,
            tipo_documento=metadados.get("tipo_documento", ""),
            ano=metadados.get("ano", ""),
            numero=metadados.get("numero"),
            caminho=caminho,
            formato="html",
            data_publicacao=data_publicacao
        )
    
    def _processar_json(self, caminho: str, metadados: Dict[str, Any]) -> Optional[DocumentoANTT]:
        """Processa um arquivo JSON e extrai seu conteúdo."""
        with open(caminho, 'r', encoding='utf-8') as file:
            try:
                dados = json.load(file)
            except json.JSONDecodeError:
                print(f"Erro ao decodificar JSON de {caminho}")
                return None
        
        # Extrair conteúdo do JSON (adaptar conforme a estrutura do JSON)
        if isinstance(dados, dict):
            # Verificar se o JSON contém um campo de texto/conteúdo
            conteudo = dados.get('conteudo', '') or dados.get('texto', '')
            
            # Se não houver campo específico, converter todo o JSON para texto
            if not conteudo:
                conteudo = json.dumps(dados, ensure_ascii=False, indent=2)
            
            # Extrair data de publicação se existir
            data_publicacao = dados.get('dataPublicacao') or dados.get('data')
            
            return DocumentoANTT(
                conteudo=conteudo,
                tipo_documento=metadados.get("tipo_documento", ""),
                ano=metadados.get("ano", ""),
                numero=metadados.get("numero"),
                caminho=caminho,
                formato="json",
                data_publicacao=data_publicacao
            )
        return None
    
    def _processar_markdown(self, caminho: str, metadados: Dict[str, Any]) -> Optional[DocumentoANTT]:
        """Processa um arquivo Markdown e extrai seu conteúdo."""
        with open(caminho, 'r', encoding='utf-8') as file:
            conteudo_md = file.read()
        
        # Converter markdown para texto
        html = markdown.markdown(conteudo_md)
        soup = BeautifulSoup(html, 'html.parser')
        texto = soup.get_text(separator='\n', strip=True)
        
        # Tentar extrair data de publicação do próprio markdown
        data_match = re.search(r'publicad[oa]\s+em\s+(\d{2}/\d{2}/\d{4})', conteudo_md, re.IGNORECASE)
        data_publicacao = data_match.group(1) if data_match else None
        
        return DocumentoANTT(
            conteudo=texto,
            tipo_documento=metadados.get("tipo_documento", ""),
            ano=metadados.get("ano", ""),
            numero=metadados.get("numero"),
            caminho=caminho,
            formato="md",
            data_publicacao=data_publicacao
        )
    
    def _criar_chunks(self, documento: DocumentoANTT) -> List[Document]:
        """Divide o documento em chunks e cria documentos no formato LangChain."""
        # Criar metadados para os chunks
        metadados = {
            "tipo_documento": documento.tipo_documento,
            "nome_tipo": self.tipos_documentos.get(documento.tipo_documento, "Documento"),
            "ano": documento.ano,
            "numero": documento.numero,
            "caminho": documento.caminho,
            "formato": documento.formato,
            "data_publicacao": documento.data_publicacao
        }
        
        # Dividir o texto em chunks
        textos_divididos = self.text_splitter.split_text(documento.conteudo)
        
        # Criar documentos LangChain
        documentos = []
        for i, texto in enumerate(textos_divididos):
            # Adicionar informação de chunk aos metadados
            meta_chunk = metadados.copy()
            meta_chunk["chunk"] = i + 1
            meta_chunk["total_chunks"] = len(textos_divididos)
            
            # Criar documento LangChain
            doc = Document(page_content=texto, metadata=meta_chunk)
            documentos.append(doc)
        
        return documentos
    
    def criar_vectorstore(self, documentos: List[Document]) -> FAISS:
        """Cria e salva um vectorstore a partir dos documentos processados."""
        logger.info(f"Criando vectorstore com {len(documentos)} documentos...")
        
        # Inicializar embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=get_openai_api_key())
        
        # Definir tamanho do lote - ajuste conforme necessário
        tamanho_lote = 50  # Número de documentos por lote
        
        # Inicializar o vectorstore com o primeiro lote
        if len(documentos) > 0:
            lotes = [documentos[i:i + tamanho_lote] for i in range(0, len(documentos), tamanho_lote)]
            logger.info(f"Processando {len(lotes)} lotes de documentos...")
            
            # Criar vectorstore com o primeiro lote
            vectorstore = FAISS.from_documents(lotes[0], embeddings)
            
            # Adicionar lotes restantes
            for i, lote in enumerate(lotes[1:], 1):
                logger.info(f"Processando lote {i+1}/{len(lotes)}...")
                if lote:  # Verificar se o lote não está vazio
                    vectorstore.add_documents(lote)
        else:
            # Criar vectorstore vazio se não houver documentos
            vectorstore = FAISS.from_documents([], embeddings)
        
        # Salvar vectorstore
        os.makedirs(self.diretorio_saida, exist_ok=True)
        vectorstore.save_local(self.diretorio_saida)
        logger.info(f"Vectorstore salvo em {self.diretorio_saida}")
        
        return vectorstore
    
    def gerar_relatorio(self, caminho_saida: str = "relatorio_documentos.json"):
        """Gera um relatório dos documentos processados."""
        relatorio = []
        
        for doc in self.documentos_processados:
            relatorio.append({
                "tipo": doc.tipo_documento,
                "nome_tipo": self.tipos_documentos.get(doc.tipo_documento, "Documento"),
                "ano": doc.ano,
                "numero": doc.numero,
                "caminho": doc.caminho,
                "formato": doc.formato,
                "data_publicacao": doc.data_publicacao,
                "tamanho_conteudo": len(doc.conteudo)
            })
        
        with open(caminho_saida, 'w', encoding='utf-8') as f:
            json.dump(relatorio, f, ensure_ascii=False, indent=2)
        
        print(f"Relatório salvo em {caminho_saida}")

    def processar_documentos(self):
        """Processa todos os documentos nos diretórios configurados"""
        documentos_processados = []
        contador = {"total": 0, "sucesso": 0, "falha": 0}
        
        try:
            for tipo_pasta in self.tipos_documentos.keys():
                pasta_tipo = os.path.join(self.diretorio_base, tipo_pasta)
                
                if not os.path.exists(pasta_tipo):
                    os.makedirs(pasta_tipo, exist_ok=True)
                    print(f"Pasta {pasta_tipo} criada.")
                    continue
                
                # Processa os anos dentro do tipo
                for ano_pasta in os.listdir(pasta_tipo):
                    caminho_ano = os.path.join(pasta_tipo, ano_pasta)
                    
                    if not os.path.isdir(caminho_ano):
                        continue
                    
                    # Encontra todos os arquivos .json neste diretório
                    arquivos_json = glob.glob(os.path.join(caminho_ano, f"{tipo_pasta}-*.json"))
                    
                    for arquivo_json in arquivos_json:
                        contador["total"] += 1
                        try:
                            with open(arquivo_json, 'r', encoding='utf-8') as f:
                                metadados = json.load(f)
                            
                            # Adiciona informações extras
                            metadados_doc = {
                                "tipo": tipo_pasta,
                                "ano": ano_pasta,
                                "caminho_json": arquivo_json,
                                "caminho_md": arquivo_json.replace('.json', '.md'),
                                "caminho_html": arquivo_json.replace('.json', '.html')
                            }
                            
                            documentos_processados.append(metadados_doc)
                            contador["sucesso"] += 1
                            
                        except Exception as e:
                            print(f"Erro ao processar {arquivo_json}: {str(e)}")
                            contador["falha"] += 1
        
        except Exception as e:
            print(f"Erro ao processar documentos: {str(e)}")
        
        print(f"Processamento concluído: {contador['total']} documentos encontrados, "
              f"{contador['sucesso']} processados com sucesso, {contador['falha']} falhas.")
        
        # Salva o relatório de documentos processados
        with open("relatorio_documentos.json", "w", encoding='utf-8') as f:
            json.dump(documentos_processados, f, ensure_ascii=False, indent=4)
        
        print(f"Relatório salvo em relatorio_documentos.json")
        return documentos_processados


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Crawler para documentos ANTT')
    parser.add_argument('--diretorio', type=str, default='dados_antt', 
                        help='Diretório contendo os documentos da ANTT')
    parser.add_argument('--saida', type=str, default=DB_FAISS_PATH,
                        help='Diretório para salvar o vectorstore')
    parser.add_argument('--openai-key', type=str, 
                        help='Chave da API OpenAI (opcional, será usada a do config.py se não fornecida)')
    
    args = parser.parse_args()
    
    # Usar a chave fornecida ou a do config.py
    api_key = args.openai_key or get_openai_api_key()
    
    # Inicializar e executar o crawler
    crawler = ANTTCrawler(args.diretorio, args.saida)
    documentos = crawler.processar_diretorio()
    
    # Criar vectorstore se houver documentos
    if documentos:
        crawler.criar_vectorstore(documentos)
        crawler.gerar_relatorio()
    else:
        logger.warning("Nenhum documento encontrado para processamento.")
