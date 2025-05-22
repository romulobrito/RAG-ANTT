import os
import json
import glob

# Configurações
DIR_DADOS = "dados_antt"
TIPOS_DOCUMENTO = ["RES", "POR", "INM", "DLB", "INC"]

def gerar_relatorio_documentos():
    """Gera um relatório dos documentos disponíveis na pasta dados_antt"""
    documentos = []
    
    print(f"Gerando relatório de documentos em {DIR_DADOS}...")
    
    # Verifica se o diretório existe
    if not os.path.exists(DIR_DADOS):
        print(f"Erro: Diretório {DIR_DADOS} não encontrado.")
        return []
    
    # Lê o arquivo documentos.json principal se existir
    documentos_json_path = os.path.join(DIR_DADOS, "documentos.json")
    if os.path.exists(documentos_json_path):
        try:
            with open(documentos_json_path, 'r', encoding='utf-8') as f:
                documentos = json.load(f)
                print(f"Carregados {len(documentos)} documentos do arquivo principal.")
            
            # Salva diretamente o relatório
            with open("relatorio_documentos.json", "w", encoding='utf-8') as f:
                json.dump(documentos, f, ensure_ascii=False, indent=4)
            print(f"Relatório salvo em relatorio_documentos.json")
            return documentos
        except Exception as e:
            print(f"Erro ao carregar o arquivo documentos.json: {str(e)}")
            documentos = []
    
    # Se não conseguiu carregar do arquivo principal, faz varredura dos diretórios
    for tipo in TIPOS_DOCUMENTO:
        tipo_dir = os.path.join(DIR_DADOS, tipo)
        if not os.path.exists(tipo_dir):
            continue
        
        for ano_dir in os.listdir(tipo_dir):
            ano_path = os.path.join(tipo_dir, ano_dir)
            if not os.path.isdir(ano_path):
                continue
            
            # Procura arquivos JSON que contêm metadados
            for json_file in glob.glob(os.path.join(ano_path, f"{tipo}-*.json")):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        doc_data = json.load(f)
                    
                    # Se for um arquivo de metadados, adiciona informações básicas
                    doc = {
                        "titulo": doc_data.get("titulo", "Sem título"),
                        "ementa": doc_data.get("ementa", "Sem ementa"),
                        "tipo": tipo,
                        "numero": doc_data.get("numero", ""),
                        "ano": ano_dir,
                        "orgao": doc_data.get("orgao", "ANTT"),
                        "arquivo_html": os.path.join(tipo, ano_dir, os.path.basename(json_file).replace(".json", ".html")),
                        "arquivo_md": os.path.join(tipo, ano_dir, os.path.basename(json_file).replace(".json", ".md"))
                    }
                    documentos.append(doc)
                except Exception as e:
                    print(f"Erro ao processar {json_file}: {str(e)}")
    
    print(f"Encontrados {len(documentos)} documentos nos diretórios.")
    
    # Salva o relatório
    with open("relatorio_documentos.json", "w", encoding='utf-8') as f:
        json.dump(documentos, f, ensure_ascii=False, indent=4)
    
    print(f"Relatório salvo em relatorio_documentos.json")
    return documentos

if __name__ == "__main__":
    documentos = gerar_relatorio_documentos()
    print(f"Processamento concluído. {len(documentos)} documentos no relatório.") 