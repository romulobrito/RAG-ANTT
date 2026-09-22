class QueryProcessor:
    def __init__(self, llm=None, template=None):
        self.llm = llm
        self.template = template
 
    def process_query(self, query, documents):
        # Exemplo simples: retorna o texto dos documentos concatenados
        context = "\n\n".join([doc.page_content for doc in documents])
        return f"Pergunta: {query}\n\nContexto:\n{context}" 