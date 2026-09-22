from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class VectorStore:
    def __init__(self, embeddings=None):
        self.embeddings = embeddings or OpenAIEmbeddings()
        self.vectorstore = None

    def create_from_documents(self, documents):
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self.vectorstore.save_local("vectorstore_local")

    def add_documents(self, documents):
        if self.vectorstore:
            self.vectorstore.add_documents(documents)
            self.vectorstore.save_local("vectorstore_local") 