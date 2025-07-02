from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def process_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return self.text_splitter.split_documents(docs)

    def process_md(self, file_path):
        loader = UnstructuredMarkdownLoader(file_path)
        docs = loader.load()
        return self.text_splitter.split_documents(docs)

    def process_folder(self, folder_path):
        import os
        docs = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                ext = file.lower().split('.')[-1]
                file_path = os.path.join(root, file)
                if ext == 'pdf':
                    docs.extend(self.process_pdf(file_path))
                elif ext == 'md':
                    docs.extend(self.process_md(file_path))
        return docs 