from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os

class MultiPDFProcessor:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def load_all_pdfs(self):
        all_docs = []
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".pdf"):
                full_path = os.path.join(self.folder_path, filename)
                print(f"Loading: {filename}")
                loader = PyMuPDFLoader(full_path)
                docs = loader.load()
                all_docs.extend(docs)
        return all_docs

    def split_documents(self, documents):
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(documents)

if __name__ == "__main__":
    processor = MultiPDFProcessor("data")
    raw_docs = processor.load_all_pdfs()
    print(f"Loaded {len(raw_docs)} raw documents from PDFs")

    text_chunks = processor.split_documents(raw_docs)
    print(f"Split into {len(text_chunks)} text chunks")
