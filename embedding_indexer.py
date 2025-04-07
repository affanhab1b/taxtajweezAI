from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from document_processor import MultiPDFProcessor  # assumes you've created this class

class EmbeddingIndexer:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def create_vectorstore(self, texts):
        vectorstore = FAISS.from_documents(texts, self.embeddings)
        vectorstore.save_local("vectorstore")  # Save the index locally
        return vectorstore

if __name__ == "__main__":
    processor = MultiPDFProcessor("data")  # folder with multiple PDFs
    docs = processor.load_all_pdfs()
    chunks = processor.split_documents(docs)

    indexer = EmbeddingIndexer()
    vectorstore = indexer.create_vectorstore(chunks)

    print("Vector store created and saved locally")
