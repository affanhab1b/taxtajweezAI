from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from rag_chain import RAGChain
from embedding_indexer import EmbeddingIndexer
# Assuming this class is defined to process PDFs
from document_processor import MultiPDFProcessor
import os


class Chatbot:
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain

    def get_response(self, user_input):
        try:
            response = self.qa_chain({"query": user_input})
            return response['result']
        except Exception as e:
            return f"An error occurred: {str(e)}"


if __name__ == "__main__":
    # Path to the saved vectorstore directory
    vectorstore_path = "vectorstore"

    # Define the embeddings to use when loading the vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Check if the vectorstore already exists
    if os.path.exists(vectorstore_path):
        # Load the existing vectorstore with embeddings
        print("Loading existing vectorstore...")
        vectorstore = FAISS.load_local(vectorstore_path, embeddings)
    else:
        # If no vectorstore exists, process documents and create the vectorstore
        print("No existing vectorstore found, creating a new one...")

        # Initialize the PDF processor and load all documents
        # Path to your folder containing PDFs (income tax manual, tax queries, etc.)
        processor = MultiPDFProcessor("data")
        raw_docs = processor.load_all_pdfs()  # Load all documents (both PDFs)
        # Split documents into smaller chunks for indexing
        chunks = processor.split_documents(raw_docs)

        # Initialize the indexer and create the vectorstore
        indexer = EmbeddingIndexer()
        # Create the FAISS vectorstore with embedded documents
        vectorstore = indexer.create_vectorstore(chunks)

        # Save the vectorstore for future use
        vectorstore.save_local(vectorstore_path)

    # Initialize the RAGChain with the created or loaded vectorstore
    rag_chain = RAGChain(vectorstore)
    qa_chain = rag_chain.create_chain()  # Create the QA chain for querying

    # Initialize the chatbot with the QA chain
    chatbot = Chatbot(qa_chain)

    # Run the chatbot interface
    print("Chatbot: Hello! How can I assist you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye!")
            break
        response = chatbot.get_response(user_input)
        print(f"Chatbot: {response}")
