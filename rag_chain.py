from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
# Importing the correct class for chat-based models
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import sys
from embedding_indexer import EmbeddingIndexer
# Assuming your class is named MultiPDFProcessor
from document_processor import MultiPDFProcessor
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

load_dotenv("apikeys.env")


# Ensure the console can print non-ASCII characters (handle encoding)
sys.stdout.reconfigure(encoding='utf-8')
# print(os.getenv("OPENAI_API_KEY"))


class RAGChain:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = self.get_llm()

    def get_llm(self):
        api_key = os.getenv("OPENAI_API_KEY")

        # Retrieve the API key from environment variables
        if os.getenv("OPENAI_API_KEY"):
            # Specify the model here and use the correct class for chat models
            return ChatOpenAI(
                # api_key=os.getenv("OPENAI_API_KEY"),
                model_kwargs={"api_key": api_key},
                temperature=0,
                model="gpt-3.5-turbo",  # Use gpt-3.5-turbo or gpt-4 for chat
            )
        else:
            raise ValueError(
                "No valid API key found! Please set one in .env file.")

    def create_chain(self):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # or "map_reduce"
            retriever=retriever,
            return_source_documents=True
        )
        return qa_chain


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

    # Example query related to taxation
    query = "What are the tax deductions available for teachers?"
    result = qa_chain({"query": query})  # Get the result based on the query
    print(f"Answer: {result['result']}")  # Print the generated response

    # # Handle and print the source documents (using either of the methods below)
    # source_docs = result['source_documents']

    # # Solution 1: Print source documents (with utf-8 encoding)
    # print(f"Source Documents: {source_docs}")
    # print(f"Source Documents: {result['source_documents']}")  # Optionally, print the source docs used
