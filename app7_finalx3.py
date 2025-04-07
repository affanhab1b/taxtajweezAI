import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from rag_chain import RAGChain
from embedding_indexer import EmbeddingIndexer
from document_processor import MultiPDFProcessor
import os

# Set page configuration
st.set_page_config(
    layout="wide",
    page_title="TaxTajweez AI - Pakistan's First Taxation Chatbot",
    initial_sidebar_state="collapsed"
)

# Initialize vectorstore
vectorstore_path = "vectorstore"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists(vectorstore_path):
    vectorstore = FAISS.load_local(vectorstore_path, embeddings)
else:
    with st.spinner("Setting up the knowledge base for the first time..."):
        processor = MultiPDFProcessor("data")
        raw_docs = processor.load_all_pdfs()
        chunks = processor.split_documents(raw_docs)
        indexer = EmbeddingIndexer()
        vectorstore = indexer.create_vectorstore(chunks)
        vectorstore.save_local(vectorstore_path)

# Initialize RAG chain
rag_chain = RAGChain(vectorstore)
qa_chain = rag_chain.create_chain()

# Chatbot class


class Chatbot:
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain

    def get_response(self, user_input):
        try:
            response = self.qa_chain({"query": user_input})
            return response['result']
        except Exception as e:
            return f"An error occurred: {str(e)}"


chatbot = Chatbot(qa_chain)

# Initialize conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Add message to history function


def add_message(role, content):
    st.session_state.conversation_history.append(
        {"role": role, "message": content})


# Custom CSS for styling with improved header positioning
st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header styling in top left */
    .header-container {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background-color: white;
        padding: 20px 30px;
        z-index: 99;
        border-bottom: 1px solid #e6e6e6;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }

    .header-title {
        font-size: 36px;
        font-weight: bold;
        margin: 0;
        color: #333;
        text-align: left;
    }

    .header-subtitle {
        font-size: 20px;
        color: #666;
        margin-top: 5px;
        text-align: left;
    }

    .header-subsubtitle {
        font-size: 16px;
        color: #666;
        margin-top: 5px;
        text-align: left;
    }

    /* Chat message styles */
    .user-message {
        background-color: #e6f7ff;
        border-radius: 18px;
        padding: 10px 16px;
        margin: 8px 0;
        margin-left: 20%;
        margin-right: 2%;
        display: inline-block;
        float: right;
        clear: both;
    }

    .bot-message {
        background-color: #f0f0f0;
        border-radius: 18px;
        padding: 10px 16px;
        margin: 8px 0;
        margin-right: 20%;
        margin-left: 2%;
        display: inline-block;
        float: left;
        clear: both;
    }

    /* Container for messages */
    .chat-container {
        height: calc(70vh - 250px);
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        padding: 50px;
    }

    /* Input area at the bottom - centered */
    .input-wrapper-centered {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        width: 70%;
        display: flex;
        justify-content: center;
        z-index: 100;
        padding: 10px;
    }

    .text-input {
        width: 100%;
        padding-right: 80px;
        padding: 10px 15px;
        border-radius: 20px;
        border: 1px solid #ccc;
        font-size: 16px;
        box-sizing: border-box;
    }

    .send-button-wrapper {
        position: absolute;
        top: 50%;
        right: 10px;
        transform: translateY(-50%);
        z-index: 2;
    }

    .send-button-wrapper button {
        background-color: #2e7d32;
        color: white;
        border-radius: 20px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        cursor: pointer;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }

    .send-button-wrapper button:hover {
        background-color: #1b5e20;
    }

</style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
<div class="header-container" style="display: flex; align-items: center;">
    <img src="https://img.icons8.com/?size=100&id=wrIcJrp64W6g&format=png&color=000000" alt="Tax Icon" style="width:60px; margin-right: 15px;">
    <div>
        <h1 class="header-title">TaxTajweez AI - Pakistan's First Taxation Chatbot</h1>
        <p class="header-subtitle">
            Built as part of a Final Year Project at Habib University, TaxTajweez AI is backed by academic research.
            You can <a href="https://journals.flvc.org/FLAIRS/article/view/135648" target="_blank">read the published paper here</a> — and then try it out for yourself below.
            <br><br>
            <strong>Disclaimer:</strong> TaxTajweez AI is not intended to replace professional tax advice. Please consult a tax advisor for personalized guidance. 
        </p>
        <p class = "header-subsubtitle">
        This project is developed independently for academic purposes and has no affiliation with the Federal Board of Revenue (FBR) or any other institution.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Main chat area - display conversation history
chat_placeholder = st.container()
with chat_placeholder:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Display conversation messages
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-message">{msg["message"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="bot-message">{msg["message"]}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Input area using centered position
with st.container():
    # Using a form for input to handle submission properly
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask me about taxes...", key="text_input", label_visibility="collapsed", placeholder="Ask me about taxes..."
        )
        submit_button = st.form_submit_button("Ask")


# Process the input when form is submitted
if submit_button and user_input:

    # --- RATE LIMITING LOGIC START ---
    import time

    MAX_QUESTIONS = 5
    COOLDOWN_SECONDS = 5 * 60 * 60  # 5 hours

    if 'question_count' not in st.session_state:
        st.session_state.question_count = 0
        st.session_state.cooldown_start = None

    if st.session_state.question_count >= MAX_QUESTIONS:
        if st.session_state.cooldown_start is None:
            st.session_state.cooldown_start = time.time()

        elapsed = time.time() - st.session_state.cooldown_start
        remaining = COOLDOWN_SECONDS - elapsed

        if remaining > 0:
            st.warning(f"You've reached the 5-question limit. Please try again in {int(remaining // 60)} minutes.")
            st.stop()
        else:
            # Reset limit after cooldown
            st.session_state.question_count = 0
            st.session_state.cooldown_start = None

    # Count this question
    st.session_state.question_count += 1
    # --- RATE LIMITING LOGIC END ---
        
    # Add user message to history
    add_message("user", user_input)

    # Get bot response
    with st.spinner('TaxTajweez AI is thinking...'):
        bot_response = chatbot.get_response(user_input)

    # Add bot response to history
    add_message("chatbot", (f"TaxTajweez AI: {bot_response}"))

    # Use rerun to update the conversation display
    st.rerun()

# LinkedIn Signature / Credit (bottom-left corner)
st.markdown("""
<div style="position: fixed; bottom: 10px; left: 20px; font-size: 13px; color: #888;">
    Built by <a href="https://www.linkedin.com/in/affan-habib/" target="_blank" style="color: #2e7d32; text-decoration: none;">Affan Habib (Connect on LinkedIn)</a>
</div>
""", unsafe_allow_html=True)