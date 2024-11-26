import streamlit as st
import os
import torch
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
import logging
from huggingface_hub import login

# Load environment variables
load_dotenv()

# Hugging Face login token
HF_TOKEN = os.getenv("HF_TOKEN", "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")
login(token=HF_TOKEN)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Multilingual PDF Text Extraction
def get_pdf_text(pdf_docs):
    """Extract text from multiple PDF documents."""
    try:
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting PDF text: {e}")
        return ""

# Text Chunking
def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Split text into manageable chunks."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            length_function=len
        )
        return text_splitter.split_text(text)
    except Exception as e:
        st.error(f"Error splitting text: {e}")
        return []

# Vector Store Creation
def get_vector_store(text_chunks):
    """Create FAISS vector store from text chunks."""
    try:
        # Use multilingual embedding model
        embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        
        # Create vector store directly from texts
        vector_store = FAISS.from_texts(
            texts=text_chunks, 
            embedding=embedding_model
        )
        
        # Save vector store locally
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# Load Existing Vector Store
def load_vector_store():
    """Load existing FAISS vector store."""
    try:
        embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        return FAISS.load_local("faiss_index", embedding_model)
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

# Conversational Retrieval Chain
def get_conversational_chain():
    """Create conversational retrieval chain."""
    try:
        # Model configuration
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        
        # Device selection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Tokenizer and Model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map=device, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        # Text Generation Pipeline
        text_generator = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            max_new_tokens=512
        )
        
        # LangChain LLM
        llm = HuggingFacePipeline(pipeline=text_generator)

        # Prompt Template
        prompt_template = """
        Context: {context}
        Question: {question}
        
        Provide a detailed, multilingual response based on the given context:
        """
        prompt = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        # Vector Store Retriever
        vector_store = load_vector_store()
        if not vector_store:
            st.error("No vector store found. Please process PDFs first.")
            return None
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        # Conversational Chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=retriever, 
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        
        return chain
    
    except Exception as e:
        st.error(f"Error creating conversational chain: {e}")
        return None

# PDF Chat Functionality
def chat_with_pdf(user_question):
    """Handle user queries against PDF documents."""
    try:
        # Load Vector Store
        vector_store = load_vector_store()
        if not vector_store:
            st.error("No vector store available.")
            return "Please upload and process PDFs first."
        
        # Retrieve Relevant Documents
        relevant_docs = vector_store.similarity_search(user_question, k=5)
        
        # Create Conversational Chain
        chain = get_conversational_chain()
        if not chain:
            return "Unable to create conversational chain."
        
        # Retrieve Chat History
        chat_history = st.session_state.get('chat_history', [])
        
        # Generate Response
        response = chain({
            "question": user_question,
            "chat_history": chat_history
        })
        
        # Update Chat History
        answer = response.get('answer', 'No answer generated.')
        chat_history.append((user_question, answer))
        st.session_state['chat_history'] = chat_history
        
        return answer
    
    except Exception as e:
        st.error(f"Error in chat processing: {e}")
        return f"An error occurred: {str(e)}"

# Streamlit Main Application
def main():
    """Streamlit application main function."""
    st.set_page_config(
        page_title="Multilingual PDF Chat", 
        page_icon="📄", 
        layout="wide"
    )
    
    st.title("📄 Multilingual PDF Chat Assistant")
    
    # Initialize Session State
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Sidebar for PDF Upload
    with st.sidebar:
        st.header("📤 PDF Upload")
        pdf_docs = st.file_uploader(
            "Upload PDF Files", 
            type=['pdf'], 
            accept_multiple_files=True
        )
        
        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    try:
                        # Extract and Process PDFs
                        text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(text)
                        get_vector_store(text_chunks)
                        st.success("PDFs Processed Successfully!")
                    except Exception as e:
                        st.error(f"PDF Processing Error: {e}")
            else:
                st.warning("Please upload PDF files first.")

    # Chat Interface
    st.subheader("💬 Chat with Your PDFs")
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        with st.spinner("Generating response..."):
            answer = chat_with_pdf(user_question)
            st.write("### Response")
            st.write(answer)

    # Chat History
    if st.session_state['chat_history']:
        st.subheader("💾 Chat History")
        for i, (q, a) in enumerate(st.session_state['chat_history'], 1):
            st.markdown(f"**Q{i}:** {q}")
            st.markdown(f"**A{i}:** {a}")
            st.divider()

if __name__ == "__main__":
    main()