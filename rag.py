import streamlit as st
import os
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.vectorstores.utils import DistanceStrategy
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Custom Embedding Wrapper
class SentenceTransformerEmbeddings:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Text chunking
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=100, 
        length_function=len
    )
    return text_splitter.split_text(text)

# Create vector store
def create_vector_store(text_chunks):
    try:
        # Use custom embedding wrapper
        embeddings = SentenceTransformerEmbeddings(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # Create vector store
        vector_store = FAISS.from_texts(
            texts=text_chunks, 
            embedding=embeddings,
            distance_strategy=DistanceStrategy.COSINE
        )
        
        # Save vector store
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"Vector store creation error: {e}")
        return None

# Load vector store
def load_vector_store():
    try:
        # Use custom embedding wrapper
        embeddings = SentenceTransformerEmbeddings(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # Load vector store
        return FAISS.load_local("faiss_index", embeddings)
    except Exception as e:
        st.error(f"Vector store loading error: {e}")
        return None

# Create QA chain
def create_qa_chain():
    try:
        # Use a smaller, more lightweight model
        model_name = "facebook/opt-125m"
        
        # Tokenizer and Model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Text Generation Pipeline
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            max_new_tokens=250
        )
        
        # LangChain LLM
        llm = HuggingFacePipeline(pipeline=pipe)
        
        # Load Vector Store
        vector_store = load_vector_store()
        if not vector_store:
            st.error("No vector store found. Process PDFs first.")
            return None
        
        # Create Retrieval QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3})
        )
        
        return qa_chain
    
    except Exception as e:
        st.error(f"QA Chain creation error: {e}")
        return None

# Main Streamlit App
def main():
    st.set_page_config(
        page_title="PDF Chat Assistant", 
        page_icon="📄", 
        layout="wide"
    )
    
    st.title("📄 Multilingual PDF Chat")
    
    # PDF Upload Section
    with st.sidebar:
        st.header("Upload PDFs")
        pdf_docs = st.file_uploader(
            "Choose PDF files", 
            type=['pdf'], 
            accept_multiple_files=True
        )
        
        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    # Extract and chunk text
                    text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(text)
                    
                    # Create vector store
                    vector_store = create_vector_store(text_chunks)
                    
                    if vector_store:
                        st.success("PDFs processed successfully!")
                    else:
                        st.error("Failed to process PDFs")
            else:
                st.warning("Please upload PDFs first")
    
    # Chat Interface
    st.header("Chat with Your PDFs")
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        # Create QA Chain
        qa_chain = create_qa_chain()
        
        if qa_chain:
            with st.spinner("Generating response..."):
                try:
                    response = qa_chain({"query": user_question})
                    st.write("### Response")
                    st.write(response['result'])
                except Exception as e:
                    st.error(f"Error generating response: {e}")
        else:
            st.warning("Please process PDFs first")

if __name__ == "__main__":
    main()