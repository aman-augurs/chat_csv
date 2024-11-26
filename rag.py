import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login

# Load environment variables
load_dotenv()

# Hugging Face login token
HF_TOKEN = os.getenv("HF_TOKEN", "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")
login(token=HF_TOKEN)

# Function to extract text from uploaded PDFs with multilingual support
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            # Extract text with better multilingual support
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate and save the FAISS vector store
def get_vector_store(text_chunks):
    # Load multilingual embedding model
    embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    
    # Create embeddings for each text chunk
    embeddings = embedding_model.encode(text_chunks)
    
    # Use FAISS to store the embeddings and associated texts
    vector_store = FAISS.from_embeddings(
        list(zip(text_chunks, embeddings))
    )
    
    # Save the vector store locally
    vector_store.save_local("faiss_index")
    return vector_store

# Function to load the FAISS vector store
def load_vector_store():
    embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    vector_store = FAISS.load_local("faiss_index", embedding_model)
    return vector_store

# Function to create a conversational retrieval chain
def get_conversational_chain():
    # Load the multilingual model
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map=device, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    
    # Create a text generation pipeline
    text_generator = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        max_new_tokens=512
    )
    
    # Convert to LangChain LLM
    llm = HuggingFacePipeline(pipeline=text_generator)

    # Define the prompt template
    prompt_template = """
    Context: {context}
    Question: {question}
    Provide a detailed, multilingual response based on the given context:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Create the conversational retrieval chain
    retriever = load_vector_store().as_retriever(search_kwargs={"k": 5})
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    
    return chain

# Function to handle user queries
def chat_with_pdf(user_question):
    # Load FAISS vector store
    vector_store = load_vector_store()
    
    # Retrieve relevant documents
    relevant_docs = vector_store.similarity_search(user_question, k=5)
    
    # Get the conversational chain
    chain = get_conversational_chain()
    
    # Maintain conversation history
    chat_history = st.session_state.get('chat_history', [])
    
    # Generate the response
    response = chain({
        "question": user_question,
        "chat_history": chat_history
    })
    
    # Update chat history
    chat_history.append((user_question, response['answer']))
    st.session_state['chat_history'] = chat_history
    
    return response['answer']

# Main Streamlit app
def main():
    st.set_page_config(page_title="Multilingual Chat with PDFs", layout="wide")
    st.title("📄 Chat with PDF Files - Multilingual Support")

    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Sidebar for file upload and processing
    with st.sidebar:
        st.header("Upload & Process Files")
        pdf_docs = st.file_uploader("Upload your PDF files here:", accept_multiple_files=True)
        if st.button("Process PDFs"):
            with st.spinner("Extracting and indexing..."):
                # Extract text and process into chunks
                text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(text)
                # Create and save FAISS index
                get_vector_store(text_chunks)
                st.success("PDFs processed successfully!")

    # Main chat interface
    st.subheader("Ask a Question")
    user_question = st.text_input("Type your question here:")
    if user_question:
        with st.spinner("Generating response..."):
            answer = chat_with_pdf(user_question)
            st.write("### Answer:")
            st.write(answer)

    # Display chat history
    st.subheader("Chat History")
    for i, (question, answer) in enumerate(st.session_state.get('chat_history', [])):
        st.write(f"**Q{i+1}:** {question}")
        st.write(f"**A{i+1}:** {answer}")
        st.divider()

if __name__ == "__main__":
    main()