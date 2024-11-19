import streamlit as st
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
import tempfile
import hashlib
import shutil

# Load environment variables
load_dotenv()

def initialize_session_state():
    """Initialize session state variables."""
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

def generate_unique_id(file):
    """Generate a unique identifier for a file."""
    return hashlib.md5(file.getvalue()).hexdigest()

def process_pdfs(uploaded_files):
    """Process multiple uploaded PDF files."""
    # Create a temporary directory for this session's vector store
    temp_dir = tempfile.mkdtemp()
    
    # Prepare to collect all documents
    all_documents = []
    
    # Process each uploaded file
    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load and process the PDF
        try:
            loader = PyPDFLoader(tmp_file_path)
            pages = loader.load()

            # Split the documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            documents = text_splitter.split_documents(pages)
            
            # Add filename as metadata to help track source
            for doc in documents:
                doc.metadata['source'] = uploaded_file.name

            all_documents.extend(documents)

            # Clean up temporary file
            os.unlink(tmp_file_path)
        
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")

    # Create vector store using Hugging Face embeddings
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Create vector store in the temporary directory
    vector_store = Chroma.from_documents(
        all_documents,
        embeddings,
        persist_directory=temp_dir
    )

    return vector_store, temp_dir

def get_conversation_chain(vector_store):
    """Create a conversation chain."""
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={
            "temperature": 0.5,
            "max_length": 512,
            "top_p": 0.9
        }
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_kwargs={
                "k": 5,  # Increase number of retrieved documents
                "filter_type": "mmr",  # Maximal Marginal Relevance
                "fetch_k": 10  # Number of documents to fetch before filtering
            }
        ),
        return_source_documents=True,
        max_tokens_limit=512
    )
    return conversation_chain

def format_response(response):
    """Format the response and extract source information."""
    answer = response['answer']
    sources = response.get('source_documents', [])
    
    formatted_response = f"{answer}\n\n"
    if sources:
        formatted_response += "Sources:\n"
        # Use a set to avoid duplicate sources
        unique_sources = set()
        for doc in sources:
            source = doc.metadata.get('source', 'Unknown source')
            unique_sources.add(source)
        
        for source in unique_sources:
            formatted_response += f"- {source}\n"
    
    return formatted_response

def handle_user_input(user_question):
    """Process user input and generate response."""
    if st.session_state.conversation is None:
        st.warning('Please upload PDF documents first.')
        return

    try:
        # Generate response
        response = st.session_state.conversation({
            'question': user_question,
            'chat_history': st.session_state.chat_history
        })

        # Format response
        formatted_response = format_response(response)
        
        # Update chat history
        st.session_state.chat_history.append((user_question, formatted_response))
        return formatted_response

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def main():
    st.title("Multi-Document PDF Chat")
    initialize_session_state()

    # Sidebar for model information
    st.sidebar.header("Model Information")
    st.sidebar.write("Using FLAN-T5 Base model")
    st.sidebar.write("Embedding model: all-MiniLM-L6-v2")
    
    # PDF upload with multiple file support
    uploaded_files = st.file_uploader(
        "Upload PDF documents", 
        type=['pdf'], 
        accept_multiple_files=True
    )

    # Process PDFs if files are uploaded
    if uploaded_files:
        with st.spinner('Processing PDFs...'):
            try:
                # Check if these are new files
                new_files = [
                    f for f in uploaded_files 
                    if generate_unique_id(f) not in 
                    [generate_unique_id(old) for old in st.session_state.uploaded_files]
                ]

                # If no new files, skip processing
                if new_files:
                    # Clear previous vector store if it exists
                    if st.session_state.vector_store:
                        try:
                            shutil.rmtree(st.session_state.vector_store.persist_directory)
                        except Exception:
                            pass

                    # Process new PDFs
                    vector_store, temp_dir = process_pdfs(uploaded_files)
                    st.session_state.vector_store = vector_store
                    st.session_state.vector_store_dir = temp_dir
                    
                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vector_store)
                    st.session_state.uploaded_files = uploaded_files
                    st.success(f'Processed {len(uploaded_files)} PDF documents!')

            except Exception as e:
                st.error(f"Error processing PDFs: {str(e)}")

    # Chat interface
    if st.session_state.conversation is not None:
        # Display uploaded files
        st.sidebar.subheader("Uploaded Documents")
        for file in st.session_state.uploaded_files:
            st.sidebar.write(file.name)

        # Chat input
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            with st.spinner('Thinking...'):
                response = handle_user_input(user_question)
                if response:
                    st.write("Response:", response)
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for question, answer in st.session_state.chat_history:
                with st.expander(f"Q: {question}"):
                    st.write(answer)

if __name__ == "__main__":
    main()