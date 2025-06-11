import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import ClarifaiEmbeddings

# --- Configuration ---
st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")
st.title("📄 RAG PDF Chatbot - OpenAI API Compatibility")


# Set up Clarifai credentials
CLARIFAI_PAT = st.secrets["CLARIFAI_PAT"]  # Store in Streamlit secrets
USER_ID = st.secrets["CLARIFAI_USER_ID"]  # Store in Streamlit secrets
APP_ID = st.secrets["CLARIFAI_APP_ID"] # Store in Streamlit secrets

print("CLARIFAI_PAT: ", CLARIFAI_PAT)


# --- Sidebar for API Key and Model ---
with st.sidebar:
    st.header("Configuration")
    api_key = CLARIFAI_PAT  # Use the stored PAT from Streamlit secrets
    base_url = "https://api.clarifai.com/v2/ext/openai/v1"
    model_name = st.text_input("Model Name", value="anthropic/completion/models/claude-sonnet-4")
    model_id = st.text_input("Clarifai Embedding Model ID", value="BAAI-bge-base-en-v15")
    st.markdown("---")
    uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    process_btn = st.button("Process Documents")

# --- Helper Functions ---
def extract_text_from_pdfs(files):
    text = ""
    for file in files:
        pdf = PdfReader(file)
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# --- Main Logic ---
if process_btn and uploaded_files and api_key:
    with st.spinner("Processing and indexing documents..."):
        # Extract and chunk text
        raw_text = extract_text_from_pdfs(uploaded_files)
        chunks = chunk_text(raw_text)


        # Clarifai Embeddings
        embeddings = ClarifaiEmbeddings(
            pat=CLARIFAI_PAT,
            user_id=USER_ID,
            app_id=APP_ID,
            model_id=model_id
        )
        vectordb = FAISS.from_texts(chunks, embeddings)

        # Store in session
        st.session_state['vectordb'] = vectordb
        st.session_state['chat_history'] = []
        st.success("Documents processed and indexed!")

# --- Chat Interface ---
if 'vectordb' in st.session_state:
    st.markdown("### Ask a question about your documents:")
    user_input = st.text_input("Your question", key="chat_input")
    if 'memory' not in st.session_state:
        st.session_state['memory'] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    if user_input:
        # Set up LLM and RAG chain
        llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            base_url=base_url
        )
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state['vectordb'].as_retriever(),
            memory=st.session_state['memory']
        )
        # Get answer
        result = rag_chain({"question": user_input, "chat_history": st.session_state['chat_history']})
        st.session_state['chat_history'].append((user_input, result['answer']))
        st.markdown(f"**Assistant:** {result['answer']}")

    # Display chat history (collapsible)
    if st.session_state.get('chat_history'):
        with st.expander("Chat History", expanded=False):
            for q, a in st.session_state['chat_history']:
                st.markdown(f"**You:** {q}")
                st.markdown(f"**Assistant:** {a}")

else:
    st.info("Upload and process PDF documents to start chatting.")

