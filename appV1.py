import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
#from langchain.embeddings import OpenAIEmbeddings

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory



# Clarifai Configuration
#os.environ["CLARIFAI_PAT"] = "b5d78d6de57f41f7ada7c9f89ce84646"
#CLARIFAI_PAT = os.getenv("CLARIFAI_PAT")
# access via key notation''  # Your Personal Access Token (PAT)
#USER_ID = "toswari-ai"
#APP_ID = os.getenv("CLARIFAI_APP_ID")

#APP_ID = "teddy-pdf-chat"

# Set up Clarifai credentials
CLARIFAI_PAT = st.secrets["CLARIFAI_PAT"]  # Store in Streamlit secrets
USER_ID = st.secrets["CLARIFAI_USER_ID"]  # Store in Streamlit secrets
APP_ID = st.secrets["CLARIFAI_APP_ID"] # Store in Streamlit secrets

print("CLARIFAI_PAT: ", CLARIFAI_PAT)


# --- Streamlit UI ---
st.title("Chat with Your PDF (Clarifai RAG + Memory)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# --- PDF Processing ---
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

# --- Vector DB Setup ---
def create_vector_db(docs, embeddings):
    return FAISS.from_texts(docs, embeddings)

# --- Main Logic ---
if uploaded_file:
    with st.spinner("Processing PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        text_chunks = [raw_text[i:i+1000] for i in range(0, len(raw_text), 1000)]  # Chunk for context

        # Clarifai API credentials
        clarifai_pat = CLARIFAI_PAT  # Replace with your actual Clarifai PAT
        clarifai_base_url = "https://api.clarifai.com/v2/ext/openai/v1"  # Clarifai's OpenAI API base URL

        # Embeddings and vector store
        embeddings = OpenAIEmbeddings(
            openai_api_key=clarifai_pat,
            openai_api_base=clarifai_base_url,
            engine="text-embedding-ada-002"  # Try using 'engine' instead of 'model'
        )
        vectordb = create_vector_db(text_chunks, embeddings)

        # Memory for chat
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # RAG Chain
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(
                model="anthropic/completion/models/claude-sonnet-4",  # Or the Clarifai-supported model you want
                openai_api_key=clarifai_pat,
                base_url=clarifai_base_url
            ),
            retriever=vectordb.as_retriever(),
            memory=memory
        )

        # Chat UI
        st.success("PDF processed! Ask questions about its content below.")

        user_input = st.text_input("You:", key="input")
        if user_input:
            response = rag_chain({"question": user_input, "chat_history": st.session_state['chat_history']})
            st.session_state['chat_history'].append((user_input, response['answer']))
            st.markdown(f"**Bot:** {response['answer']}")

        # Display chat history
        if st.session_state['chat_history']:
            st.markdown("---")
            st.markdown("#### Chat History")
            for i, (q, a) in enumerate(st.session_state['chat_history']):
                st.markdown(f"**You:** {q}")
                st.markdown(f"**Bot:** {a}")
