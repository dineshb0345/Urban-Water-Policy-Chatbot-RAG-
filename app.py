import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -------------------------------------------------
# Directories
# -------------------------------------------------
UPLOAD_DIR = "temp_uploads"
VECTOR_DIR = "vectorstore"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()

# -------------------------------------------------
# Load Vectorstore
# -------------------------------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        VECTOR_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

# -------------------------------------------------
# Sidebar - Upload PDFs
# -------------------------------------------------
st.sidebar.header("📄 Upload Water Policy PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Upload policy documents",
    type="pdf",
    accept_multiple_files=True
)

# -------------------------------------------------
# PDF Processing
# -------------------------------------------------
def process_pdfs(pdf_files):
    docs = []

    for pdf in pdf_files:
        path = os.path.join(UPLOAD_DIR, pdf.name)
        with open(path, "wb") as f:
            f.write(pdf.getbuffer())

        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DIR)

if st.sidebar.button("Process Documents"):
    if uploaded_files:
        with st.spinner("Processing documents..."):
            process_pdfs(uploaded_files)
        st.sidebar.success("Documents indexed successfully ✅")
    else:
        st.sidebar.warning("Please upload at least one PDF")

# -------------------------------------------------
# Load Vectorstore
# -------------------------------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        VECTOR_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title="Urban Water Policy RAG Bot", layout="wide")
st.title("💧 Urban Water Policy Chatbot (RAG)")

# -----------------------------
# Initialize chat memory
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Display chat history
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# 3️⃣ User input
# -----------------------------
query = st.chat_input(
    "Ask about urban water policy…",
    key="main_chat_input"
)

if query:
    # Save & display user message
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )
    with st.chat_message("user"):
        st.markdown(query)

    db = load_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 4})

    # Retrieve relevant documents
    docs = retriever.invoke(query)

    context_text = ""
    citations = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown document")
        page = doc.metadata.get("page", "N/A")

        context_text += f"\nSource {i}:\n{doc.page_content}\n"
        citations.append(f"[{i}] {os.path.basename(source)} (Page {page})")

    prompt = ChatPromptTemplate.from_template("""
You are an Urban Water Policy Assistant.

Answer the question ONLY using the sources below.
Do NOT mention document IDs or internal references.
If the answer is not present, say:
"I could not find this information in the uploaded policy documents."

Sources:
{context}

Question:
{question}
""")

    llm = ChatOllama(
        model="llama3",
        temperature=0
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    with st.spinner("Fetching policy information..."):
        answer = chain.invoke({
            "context": context_text,
            "question": query
        })

    # Display & save assistant reply
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    # Sources (outside chat bubble)
    st.markdown("### 📚 Sources")
    for cite in citations:
        st.write(cite)



