import os
import io
import hashlib
import pickle
import shutil

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="StudyMate - AI Study Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 StudyMate")
st.subheader("Your AI-Powered Study Assistant")

st.write(
    "Upload a PDF from any subject, ask questions, "
    "understand difficult concepts, and prepare for exams."
)


# --------------------------------------------------
# 2. LOAD ENVIRONMENT VARIABLES & SETUP CACHE DIR
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error(
        "Gemini API key not found. Please add your key "
        "to the .env file and restart the application."
    )
    st.stop()

# Local folder to cache processed vector indices for instant reload
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache_index")
os.makedirs(CACHE_DIR, exist_ok=True)


# --------------------------------------------------
# 3. LOAD CLOUD EMBEDDING MODEL (FAST & LAPTOP SAFE)
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    """Uses Google Cloud Embeddings API (gemini-embedding-2).

    0% local CPU usage, no thermal strain, and processes 100+ chunks in seconds.
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=API_KEY
    )


# --------------------------------------------------
# 4. LOAD GEMINI LLM
# --------------------------------------------------

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        google_api_key=API_KEY,
        temperature=0.2
    )


# --------------------------------------------------
# 5. SESSION STATE INITIALIZATION
# --------------------------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "processed_hash" not in st.session_state:
    st.session_state.processed_hash = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# --------------------------------------------------
# 6. PDF UPLOAD & CACHED PROCESSING
# --------------------------------------------------

st.sidebar.header("📄 Upload Study Material")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    # Re-process or reload if new file or not in session
    if file_hash != st.session_state.processed_hash:
        file_cache_dir = os.path.join(CACHE_DIR, file_hash)
        chunks_cache_file = os.path.join(file_cache_dir, "chunks.pkl")

        # 6a. Check if this PDF's FAISS index was already cached on disk
        if os.path.exists(file_cache_dir) and os.path.exists(chunks_cache_file):
            with st.spinner("⚡ Loading from cache (instant)..."):
                try:
                    embeddings = load_embedding_model()
                    vector_store = FAISS.load_local(
                        file_cache_dir,
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    with open(chunks_cache_file, "rb") as f:
                        chunks = pickle.load(f)

                    st.session_state.vector_store = vector_store
                    st.session_state.chunks = chunks
                    st.session_state.processed_hash = file_hash
                    st.session_state.pdf_name = uploaded_file.name

                    st.sidebar.success("⚡ Loaded instantly from cache!")
                    st.sidebar.write(f"Cached chunks: {len(chunks)}")
                except Exception as cache_err:
                    st.sidebar.warning(f"Cache reload failed ({cache_err}). Re-indexing...")

        # 6b. If not cached, process PDF and generate embeddings via API
        if file_hash != st.session_state.processed_hash:
            with st.spinner("⚡ Extracting text & generating cloud embeddings..."):
                try:
                    pdf_reader = PdfReader(io.BytesIO(file_bytes))
                    documents = []

                    for page_number, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            documents.append(
                                Document(
                                    page_content=page_text,
                                    metadata={
                                        "page": page_number + 1,
                                        "source": uploaded_file.name
                                    }
                                )
                            )

                    if not documents:
                        st.error(
                            "No readable text was found. "
                            "Please upload a text-based PDF."
                        )
                        st.stop()

                    # Split text into manageable chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800,
                        chunk_overlap=120
                    )
                    chunks = text_splitter.split_documents(documents)

                    # Compute embeddings in Google Cloud (fast, 0% CPU strain)
                    embeddings = load_embedding_model()
                    vector_store = FAISS.from_documents(chunks, embeddings)

                    # Save to local disk cache for instant reuse on page refresh
                    os.makedirs(file_cache_dir, exist_ok=True)
                    vector_store.save_local(file_cache_dir)
                    with open(chunks_cache_file, "wb") as f:
                        pickle.dump(chunks, f)

                    # Store in session state
                    st.session_state.vector_store = vector_store
                    st.session_state.chunks = chunks
                    st.session_state.processed_hash = file_hash
                    st.session_state.pdf_name = uploaded_file.name

                    st.sidebar.success("✅ PDF processed & cached!")
                    st.sidebar.write(f"Pages with text: {len(documents)}")
                    st.sidebar.write(f"Text chunks created: {len(chunks)}")

                except Exception as error:
                    st.error(f"PDF processing failed: {error}")
                    st.stop()
    else:
        st.sidebar.success("✅ PDF ready.")

else:
    st.session_state.vector_store = None
    st.session_state.chunks = []
    st.session_state.processed_hash = None
    st.session_state.pdf_name = None

# Sidebar cache management
st.sidebar.divider()
if st.sidebar.button("🗑️ Clear Cache", help="Remove cached PDF indexes to free disk space"):
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR, exist_ok=True)
    st.session_state.vector_store = None
    st.session_state.chunks = []
    st.session_state.processed_hash = None
    st.sidebar.info("Cache cleared. Re-upload your PDF.")
    st.rerun()


# --------------------------------------------------
# 7. CHECK PDF AVAILABILITY
# --------------------------------------------------

if st.session_state.vector_store is None:
    st.info("👈 Upload a PDF from the sidebar to start.")
    st.markdown(
        """
        ### What can StudyMate do?

        - 💬 **Ask questions** about your PDF with fast RAG retrieval.
        - 💡 **Explain concepts** simply with step-by-step clarity.
        - 📝 **Summarize documents** in seconds with one-shot AI summaries.
        - 🎯 **Generate exam questions** to test your knowledge.
        - 📖 **Show source page numbers** for verifiable learning.
        """
    )
    st.stop()

st.success(f"Currently studying: **{st.session_state.pdf_name}**")


# --------------------------------------------------
# 8. RETRIEVE RELEVANT CHUNKS
# --------------------------------------------------

def retrieve_context(question, top_k=4):
    vector_store = st.session_state.vector_store
    k_to_fetch = min(top_k, max(1, vector_store.index.ntotal))

    relevant_documents = vector_store.similarity_search(
        question,
        k=k_to_fetch
    )

    context_parts = []
    source_pages = []

    for document in relevant_documents:
        page_number = document.metadata.get("page", "Unknown")
        context_parts.append(
            f"[Page {page_number}]\n{document.page_content}"
        )
        source_pages.append(page_number)

    context = "\n\n".join(context_parts)
    return context, sorted(set(source_pages))


# --------------------------------------------------
# 9. CALL GEMINI HELPER
# --------------------------------------------------

def ask_gemini(prompt):
    """Invokes Gemini LLM and cleanly extracts the response text."""
    llm = load_llm()
    response = llm.invoke(prompt)
    content = response.content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text"):
                parts.append(str(item.text))
        return "".join(parts).strip()
    return str(content).strip()


# --------------------------------------------------
# 10. PROMPT BUILDERS
# --------------------------------------------------

def build_qa_prompt(question):
    context, source_pages = retrieve_context(question, top_k=5)
    prompt = f"""
You are StudyMate, a helpful and patient AI study assistant.

Answer the student's question using only the provided PDF context.

Instructions:
1. Explain clearly in simple, easy-to-understand English.
2. Use bullet points, bold headings, and real-world examples where helpful.
3. Do not invent facts or extrapolate beyond what the context supports.
4. If the context does not contain sufficient details to answer, state clearly that the uploaded PDF does not provide enough information.

STUDENT QUESTION:
{question}

RETRIEVED PDF CONTEXT:
{context}

Provide a well-structured, clear answer:
"""
    return prompt, source_pages


def build_summary_prompt():
    all_chunks = st.session_state.chunks
    # Single-shot summarization: pass combined text to Gemini Flash
    # Gemini Flash has a 1,000,000+ token context window, easily fitting full study PDFs.
    combined_context = "\n\n".join(doc.page_content for doc in all_chunks)

    prompt = f"""
You are StudyMate, an expert academic tutor.

Create a comprehensive, well-organized study summary based on the provided PDF content.

Please structure your summary with:
1. **Overview & Subject Objective**: A concise 2-3 sentence summary of the material.
2. **Key Concepts & Definitions**: Important terms and fundamental concepts explained simply.
3. **Core Topics Breakdown**: Detailed breakdown of the main themes using headings and bullet points.
4. **Summary & High-Yield Exam Takeaways**: Key conclusions and crucial points for exam revision.

Do not invent facts outside the provided content.

PDF CONTENT:
{combined_context}
"""
    pages = sorted(
        set(
            doc.metadata.get("page")
            for doc in all_chunks
            if doc.metadata.get("page") is not None
        )
    )
    return prompt, pages


def build_exam_questions_prompt():
    context, source_pages = retrieve_context(
        "Important concepts, definitions, formulas, key themes, and core principles",
        top_k=8
    )
    prompt = f"""
You are an academic exam preparation assistant.

Generate 5 high-yield exam preparation questions based only on the provided PDF content.

Include:
- 3 Conceptual / Short-Answer Questions
- 2 Analytical / In-depth Descriptive Questions

Do not provide the answers.
Do not invent topics outside the provided context.

PDF CONTENT:
{context}
"""
    return prompt, source_pages


# --------------------------------------------------
# 11. STUDYMATE USER INTERFACE
# --------------------------------------------------

st.divider()
st.header("💬 Study Assistant")

task_type = st.selectbox(
    "Choose a task:",
    [
        "Ask a Question",
        "Summarize PDF",
        "Generate Exam Questions"
    ]
)

if task_type == "Ask a Question":
    user_question = st.text_area(
        "Enter your question:",
        placeholder="Example: Explain supervised learning with an example."
    )
else:
    user_question = ""


if st.button("Generate Response", type="primary"):
    try:
        if task_type == "Ask a Question":
            if not user_question.strip():
                st.warning("Please enter a question.")
                st.stop()
            prompt, source_pages = build_qa_prompt(user_question)

        elif task_type == "Summarize PDF":
            prompt, source_pages = build_summary_prompt()

        else:
            prompt, source_pages = build_exam_questions_prompt()

        with st.spinner("StudyMate is preparing your response... ⚡"):
            answer = ask_gemini(prompt)

        st.subheader("📝 StudyMate's Response")
        st.markdown(answer)

        st.divider()
        st.subheader("📖 Source References")

        if source_pages:
            page_text = ", ".join(str(p) for p in source_pages)
            st.write(f"Relevant PDF pages: **{page_text}**")
        else:
            st.write("No source pages available.")

    except Exception as error:
        st.error(f"Something went wrong: {error}")
        st.info(
            "Check your internet connection, API key, API quota, and installed packages."
        )


# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "StudyMate | Streamlit | Fast Cloud RAG | LangChain | "
    "Google Embeddings | FAISS | Gemini Flash"
)
