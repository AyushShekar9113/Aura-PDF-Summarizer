import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
import tempfile
import os
import time
import io
import json
import collections
import re

# ----------------------------
# Load env + basic config
# ----------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

st.set_page_config(page_title="Aura RAG — PDF Summarizer", layout="wide", page_icon="🤖")
# st.sidebar.checkbox("Dark mode", value=st.session_state.get("dark_mode", False), key="darkest_mode")
def apply_theme():
    if st.session_state.dark_mode:
        dark_css = """
        body, .stApp, .css-1d391kg { background: #0f1724; color: #e6eef8; }
        .user { background:#155e75; color: white; }
        .bot  { background:#0b1220; color: #e6eef8; border:1px solid #233247; }
        .stTextInput>div>div>input { background: #1f2937; color: white; }
        .stButton>button { background-color: #155e75; color: white; border: none; }
        .stSelectbox>div>div>div>div { background: #1f2937; color: white; }
        .stFileUploader>div>div>input { background: #1f2937; color: white; }
        """
    else:
        dark_css = """
        body, .stApp { background: linear-gradient(120deg,#f7fbff,#ffffff); color: #000; }
        .user { background:#d1e7ff; color: black; }
        .bot  { background:#f7f7fb; color: black; }
        """
    st.markdown(f"<style>{dark_css}</style>", unsafe_allow_html=True)

# apply_theme()

# ----------------------------
# CSS / Theme toggles
# ----------------------------
def local_css(css: str):
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

base_css = """
/* Chat bubbles */
.user { background:#d1e7ff; padding:12px; border-radius:12px; margin:8px 0; text-align:right; }
.bot  { background:#f7f7fb; padding:12px; border-radius:12px; margin:8px 0; text-align:left; }
.header { font-size:20px; font-weight:700; margin-bottom:8px; }
.small { font-size:12px; color: #666; }
"""

light_css = """
body { background: linear-gradient(120deg,#f7fbff,#ffffff); font-family: 'Inter', sans-serif; }
"""

dark_css = """
body { background: linear-gradient(120deg,#0f1724,#071122); color: #e6eef8; font-family: 'Inter', sans-serif; }
.user { background:#155e75; color: white; }
.bot  { background:#0b1220; color: #e6eef8; border:1px solid #233247; }
"""

# ----------------------------
# Session state init
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of (user, bot, timestamp)
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "total_response_time" not in st.session_state:
    st.session_state.total_response_time = 0.0
if "vectorstore_cache" not in st.session_state:
    st.session_state.vectorstore_cache = {}  # path -> FAISS obj
if "pdf_meta" not in st.session_state:
    st.session_state.pdf_meta = {}  # store pages or metadata for preview
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_relevant_docs" not in st.session_state:
    st.session_state.last_relevant_docs = []

# ----------------------------
# Sidebar - navigation
# ----------------------------
with st.sidebar:
    st.header("Aura RAG - Controls")
    page = st.radio("Go to", ["Chat", "Analytics", "Export Summary"])
    st.write("---")
    st.checkbox("Dark mode", value=st.session_state.dark_mode, key="dark_mode")
    # st.write("Model:")
    # model_choice = st.selectbox("LLM model", ["gemini-2.5-flash", "gemini-1.5-pro"], index=0)
    # st.write("Embedding model: sentence-transformers/all-MiniLM-L6-v2")
    # st.write("---")
    # st.caption("Note: Make sure GOOGLE_API_KEY is set in .env")

# Apply CSS
local_css(base_css)
if st.session_state.dark_mode:
    local_css(dark_css)
else:
    local_css(light_css)

# ----------------------------
# Utility helpers
# ----------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text

def top_keywords(texts, top_k=10):
    stop = set([
        "the","and","is","in","to","of","a","for","with","that","on","as","are","this","it","by","an",
        "be","or","from","we","can","which","have","has"
    ])
    counter = collections.Counter()
    for t in texts:
        words = re.findall(r"[A-Za-z]{3,}", t.lower())
        for w in words:
            if w not in stop:
                counter[w] += 1
    return counter.most_common(top_k)

# ----------------------------
# Caching vectorstore creation
# ----------------------------
@st.cache_resource
def create_vectorstore_from_file(path: str):
    """
    Create FAISS vectorstore from a PDF path.
    This function is cached by streamlit to avoid reembedding each time.
    """
    loader = PyPDFLoader(path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    # safety check
    if not docs:
        raise ValueError("No text was extracted from the PDF (maybe scanned pages?)")

    # embeddings: force CPU to avoid meta tensor errors
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = FAISS.from_documents(docs, embedding)
    # also return pages/documents for preview
    return vectorstore, pages, docs

# ----------------------------
# LLM wrapper (synchronous)
# ----------------------------
def answer_with_llm(context: str, question: str, model_name: str = "gemini-2.5-flash"):
    """
    Run the prompt through the LLM and return the final text and optionally timings.
    We'll also return a short "summary" for export use.
    """
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=GOOGLE_API_KEY)
    template = PromptTemplate.from_template(
        """You are Aura — a helpful assistant that answers using only the provided PDF content.
Use plain, concise language and if you don't know, say you don't know.

PDF Context:
{context}

Question:
{question}

Answer concisely:"""
    )
    parser = StrOutputParser()
    chain = template | llm | parser

    # measure time
    t0 = time.time()
    result = chain.invoke({
        "context": context,
        "question": question
    })
    response_time = time.time() - t0
    return result, response_time

# ----------------------------
# Streaming helper (simulated safe streaming)
# ----------------------------
def stream_text_to_container(text: str, container, delay: float = 0.01):
    """
    Simulate streaming by writing words gradually to container.
    This avoids relying on LLM streaming API (which may differ by provider).
    """
    words = text.split()
    out = ""
    for w in words:
        out += w + " "
        container.markdown(f"<div class='bot'>{out}</div>", unsafe_allow_html=True)
        time.sleep(delay)
    # return final
    container.markdown(f"<div class='bot'>{out}</div>", unsafe_allow_html=True)

# ----------------------------
# Pages
# ----------------------------
if page == "Chat":
    # Main Chat UI
    st.markdown("<div class='header'>Aura — PDF Summarizer & QA</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload PDF (selectable text PDFs work best)", type=["pdf"])
        user_query = st.text_input("Ask a question about the PDF", key="user_query_box")
        submit_btn = st.button("Ask Aura")
    with col2:
        st.markdown("**Quick actions**")
        if st.button("Summarize whole PDF"):
            st.session_state.user_wants_summary = True
        if st.button("Clear Chat"):
            st.session_state.history = []
            st.session_state.query_count = 0
            st.session_state.total_response_time = 0.0
            st.session_state.last_answer = ""
            st.session_state.last_relevant_docs = []
            st.rerun()

    # If new PDF uploaded, (re)create vectorstore and cache
    if uploaded_file:
        # save to tmp path
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(uploaded_file.read())
        tmp.flush()
        tmp_path = tmp.name

        # If cached path differs or not in cache, create vectorstore
        cache_key = tmp_path  # we use path since new tmp file each upload; caching still helps within same process
        try:
            vectorstore, pages, docs = create_vectorstore_from_file(tmp_path)
            st.session_state.pdf_meta = {
                "path": tmp_path,
                "pages": pages,
                "docs_count": len(docs)
            }
        except Exception as e:
            st.error(f"Failed to create vectorstore: {e}")
            st.stop()

    # Handle summary request or user query
    if uploaded_file and (submit_btn or st.session_state.get("user_wants_summary", False) or user_query):
        # Determine the question
        question = user_query.strip() if user_query else ""
        if st.session_state.get("user_wants_summary", False):
            # Build context: use all docs concatenated (or top relevant docs)
            docs = create_vectorstore_from_file(st.session_state.pdf_meta["path"])[2]
            context_text = "\n\n".join([d.page_content for d in docs])
            question = "Please provide a concise full summary of the full PDF."
            st.session_state.user_wants_summary = False
        else:
            # Use retriever to get top relevant docs
            retriever = vectorstore.as_retriever()
            relevant_docs = retriever.invoke(question)
            st.session_state.last_relevant_docs = relevant_docs
            context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Run LLM and stream the result
        placeholder = st.empty()
        with placeholder.container():
            st.markdown(f"<div class='user'>{clean_text(question)}</div>", unsafe_allow_html=True)
            bot_container = st.empty()

        # Get full response (synchronous call)
        try:
            result_text, response_time = answer_with_llm(context=context_text, question=question)
        except Exception as e:
            st.error(f"LLM call failed: {e}")
            st.stop()

        # Streaming simulation to UI
        stream_text_to_container(result_text, bot_container, delay=0.005)

        # Update analytics + memory
        st.session_state.history.append((question, result_text, time.time()))
        st.session_state.query_count += 1
        st.session_state.total_response_time += response_time
        st.session_state.last_answer = result_text

        # Translate and show translated result below (optional)
        selected_lang = st.sidebar.selectbox("Translated language (quick)", ["English","Hindi","Marathi","Tamil","Telugu","Gujarati","Kannada"], index=0)
        lang_map = {"English":"en","Hindi":"hi","Marathi":"mr","Tamil":"ta","Telugu":"te","Gujarati":"gu","Kannada":"kn"}
        dest_code = lang_map.get(selected_lang, "en")
        try:
            translated = GoogleTranslator(source="auto", target=dest_code).translate(result_text) if dest_code != "en" else result_text
        except Exception:
            translated = result_text

        with st.expander("Translated Answer"):
            st.write(translated)
        


    # ----------------------------
    # Add download buttons for this answer
    # ----------------------------
    # Original (English) answer
        st.download_button(
            label="📥 Download Answer (English)",
            data=result_text.encode("utf-8"),
            file_name="aura_answer_en.txt",
            mime="text/plain"
        )

        # Translated answer (if different from English)
        if translated != result_text:
            st.download_button(
                label=f"📥 Download Answer ({selected_lang})",
                data=translated.encode("utf-8"),
                file_name=f"aura_answer_{selected_lang}.txt",
                mime="text/plain"
            )


        # Show source references (page snippets)
        with st.expander("Source snippets (from PDF)"):
            if st.session_state.last_relevant_docs:
                for i, doc in enumerate(st.session_state.last_relevant_docs[:5]):
                    st.markdown(f"**Snippet {i+1}:**")
                    st.write(doc.page_content[:400] + "...")
            else:
                st.write("No relevant doc snippets available.")

    # Show conversation history on the right column-like area
    st.markdown("---")
    st.markdown("### Conversation History")
    for q, a, ts in reversed(st.session_state.history[-20:]):
        st.markdown(f"<div class='user'>{clean_text(q)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bot'>{clean_text(a)}</div>", unsafe_allow_html=True)

# ----------------------------
# Analytics page
# ----------------------------
elif page == "Analytics":
    st.header("📊 Analytics")
    st.markdown("Usage statistics for this session (in-memory):")
    st.write(f"Total queries: **{st.session_state.query_count}**")
    avg_time = (st.session_state.total_response_time / st.session_state.query_count) if st.session_state.query_count > 0 else 0.0
    st.write(f"Average response time (s): **{avg_time:.2f}**")

    # Top keywords from all stored answers
    all_answers = [a for (_q, a, _ts) in st.session_state.history]
    if all_answers:
        keywords = top_keywords(all_answers, top_k=20)
        st.markdown("Top keywords from AI answers")
        st.table(keywords)
    else:
        st.write("No answers yet to compute keywords.")

    # PDF metadata
    if st.session_state.pdf_meta:
        st.markdown("PDF Info")
        st.write(f"Extracted chunks: **{st.session_state.pdf_meta.get('docs_count', 'N/A')}**")
    else:
        st.write("No PDF loaded in this session.")


elif page == "Export Summary":
    st.header("📦 Export / Download")
    if not st.session_state.history:
        st.info("No conversation yet — ask a question in Chat tab first.")
    else:
        st.write("Create a downloadable summary of the conversation and the last model answer.")

        # Build transcript
        transcript = []
        for q, a, ts in st.session_state.history:
            tstr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
            transcript.append(f"[{tstr}] Q: {q}\nA: {a}\n")

        transcript_text = "\n\n".join(transcript)

        # Create short summary by asking LLM to summarize the transcript
        st.write("Generating short summary (this will call the LLM)...")
        try:
            summ_prompt = "Summarize the following conversation in 6-8 lines, focusing on the main points and any concrete actions:\n\n" + transcript_text
            summary, _ = answer_with_llm(context=transcript_text, question="Please provide a concise summary of the conversation.")
        except Exception as e:
            st.error(f"Failed to summarize conversation: {e}")
            summary = st.session_state.last_answer or "No summary available."

        st.subheader("Conversation Summary")
        st.write(summary)

        # Provide downloads
        txt_bytes = summary.encode("utf-8")
        st.download_button("Download summary (.txt)", data=txt_bytes, file_name="aura_summary.txt", mime="text/plain")

        # Full transcript
        t_bytes = transcript_text.encode("utf-8")
        st.download_button("Download full transcript (.txt)", data=t_bytes, file_name="aura_transcript.txt", mime="text/plain")
