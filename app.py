import streamlit as st
import os
from pathlib import Path

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic RAG — AI Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── INITIALIZE SESSION STATE ─────────────────────────────────────────────────
if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state["GOOGLE_API_KEY"] = None
if "TAVILY_API_KEY" not in st.session_state:
    st.session_state["TAVILY_API_KEY"] = None
if "api_configured" not in st.session_state:
    st.session_state["api_configured"] = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #0f1629;
    --bg-card: #141c2e;
    --accent-cyan: #00d4ff;
    --accent-purple: #7c3aed;
    --accent-green: #10b981;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --border: rgba(0, 212, 255, 0.15);
}

.stApp { background: var(--bg-primary); font-family: 'DM Sans', sans-serif; }
.stApp > header { background: transparent; }
#MainMenu, footer, header { visibility: hidden; }

section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}

.api-banner {
    background: linear-gradient(135deg, #0f1629 0%, #1a0a2e 100%);
    border: 1px solid rgba(0, 212, 255, 0.3);
    border-radius: 12px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1rem;
}
.api-banner .banner-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #00d4ff;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 0.3rem;
}

.rag-header {
    background: linear-gradient(135deg, #0f1629 0%, #1a0a2e 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.rag-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(0,212,255,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.rag-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    color: var(--accent-cyan);
    margin: 0;
    letter-spacing: -0.5px;
}
.rag-header p {
    color: var(--text-secondary);
    margin: 0.5rem 0 0 0;
    font-size: 0.95rem;
}

.info-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.info-card .label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent-cyan);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.3rem;
}
.info-card .value {
    color: var(--text-primary);
    font-size: 0.95rem;
}

.chat-container {
    max-height: 500px;
    overflow-y: auto;
    padding: 0.5rem 0;
}
.chat-msg {
    padding: 1rem 1.2rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    font-size: 0.9rem;
    line-height: 1.7;
}
.chat-user {
    background: rgba(124, 58, 237, 0.15);
    border: 1px solid rgba(124, 58, 237, 0.3);
    border-left: 3px solid #7c3aed;
    color: var(--text-primary);
}
.chat-assistant {
    background: rgba(0, 212, 255, 0.05);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-left: 3px solid #00d4ff;
    color: var(--text-primary);
}
.chat-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.5rem;
}
.label-user { color: #7c3aed; }
.label-ai { color: #00d4ff; }

.source-box {
    background: rgba(16, 185, 129, 0.05);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin-top: 0.8rem;
    font-size: 0.78rem;
    color: #10b981;
    font-family: 'Space Mono', monospace;
}

.step-flow {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-secondary);
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.step-active { color: var(--accent-cyan); }
.step-sep { color: rgba(148,163,184,0.3); }

.stTextInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.1) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #00d4ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important;
}

.stFileUploader > div {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.8rem;
    margin-bottom: 1.5rem;
}
.metric-item {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-num {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    color: var(--accent-cyan);
    font-weight: 700;
}
.metric-label {
    font-size: 0.7rem;
    color: var(--text-secondary);
    margin-top: 0.2rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

hr { border-color: var(--border) !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--accent-purple); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
    <h1>🧠 Agentic RAG System</h1>
    <p>Upload documents → Ask questions → AI retrieves answers with citations + web search fallback</p>
    <div style="margin-top:1rem;">
        <span class="step-flow">
            <span class="step-active">① UPLOAD DOCUMENT</span>
            <span class="step-sep">──▶</span>
            <span class="step-active">② EMBED & INDEX</span>
            <span class="step-sep">──▶</span>
            <span class="step-active">③ SEMANTIC SEARCH</span>
            <span class="step-sep">──▶</span>
            <span class="step-active">④ LLM + CITATIONS</span>
            <span class="step-sep">──▶</span>
            <span class="step-active">⑤ WEB FALLBACK</span>
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── FIX 4: API Keys on Main Page ─────────────────────────────────────────────
if not st.session_state.get("api_configured"):

    st.markdown("""
    <div class="api-banner">
        <div class="banner-title">🔐 Step 1 — Enter your API keys to get started</div>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3 = st.columns([2, 2, 1])
    with k1:
        gemini_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="AIza...",
            key="gemini_input"
        )
    with k2:
        tavily_key_input = st.text_input(
            "Tavily API Key (optional)",
            type="password",
            placeholder="tvly-...",
            key="tavily_input"
        )
    with k3:
        st.markdown("<div style='margin-top:1.75rem'>", unsafe_allow_html=True)
        if st.button("🔐 Save Keys", use_container_width=True):
            if gemini_key_input.strip():
                st.session_state["GOOGLE_API_KEY"] = gemini_key_input.strip()
                st.session_state["TAVILY_API_KEY"] = tavily_key_input.strip()
                st.session_state["api_configured"] = True
                os.environ["GOOGLE_API_KEY"] = gemini_key_input.strip()
                os.environ["TAVILY_API_KEY"] = tavily_key_input.strip()
                st.success("✅ API Keys saved! You can now upload a document.")
                st.rerun()
            else:
                st.error("⚠️ Gemini API key is required.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

else:
    # Compact connected bar with option to change keys
    c1, c2 = st.columns([5, 1])
    with c1:
        st.markdown("""
        <div style="font-family:'Space Mono',monospace; font-size:0.75rem;
             color:#10b981; padding:0.4rem 0;">
            🟢 API CONNECTED — Ready to process documents
        </div>
        """, unsafe_allow_html=True)
    with c2:
        if st.button("🔑 Change Keys", use_container_width=True):
            st.session_state["api_configured"] = False
            st.session_state["GOOGLE_API_KEY"] = None
            st.session_state["TAVILY_API_KEY"] = None
            st.rerun()
    st.divider()


# ── Sidebar — Settings & Status only (no API keys needed here) ───────────────
with st.sidebar:

    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.7rem;
         color:#00d4ff; text-transform:uppercase; letter-spacing:3px;
         margin-bottom:1rem;">
        ⚙ RAG SETTINGS
    </div>
    """, unsafe_allow_html=True)

    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100,
                           help="Smaller = more precise, Larger = more context")
    chunk_overlap = st.slider("Chunk Overlap", 0, 400, 200, 50)
    top_k = st.slider("Top K Chunks", 2, 10, 5,
                      help="How many chunks to retrieve")
    use_web_fallback = st.checkbox("🌐 Enable Web Search Fallback", value=True,
                                   help="Search web if document context is insufficient")

    st.markdown("---")

    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.7rem;
         color:#00d4ff; text-transform:uppercase; letter-spacing:3px;
         margin-bottom:1rem;">
        📊 SYSTEM STATUS
    </div>
    """, unsafe_allow_html=True)

    api_status = "🟢 CONNECTED" if st.session_state.get("api_configured") else "🔴 NO API KEY"
    doc_status = (
        f"🟢 INDEXED ({st.session_state.get('chunk_count', 0)} chunks)"
        if st.session_state.get("doc_processed")
        else "🟡 NO DOCUMENT"
    )

    st.markdown(f"""
    <div class="info-card">
        <div class="label">API Status</div>
        <div class="value">{api_status}</div>
    </div>
    <div class="info-card">
        <div class="label">Document Status</div>
        <div class="value">{doc_status}</div>
    </div>
    <div class="info-card">
        <div class="label">Vector DB</div>
        <div class="value">FAISS (Local)</div>
    </div>
    <div class="info-card">
        <div class="label">Embeddings</div>
        <div class="value">all-MiniLM-L6-v2</div>
    </div>
    <div class="info-card">
        <div class="label">LLM</div>
        <div class="value">Gemini 1.5 Pro</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main Content ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.4], gap="large")


# ── LEFT COLUMN — Document Upload ────────────────────────────────────────────
with col_left:
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.75rem;
         color:#00d4ff; text-transform:uppercase; letter-spacing:2px;
         margin-bottom:1rem;">
        📄 DOCUMENT UPLOAD & INDEXING
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PDF, TXT, or DOCX",
        type=["pdf", "txt", "docx"],
        help="Your document will be chunked, embedded, and stored in FAISS vector DB"
    )

    if uploaded_file:
        if st.button("⚡ Process & Index Document", use_container_width=True):
            if not st.session_state.get("api_configured"):
                st.error("⚠️ Please enter your Gemini API key at the top of the page first.")
            else:
                with st.spinner("🔄 Processing document..."):
                    try:
                        from rag_engine import process_document
                        vectorstore, num_chunks = process_document(
                            uploaded_file,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        st.session_state.vectorstore = vectorstore
                        st.session_state.doc_processed = True
                        st.session_state.doc_name = uploaded_file.name
                        st.session_state.chunk_count = num_chunks
                        st.success(f"✅ Document indexed! {num_chunks} chunks created.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
                        st.info("Make sure you have installed all requirements: `pip install -r requirements.txt`")

    if st.session_state.get("doc_processed"):
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-item">
                <div class="metric-num">{st.session_state.chunk_count}</div>
                <div class="metric-label">Chunks</div>
            </div>
            <div class="metric-item">
                <div class="metric-num">{top_k}</div>
                <div class="metric-label">Retrieved</div>
            </div>
            <div class="metric-item">
                <div class="metric-num">384</div>
                <div class="metric-label">Dimensions</div>
            </div>
        </div>
        <div class="info-card">
            <div class="label">Indexed Document</div>
            <div class="value">📄 {st.session_state.doc_name}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.75rem;
         color:#00d4ff; text-transform:uppercase; letter-spacing:2px;
         margin-bottom:1rem;">
        🏗️ RAG PIPELINE ARCHITECTURE
    </div>
    <div class="info-card" style="font-family:'Space Mono',monospace; font-size:0.72rem; color:#94a3b8; line-height:2;">
        📄 Document Upload<br>
        &nbsp;&nbsp;&nbsp;↓<br>
        ✂️ Text Chunking (RecursiveCharacterTextSplitter)<br>
        &nbsp;&nbsp;&nbsp;↓<br>
        🔢 Embedding (sentence-transformers/MiniLM)<br>
        &nbsp;&nbsp;&nbsp;↓<br>
        🗄️ FAISS Vector Store (cosine similarity)<br>
        &nbsp;&nbsp;&nbsp;↓<br>
        ❓ User Query → Query Embedding<br>
        &nbsp;&nbsp;&nbsp;↓<br>
        🔍 Top-K Semantic Search<br>
        &nbsp;&nbsp;&nbsp;↓<br>
        🤖 Gemini 1.5 Pro (Grounded Generation)<br>
        &nbsp;&nbsp;&nbsp;↓ (if context insufficient)<br>
        🌐 Tavily Web Search Agent<br>
        &nbsp;&nbsp;&nbsp;↓<br>
        💬 Answer + Source Citations
    </div>
    """, unsafe_allow_html=True)


# ── RIGHT COLUMN — Chat Interface ────────────────────────────────────────────
with col_right:
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.75rem;
         color:#00d4ff; text-transform:uppercase; letter-spacing:2px;
         margin-bottom:1rem;">
        💬 AI RESEARCH ASSISTANT
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.messages:
        chat_html = '<div class="chat-container">'
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_html += f"""
                <div class="chat-msg chat-user">
                    <div class="chat-label label-user">▶ YOU</div>
                    {msg["content"]}
                </div>
                """
            else:
                sources_html = ""
                if "sources" in msg and msg["sources"]:
                    sources_html = '<div class="source-box">📎 Sources: ' + \
                                   " | ".join(msg["sources"]) + '</div>'
                web_html = ""
                if msg.get("used_web"):
                    web_html = '<div class="source-box" style="color:#7c3aed; border-color:rgba(124,58,237,0.3); background:rgba(124,58,237,0.05);">🌐 Web search used as fallback</div>'
                chat_html += f"""
                <div class="chat-msg chat-assistant">
                    <div class="chat-label label-ai">🧠 AI ASSISTANT</div>
                    {msg["content"]}
                    {sources_html}
                    {web_html}
                </div>
                """
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem;
             color:#475569; font-size:0.9rem; font-family:'DM Sans',sans-serif;">
            <div style="font-size:2.5rem; margin-bottom:1rem;">🧠</div>
            <div style="font-weight:500; color:#64748b; margin-bottom:0.5rem;">
                Ready to answer questions from your documents
            </div>
            <div style="font-size:0.8rem;">
                Upload a PDF → Process it → Start asking questions
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if st.session_state.get("doc_processed"):
        st.markdown("""
        <div style="font-family:'Space Mono',monospace; font-size:0.65rem;
             color:#94a3b8; text-transform:uppercase; letter-spacing:2px;
             margin-bottom:0.6rem;">
            ⚡ QUICK QUESTIONS
        </div>
        """, unsafe_allow_html=True)

        q_col1, q_col2 = st.columns(2)
        with q_col1:
            if st.button("📋 Summarize document", use_container_width=True):
                st.session_state.pending_query = "Give me a comprehensive summary of this document"
        with q_col2:
            if st.button("🔑 Key findings", use_container_width=True):
                st.session_state.pending_query = "What are the key findings or main points?"

        q_col3, q_col4 = st.columns(2)
        with q_col3:
            if st.button("❓ Main conclusion", use_container_width=True):
                st.session_state.pending_query = "What is the main conclusion of this document?"
        with q_col4:
            if st.button("📊 Data & stats", use_container_width=True):
                st.session_state.pending_query = "What are the important data points or statistics mentioned?"

    query = st.text_input(
        "Ask anything about your document...",
        placeholder="What does this document say about...?",
        key="chat_input"
    )

    if "pending_query" in st.session_state:
        query = st.session_state.pending_query
        del st.session_state.pending_query

    col_send, col_web = st.columns([3, 1])
    with col_send:
        send_clicked = st.button("🚀 Send Query", use_container_width=True)
    with col_web:
        web_color = '#10b981' if use_web_fallback else '#64748b'
        web_label = '🌐 WEB ON' if use_web_fallback else '🌐 WEB OFF'
        st.markdown(f"""
        <div style="padding-top:0.5rem; text-align:center; font-size:0.7rem;
             font-family:'Space Mono',monospace; color:{web_color};">
            {web_label}
        </div>
        """, unsafe_allow_html=True)

    if send_clicked and query:
        if not st.session_state.get("api_configured"):
            st.error("⚠️ Please enter your Gemini API key at the top of the page.")
        elif not st.session_state.get("doc_processed"):
            st.warning("⚠️ Please upload and process a document first.")
        else:
            st.session_state.messages.append({"role": "user", "content": query})

            with st.spinner("🔍 Searching document + generating answer..."):
                try:
                    from rag_engine import answer_query
                    result = answer_query(
                        query=query,
                        vectorstore=st.session_state.vectorstore,
                        top_k=top_k,
                        use_web=use_web_fallback and bool(os.environ.get("TAVILY_API_KEY"))
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                        "used_web": result.get("used_web", False)
                    })
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error generating answer: {str(e)}\n\nMake sure you've installed requirements and your API key is valid.",
                        "sources": []
                    })
            st.rerun()