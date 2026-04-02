# 🧠 Agentic RAG System — Personal AI Research Assistant

> Upload any document → Ask questions in natural language → Get grounded answers with citations + web search fallback

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green)
![Gemini](https://img.shields.io/badge/Gemini-1.5_Pro-orange)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-purple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

---

## 🏗️ Architecture

```
PDF/TXT/DOCX Upload
        ↓
RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
        ↓
HuggingFace Embeddings (all-MiniLM-L6-v2 → 384 dimensions)
        ↓
FAISS Vector Store (cosine similarity indexing)
        ↓
User Query → Query Embedding
        ↓
Top-K Semantic Search (k=5)
        ↓
Gemini 1.5 Pro (Grounded Generation with context injection)
        ↓ (if context insufficient ↓)
Tavily Web Search Agent (autonomous web fallback)
        ↓
Grounded Answer + Source Citations
```

---

## ⚙️ Setup

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAMEdarshankurali/agentic-rag-system
cd agentic-rag-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Get API Keys (both free)
# Gemini: https://aistudio.google.com/app/apikey
# Tavily: https://tavily.com (free tier)

# 4. Run the app
streamlit run app.py
```

---

## 🔑 API Keys Required

| Service | Purpose | Cost | Link |
|---------|---------|------|------|
| Google Gemini | LLM for answer generation | Free | [Get Key](https://aistudio.google.com) |
| Tavily Search | Web search fallback agent | Free (1000 req/month) | [Get Key](https://tavily.com) |

Enter both keys in the sidebar after launching the app.

---

## 🚀 Features

- **📄 Multi-format Support** — PDF, TXT, DOCX
- **🔢 Semantic Chunking** — Recursive splitting with configurable overlap
- **🗄️ FAISS Vector DB** — Lightning-fast cosine similarity search
- **🤖 Grounded Generation** — Answers strictly from document context
- **📎 Source Citations** — Shows which chunks were used
- **🌐 Agentic Web Fallback** — Searches web when document is insufficient
- **⚙️ Configurable** — Adjust chunk size, overlap, and top-K from UI

---

## 📁 Project Structure

```
agentic-rag-system/
├── app.py              ← Streamlit UI (main entry point)
├── rag_engine.py       ← Core RAG pipeline logic
├── requirements.txt    ← Python dependencies
└── README.md
```

---

## 🧠 Key Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| **RAG** | FAISS retrieval + Gemini generation |
| **Embeddings** | all-MiniLM-L6-v2 (384-dim) |
| **Vector Search** | Cosine similarity via FAISS |
| **Agentic AI** | Tavily web search agent as autonomous fallback |
| **Grounding** | System prompt enforces context-only answers |
| **Chunking Strategy** | Recursive splitting with overlap |

---

## 💬 Example Questions to Try

After uploading a research paper or report:
- *"What is the main conclusion of this document?"*
- *"Summarize the methodology section"*
- *"What are the key findings?"*
- *"What data or statistics are mentioned?"*

---

## 🏷️ Tech Stack

```
LLM           : Google Gemini 1.5 Pro
Embeddings    : sentence-transformers/all-MiniLM-L6-v2
Vector DB     : FAISS (Facebook AI Similarity Search)
Framework     : LangChain
Web Agent     : Tavily Search API
UI            : Streamlit
Language      : Python 3.10+
```


