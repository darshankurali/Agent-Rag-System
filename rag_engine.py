import os
import io
import tempfile
from typing import List, Dict, Any, Tuple


# ── STEP 1: Document Loading & Chunking ─────────────────────────────────────
def load_document(uploaded_file) -> str:
    """Load text from PDF, TXT, or DOCX file."""
    file_ext = uploaded_file.name.split('.')[-1].lower()

    if file_ext == 'txt':
        return uploaded_file.read().decode('utf-8')

    elif file_ext == 'pdf':
        from langchain_community.document_loaders import PyPDFLoader
        # Save to temp file since PyPDFLoader needs a path
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        os.unlink(tmp_path)  # Clean up temp file
        return "\n\n".join([doc.page_content for doc in docs])

    elif file_ext == 'docx':
        from docx import Document
        doc = Document(io.BytesIO(uploaded_file.read()))
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

    else:
        raise ValueError(f"Unsupported file type: {file_ext}")


def chunk_text(text: str, chunk_size: int = 1000,
               chunk_overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks using RecursiveCharacterTextSplitter.

    Why overlap? — Preserves context at chunk boundaries so answers
    don't get cut off mid-sentence.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Tries to split on these separators in order
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


# ── STEP 2: Embedding & Vector Store ────────────────────────────────────────
def create_embeddings():
    """
    Load HuggingFace sentence-transformer embedding model.

    Model: all-MiniLM-L6-v2
    - Dimensions: 384
    - Size: 80MB (lightweight, runs on CPU)
    - Speed: ~14,000 sentences/second on CPU
    - Great accuracy for semantic similarity
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}  # cosine similarity
    )


def build_vectorstore(chunks: List[str], embeddings) -> Any:
    """
    Build FAISS vector store from text chunks.

    FAISS = Facebook AI Similarity Search
    - Stores dense vector representations of each chunk
    - Enables fast cosine similarity search at query time
    - All runs locally — no external API needed
    """
    from langchain_community.vectorstores import FAISS

    return FAISS.from_texts(texts=chunks, embedding=embeddings)


def process_document(uploaded_file, chunk_size: int = 1000,
                     chunk_overlap: int = 200) -> Tuple[Any, int]:
    """
    Full pipeline: Upload → Load → Chunk → Embed → Index
    Returns: (vectorstore, num_chunks)
    """
    # Load raw text
    raw_text = load_document(uploaded_file)

    # Chunk into overlapping pieces
    chunks = chunk_text(raw_text, chunk_size, chunk_overlap)

    # Create embeddings model
    embeddings = create_embeddings()

    # Build FAISS index
    vectorstore = build_vectorstore(chunks, embeddings)

    return vectorstore, len(chunks)


# ── STEP 3: Retrieval ────────────────────────────────────────────────────────
def retrieve_context(query: str, vectorstore,
                     top_k: int = 5) -> List[Dict]:
    """
    Semantic search: Embed the query → Find top-K similar chunks.

    How it works:
    1. Query text is embedded into a 384-dim vector
    2. Cosine similarity is computed against all chunk vectors in FAISS
    3. Top-K most similar chunks are returned
    """
    docs_with_scores = vectorstore.similarity_search_with_score(
        query, k=top_k)

    results = []
    for i, (doc, score) in enumerate(docs_with_scores):
        results.append({
            "chunk_id": i + 1,
            "content": doc.page_content,
            "similarity": float(score),
            "source": f"Chunk {i+1} (similarity: {score:.3f})"
        })

    return results


# ── STEP 4: LLM Generation (Grounded) ───────────────────────────────────────
def generate_answer(query: str, context_chunks: List[Dict]) -> str:
    """
    Generate grounded answer using Gemini Pro with retrieved context.

    Key: The LLM is INSTRUCTED to use only the provided context.
    This prevents hallucination and grounds the answer in the document.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.schema import HumanMessage, SystemMessage

    # Build context string from retrieved chunks
    context_text = ""
    for chunk in context_chunks:
        context_text += f"\n--- Context {chunk['chunk_id']} ---\n"
        context_text += chunk['content'] + "\n"

    # System prompt — instructs LLM to be grounded
    system_prompt = """You are an intelligent research assistant. 
Answer the user's question based STRICTLY on the provided document context.

Rules:
1. Only use information from the context below
2. If the answer is not in the context, say "I couldn't find this in the document"
3. Always cite which context section you used (e.g., "According to Context 1...")
4. Be precise, clear, and structured
5. If the context is partially relevant, use what's available and note limitations"""

    # Human prompt with context injection
    human_prompt = f"""Context from document:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context above."""

    # ✅ FIX: explicitly pass google_api_key from environment
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.2,
        max_tokens=2048,
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    response = llm(messages)
    return response.content


# ── STEP 5: Web Search Fallback (Agentic) ───────────────────────────────────
def needs_web_fallback(answer: str) -> bool:
    """
    Check if LLM answer indicates insufficient document context.
    If so, trigger the web search agent as fallback.
    """
    insufficient_phrases = [
        "couldn't find",
        "not in the document",
        "not mentioned",
        "no information",
        "I don't have",
        "not available in",
        "cannot find"
    ]
    return any(phrase.lower() in answer.lower()
               for phrase in insufficient_phrases)


def web_search_agent(query: str) -> str:
    """
    Agentic Web Search Fallback using Tavily API.

    Tavily is designed for LLM use — returns clean, relevant results
    without ads or irrelevant content.
    """
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate

    # ✅ FIX: ensure Tavily key is set in environment before tool initializes
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if not tavily_key:
        return "Web search unavailable — Tavily API key not configured."

    os.environ["TAVILY_API_KEY"] = tavily_key

    # Tavily search tool
    search_tool = TavilySearchResults(
        max_results=5,
        include_answer=True,
        include_raw_content=False
    )

    # ✅ FIX: explicitly pass google_api_key from environment
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.3,
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )

    # Agent prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research agent. Search the web to find accurate,
         up-to-date information to answer the user's question.
         Always cite your sources and be factual."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    # Create and run agent
    agent = create_tool_calling_agent(llm, [search_tool], prompt)
    executor = AgentExecutor(agent=agent, tools=[search_tool], verbose=False)

    result = executor.invoke({"input": query})
    return result["output"]


# ── MAIN ANSWER FUNCTION ─────────────────────────────────────────────────────
def answer_query(query: str, vectorstore, top_k: int = 5,
                 use_web: bool = True) -> Dict[str, Any]:
    """
    Complete RAG pipeline:
    Query → Retrieve → Generate → [Web Fallback if needed]
    """
    # Step 1: Retrieve relevant chunks
    context_chunks = retrieve_context(query, vectorstore, top_k)

    # Step 2: Generate grounded answer
    answer = generate_answer(query, context_chunks)

    # Step 3: Web fallback if document context was insufficient
    used_web = False
    if use_web and needs_web_fallback(answer):
        try:
            web_answer = web_search_agent(query)
            answer = f"{answer}\n\n**🌐 Additional information from web search:**\n{web_answer}"
            used_web = True
        except Exception as e:
            pass  # Web search failed — use document answer only

    # Step 4: Format source citations
    sources = [
        f"Page chunk {c['chunk_id']} (relevance: {1 - c['similarity']:.1%})"
        for c in context_chunks[:3]
    ]

    return {
        "answer": answer,
        "sources": sources,
        "used_web": used_web,
        "num_chunks_used": len(context_chunks)
    }