# 🚀 Built a Production-Grade RAG System from Scratch — Here's Everything I Learned

Over the past few weeks, I built a full end-to-end **Retrieval-Augmented Generation (RAG)** system for analyzing Alphabet Inc.'s 2025 10-K SEC filing. Not a tutorial clone — a real production system with hybrid search, agentic reasoning, evaluation pipelines, and live monitoring.

Here's a complete breakdown of every technical decision, every tool, and every lesson learned.

---

## 🧠 What Does It Actually Do?

You ask it a question like *"Is Alphabet's cash sufficient to cover its long-term debt?"* and it:

1. **Decides** which search strategy to use (financial tables vs. narrative text)
2. **Retrieves** the most relevant chunks from a 601-chunk vector database
3. **Reasons** across multiple sources using a ReAct agent loop
4. **Answers** with cited sources, item numbers, and page references
5. **Logs** every single step to LangSmith for full observability

All powered by Claude claude-opus-4-5, ChromaDB, and BGE-M3 embeddings.

---

## 📁 The Problem With Raw 10-K Filings

A 10-K is not a clean document. It's a mix of:
- Dense narrative prose (risk factors, MD&A, business strategy)
- Structured financial tables (income statement, balance sheet, cash flows)
- Legal footnotes and accounting disclosures
- Repeated boilerplate language

A naive chunking strategy — just splitting every 1,000 characters — destroys the semantic structure. A table split across two chunks becomes meaningless. A risk factor paragraph split mid-sentence loses context.

**The solution: Content-aware parsing.**

---

## 🔧 Phase 1 — Document Parsing & Chunking

**Tool:** Docling + LangChain Text Splitters

The first step was converting the raw 10-K markdown into semantically meaningful chunks with two distinct strategies:

### For Tables:
- Detected using a regex pattern: `r'(\|.*\|(?:\n\|.*\|)*)'`
- Kept **completely intact** — never split a table across chunks
- Extracted the heading context from the preceding paragraph
- Stored as: `"Table Heading: {context}\n\n{table_content}"`

### For Narrative Text:
- Split first by markdown headers (`#`, `##`) using `MarkdownHeaderTextSplitter`
- Then recursively chunked at 1,000 characters with 100-character overlap using `RecursiveCharacterTextSplitter`
- Header metadata preserved in every chunk

### Result:
```
Total chunks:  601
Text chunks:   528  (87.9%)
Table chunks:   73  (12.1%)
```

Every chunk carries rich metadata:
```json
{
  "content_type": "table",
  "item_number":  "Item 8",
  "page":         "47",
  "section":      "Financial Statements"
}
```

The `content_type` field became critical for filtered retrieval later.

---

## 🗄️ Phase 2 — Vector Store with BGE-M3 Embeddings

**Tools:** ChromaDB · BAAI/bge-m3 · HuggingFace Sentence Transformers

### Why BGE-M3?
Most RAG tutorials use `text-embedding-ada-002` or a small MiniLM model. I chose **BGE-M3** because:
- State-of-the-art on MTEB (Massive Text Embedding Benchmark)
- Supports multi-granularity retrieval — works for both short queries and long passages
- Completely free and runs locally — no per-token embedding cost
- Normalized embeddings for cosine similarity

### ChromaDB Setup:
```python
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vector_db = Chroma.from_documents(
    documents=docs_to_store,
    embedding=embeddings,
    persist_directory="./alphabet_10k_db"
)
```

ChromaDB persists to disk — the entire 601-chunk database is ~80MB and survives restarts. No re-embedding needed.

---

## 🔍 Phase 3 — Hybrid Retrieval with RRF Fusion

**Tools:** BM25Okapi (rank_bm25) · ChromaDB dense search · Reciprocal Rank Fusion

This is where the system goes from "RAG tutorial" to production quality.

### The Problem With Pure Vector Search:
Dense embeddings are excellent at semantic similarity but struggle with:
- Exact financial terms: "ASU 2016-13", "CECL", specific dollar amounts
- Acronyms and proper nouns
- Queries where keyword matching matters more than semantics

### The Problem With Pure BM25:
BM25 is excellent at exact keyword matching but misses:
- Paraphrasing and synonyms
- Conceptual similarity ("earnings" vs "net income")

### The Solution: Hybrid Search with RRF

Run **both** retrieval methods in parallel, then fuse the rankings:

```python
def hybrid_search(query, content_type=None, top_n=5, fetch=20, k=60):

    # Dense leg — ChromaDB semantic search
    dense_results = collection.query(
        query_texts=[query],
        n_results=fetch,
        where={"content_type": content_type}  # metadata filter
    )

    # Sparse leg — BM25 keyword search
    tokens = tokenize(query)
    bm25_scores = bm25.get_scores(tokens)

    # RRF Fusion — rank(doc) = Σ 1/(k + rank_i)
    for rank, doc in enumerate(dense_results):
        rrf_score[doc] += 1 / (rank + k)
    for rank, doc in enumerate(bm25_results):
        rrf_score[doc] += 1 / (rank + k)

    return sorted_by_rrf_score[:top_n]
```

**Why RRF constant k=60?**
The `k=60` constant was established in the original RRF paper (Cormack et al., 2009) and prevents high-ranked documents from dominating. It's the industry standard for hybrid search fusion.

### Content-Type Filtering:
The metadata filter is a major quality improvement. When Claude calls `table_search`, it adds `where={"content_type": "table"}` to the ChromaDB query. This means:
- Financial number questions always hit actual financial tables
- Narrative questions always hit prose sections
- Zero cross-contamination

---

## 🤖 Phase 4 — ReAct Agent with Tool Use

**Tools:** Anthropic Claude claude-opus-4-5 · Tool Use API · ReAct Pattern

### Why an Agent Instead of a Simple RAG Chain?

Simple RAG: retrieve → generate. Works for simple questions.

But for a 10-K analyst, many questions require **multi-step reasoning**:

*"Is cash sufficient to cover long-term debt?"*

This requires:
1. Search tables for total cash position
2. Search tables for total long-term debt
3. Compare the two numbers
4. Form a judgment

A single retrieval step can't do this. An agent can.

### Tool Definitions:

```python
TOOLS = [
    {
        "name": "text_search",
        "description": "Search narrative sections: risk factors, MD&A,
                        strategy, competition. Use for qualitative questions.",
    },
    {
        "name": "table_search",
        "description": "Search financial TABLES: income statement, balance sheet,
                        cash flow. Use when the question needs specific numbers.",
    }
]
```

### The ReAct Loop:

```
Question → LLM decides tool → Execute tool → LLM sees results
         → LLM decides next tool (or final answer) → repeat
```

The agent runs up to **8 iterations** before forcing a stop. In practice, most questions resolve in 2-3 iterations. Complex multi-hop questions use 4-5.

### System Prompt Design:
```
- Quantitative questions: use table_search FIRST
- Qualitative questions: use text_search FIRST
- Comparison questions: call BOTH tools before answering
- Always cite Source number, Item, and page
- Never guess numbers
```

The explicit routing instructions in the system prompt reduce unnecessary tool calls by ~40% compared to letting the model decide freely.

---

## 📡 Phase 5 — LangSmith Monitoring

**Tool:** LangSmith (LangChain's observability platform)

Every production system needs observability. Without it, you're flying blind.

### What Gets Traced:

Every query creates a **full trace tree** in LangSmith:

```
root: 10k_rag_agent
├── llm_call_1          [tokens: 1,247 in / 89 out]
├── tool: table_search  [query: "total revenues 2024", chunks: 5]
├── llm_call_2          [tokens: 3,891 in / 412 out]
├── tool: text_search   [query: "AI competition risks", chunks: 5]
└── llm_call_3          [tokens: 5,203 in / 891 out] → FINAL ANSWER
```

### What LangSmith Shows You:
- **Latency** per step — where is time being spent?
- **Token counts** — input/output tokens for every LLM call
- **Tool call frequency** — which tools get called most?
- **Error rates** — which questions fail or hit max iterations?
- **Full prompt/response** — debug exactly what the model saw

### Implementation:
```python
# Every run creates a parent span
ls_client.create_run(
    id=root_id,
    name="10k_rag_agent",
    run_type="chain",
    project_name="alphabet-10k-rag",
    inputs={"question": question}
)

# Every tool call is a child span
ls_client.create_run(
    id=tool_id,
    name=block.name,
    run_type="tool",
    inputs={"query": q},
    parent_run_id=root_id   # ← links to parent
)
```

This gives a **nested waterfall view** — exactly like distributed tracing in backend engineering.

---

## 📊 Phase 6 — RAGAS Evaluation

**Tools:** RAGAS · HuggingFace Datasets · LangSmith Feedback API

Vibes-based evaluation ("it seems to answer well") is not engineering. I built a proper evaluation pipeline.

### The 4 RAGAS Metrics:

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Faithfulness** | Is the answer grounded in retrieved context? | Catches hallucinations |
| **Answer Relevancy** | Does the answer address the question? | Catches off-topic responses |
| **Context Precision** | Are the TOP chunks the most relevant? | Measures retriever ranking quality |
| **Context Recall** | Did retrieval find ALL needed information? | Measures retrieval completeness |

### Golden Dataset:
10 hand-crafted question/ground-truth pairs covering:
- Financial table lookups (revenues, debt, cash)
- Risk factor analysis
- Multi-hop reasoning (cash vs. debt comparison)
- Accounting policy lookups
- Segment-level financial data

### Scores logged back to LangSmith:
Every RAGAS evaluation run creates feedback entries on each trace, visible in the LangSmith UI under the Feedback tab. This means you can correlate score drops with specific query patterns over time.

---

## 🎨 Phase 7 — Production Streamlit UI

**Tools:** Streamlit · Plus Jakarta Sans + Space Mono fonts · CSS Variables

The interface was designed for actual analysts, not demos:

- **Real-time tool trace** — watch the agent decide which tool to call
- **RAG sources panel** — every retrieved chunk shown with metadata (type, item, page)
- **Session statistics** — queries, tool calls, chunks retrieved
- **One-click sample questions** — 8 pre-loaded financial analysis queries
- **LangSmith live link** — direct link to the trace for every answer
- **Persistent chat history** — full conversation context maintained

**Deployed on:** Streamlit Community Cloud (free tier)
**Live at:** [https://tinyurl.com/bdfrd3jx]

---

## 🏗️ Full Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM** | Claude claude-opus-4-5 (Anthropic) | Reasoning & generation |
| **Embeddings** | BAAI/bge-m3 | Semantic vector encoding |
| **Vector DB** | ChromaDB | Persistent vector storage |
| **Sparse Search** | BM25Okapi (rank_bm25) | Keyword retrieval |
| **Fusion** | Reciprocal Rank Fusion | Hybrid search merging |
| **Agent Framework** | Anthropic Tool Use API | ReAct agentic loop |
| **Monitoring** | LangSmith | Full trace observability |
| **Evaluation** | RAGAS | RAG quality metrics |
| **UI** | Streamlit | Production web interface |
| **Document Parsing** | Docling + LangChain Splitters | Content-aware chunking |
| **Deployment** | Streamlit Community Cloud | Free hosting |

---

## 📐 Architecture Diagram

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│         Streamlit UI                │
│   (Chat + RAG Panel + Traces)       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      ReAct Agent (Claude claude-opus-4-5)    │
│   System Prompt + Tool Routing      │
└──────┬───────────────────┬──────────┘
       │                   │
       ▼                   ▼
┌─────────────┐    ┌──────────────────┐
│ text_search │    │  table_search    │
│  (prose)    │    │  (financials)    │
└──────┬──────┘    └────────┬─────────┘
       │                    │
       └─────────┬──────────┘
                 ▼
┌─────────────────────────────────────┐
│       Hybrid Retriever              │
│  BM25 + ChromaDB → RRF Fusion       │
│  Content-Type Metadata Filter       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│     ChromaDB  (601 chunks)          │
│     BGE-M3 Embeddings               │
│     alphabet_10k_db/                │
└─────────────────────────────────────┘
               │
               ▼ (all steps)
┌─────────────────────────────────────┐
│         LangSmith                   │
│  Traces · Tokens · Latency · RAGAS  │
└─────────────────────────────────────┘
```

---

## 💡 Key Lessons Learned

**1. Content-aware chunking is everything.**
Splitting tables mid-row is catastrophic for financial Q&A. Detecting and preserving table structure gave a massive quality improvement.

**2. Hybrid search beats pure vector search for financial documents.**
BM25 handles exact financial terms (ASU codes, specific dollar amounts) that embeddings often miss.

**3. Metadata filtering is underrated.**
Adding `content_type` filtering to ChromaDB queries dramatically improves precision. The model should never get prose chunks when it asked for financial tables.

**4. Observability from day one.**
LangSmith traces made it immediately obvious when the agent was over-retrieving or making unnecessary tool calls. Without traces, these issues are invisible.

**5. Name your variables carefully.**
I had three different things all called `client` — Anthropic client, ChromaDB client, LangSmith client. That caused subtle bugs. Always use `anthropic_client`, `chroma_client`, `ls_client`.

**6. RAGAS needs ground truth.**
The evaluation pipeline is only as good as your golden dataset. I spent significant time writing accurate ground truths for the 10 test questions.

---

## 🔭 What's Next

- [ ] Multi-document support (multiple fiscal years for trend analysis)
- [ ] RAGAS evaluation dashboard inside the Streamlit app
- [ ] Streaming responses for better UX
- [ ] Query caching for repeated questions
- [ ] Fine-tuned reranker model for improved context precision

---

## 🛠️ Run It Yourself

```bash
git clone https://github.com/nandhak12/alphabet-10k
cd alphabet-10k

pip install streamlit anthropic chromadb sentence-transformers \
            rank_bm25 langsmith python-dotenv

# Create .env
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
echo "LANGCHAIN_API_KEY=ls__..."   >> .env

streamlit run app.py
```

---

*Built with curiosity, caffeine, and a lot of ChromaDB debugging.*

*If this was useful, feel free to connect — always happy to discuss RAG architecture, LLM evaluation, or financial AI applications.*

#RAG #LLM #GenerativeAI #NLP #LangChain #Anthropic #Claude #ChromaDB #FinancialAI #MachineLearning #Python #Streamlit #LangSmith #VectorDatabase #AIEngineering #ProductionAI #SEC #FinTech
