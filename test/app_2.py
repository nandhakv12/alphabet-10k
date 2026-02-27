# ============================================================
# app.py — Alphabet 10-K Analyst  (Vibrant UI)
#
# SETUP:
#   .env:
#       ANTHROPIC_API_KEY=sk-ant-...
#       LANGCHAIN_API_KEY=ls__...   (optional)
#   pip install streamlit anthropic chromadb sentence-transformers rank_bm25 langsmith python-dotenv
#   streamlit run app.py
# ============================================================

import os, re, time, uuid
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

# ── LangSmith ────────────────────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = "alphabet-10k-rag"
os.environ["LANGCHAIN_API_KEY"]    = os.environ.get("LANGCHAIN_API_KEY", "")

LANGSMITH_ON = False
ls_client    = None
try:
    from langsmith import Client as LangSmithClient
    if os.environ.get("LANGCHAIN_API_KEY", "").strip():
        ls_client    = LangSmithClient()
        LANGSMITH_ON = True
except Exception:
    pass

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alphabet 10-K Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# VIBRANT CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg:        #ffffff;
    --surface:   #f9f9fc;
    --surface2:  #f3f3f8;
    --border:    #e4e4ef;
    --border2:   #d8d8eb;
    --text:      #0f0f1a;
    --muted:     #7070a0;
    --purple:    #8b5cf6;
    --pink:      #ec4899;
    --cyan:      #0891b2;
    --green:     #059669;
    --orange:    #d97706;
    --blue:      #2563eb;
    --grad1:     linear-gradient(135deg, #8b5cf6, #ec4899);
    --grad2:     linear-gradient(135deg, #06b6d4, #3b82f6);
    --grad3:     linear-gradient(135deg, #10b981, #06b6d4);
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}
.stApp {
    background: #ffffff  !important;
    background-image:
        radial-gradient(ellipse 600px 400px at 20% 0%, rgba(139,92,246,.07) 0%, transparent 70%),
        radial-gradient(ellipse 500px 400px at 80% 100%, rgba(6,182,212,.05) 0%, transparent 70%) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Main padding ── */
.block-container { padding: 1.5rem 2rem !important; max-width: 1050px; }

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 14px !important;
    padding: .9rem 1rem !important;
}
[data-testid="metric-container"] label {
    color: var(--muted) !important;
    font-size: .65rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: .1em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* ── Sidebar buttons ── */
.stButton > button {
    background: var(--surface2) !important;
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: .78rem !important;
    font-weight: 500 !important;
    padding: .5rem .9rem !important;
    transition: all .2s ease !important;
    text-align: left !important;
    line-height: 1.4 !important;
}
.stButton > button:hover {
    background: var(--surface) !important;
    color: var(--text) !important;
    border-color: var(--purple) !important;
    box-shadow: 0 0 12px rgba(139,92,246,.2) !important;
    transform: translateX(3px) !important;
}

/* ── Text input ── */
.stTextInput input {
    background: var(--surface2) !important;
    border: 1.5px solid var(--border2) !important;
    border-radius: 16px !important;
    color: var(--text) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: .95rem !important;
    padding: .9rem 1.3rem !important;
    transition: all .2s !important;
}
.stTextInput input:focus {
    border-color: var(--purple) !important;
    box-shadow: 0 0 0 4px rgba(139,92,246,.15), 0 0 20px rgba(139,92,246,.1) !important;
}
.stTextInput input::placeholder { color: #c0c0d8 !important; }

/* ── Send button ── */
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #8b5cf6, #ec4899) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 16px !important;
    font-weight: 700 !important;
    font-size: .9rem !important;
    letter-spacing: .02em !important;
    height: 54px !important;
    transition: all .2s !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    opacity: .9 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(139,92,246,.4) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    margin: .5rem 0 !important;
}
[data-testid="stExpander"] details summary {
    background: var(--surface) !important;
    color: var(--muted) !important;
    font-size: .72rem !important;
    font-weight: 700 !important;
    font-family: 'Space Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: .1em;
    padding: .9rem 1.2rem !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stExpander"] details summary:hover { color: var(--purple) !important; }
[data-testid="stExpander"] details[open] summary {
    color: var(--purple) !important;
    border-bottom: 1px solid var(--border2) !important;
}

/* ── Success / warning / info ── */
[data-testid="stNotification"] { border-radius: 12px !important; }
.stSuccess { background: rgba(16,185,129,.1) !important; border-color: var(--green) !important; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: .8rem 0 !important; }

/* ── Captions ── */
.stCaption {
    color: var(--muted) !important;
    font-size: .68rem !important;
    font-family: 'Space Mono', monospace !important;
    letter-spacing: .06em;
    text-transform: uppercase;
}

/* ── Code ── */
code {
    font-family: 'Space Mono', monospace !important;
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 6px !important;
    padding: .1em .4em !important;
    color: var(--cyan) !important;
    font-size: .82em !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f3f3f8; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--purple); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
import os
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "alphabet_10k_db")
COLLECTION_NAME = "langchain"
MODEL           = "claude-opus-4-5"

TOOLS = [
    {
        "name": "text_search",
        "description": "Search narrative 10-K sections: risk factors, MD&A, strategy, competition.",
        "input_schema": {"type":"object","properties":{"query":{"type":"string"}},"required":["query"]},
    },
    {
        "name": "table_search",
        "description": "Search financial TABLES: income statement, balance sheet, cash flow, footnotes.",
        "input_schema": {"type":"object","properties":{"query":{"type":"string"}},"required":["query"]},
    },
]

SYSTEM_PROMPT = """You are a senior financial analyst specializing in SEC 10-K filings.
You have access to Alphabet Inc.'s 2025 10-K filing through two search tools.

Guidelines:
- Quantitative questions (numbers, ratios): use table_search first.
- Qualitative questions (risks, strategy): use text_search first.
- Comparison questions: call BOTH tools before answering.
- Always cite Source number, Item, and page in your final answer.
- Never guess numbers — say so if tools return nothing useful.
- Use markdown formatting with **bold** key numbers and clear headers."""

SAMPLE_QS = [
    "💰 What were total revenues for fiscal 2024?",
    "🤖 What are the main AI competition risks?",
    "💵 Is cash sufficient to cover long-term debt?",
    "📈 What was Google Services operating income?",
    "🧾 What are unrecognized tax benefits?",
    "🏗️ What is the capex plan for 2025?",
    "🔄 Share repurchase details for fiscal 2024?",
    "📚 What does ASU 2016-13 refer to?",
]

# ── color map for tool types ──────────────────────────────────
TOOL_COLORS  = {"text_search": "#06b6d4",  "table_search": "#f59e0b"}
TOOL_ICONS   = {"text_search": "🔍",        "table_search": "📊"}
TYPE_COLORS  = {"text": "#06b6d4",          "table": "#f59e0b"}
TYPE_ICONS   = {"text": "📝",               "table": "📊"}
TYPE_LABELS  = {"text": "TEXT",             "table": "TABLE"}

# ─────────────────────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚡ Loading retrieval engine…")
def load_retriever():
    client   = chromadb.PersistentClient(path=CHROMA_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-m3", normalize_embeddings=True)
    col = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
    res = col.get(include=["documents","metadatas"])
    def _tok(t):
        return re.sub(r"[^a-z0-9\s]"," ",t.lower()).split()
    bm25 = BM25Okapi([_tok(d) for d in res["documents"]])
    return col, bm25, res["documents"], res["metadatas"], _tok

@st.cache_resource
def get_anthropic():
    key = os.environ.get("ANTHROPIC_API_KEY","").strip()
    if not key:
        st.error("❌  ANTHROPIC_API_KEY missing. Add it to your .env file.")
        st.stop()
    return anthropic.Anthropic(api_key=key)

collection, bm25, corpus_docs, corpus_meta, _tok = load_retriever()
anthropic_client = get_anthropic()

# ─────────────────────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────────────────────
def hybrid_search(query, content_type=None, top_n=5, fetch=20, k=60):
    where = {"content_type": content_type} if content_type else None
    kw    = dict(query_texts=[query],
                 n_results=min(fetch, collection.count()),
                 include=["documents","metadatas"])
    if where:
        kw["where"] = where
    dr          = collection.query(**kw)
    dense_docs  = dr["documents"][0]
    dense_metas = dr["metadatas"][0]

    tokens = _tok(query)
    scores = bm25.get_scores(tokens)
    sparse = sorted(
        [(s,i) for i,(s,m) in enumerate(zip(scores,corpus_meta))
         if not content_type or m.get("content_type")==content_type],
        reverse=True)[:fetch]

    rrf, rrf_data = {}, {}
    for rank,(doc,meta) in enumerate(zip(dense_docs,dense_metas)):
        key           = doc[:120]
        rrf[key]      = rrf.get(key,0) + 1/(rank+k)
        rrf_data[key] = {"content":doc,"metadata":meta}
    for rank,(_,idx) in enumerate(sparse):
        key           = corpus_docs[idx][:120]
        rrf[key]      = rrf.get(key,0) + 1/(rank+k)
        rrf_data[key] = {"content":corpus_docs[idx],"metadata":corpus_meta[idx]}

    ranked = sorted(rrf, key=rrf.__getitem__, reverse=True)
    return [rrf_data[k] for k in ranked[:top_n]]


def execute_tool(name, tool_input):
    q = tool_input["query"]
    if name == "text_search":
        chunks = hybrid_search(q, content_type="text", top_n=5) or hybrid_search(q, top_n=5)
    elif name == "table_search":
        chunks = hybrid_search(q, content_type="table", top_n=5) or hybrid_search(q, top_n=5)
    else:
        return f"Unknown tool: {name}", []
    if not chunks:
        return "No relevant content found.", []
    parts = []
    for i,c in enumerate(chunks,1):
        m = c["metadata"]
        parts.append(
            f"[{i}] Item {m.get('item_number','?')} | "
            f"page {m.get('page','?')} | {m.get('content_type','?')}\n"
            f"{c['content'].strip()}")
    return "\n\n---\n\n".join(parts), chunks

# ─────────────────────────────────────────────────────────────
# LANGSMITH
# ─────────────────────────────────────────────────────────────
def ls_start(run_id, name, run_type, inputs, parent_id=None):
    if not LANGSMITH_ON: return
    try:
        ls_client.create_run(id=run_id, name=name, run_type=run_type,
                             project_name="alphabet-10k-rag",
                             inputs=inputs, parent_run_id=parent_id)
    except Exception: pass

def ls_end(run_id, outputs=None):
    if not LANGSMITH_ON: return
    try:
        ls_client.update_run(run_id, outputs=outputs or {}, end_time=time.time())
    except Exception: pass

# ─────────────────────────────────────────────────────────────
# RAG SOURCES PANEL  — pure Streamlit, styled via CSS variables
# ─────────────────────────────────────────────────────────────
def render_sources(chunks: list, trace_lines: list, elapsed: float = None):
    if not chunks and not trace_lines:
        return

    # deduplicate
    seen, unique = set(), []
    for c in chunks:
        k = c["content"][:100]
        if k not in seen:
            seen.add(k)
            unique.append(c)

    title = (
        f"🗂  {len(unique)} sources · {len(trace_lines)} searches"
        + (f" · ⚡ {elapsed}s" if elapsed else "")
    )

    with st.expander(title, expanded=True):

        # ── Search trace ──────────────────────────────────────
        if trace_lines:
            st.caption("— search queries —")
            for tl in trace_lines:
                icon  = TOOL_ICONS.get(tl["tool"], "🔧")
                c1,c2,c3 = st.columns([.04, .20, .76])
                c1.write(icon)
                c2.write(f"`{tl['tool']}`")
                c3.caption(f'"{tl["query"][:85]}"')
            st.divider()

        # ── Source chunks ─────────────────────────────────────
        st.caption("— retrieved context chunks —")

        for i, c in enumerate(unique, 1):
            m      = c["metadata"]
            ctype  = m.get("content_type", m.get("type","text"))
            item   = m.get("item_number", m.get("section","—"))
            page   = m.get("page","—")
            prev   = c["content"][:300].replace("\n"," ").strip()

            icon   = TYPE_ICONS.get(ctype, "📄")
            label  = TYPE_LABELS.get(ctype, ctype.upper())

            # Row: icon | SOURCE N | badge | item | page
            r1, r2, r3, r4, r5 = st.columns([.04, .15, .14, .15, .52])
            r1.write(icon)
            r2.write(f"**Source {i}**")
            r3.write(f"`{label}`")
            r4.caption(f"Item {item}")
            r5.caption(f"pg {page}")

            # Preview as blockquote
            st.write(f"> {prev}…")

            if i < len(unique):
                st.divider()

        # ── LangSmith ─────────────────────────────────────────
        if LANGSMITH_ON:
            st.divider()
            st.write("📡 [View trace in LangSmith →](https://smith.langchain.com)")

# ─────────────────────────────────────────────────────────────
# AGENT LOOP
# ─────────────────────────────────────────────────────────────
def run_agent(question, status_ph, answer_ph):
    messages    = [{"role":"user","content":question}]
    iteration   = 0
    all_chunks  = []
    trace_lines = []

    root_id = str(uuid.uuid4())
    ls_start(root_id, "10k_rag_agent", "chain", {"question": question})

    while iteration < 8:
        iteration += 1

        with status_ph.container():
            st.spinner(f"Thinking… iteration {iteration}")
            for tl in trace_lines:
                icon = TOOL_ICONS.get(tl["tool"], "🔧")
                st.write(f"{icon} `{tl['tool']}` → *\"{tl['query'][:70]}\"*")

        llm_id = str(uuid.uuid4())
        ls_start(llm_id, f"llm_{iteration}", "llm",
                 {"model":MODEL, "iteration":iteration}, parent_id=root_id)

        response = anthropic_client.messages.create(
            model=MODEL, max_tokens=4096,
            system=SYSTEM_PROMPT, tools=TOOLS, messages=messages)
        messages.append({"role":"assistant","content":response.content})

        ls_end(llm_id, outputs={
            "stop_reason":   response.stop_reason,
            "input_tokens":  response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        })

        if response.stop_reason == "end_turn":
            status_ph.empty()
            for block in response.content:
                if hasattr(block, "text"):
                    with answer_ph.container():
                        st.write(block.text)
                    ls_end(root_id, outputs={"answer":block.text[:400],"iterations":iteration})
                    return block.text, all_chunks, trace_lines
            ls_end(root_id, outputs={"answer":"none"})
            return "No answer returned.", all_chunks, trace_lines

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    q = block.input.get("query","")
                    trace_lines.append({"iter":iteration,"tool":block.name,"query":q})

                    with status_ph.container():
                        for tl in trace_lines:
                            icon = TOOL_ICONS.get(tl["tool"],"🔧")
                            st.write(f"{icon} `{tl['tool']}` → *\"{tl['query'][:70]}\"*")

                    tid = str(uuid.uuid4())
                    ls_start(tid, block.name, "tool",
                             {"query":q,"tool":block.name}, parent_id=root_id)

                    result_str, chunks = execute_tool(block.name, block.input)
                    all_chunks.extend(chunks)
                    ls_end(tid, outputs={"num_chunks":len(chunks)})

                    tool_results.append({
                        "type":"tool_result",
                        "tool_use_id":block.id,
                        "content":result_str,
                    })
            messages.append({"role":"user","content":tool_results})
        else:
            break

    ls_end(root_id, outputs={"answer":"max_iterations"})
    return "Max iterations reached.", all_chunks, trace_lines

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
for k, v in [("messages",[]),("total_queries",0),("total_tools",0),("total_chunks",0)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Brand header ──────────────────────────────────────────
    st.markdown("""
    <div style="padding: .2rem .5rem 1.2rem .5rem">
        <div style="
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 1.15rem; font-weight: 800;
            background: linear-gradient(135deg, #8b5cf6, #ec4899);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; letter-spacing: -.02em;
        ">Alphabet 10-K</div>
        <div style="
            font-family: 'Space Mono', monospace;
            font-size: .62rem; color: #6b6b8a;
            margin-top: 3px; letter-spacing: .08em; text-transform: uppercase;
        ">Intelligence Layer · 2025 Filing</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── LangSmith status ──────────────────────────────────────
    if LANGSMITH_ON:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(16,185,129,.12), rgba(6,182,212,.08));
            border: 1px solid rgba(16,185,129,.3); border-radius: 10px;
            padding: 8px 14px; display: flex; align-items: center; gap: 8px;
        ">
            <div style="width:8px;height:8px;background:#10b981;border-radius:50%;
                        box-shadow:0 0 8px #10b981;flex-shrink:0"></div>
            <span style="font-family:'Plus Jakarta Sans',sans-serif;font-size:.72rem;
                         color:#10b981;font-weight:600">LangSmith Connected</span>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Tracing every query")
    else:
        st.markdown("""
        <div style="
            background: rgba(107,107,138,.06); border: 1px solid #1e1e35;
            border-radius: 10px; padding: 8px 14px;
            display: flex; align-items: center; gap: 8px;
        ">
            <div style="width:8px;height:8px;background:#2e2e4a;border-radius:50%;flex-shrink:0"></div>
            <span style="font-family:'Plus Jakarta Sans',sans-serif;font-size:.72rem;
                         color:#6b6b8a;font-weight:500">LangSmith Off</span>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Add LANGCHAIN_API_KEY to .env")

    st.divider()

    # ── Stats ─────────────────────────────────────────────────
    st.markdown("<p style='font-family:Space Mono,monospace;font-size:.62rem;"
                "color:#6b6b8a;text-transform:uppercase;letter-spacing:.1em;"
                "margin:0 0 8px 0'>Index</p>", unsafe_allow_html=True)
    st.metric("Chunks", f"{collection.count():,}")

    st.markdown("<p style='font-family:Space Mono,monospace;font-size:.62rem;"
                "color:#6b6b8a;text-transform:uppercase;letter-spacing:.1em;"
                "margin:8px 0'>Session</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Queries",    st.session_state.total_queries)
    c2.metric("Tools",      st.session_state.total_tools)
    st.metric("Chunks hit", st.session_state.total_chunks)

    st.divider()

    # ── Sample questions ──────────────────────────────────────
    st.markdown("<p style='font-family:Space Mono,monospace;font-size:.62rem;"
                "color:#6b6b8a;text-transform:uppercase;letter-spacing:.1em;"
                "margin:0 0 8px 0'>Try these</p>", unsafe_allow_html=True)
    for q in SAMPLE_QS:
        if st.button(q, key=f"sq_{q[:16]}", use_container_width=True):
            st.session_state["pending_q"] = q

    st.divider()
    if st.button("🗑  Clear chat", use_container_width=True):
        for k in ["messages","total_queries","total_tools","total_chunks"]:
            st.session_state[k] = [] if k=="messages" else 0
        st.rerun()

    st.markdown(f"""
    <div style="font-family:'Space Mono',monospace;font-size:.58rem;
                color:#2e2e4a;margin-top:.5rem;line-height:1.7">
        {MODEL}<br>BM25 + ChromaDB · RRF Fusion<br>BGE-M3 Embeddings
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(139,92,246,.08) 0%, rgba(236,72,153,.06) 50%, rgba(6,182,212,.05) 100%);
    border: 1px solid rgba(139,92,246,.2);
    border-radius: 20px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
">
    <div style="position:absolute;top:-40px;right:-20px;
                width:200px;height:200px;border-radius:50%;
                background:radial-gradient(circle, rgba(139,92,246,.15), transparent 70%)">
    </div>
    <div style="position:absolute;bottom:-30px;left:60%;
                width:150px;height:150px;border-radius:50%;
                background:radial-gradient(circle, rgba(6,182,212,.1), transparent 70%)">
    </div>
    <div style="position:relative">
        <div style="
            display:flex; align-items:center; gap:10px; margin-bottom:.5rem;
        ">
            <div style="
                background:linear-gradient(135deg,#8b5cf6,#ec4899);
                border-radius:10px; padding:8px 10px; font-size:1.1rem;
            ">📊</div>
            <div>
                <div style="
                    font-family:'Plus Jakarta Sans',sans-serif;
                    font-size:1.5rem; font-weight:800; color:#f0f0ff;
                    letter-spacing:-.03em; line-height:1;
                ">Alphabet 10-K Analyst</div>
                <div style="
                    font-family:'Space Mono',monospace;
                    font-size:.68rem; color:#6b6b8a;
                    margin-top:4px; letter-spacing:.04em;
                ">HYBRID RAG · REACT AGENT · SEC INTELLIGENCE · 2025 ANNUAL REPORT</div>
            </div>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:.8rem">
            <span style="background:rgba(139,92,246,.15);border:1px solid rgba(139,92,246,.3);
                         border-radius:20px;padding:3px 12px;font-size:.7rem;
                         color:#a78bfa;font-weight:600;font-family:'Plus Jakarta Sans',sans-serif">
                BM25 + Vector Search
            </span>
            <span style="background:rgba(6,182,212,.1);border:1px solid rgba(6,182,212,.25);
                         border-radius:20px;padding:3px 12px;font-size:.7rem;
                         color:#67e8f9;font-weight:600;font-family:'Plus Jakarta Sans',sans-serif">
                RRF Fusion
            </span>
            <span style="background:rgba(236,72,153,.1);border:1px solid rgba(236,72,153,.25);
                         border-radius:20px;padding:3px 12px;font-size:.7rem;
                         color:#f9a8d4;font-weight:600;font-family:'Plus Jakarta Sans',sans-serif">
                ReAct Agent
            </span>
            <span style="background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.25);
                         border-radius:20px;padding:3px 12px;font-size:.7rem;
                         color:#6ee7b7;font-weight:600;font-family:'Plus Jakarta Sans',sans-serif">
                601 Chunks
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CHAT HISTORY
# ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])
        if msg.get("chunks"):
            render_sources(msg["chunks"], msg.get("trace_lines",[]))

# ─────────────────────────────────────────────────────────────
# INPUT BAR
# ─────────────────────────────────────────────────────────────
st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
with st.form("chat_form", clear_on_submit=True):
    cols = st.columns([.80, .20])
    user_input = cols[0].text_input(
        "q",
        placeholder="Ask anything about Alphabet's financials, risks, or strategy…",
        label_visibility="collapsed")
    submitted = cols[1].form_submit_button("Ask ✦", use_container_width=True)

if "pending_q" in st.session_state:
    user_input = st.session_state.pop("pending_q")
    submitted  = True

# ─────────────────────────────────────────────────────────────
# PROCESS
# ─────────────────────────────────────────────────────────────
if submitted and user_input.strip():
    question = user_input.strip()
    st.session_state.messages.append({"role":"user","content":question})
    st.session_state.total_queries += 1

    with st.chat_message("user"):
        st.write(question)

    status_ph = st.empty()
    answer_ph = st.empty()

    t0 = time.time()
    answer, chunks, trace_lines = run_agent(question, status_ph, answer_ph)
    elapsed = round(time.time() - t0, 1)

    st.session_state.total_tools  += len(trace_lines)
    st.session_state.total_chunks += len(chunks)

    render_sources(chunks, trace_lines, elapsed=elapsed)

    st.session_state.messages.append({
        "role":        "assistant",
        "content":     answer,
        "chunks":      chunks,
        "trace_lines": trace_lines,
    })
    st.rerun()
