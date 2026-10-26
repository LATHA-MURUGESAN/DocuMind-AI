import io
import hashlib

import chromadb
import ollama
import streamlit as st
from pypdf import PdfReader
from docx import Document

st.set_page_config(page_title="DocuMind | AI Knowledge Assistant", page_icon="✦", layout="wide")

CHAT_MODEL = "llama3.2"
EMBED_MODEL = "embeddinggemma"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#182230; --muted:#687386; --line:#e5eaf2; --violet:#635bff; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:radial-gradient(circle at 8% 0%,rgba(99,91,255,.08),transparent 26rem),#f5f7fb; color:var(--ink); }
[data-testid="stHeader"] { background:rgba(245,247,251,.82); }
[data-testid="stSidebar"] { background:#111827; border-right:1px solid rgba(255,255,255,.08); }
[data-testid="stSidebar"] * { color:#e8edf7; }
[data-testid="stSidebar"] .stMarkdown p { color:#aeb8ca; }
[data-testid="stSidebar"] code { background:#252f42; color:#d7d3ff; border-radius:6px; }
[data-testid="stSidebar"] [data-testid="stAlert"] { background:#202b3e; border:1px solid #344157; }
.block-container { max-width:1220px; padding-top:2.2rem; padding-bottom:3rem; }
.brand-row { display:flex; align-items:center; gap:12px; margin-bottom:1.5rem; }
.brand-mark { width:46px;height:46px;border-radius:15px;display:flex;align-items:center;justify-content:center;color:white;font-size:23px;font-weight:800;background:linear-gradient(135deg,#817aff,#5148df);box-shadow:0 9px 22px rgba(99,91,255,.28); }
.brand-name { font-family:'Manrope',sans-serif;font-size:20px;font-weight:800;color:#182230;letter-spacing:-.5px; }
.brand-sub { font-size:12px;color:#8490a3;margin-top:1px; }
.hero { padding:27px 30px;border-radius:24px;color:white;margin-bottom:22px;background:linear-gradient(118deg,#211b54 0%,#4e46d8 58%,#827cff 100%);position:relative;overflow:hidden;box-shadow:0 15px 36px rgba(65,59,180,.16); }
.hero:after { content:"";position:absolute;width:260px;height:260px;border-radius:50%;border:1px solid rgba(255,255,255,.16);right:-55px;top:-115px;box-shadow:0 0 0 28px rgba(255,255,255,.04),0 0 0 60px rgba(255,255,255,.035); }
.eyebrow { font-size:11px;font-weight:700;letter-spacing:1.7px;text-transform:uppercase;color:#c9c7ff; }
.hero h1 { font-family:'Manrope',sans-serif;font-size:clamp(26px,3.2vw,39px);line-height:1.15;margin:9px 0 10px;color:#fff;letter-spacing:-1.2px; }
.hero p { margin:0;color:#e2e1ff;font-size:14px;max-width:650px; }
.section-label { color:#697386;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1.3px;margin:5px 0 12px; }
.panel { background:rgba(255,255,255,.92);border:1px solid var(--line);border-radius:20px;padding:21px 22px;margin-bottom:18px;box-shadow:0 5px 18px rgba(28,39,60,.035); }
.panel-title { font-family:'Manrope',sans-serif;font-size:17px;font-weight:800;color:#202b3c;margin-bottom:4px; }
.panel-sub { color:#8490a3;font-size:12px;margin-bottom:15px; }
.stat-card { background:white;border:1px solid var(--line);border-radius:16px;padding:15px 17px;box-shadow:0 5px 16px rgba(28,39,60,.025);min-height:91px; }
.stat-label { color:#8490a3;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.7px; }
.stat-value { font-family:'Manrope',sans-serif;color:#202b3c;font-size:23px;font-weight:800;margin-top:6px; }
.stat-note { color:#8490a3;font-size:11px;margin-top:1px; }
.file-badge { display:inline-block;background:#f0efff;color:#5b52df;padding:5px 9px;border-radius:8px;font-size:11px;font-weight:700; }
.success-note { background:#e9fbf2;border:1px solid #c8f1dd;color:#176747;border-radius:12px;padding:11px 13px;font-size:13px; }
.hint-card { background:#f7f8ff;border:1px dashed #d5d4ff;border-radius:14px;padding:13px 15px;color:#636b80;font-size:13px; }
.stButton > button { border-radius:11px;min-height:44px;font-weight:700;border:0;background:linear-gradient(135deg,#6c63ff,#5148df);color:white;box-shadow:0 7px 16px rgba(99,91,255,.18);transition:transform .15s ease; }
.stButton > button:hover { color:white;border:0;transform:translateY(-1px); }
.stTextInput input { border-radius:12px;min-height:46px;border-color:#dfe4ee; }
[data-testid="stFileUploader"] { background:#fafbff;border:1px dashed #c9c9f7;border-radius:15px;padding:10px; }
[data-testid="stChatMessage"] { border:1px solid #e7eaf2;border-radius:16px;padding:13px 15px;background:white;margin-bottom:10px; }
.footer { text-align:center;color:#9aa3b2;font-size:11px;padding-top:22px; }
div[data-testid="stExpander"] { border:1px solid #e5eaf2;border-radius:12px;background:#fff; }
@media(max-width:768px) { .block-container{padding-top:1rem}.hero{padding:22px}.panel{padding:17px} }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_chroma_client():
    return chromadb.Client()

def extract_text(uploaded_file):
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith(".txt"):
        return raw.decode("utf-8", errors="ignore")
    if name.endswith(".docx"):
        doc = Document(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError("Unsupported file type. Upload a PDF, TXT, or DOCX file.")

def split_into_chunks(text, chunk_size=900, overlap=150):
    text = " ".join(text.split())
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks

def build_index(uploaded_file):
    text = extract_text(uploaded_file)
    if not text.strip():
        raise ValueError("No readable text was found. Scanned PDFs need OCR first.")
    chunks = split_into_chunks(text)
    if not chunks:
        raise ValueError("The uploaded file did not contain readable text.")
    file_key = hashlib.sha256(uploaded_file.name.encode("utf-8") + uploaded_file.getvalue()).hexdigest()[:16]
    collection_name = f"rag_{file_key}"
    client = get_chroma_client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name)
    result = ollama.embed(model=EMBED_MODEL, input=chunks)
    embeddings = result["embeddings"]
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"chunk": i, "source": uploaded_file.name} for i in range(len(chunks))],
    )
    return collection, len(chunks), len(text)

def answer_question(collection, question):
    query_embedding = ollama.embed(model=EMBED_MODEL, input=question)["embeddings"][0]
    results = collection.query(query_embeddings=[query_embedding], n_results=4, include=["documents","metadatas","distances"])
    docs = results["documents"][0] if results["documents"] else []
    metadata = results["metadatas"][0] if results["metadatas"] else []
    context = "\n\n---\n\n".join(f"[Chunk {m.get('chunk','?')}]\n{doc}" for doc,m in zip(docs,metadata))
    prompt = f"""Answer the user's question using ONLY the context below.
If the answer is not present in the context, say:
"I couldn't find that information in the uploaded document."
Do not invent facts.

CONTEXT:
{context}

QUESTION:
{question}
"""
    response = ollama.chat(model=CHAT_MODEL, messages=[
        {"role":"system","content":"You answer questions grounded in supplied document excerpts."},
        {"role":"user","content":prompt},
    ])
    return response["message"]["content"], docs, metadata

with st.sidebar:
    st.markdown('<div style="padding:10px 2px 22px"><div style="font-size:11px;letter-spacing:2px;color:#9aa5bb;font-weight:700">DOCUMIND / LOCAL AI</div><div style="font-family:Manrope,sans-serif;font-size:24px;font-weight:800;color:white;margin-top:6px">Knowledge Studio</div><div style="font-size:12px;color:#aeb8ca;margin-top:5px">Your documents. Your private assistant.</div></div>', unsafe_allow_html=True)
    st.markdown("### ⚙️ Model settings")
    st.markdown(f"**Chat model**  \n`{CHAT_MODEL}`")
    st.markdown(f"**Embedding model**  \n`{EMBED_MODEL}`")
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("1. Upload a document\n2. Build the searchable index\n3. Ask a question\n4. Inspect the retrieved sources")
    st.markdown("---")
    st.markdown("**Privacy**")
    st.caption("Files are processed by your local Python app and Ollama service.")
    if st.button("↻ Clear current session", use_container_width=True):
        for key in ["rag_collection","rag_file_name","rag_file_hash","rag_chunk_count","rag_char_count","rag_history"]:
            st.session_state.pop(key, None)
        st.rerun()

st.markdown("""
<div class="brand-row"><div class="brand-mark">✦</div><div><div class="brand-name">DocuMind</div><div class="brand-sub">AI knowledge assistant · runs locally</div></div></div>
<div class="hero"><div class="eyebrow">YOUR PERSONAL DOCUMENT ASSISTANT</div><h1>Knowledge, without<br>the searching.</h1><p>Upload a document, ask questions in plain language, and get answers grounded in the content you provide.</p></div>
""", unsafe_allow_html=True)

m1,m2,m3 = st.columns(3)
with m1:
    st.markdown('<div class="stat-card"><div class="stat-label">Chat model</div><div class="stat-value">Llama 3.2</div><div class="stat-note">Local answer generation</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="stat-card"><div class="stat-label">Vector search</div><div class="stat-value">ChromaDB</div><div class="stat-note">Semantic retrieval</div></div>', unsafe_allow_html=True)
with m3:
    if "rag_collection" in st.session_state:
        val, note = str(st.session_state["rag_chunk_count"]), "Text sections indexed"
    else:
        val, note = "Ready", "Upload a file to begin"
    st.markdown(f'<div class="stat-card"><div class="stat-label">Document status</div><div class="stat-value">{val}</div><div class="stat-note">{note}</div></div>', unsafe_allow_html=True)

st.write("")
left,right = st.columns([0.92,1.08], gap="large")
with left:
    st.markdown('<div class="section-label">01 / Add your knowledge</div><div class="panel"><div class="panel-title">Upload a document</div><div class="panel-sub">Choose a file to create your private searchable knowledge base.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop a PDF, DOCX, or TXT file here", type=["pdf","txt","docx"], help="Scanned PDFs need OCR.", label_visibility="collapsed")
    if uploaded_file:
        st.markdown(f'<div style="margin:12px 0 8px"><span class="file-badge">FILE READY</span> <span style="font-size:13px;font-weight:700;color:#30394b">{uploaded_file.name}</span></div>', unsafe_allow_html=True)
        st.caption(f"{uploaded_file.size:,} bytes · ready to process")
    process_clicked = st.button("✦  Process document", type="primary", use_container_width=True, disabled=uploaded_file is None)
    st.markdown("</div>", unsafe_allow_html=True)
    if process_clicked and uploaded_file:
        with st.spinner("Reading document, creating embeddings, and building the index..."):
            try:
                collection, chunk_count, char_count = build_index(uploaded_file)
                st.session_state["rag_collection"] = collection
                st.session_state["rag_file_name"] = uploaded_file.name
                st.session_state["rag_file_hash"] = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
                st.session_state["rag_chunk_count"] = chunk_count
                st.session_state["rag_char_count"] = char_count
                st.session_state["rag_history"] = []
                st.success(f"Document indexed successfully — {chunk_count} chunks created.")
            except Exception as exc:
                st.error(f"Could not process the document: {exc}")
                st.info("Check that Ollama is running and both model names match `ollama list`.")
    st.markdown('<div class="hint-card">💡 <b>Tip:</b> Ask specific questions, such as “What is the leave policy?” or “Summarize the main responsibilities.”</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-label">02 / Ask your document</div><div class="panel"><div class="panel-title">Chat with your file</div><div class="panel-sub">Answers are based on the text retrieved from your document.</div>', unsafe_allow_html=True)
    if "rag_collection" not in st.session_state:
        st.markdown('<div style="text-align:center;padding:35px 12px 28px"><div style="font-size:36px;margin-bottom:10px">💬</div><div style="font-weight:800;color:#30394b;font-size:16px">Your conversation starts here</div><div style="font-size:13px;color:#8490a3;margin-top:6px">Upload and process a document first.</div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="success-note">✓ Indexed: <b>{st.session_state["rag_file_name"]}</b> · {st.session_state["rag_chunk_count"]} searchable chunks</div>', unsafe_allow_html=True)
        with st.form("question_form", clear_on_submit=True):
            question = st.text_input("Ask a question", placeholder="e.g. What is the leave policy?", label_visibility="collapsed")
            submitted = st.form_submit_button("Ask question  →", type="primary", use_container_width=True)
        if submitted:
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Searching relevant passages and generating an answer..."):
                    try:
                        answer,docs,metadata = answer_question(st.session_state["rag_collection"], question.strip())
                        st.session_state["rag_history"].append({"question":question.strip(),"answer":answer,"docs":docs,"metadata":metadata})
                    except Exception as exc:
                        st.error(f"Could not generate an answer: {exc}")
        for item in reversed(st.session_state.get("rag_history",[])):
            with st.chat_message("assistant", avatar="🤖"):
                st.write(item["question"])
            with st.chat_message("assistant"):
                st.write(item["answer"])
                with st.expander("View retrieved source passages"):
                    for i,doc in enumerate(item["docs"]):
                        chunk_number = item["metadata"][i].get("chunk","?") if i < len(item["metadata"]) else "?"
                        st.markdown(f"**Source passage · chunk {chunk_number}**")
                        st.write(doc)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="footer">DOCUMIND · LOCAL RAG DEMO · STREAMLIT + CHROMADB + OLLAMA</div>', unsafe_allow_html=True)
