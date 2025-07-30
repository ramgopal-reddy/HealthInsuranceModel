import os, re, json, pickle, requests
from urllib.parse import urlparse
from hashlib import sha256
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from docx import Document
from email import message_from_bytes
from io import BytesIO
import fitz  # PyMuPDF
import numpy as np
import faiss
import concurrent.futures

# -------------------------
# HARDCODED KEYS
# -------------------------
HF_TOKEN = "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"
OPENROUTER_API_KEY = "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"
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
API_TOKEN = "b3e3b79e7611d2b1b66a032cee801cfb7481c8b537337fd7c3c5ab6a78c5b8b7"

os.makedirs("faiss_indexes", exist_ok=True)
app = FastAPI(title="LLM-Powered Insurance API")

# -------------------------
# Models
# -------------------------

class RunRequest(BaseModel):
    documents: str
    questions: list[str]

# -------------------------
# Auth + Utils
# -------------------------

def check_token(auth_header: str):
    if not auth_header.startswith("Bearer ") or auth_header.split(" ")[1] != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing token")

def compute_hash(text: str):
    return sha256(text.encode()).hexdigest()

# -------------------------
# Document Parsers
# -------------------------

def extract_text_from_pdf_url(url):
    r = requests.get(url); r.raise_for_status()
    doc = fitz.open(stream=BytesIO(r.content), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def extract_text_from_docx_url(url):
    r = requests.get(url); r.raise_for_status()
    doc = Document(BytesIO(r.content))
    return "\n".join(para.text for para in doc.paragraphs)

def extract_text_from_eml_url(url):
    r = requests.get(url); r.raise_for_status()
    msg = message_from_bytes(r.content)
    return msg.get_payload()

def get_text_from_blob(url):
    path = urlparse(url).path.lower()
    if path.endswith(".pdf"):
        return extract_text_from_pdf_url(url)
    elif path.endswith(".docx"):
        return extract_text_from_docx_url(url)
    elif path.endswith(".eml"):
        return extract_text_from_eml_url(url)
    else:
        raise ValueError("Unsupported file format (must be PDF, DOCX, or EML)")

# -------------------------
# Embedding + FAISS
# -------------------------

def chunk_text(text, max_words=200):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def embed_text(texts):
    if isinstance(texts, str):
        texts = [texts]
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    response = requests.post(
        f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_EMBEDDING_MODEL}",
        headers=headers,
        json={"inputs": texts},
        timeout=15
    )
    response.raise_for_status()
    return response.json()

def build_or_load_faiss(doc_text: str):
    doc_id = compute_hash(doc_text)
    index_file = f"faiss_indexes/{doc_id}.index"
    chunks_file = f"faiss_indexes/{doc_id}_chunks.pkl"

    if os.path.exists(index_file) and os.path.exists(chunks_file):
        index = faiss.read_index(index_file)
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)
        return chunks, index

    chunks = chunk_text(doc_text)
    chunks = chunks[:40]  # limit for performance

    embeddings = embed_text(chunks)
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, index_file)
    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)

    return chunks, index

def get_top_k_chunks(query, chunks, index, k=5):
    query_vec = np.array(embed_text(query)).astype("float32").reshape(1, -1)
    _, I = index.search(query_vec, k)
    return [chunks[i] for i in I[0]]

# -------------------------
# OpenRouter LLM
# -------------------------

def generate_decision(user_query, retrieved_clauses):
    prompt = f"""
You are a health insurance assistant. Based on the user query and the retrieved policy clauses, make a decision.

## User Query:
{user_query}

## Retrieved Clauses:
{retrieved_clauses}

## Task:
1. Decide if the case should be APPROVED or REJECTED.
2. If approved, specify payout amount if mentioned.
3. Justify using exact clause references.
4. Return only JSON in format:

{{
  "decision": "approved/rejected",
  "amount": "if mentioned",
  "justification": "reason based on clause"
}}
"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://render.com",
        "X-Title": "HackRxBot"
    }
    body = {
        "model": "mistralai/mixtral-8x7b",
        "messages": [
            {"role": "system", "content": "You are a helpful insurance assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body, timeout=30)
        res.raise_for_status()
        message = res.json()["choices"][0]["message"]["content"]
        return parse_json(message)
    except Exception as e:
        return {"error": str(e)}

def parse_json(text):
    try:
        json_str = re.search(r'{.*}', text, re.DOTALL).group()
        return json.loads(json_str)
    except Exception:
        return {"error": "Invalid JSON", "raw": text}

# -------------------------
# FastAPI Routes
# -------------------------

@app.get("/")
def home():
    return {
        "status": "LLM-Powered Insurance API running",
        "hf": True,
        "openrouter": True
    }

@app.post("/hackrx/run")
def run_handler(request: RunRequest, authorization: str = Header(...)):
    check_token(authorization)

    try:
        text = get_text_from_blob(request.documents)
        chunks, index = build_or_load_faiss(text)

        def handle_question(q):
            context = "\n\n".join(get_top_k_chunks(q, chunks, index))
            result = generate_decision(q, context)
            if "error" in result:
                return f"Error: {result['error']}"
            return result.get("justification", "No answer")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(handle_question, request.questions))

        return {"answers": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
