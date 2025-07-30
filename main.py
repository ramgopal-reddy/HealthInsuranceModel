import os, re, json
from urllib.parse import urlparse
from hashlib import sha256
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from docx import Document
from email import message_from_bytes
from dotenv import load_dotenv
from io import BytesIO
import requests
import fitz  # PyMuPDF
import numpy as np
import faiss
import concurrent.futures

# Load environment variables
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not HF_API_TOKEN or not OPENROUTER_API_KEY:
    raise ValueError("Missing HF_API_TOKEN or OPENROUTER_API_KEY")

# App and token
app = FastAPI(title="LLM Insurance API")
API_TOKEN = "b3e3b79e7611d2b1b66a032cee801cfb7481c8b537337fd7c3c5ab6a78c5b8b7"
os.makedirs("faiss_indexes", exist_ok=True)

# -------------------------
# Models
# -------------------------
class RunRequest(BaseModel):
    documents: str
    questions: list[str]

# -------------------------
# Auth & Utils
# -------------------------
def check_token(auth_header: str):
    if not auth_header.startswith("Bearer ") or auth_header.split(" ")[1] != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing token")

def compute_hash(text: str):
    return sha256(text.encode()).hexdigest()

# -------------------------
# Document Parsing
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
# Embedding via Hugging Face API
# -------------------------
def embed_text(text):
    url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    response = requests.post(url, headers=headers, json={"inputs": text})
    if response.status_code != 200:
        raise Exception("Hugging Face Embedding API failed")
    return np.array(response.json()[0], dtype="float32")

def chunk_text(text, max_words=200):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def build_or_load_faiss(doc_text):
    doc_id = compute_hash(doc_text)
    index_file = f"faiss_indexes/{doc_id}.index"
    chunks_file = f"faiss_indexes/{doc_id}.txt"

    if os.path.exists(index_file) and os.path.exists(chunks_file):
        index = faiss.read_index(index_file)
        with open(chunks_file, "r") as f:
            chunks = [line.strip() for line in f.readlines()]
        return chunks, index

    chunks = chunk_text(doc_text)[:40]
    embeddings = [embed_text(c) for c in chunks]
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    faiss.write_index(index, index_file)
    with open(chunks_file, "w") as f:
        f.write("\n".join(chunks))
    return chunks, index

def get_top_k_chunks(query, chunks, index, k=5):
    query_vec = embed_text(query).reshape(1, -1)
    _, I = index.search(query_vec, k)
    return [chunks[i] for i in I[0]]

# -------------------------
# Generate with OpenRouter + DeepSeek
# -------------------------
def generate_decision(query, context):
    prompt = f"""
You are a health insurance assistant.

## User Query:
{query}

## Retrieved Clauses:
{context}

## Task:
1. Decide if the case is APPROVED or REJECTED.
2. If approved, state payout amount (if any).
3. Justify decision using clause content.
4. Reply ONLY with JSON:
{{
  "decision": "approved/rejected",
  "amount": "if mentioned",
  "justification": "brief explanation"
}}
"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    if r.status_code != 200:
        return {"error": f"LLM call failed: {r.text}"}
    try:
        response_text = r.json()['choices'][0]['message']['content']
        return json.loads(re.search(r"{.*}", response_text, re.DOTALL).group())
    except:
        return {"error": "Failed to parse LLM JSON", "raw": r.text}

# -------------------------
# API Routes
# -------------------------
@app.get("/")
def root():
    return {"status": "LLM-Powered Insurance API running"}

@app.post("/hackrx/run")
def run_handler(request: RunRequest, authorization: str = Header(...)):
    check_token(authorization)
    try:
        text = get_text_from_blob(request.documents)
        chunks, index = build_or_load_faiss(text)

        def process(q):
            context = "\n\n".join(get_top_k_chunks(q, chunks, index))
            res = generate_decision(q, context)
            if "error" in res:
                return f"Error: {res['error']}"
            return res.get("justification", "No answer.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            answers = list(executor.map(process, request.questions))

        return {"answers": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
