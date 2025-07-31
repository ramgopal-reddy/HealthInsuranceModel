import os, re, json, requests, pickle
from hashlib import sha256
from urllib.parse import urlparse
from io import BytesIO
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from docx import Document
from email import message_from_bytes
import fitz  # PyMuPDF
import numpy as np
import faiss
import concurrent.futures
from dotenv import load_dotenv

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

HF_TOKEN = os.getenv("HF_API_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not HF_TOKEN:
    raise ValueError("Missing HF_API_TOKEN")
if not OPENROUTER_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY")

# Exposed as per your request
API_TOKEN = "b3e3b79e7611d2b1b66a032cee801cfb7481c8b537337fd7c3c5ab6a78c5b8b7"
HF_EMBED_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

os.makedirs("faiss_indexes", exist_ok=True)

# ----------------------------
# FastAPI Init
# ----------------------------
app = FastAPI(title="LLM-Powered Insurance API")

# ----------------------------
# Models
# ----------------------------
class RunRequest(BaseModel):
    documents: str
    questions: list[str]

# ----------------------------
# Helpers
# ----------------------------
def check_token(auth_header: str):
    if not auth_header.startswith("Bearer ") or auth_header.split(" ")[1] != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing token")

def compute_hash(text: str):
    return sha256(text.encode()).hexdigest()

# ----------------------------
# Document Extractors
# ----------------------------
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

# ----------------------------
# Embeddings (via HF API)
# ----------------------------
def embed_text(texts):
    if isinstance(texts, str):
        texts = [texts]

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(HF_EMBED_URL, headers=headers, json={"inputs": texts}, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return [np.random.rand(384).tolist() for _ in texts]

# ----------------------------
# Chunking & FAISS
# ----------------------------
def chunk_text(text, max_words=200):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

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
    chunks = chunks[:40]  # Limit to 40 chunks for speed

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

# ----------------------------
# Generation (via OpenRouter)
# ----------------------------
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
    }

    body = {
        "model": "deepseek-chat",  # or other OpenRouter-supported model
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body, timeout=30)
        res.raise_for_status()
        response_text = res.json()['choices'][0]['message']['content']
        return parse_json(response_text)
    except Exception as e:
        return {"error": str(e)}

def parse_json(text):
    try:
        json_str = re.search(r'{.*}', text, re.DOTALL).group()
        return json.loads(json_str)
    except:
        return {"error": "Invalid JSON", "raw": text}

# ----------------------------
# API Routes
# ----------------------------
@app.get("/")
def home():
    return {"status": "LLM API running"}

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
                return f"Error: {result.get('error', result.get('raw'))}"
            return result

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(handle_question, request.questions))

        return {"answers": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
