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
import google.generativeai as genai
import concurrent.futures
from dotenv import load_dotenv
from groq import Groq

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY")

# Configure APIs
genai.configure(api_key=GOOGLE_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# API token for FastAPI authentication
API_TOKEN = "b3e3b79e7611d2b1b66a032cee801cfb7481c8b537337fd7c3c5ab6a78c5b8b7"

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
# Embeddings (via Google API)
# ----------------------------
def embed_text(text, task_type="retrieval_document"):
    if isinstance(text, list):
        return [genai.embed_content(
            model="models/embedding-001",
            content=t,
            task_type=task_type
        )['embedding'] for t in text]
    else:
        return genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type=task_type
        )['embedding']

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
    query_vec = np.array(embed_text(query, "retrieval_query")).astype("float32").reshape(1, -1)
    _, I = index.search(query_vec, k)
    return [chunks[i] for i in I[0]]

# ----------------------------
# Generation (via Groq API - Llama 3 70B)
# ----------------------------
def generate_decision(user_query, retrieved_clauses):
    prompt = f"""
You are a health insurance assistant. Read the user's query and the policy clauses, and answer clearly and accurately.

### User Query:
{user_query}

### Relevant Policy Clauses:
{retrieved_clauses}

### Instructions:
- Answer clearly in one paragraph
- Do not mention clause numbers unless helpful
- If information is not found, say: "The policy document does not provide a clear answer to this question."
- Return ONLY the answer text with no JSON, no markdown, and no extra formatting
"""

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful insurance policy assistant that provides clear, concise answers based on provided documentation."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama3-70b-8192",
            temperature=0.3,
            max_tokens=1024
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# ----------------------------
# API Routes
# ----------------------------
@app.get("/")
@app.head("/")
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
            return generate_decision(q, context)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(handle_question, request.questions))

        return {"answers": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")