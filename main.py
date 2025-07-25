from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from urllib.parse import urlparse
from hashlib import sha256
from dotenv import load_dotenv
from email import message_from_bytes
from docx import Document
import google.generativeai as genai
import numpy as np
import faiss
import fitz  # PyMuPDF
import requests
import json
import os
import re
from io import BytesIO
import concurrent.futures

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

API_TOKEN = "b3e3b79e7611d2b1b66a032cee801cfb7481c8b537337fd7c3c5ab6a78c5b8b7"
app = FastAPI(title="LLM-Powered Query System")

# In-memory cache to speed up repeated document loads
DOC_CACHE = {}

# -------------------------
# Models
# -------------------------

class RunRequest(BaseModel):
    documents: str
    questions: list[str]

# -------------------------
# Utils
# -------------------------

def check_token(auth_header: str):
    if not auth_header.startswith("Bearer ") or auth_header.split(" ")[1] != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing authorization token")

def compute_url_hash(url: str) -> str:
    return sha256(url.encode()).hexdigest()

def extract_text_from_pdf_url(url):
    response = requests.get(url)
    response.raise_for_status()
    doc = fitz.open(stream=BytesIO(response.content), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def extract_text_from_docx_url(url):
    response = requests.get(url)
    response.raise_for_status()
    doc = Document(BytesIO(response.content))
    return "\n".join(para.text for para in doc.paragraphs)

def extract_text_from_eml_url(url):
    response = requests.get(url)
    response.raise_for_status()
    msg = message_from_bytes(response.content)
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
# Embedding & Retrieval
# -------------------------

def chunk_text(text, max_words=200):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def embed_text(text, task_type="retrieval_document"):
    return genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type=task_type
    )['embedding']

def build_faiss_index(chunks):
    embeddings = [embed_text(chunk, "retrieval_document") for chunk in chunks]
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    return index, embeddings

def get_top_k_chunks(query, chunks, embeddings, index, k=5):
    query_vector = np.array(embed_text(query, "retrieval_query")).astype("float32").reshape(1, -1)
    _, I = index.search(query_vector, k)
    return [chunks[i] for i in I[0]]

# -------------------------
# Gemini Response
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
3. Give a short justification using exact clause references.
4. Output result in the following JSON format:

{{
  "decision": "approved/rejected",
  "amount": "if mentioned",
  "justification": "clear reason based on clause"
}}
"""
    try:
        response = model.generate_content(prompt)
        return parse_json_from_response(response.text)
    except Exception as e:
        return {"error": f"Gemini error: {str(e)}"}

def parse_json_from_response(response_text):
    try:
        json_str = re.search(r'{.*}', response_text, re.DOTALL).group()
        return json.loads(json_str)
    except:
        return {"error": "Failed to parse Gemini response", "raw": response_text}

# -------------------------
# FastAPI Routes
# -------------------------

@app.get("/")
def root():
    return {"message": "LLM-Powered Insurance API is running."}

@app.post("/hackrx/run")
def run_handler(request: RunRequest, authorization: str = Header(...)):
    check_token(authorization)

    try:
        doc_id = compute_url_hash(request.documents)

        if doc_id in DOC_CACHE:
            chunks, index, embeddings = DOC_CACHE[doc_id]
        else:
            raw_text = get_text_from_blob(request.documents)
            chunks = chunk_text(raw_text)
            index, embeddings = build_faiss_index(chunks)
            DOC_CACHE[doc_id] = (chunks, index, embeddings)

        def handle_question(q):
            top_chunks = get_top_k_chunks(q, chunks, embeddings, index)
            context = "\n\n".join(top_chunks)
            return generate_decision(q, context)

        # Parallelize for multiple questions
        with concurrent.futures.ThreadPoolExecutor() as executor:
            answers = list(executor.map(handle_question, request.questions))

        return {"answers": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
