from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import os
import requests
import fitz  # PyMuPDF
from docx import Document
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import json
import re
from email import message_from_bytes
from io import BytesIO

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# Bearer token for validation
API_TOKEN = "b3e3b79e7611d2b1b66a032cee801cfb7481c8b537337fd7c3c5ab6a78c5b8b7"

app = FastAPI(title="LLM Insurance Assistant")

# ---------------------------------------
# Data Models
# ---------------------------------------

class RunRequest(BaseModel):
    documents: str  # Blob URL (PDF, DOCX, EML)
    questions: list[str]

# ---------------------------------------
# Utility Functions
# ---------------------------------------

def check_token(auth_header: str):
    if not auth_header.startswith("Bearer ") or auth_header.split(" ")[1] != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing authorization token")

def extract_text_from_pdf_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("Failed to download PDF")
    doc = fitz.open(stream=BytesIO(response.content), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def extract_text_from_docx_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("Failed to download DOCX")
    doc = Document(BytesIO(response.content))
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_eml_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("Failed to download EML")
    msg = message_from_bytes(response.content)
    return msg.get_payload()

def get_text_from_blob(url):
    if url.endswith(".pdf"):
        return extract_text_from_pdf_url(url)
    elif url.endswith(".docx"):
        return extract_text_from_docx_url(url)
    elif url.endswith(".eml"):
        return extract_text_from_eml_url(url)
    else:
        raise ValueError("Unsupported file format (must be PDF, DOCX, or EML)")

def chunk_text(text, max_words=200):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def embed_text(text, task_type):
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
    response = model.generate_content(prompt)
    return parse_json_from_response(response.text)

def parse_json_from_response(response_text):
    try:
        json_str = re.search(r'{.*}', response_text, re.DOTALL).group()
        return json.loads(json_str)
    except:
        return {"error": "Failed to parse Gemini response", "raw": response_text}

# ---------------------------------------
# API ROUTES
# ---------------------------------------

@app.get("/")
def root():
    return {"message": "LLM-Powered Insurance API is running."}

@app.post("/hackrx/run")
def run_handler(request: RunRequest, authorization: str = Header(...)):
    check_token(authorization)
    
    try:
        text = get_text_from_blob(request.documents)
        chunks = chunk_text(text)
        index, embeddings = build_faiss_index(chunks)

        answers = []
        for question in request.questions:
            top_chunks = get_top_k_chunks(question, chunks, embeddings, index)
            context = "\n\n".join(top_chunks)
            decision = generate_decision(question, context)
            answers.append(decision)

        return {"answers": answers}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
