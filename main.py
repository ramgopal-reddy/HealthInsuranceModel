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
import google.generativeai as genai
import concurrent.futures
import pickle
from typing import List, Dict, Optional
from nltk.tokenize import sent_tokenize
import nltk

# Download NLTK data (run once)
nltk.download('punkt')

# -------------------------
# Enhanced Configuration
# -------------------------

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
# Using a more capable model for better accuracy
model = genai.GenerativeModel("gemini-1.5-pro")

API_TOKEN = "b3e3b79e7611d2b1b66a032cee801cfb7481c8b537337fd7c3c5ab6a78c5b8b7"

# Cache folder for FAISS indexes
os.makedirs("faiss_indexes", exist_ok=True)

app = FastAPI(title="Enhanced LLM-Powered Insurance API")

# -------------------------
# Enhanced Models
# -------------------------

class RunRequest(BaseModel):
    documents: str
    questions: List[str]
    policy_type: Optional[str] = "health"  # Added policy type for better context

# -------------------------
# Enhanced Document Processing
# -------------------------

def semantic_chunking(text: str, max_length: int = 500) -> List[str]:
    """Improved chunking that preserves semantic boundaries"""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_length and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def clean_text(text: str) -> str:
    """Clean and normalize text for better processing"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove header/footer artifacts
    text = re.sub(r'Page \d+ of \d+', '', text)
    return text

# -------------------------
# Enhanced Embedding + Retrieval
# -------------------------

def embed_text_batch(texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
    """Batch process embeddings for efficiency"""
    # Gemini's embedding API has limits, so we process in batches
    batch_size = 10
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = genai.embed_content(
            model="models/embedding-001",
            content=batch,
            task_type=task_type
        )
        embeddings.extend(response['embedding'])
    return embeddings

def build_enhanced_faiss_index(doc_text: str) -> tuple:
    """Build a more sophisticated FAISS index with metadata"""
    doc_id = compute_hash(doc_text)
    index_file = f"faiss_indexes/{doc_id}.index"
    chunks_file = f"faiss_indexes/{doc_id}_chunks.pkl"
    metadata_file = f"faiss_indexes/{doc_id}_meta.pkl"

    if os.path.exists(index_file) and os.path.exists(chunks_file):
        index = faiss.read_index(index_file)
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
        return chunks, index, metadata

    # Enhanced preprocessing
    cleaned_text = clean_text(doc_text)
    chunks = semantic_chunking(cleaned_text)
    
    # Generate embeddings in batches for efficiency
    embeddings = embed_text_batch(chunks)
    
    # Add metadata about each chunk (position in document)
    metadata = [{"position": i, "length": len(chunk)} for i, chunk in enumerate(chunks)]
    
    # Build FAISS index
    dim = len(embeddings[0])
    index = faiss.IndexFlatIP(dim)  # Using Inner Product for better similarity
    index.add(np.array(embeddings).astype("float32"))
    
    # Save everything
    faiss.write_index(index, index_file)
    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)
    
    return chunks, index, metadata

def get_relevant_chunks(query: str, chunks: List[str], index, metadata: List[Dict], k: int = 5) -> List[str]:
    """Enhanced retrieval with score thresholding"""
    query_vec = np.array(embed_text_batch([query], "retrieval_query")[0]).astype("float32").reshape(1, -1)
    scores, indices = index.search(query_vec, k*2)  # Get more candidates for filtering
    
    # Filter by similarity score threshold
    threshold = 0.7  # Adjusted based on testing
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if score > threshold and idx < len(chunks):
            results.append({
                "text": chunks[idx],
                "score": float(score),
                "position": metadata[idx]["position"]
            })
    
    # Sort by position to maintain document flow
    results.sort(key=lambda x: x["position"])
    return [r["text"] for r in results[:k]]  # Return top k after filtering

# -------------------------
# Enhanced Decision Generation
# -------------------------

def generate_structured_decision(user_query: str, retrieved_clauses: List[str], policy_type: str) -> Dict:
    """Enhanced decision generation with structured output"""
    context = "\n\n".join([f"CLAUSE {i+1}:\n{clause}" for i, clause in enumerate(retrieved_clauses)])
    
    prompt = f"""
ROLE: You are a senior {policy_type} insurance claims adjuster with 20 years of experience.
TASK: Evaluate the claim based on the policy documents and provide a detailed decision.

POLICY DOCUMENT EXCERPTS:
{context}

CLAIM DETAILS:
{user_query}

INSTRUCTIONS:
1. Carefully analyze each relevant policy clause
2. Determine coverage eligibility
3. If covered, calculate the payable amount based on policy terms
4. If not covered, specify the exact exclusion clause
5. Provide clear justification referencing specific clauses

OUTPUT FORMAT (JSON):
{{
  "decision": "APPROVED|REJECTED|PENDING",
  "amount": float|null,
  "currency": "USD|EUR|etc"|null,
  "justification": "Detailed explanation referencing clauses",
  "clauses_used": ["list of clause numbers referenced"],
  "confidence": "HIGH|MEDIUM|LOW"
}}
"""
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,  # Lower for more deterministic outputs
                "max_output_tokens": 1024
            }
        )
        result = parse_enhanced_json(response.text)
        
        # Validation layer
        if not validate_decision(result):
            raise ValueError("Invalid decision format")
            
        return result
    except Exception as e:
        return {
            "error": str(e),
            "fallback_decision": {
                "decision": "PENDING",
                "justification": "System error - requires manual review",
                "confidence": "LOW"
            }
        }

def parse_enhanced_json(text: str) -> Dict:
    """More robust JSON parsing"""
    try:
        # Handle common formatting issues
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        
        # Ensure required fields
        if "decision" not in data:
            raise ValueError("Missing decision field")
            
        return data
    except json.JSONDecodeError:
        # Fallback parsing for malformed JSON
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        return {"error": "Invalid JSON format", "raw": text}

def validate_decision(decision: Dict) -> bool:
    """Validate the decision structure"""
    required_fields = ["decision", "justification"]
    if not all(field in decision for field in required_fields):
        return False
        
    if decision["decision"] not in ["APPROVED", "REJECTED", "PENDING"]:
        return False
        
    return True

# -------------------------
# Enhanced API Endpoints
# -------------------------

@app.post("/hackrx/run")
def enhanced_run_handler(request: RunRequest, authorization: str = Header(...)):
    check_token(authorization)

    try:
        # Get and process document
        text = get_text_from_blob(request.documents)
        chunks, index, metadata = build_enhanced_faiss_index(text)

        def process_question(q: str) -> Dict:
            # Enhanced retrieval
            context_chunks = get_relevant_chunks(q, chunks, index, metadata, k=5)
            
            # Generate structured decision
            decision = generate_structured_decision(q, context_chunks, request.policy_type)
            
            # Error handling fallback
            if "error" in decision:
                return {
                    "question": q,
                    "error": decision["error"],
                    "fallback": decision.get("fallback_decision", {}),
                    "status": "error"
                }
            return {
                "question": q,
                "response": decision,
                "status": "success"
            }

        # Process questions in parallel with error handling
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_question, request.questions))

        return {
            "results": results,
            "document_hash": compute_hash(text),
            "chunks_processed": len(chunks)
        }

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Document download error: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "solution": "Please check the document format and try again"
            }
        )