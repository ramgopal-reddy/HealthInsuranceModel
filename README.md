# HealthInsuranceModel 🚀 – Intelligent Document Q&A

**A HackRx Submission**  
LLM-Powered Intelligent Query–Retrieval System

---

## 📌 Problem Statement

Enterprises in **insurance**, **legal**, and **HR** are drowning in massive, complex documents.  
Manually finding specific clauses or answers is:

- **Slow & Inefficient** – Experts spend hours searching for information.
- **Costly** – Wasted manpower and reliance on specialists.
- **Risky** – Misinterpretations lead to compliance failures, legal liabilities, and poor decisions.

The core challenge: **Unlock valuable structured knowledge trapped inside unstructured text**.

---

## 💡 Solution – ClausePilot

ClausePilot is an **Intelligent Retrieval-Augmented Generation (RAG) system** that transforms dense documents into interactive knowledge bases.

- Upload or provide a document URL.
- Ask **natural language questions**.
- Get **instant, accurate, explainable answers** with structured justifications.

---

## 🛠️ Tech Stack

| Component       | Technology Used             |
| --------------- | --------------------------- |
| Backend API     | FastAPI                     |
| LLM             | Google Gemini 1.5 Flash     |
| Embeddings      | Google embedding-001        |
| Vector Database | FAISS (in-memory, cached)   |
| Parsing         | PyMuPDF, python-docx        |
| Concurrency     | Python `concurrent.futures` |

---

## ⚙️ How It Works

1. **📥 Ingestion & Parsing** – Downloads document (.pdf, .docx, .eml) & extracts text.
2. **✂️ Chunking & Embedding** – Splits text into chunks & generates embeddings.
3. **🧠 Indexing** – Stores embeddings in FAISS index for fast semantic search.
4. **🔍 Retrieval** – Converts query into an embedding & retrieves relevant chunks.
5. **💬 Generation & Justification** – Sends retrieved chunks + question to Gemini LLM for a precise answer & explanation in JSON format.

---

## ✨ Key Advantages

- **High Accuracy & Explainability** – JSON answers with clause references.
- **Token Efficiency** – Only relevant chunks sent to the LLM.
- **Speed** – Cached FAISS index & concurrent processing.
- **Modularity** – Swap in other LLMs or vector DBs.

---

## 🔮 Future Enhancements

- Managed vector DB (Pinecone) for scalability.
- Interactive web UI for real-time Q&A.
- Comparative queries across multiple documents.
- Hybrid semantic + keyword search.

---

## 📂 Project Structure

```
ClausePilot/
├── main.py
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1️⃣ Clone the repo

```bash
git clone https://github.com/ramgopal-reddy/HealthInsuranceModel.git
cd HealthInsuranceModel
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Set environment variables

Create a `.env` file with your API keys:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4️⃣ Run the API

```bash
uvicorn app.main:app --reload
```

API will be live at: `http://127.0.0.1:8000` (similar)

---

## 📬 API Usage

**POST** `/hackrx/run`  
Send:

```json
{
  "document_url": "https://example.com/sample.pdf",
  "questions": [
    "What is the coverage period?",
    "What is the cancellation policy?"
  ]
}
```

Returns:

```json
[
  {
    "question": "...",
    "answer": "...",
    "justification": "..."
  }
]
```

---

## 🧑‍💻 Team –

- **Lead AI/ML Engineer** – Architected RAG pipeline & optimized embeddings.
- **Backend & API Developer** – Built scalable, low-latency FastAPI service.

---

## 📜 License

MIT License – see [LICENSE](LICENSE) for details.
