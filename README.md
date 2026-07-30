# 🦙 Hermes 3 Hybrid RAG Engine (Local Vector Search + Web Search Fallback)

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![LLM Engine](https://img.shields.io/badge/LLM-Hermes%203%20(Ollama)-orange)
![Vector DB](https://img.shields.io/badge/VectorDB-ChromaDB-purple)

A production-ready Hybrid Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **ChromaDB**, and **Hermes 3** (via Ollama).

This system features an **intelligent threshold decision-maker**: it prioritizes ultra-fast local document retrieval via embeddings and dynamically falls back to real-time Web Search (DuckDuckGo) when local context relevancy is insufficient.

---

## 🏛️ System Architecture

```text
                  +------------------------+
                  |       User Query       |
                  +-----------+------------+
                              |
                              v
                +----------------------------+
                | Local ChromaDB Vector Store|
                +-------------+--------------+
                              |
                     (Cosine Distance)
                              |
            +-----------------+-----------------+
            |                                   |
            v                                   v
  [ Distance <= 0.50 ]                [ Distance > 0.50 ]
 (High Local Relevancy)              (Low/No Local Context)
            |                                   |
            v                                   v
  +-------------------+               +-------------------+
  |  Extract Context  |               |  Web Search API   |
  |   from Local DB   |               |    (DuckDuckGo)   |
  +---------+---------+               +---------+---------+
            |                                   |
            +-----------------+-----------------+
                              |
                              v
                  +------------------------+
                  |  Hermes 3 LLM Engine   |
                  |  (Context + Citation)  |
                  +-----------+------------+
                              |
                              v
                  +------------------------+
                  |  Structured JSON API   |
                  +------------------------+
```

## ✨ Key Features

- **Dynamic Threshold Decision Engine**: Uses Cosine Distance scoring (all-MiniLM-L6-v2) to intelligently evaluate if local context is sufficient (LOCAL_RAG) or if an internet search fallback is required (WEB_FALLBACK).
- **Local & Privacy-First Architecture**: Powered by local Hermes 3 inference via Ollama and persistent ChromaDB vector store.
- **Transparent Source Citation**: Forces Hermes 3 to explicitly cite whether the information was retrieved from local documents or real-time web results.
- **RESTful API with FastAPI**: Built-in auto-generated interactive documentation (Swagger UI).

## 🚀 Getting Started

### Prerequisites

- Install Python 3.10+
- Install & run Ollama locally:

```bash
ollama pull hermes3
```

### Installation

1. Clone the repository:

```bash
git clone https://github.com/username/hermes-hybrid-rag.git
cd hermes-hybrid-rag
```

2. Setup virtual environment & install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the API Server:

```bash
uvicorn app.main:app --reload
```

4. Open http://127.0.0.1:8000/docs in your browser to access the Interactive API Docs (Swagger UI).

## 📌 API Endpoints

### 1. `POST /chat`

Submits a query to the hybrid engine.

**Request:**

```json
{
  "query": "Apa itu projek SMART TASK?"
}
```

**Response Example:**

```json
{
  "query": "Apa itu projek SMART TASK?",
  "answer": "Projek SMART TASK adalah aplikasi manajemen prioritas tugas berbasis Eisenhower Matrix. [Sumber: smart_task_docs.pdf]",
  "source_type": "LOCAL_RAG",
  "best_distance": 0.3307,
  "sources": ["smart_task_docs.pdf"]
}
```

### 2. `POST /ingest`

Ingests custom raw documents into the local ChromaDB vector store.

## 🛠️ Tech Stack

- **LLM:** Hermes 3 (8B) via Ollama
- **Framework:** FastAPI / Pydantic
- **Vector DB:** ChromaDB
- **Embeddings:** HuggingFace sentence-transformers/all-MiniLM-L6-v2
- **Web Search:** ddgs (DuckDuckGo Search)

---
```
