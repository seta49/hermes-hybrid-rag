# app/main.py
import sys
import os

# Memastikan root directory terdaftar di sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.services.rag_engine import process_rag_query
from app.services.vectorstore import add_documents

app = FastAPI(
    title="Hermes Hybrid RAG Engine",
    description="Local RAG system with dynamic web search fallback powered by Hermes 3",
    version="1.0.0"
)

# Schema request untuk endpoint /chat
class ChatRequest(BaseModel):
    query: str = Field(..., example="Apa itu projek SMART TASK?")

# Schema request untuk endpoint /ingest
class IngestRequest(BaseModel):
    documents: list[str] = Field(..., example=["Teks dokumen baru yang ingin dimasukkan."])
    sources: list[str] = Field(..., example=["manual_v2.txt"])

@app.get("/")
def root_status():
    """Endpoint status server sederhana"""
    return {
        "status": "online",
        "service": "Hermes Hybrid RAG Engine",
        "llm_model": "Hermes 3 (via Ollama)"
    }

@app.post("/ingest")
def ingest_documents_endpoint(payload: IngestRequest):
    """
    Endpoint untuk memasukkan dokumen baru ke Vector DB (ChromaDB)
    """
    if len(payload.documents) != len(payload.sources):
        raise HTTPException(
            status_code=400, 
            detail="Jumlah 'documents' dan 'sources' harus sama."
        )
    
    # Generate ID unik untuk setiap dokumen
    ids = [f"doc_{os.urandom(4).hex()}" for _ in range(len(payload.documents))]
    metadatas = [{"source": src} for src in payload.sources]
    
    try:
        add_documents(payload.documents, metadatas, ids)
        return {
            "status": "success",
            "message": f"Berhasil menambahkan {len(payload.documents)} dokumen ke VectorDB.",
            "inserted_ids": ids
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memasukkan dokumen: {str(e)}")

@app.post("/chat")
def chat_endpoint(payload: ChatRequest):
    """
    Endpoint utama pencarian hybrid (Local Vector Search -> Web Fallback -> Hermes 3)
    """
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query tidak boleh kosong.")
    
    try:
        result = process_rag_query(payload.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error memproses RAG query: {str(e)}")