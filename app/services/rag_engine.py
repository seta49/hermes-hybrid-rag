import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))



# app/services/rag_engine.py
from app.services.vectorstore import search_similar_docs
from app.services.search import web_search
from app.core.llm import query_hermes

# Ambang batas jarak kemainginan Cosine (Distance)
# Nilai di bawah 0.50 dianggap relevansi tinggi
DISTANCE_THRESHOLD = 0.50

def process_rag_query(query: str) -> dict:
    """
    Memproses query user dengan logika fallback RAG yang dinamis.
    """
    # 1. Cari dulu di Vector Store lokal
    local_results = search_similar_docs(query, n_results=2)
    
    docs = local_results.get('documents', [[]])[0]
    distances = local_results.get('distances', [[]])[0]
    metadatas = local_results.get('metadatas', [[]])[0]
    
    source_type = "LOCAL_RAG"
    context = ""
    used_sources = []
    best_distance = distances[0] if distances else 1.0

    # 2. Cek apakah dokumen lokal cukup relevan berdasarkan Threshold
    if distances and best_distance <= DISTANCE_THRESHOLD:
        # Pake data lokal
        context_list = []
        for doc, meta in zip(docs, metadatas):
            src_name = meta.get('source', 'Local Document')
            context_list.append(f"[Source: {src_name}]\n{doc}")
            used_sources.append(src_name)
        context = "\n\n".join(context_list)
    else:
        # Fallback ke pencarian internet
        source_type = "WEB_FALLBACK"
        search_results = web_search(query, max_results=2)
        if search_results:
            context = search_results
            used_sources = ["DuckDuckGo Web Search"]
        else:
            context = "Tidak ditemukan konteks yang relevan di internet."
            used_sources = ["None"]

    # 3. Buat System Prompt untuk Hermes 3
    system_prompt = (
        "Kamu adalah asisten cerdas yang jujur dan akurat. "
        "Jawab pertanyaan pengguna secara ringkas berdasarkan KONTEKS yang diberikan di bawah ini. "
        "Jika informasi tidak ada di dalam konteks, katakan bahwa kamu tidak tahu secara jujur. "
        "Selalu sebutkan sumber informasi yang kamu pakai di akhir jawaban.\n\n"
        f"--- KONTEKS ---\n{context}"
    )

    prompt = f"Pertanyaan: {query}"
    
    # 4. Minta Hermes 3 menghasilkan jawaban
    answer = query_hermes(prompt, system_prompt=system_prompt)

    return {
        "query": query,
        "answer": answer,
        "source_type": source_type,
        "best_distance": round(best_distance, 4) if distances else None,
        "sources": list(set(used_sources))
    }

# Testing RAG Engine
if __name__ == "__main__":
    print("--- Testing Hybrid RAG Engine ---")
    
    # Tes 1: Query yang datanya ADA di ChromaDB lokal kita (dari step 3)
    query_lokal = "Apa itu projek SMART TASK?"
    print(f"\n1. Menguji Query Lokal: '{query_lokal}'")
    res1 = process_rag_query(query_lokal)
    print(f"- Decision  : {res1['source_type']} (Distance: {res1['best_distance']})")
    print(f"- Sources   : {res1['sources']}")
    print(f"- Answer    :\n{res1['answer']}\n")

    print("=" * 60)

    # Tes 2: Query yang datanya TIDAK ADA di lokal (Harus Fallback ke Web Search)
    query_web = "Siapa juara Champions League tahun 2024?"
    print(f"\n2. Menguji Query Fallback Web: '{query_web}'")
    res2 = process_rag_query(query_web)
    print(f"- Decision  : {res2['source_type']} (Distance: {res2['best_distance']})")
    print(f"- Sources   : {res2['sources']}")
    print(f"- Answer    :\n{res2['answer']}")