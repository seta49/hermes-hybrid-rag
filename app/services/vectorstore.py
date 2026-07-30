# app/services/vectorstore.py
import chromadb
from chromadb.utils import embedding_functions

# 1. Gunakan model embedding lokal yang ringan & cepat
# Model ini akan di-download otomatis oleh SentenceTransformers saat pertama kali dijalankan
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 2. Inisialisasi ChromaDB dengan penyimpanan lokal (persistent)
# Data akan tersimpan di folder 'chroma_data'
client = chromadb.PersistentClient(path="./chroma_data")

# 3. Buat atau ambil collection khusus dokumen lokal
collection = client.get_or_create_collection(
    name="local_knowledge_base",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"} # Menggunakan Cosine Distance
)

def add_documents(docs: list[str], metadatas: list[dict], ids: list[str]):
    """
    Memasukkan daftar dokumen ke dalam Vector Database.
    
    :param docs: Teks isi dokumen
    :param metadatas: Informasi tambahan (misal: {"source": "manual_v1.pdf"})
    :param ids: ID unik tiap dokumen (misal: ["doc_1", "doc_2"])
    """
    collection.add(
        documents=docs,
        metadatas=metadatas,
        ids=ids
    )

def search_similar_docs(query: str, n_results: int = 2):
    """
    Mencari dokumen yang paling relevan dengan query user.
    Meringkas hasil ke bentuk teks, jarak (distance), dan metadata.
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results

# File ini bisa dijalankan langsung untuk testing
if __name__ == "__main__":
    print("--- Testing Vectorstore Service ---")
    
    # Contoh data dummy buat dites
    sample_docs = [
        "Ahmad Dahlan University (UAD) is a private university located in Yogyakarta, Indonesia.",
        "The SMART TASK project is a priority management application based on the Eisenhower Matrix.",
        "Thermal repasting on laptops helps reduce CPU temperatures under heavy loads."
    ]
    sample_meta = [
        {"source": "uad_info.txt"},
        {"source": "smart_task_docs.pdf"},
        {"source": "hardware_guide.md"}
    ]
    sample_ids = ["id_1", "id_2", "id_3"]
    
    print("1. Memasukkan dokumen sampel ke ChromaDB...")
    add_documents(sample_docs, sample_meta, sample_ids)
    print("   Selesai!")
    
    # Tes pencarian
    test_query = "What is SMART TASK?"
    print(f"\n2. Melakukan pencarian untuk query: '{test_query}'")
    search_res = search_similar_docs(test_query, n_results=1)
    
    found_doc = search_res['documents'][0][0]
    found_dist = search_res['distances'][0][0]
    found_src = search_res['metadatas'][0][0]['source']
    
    print("\nHasil Pencarian Vector:")
    print(f"- Dokumen  : {found_doc}")
    print(f"- Distance : {found_dist:.4f} (Makin kecil nilainya, makin mirip/relevan)")
    print(f"- Sumber   : {found_src}")