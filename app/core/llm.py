# app/core/llm.py
import requests

# Endpoint bawaan dari Ollama lokal
OLLAMA_URL = "http://localhost:11434/api/generate"

def query_hermes(prompt: str, system_prompt: str = "", model_name: str = "hermes3:3b") -> str:
    """
    Helper function untuk mengirim prompt ke Hermes 3 via REST API Ollama.
    
    :param prompt: Pertanyaan / input dari pengguna
    :param system_prompt: Instruksi peran / konteks untuk Hermes 3
    :param model_name: Nama model di Ollama (default: hermes3)
    :return: Jawaban teks dari Hermes 3
    """
    payload = {
        "model": model_name,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False  # Kita set False agar mendapat jawaban utuh langsung
    }
    
    try:
        # Kirim request HTTP POST ke Ollama
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()  # Cek jika ada HTTP error (4xx / 5xx)
        
        # Ambil respon teks dari JSON
        result = response.json()
        return result.get("response", "").strip()
        
    except requests.exceptions.ConnectionError:
        return "Error: Tidak dapat terhubung ke Ollama. Pastikan Ollama sudah berjalan (ollama serve)."
    except requests.exceptions.Timeout:
        return "Error: Request ke Hermes 3 timeout (terlalu lama memproses)."
    except Exception as e:
        return f"Error tidak terduga saat menghubungi Hermes 3: {str(e)}"

# Code di bawah ini hanya akan berjalan kalau file ini dieksekusi langsung
if __name__ == "__main__":
    print("--- Testing Helper Hermes 3 ---")
    test_system = "Kamu adalah asisten AI yang ramah dan to-the-point."
    test_prompt = "Jelaskan ringkas apa itu RAG (Retrieval-Augmented Generation) dalam 2 kalimat."
    
    print(f"Prompt: {test_prompt}\n")
    print("Menghubungi Hermes 3...")
    
    response = query_hermes(prompt=test_prompt, system_prompt=test_system)
    print("\nHasil Respon:")
    print(response)