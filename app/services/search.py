# app/services/search.py
from ddgs import DDGS

def web_search(query: str, max_results: int = 3) -> str:
    """
    Melakukan pencarian ke internet via DuckDuckGo.
    
    :param query: Kata kunci pencarian
    :param max_results: Jumlah hasil maksimal yang ingin diambil
    :return: Teks gabungan berisi judul, snippet, dan URL sumber
    """
    try:
        # Inisialisasi DuckDuckGo Search
        ddgs = DDGS()
        results = ddgs.text(query, max_results=max_results)
        
        if not results:
            return ""
        
        snippets = []
        for i, r in enumerate(results, 1):
            title = r.get('title', 'No Title')
            body = r.get('body', 'No Description')
            url = r.get('href', '')
            snippets.append(f"[{i}] Title: {title}\nSnippet: {body}\nSource: {url}")
        
        # Gabungkan semua snippet menjadi satu teks panjang untuk context LLM
        return "\n\n".join(snippets)
        
    except Exception as e:
        print(f"Error saat melakukan Web Search: {str(e)}")
        return ""

# Pengujian langsung
if __name__ == "__main__":
    print("--- Testing Web Search Service ---")
    test_query = "Siapa pemenang Ballon d'Or 2024?"
    
    print(f"Mencari informasi di web untuk: '{test_query}'...\n")
    search_result = web_search(test_query, max_results=2)
    
    if search_result:
        print("Hasil Web Search:")
        print(search_result)
    else:
        print("Gagal mengambil hasil pencarian web.")