import json
import math
import os
import numpy as np
from src.search.nlp import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_resources(index_path='data/index.json', pubs_path='data/scraper_results.json'):
    with open(index_path, 'r', encoding='utf-8') as f:
        index = json.load(f)
    with open(pubs_path, 'r', encoding='utf-8') as f:
        publications = json.load(f)
    return index, publications

def get_custom_ranking(query, index, total_docs):
    """
    Calculates TF-IDF scores manually and ranks documents using Cosine Similarity.
    """
    query_tokens = preprocess(query)
    if not query_tokens:
        return []

    # Build Query Vector
    # We treat the query as a 'mini-document'
    query_vector = {}
    for token in query_tokens:
        query_vector[token] = query_vector.get(token, 0) + 1

    scores = {} # Document scores: {url: dot_product_sum}
    
    # Calculate scores using the formula: TF * log10(N/DF)
    for token, q_tf in query_vector.items():
        if token in index:
            df = index[token]['df']
            idf = math.log10(total_docs / df)
            
            for url, doc_tf in index[token]['postings'].items():
                # Simple Vector Space Model ranking
                tfidf_score = doc_tf * idf
                scores[url] = scores.get(url, 0) + (tfidf_score * (q_tf * idf))

    # Sort results by score descending
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results

def get_sklearn_ranking(query, publications):
    """
    Uses Scikit-Learn to calculate TF-IDF and Cosine Similarity for comparison.
    """
    # Prepare corpus
    corpus = [f"{p.get('title', '')} {p.get('abstract', '')}" for p in publications]
    urls = [p.get('url') for p in publications]
    
    vectorizer = TfidfVectorizer(stop_words=None) # We already preprocessed, or let sklearn handle it
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    query_vec = vectorizer.transform([query])
    
    # Calculate cosine similarity between query and all docs
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Rank them
    ranked_indices = cosine_sim.argsort()[::-1]
    results = [(urls[i], cosine_sim[i]) for i in ranked_indices if cosine_sim[i] > 0]
    return results


# ---------------------------------------------------------------------------
# REQ-B40 — Document-Document Similarity Matrix
# ---------------------------------------------------------------------------

def build_similarity_matrix(
    publications: list[dict],
    output_dir: str = 'data',
    top_k: int | None = None,
) -> tuple[list[str], np.ndarray]:
    """
    Builds a Document × Document cosine-similarity matrix using TF-IDF vectors
    and persists it to disk (REQ-B40).

    Each cell (i, j) holds the cosine similarity between document i and
    document j, where 1.0 means identical content and 0.0 means no overlap.
    The diagonal is always 1.0 (a document is fully similar to itself).

    The result is saved in two formats:
    - ``similarity_matrix.npy``  – raw NumPy array for fast numeric access.
    - ``similarity_matrix.json`` – human-readable sparse representation that
      lists, for every document, its most similar neighbours (score > 0),
      sorted by descending similarity.  If *top_k* is given, only the top-k
      neighbours per document are stored.

    Args:
        publications: List of publication dicts (must have 'url', 'title',
                      'abstract' keys).
        output_dir:   Directory where output files are written.
        top_k:        Optional limit on the number of neighbours stored in the
                      JSON output.  Has no effect on the .npy matrix.

    Returns:
        (doc_ids, matrix) where doc_ids is the ordered list of URLs and
        matrix is the N×N NumPy similarity array.
    """
    corpus = [
        f"{p.get('title', '')} {p.get('abstract', '')}"
        for p in publications
    ]
    doc_ids = [p.get('url') for p in publications]

    if not corpus:
        print("[REQ-B40] No publications found — similarity matrix not built.")
        return doc_ids, np.empty((0, 0))

    # Build TF-IDF matrix (docs × terms)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Compute full N×N cosine similarity matrix
    sim_matrix = cosine_similarity(tfidf_matrix)  # shape: (N, N)

    os.makedirs(output_dir, exist_ok=True)

    # --- Persist as NumPy binary (.npy) for fast programmatic access ---
    npy_path = os.path.join(output_dir, "similarity_matrix.npy")
    np.save(npy_path, sim_matrix)

    # --- Persist as sparse JSON for human-readable neighbour lookup ---
    sparse: dict[str, list[dict]] = {}
    for i, url in enumerate(doc_ids):
        row = sim_matrix[i]
        # Exclude the document itself (diagonal) and zero-similarity docs
        neighbours = [
            {"url": doc_ids[j], "score": round(float(row[j]), 6)}
            for j in range(len(doc_ids))
            if j != i and row[j] > 0.0
        ]
        # Sort by descending similarity
        neighbours.sort(key=lambda x: x["score"], reverse=True)
        if top_k is not None:
            neighbours = neighbours[:top_k]
        sparse[url] = neighbours

    json_path = os.path.join(output_dir, "similarity_matrix.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({"doc_ids": doc_ids, "neighbours": sparse}, f, ensure_ascii=False, indent=2)

    n = len(doc_ids)
    print(f"[REQ-B40] Similarity matrix built: {n}×{n} docs")
    print(f"          Binary  → {npy_path}")
    print(f"          Sparse  → {json_path}")

    return doc_ids, sim_matrix


def load_similarity_matrix(
    output_dir: str = 'data',
) -> tuple[list[str], np.ndarray]:
    """
    Loads the persisted similarity matrix from disk.

    Returns:
        (doc_ids, matrix) where doc_ids is the ordered list of URLs and
        matrix is the N×N NumPy similarity array.
    """
    npy_path = os.path.join(output_dir, "similarity_matrix.npy")
    json_path = os.path.join(output_dir, "similarity_matrix.json")

    matrix = np.load(npy_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data["doc_ids"], matrix


def get_similar_documents(
    url: str,
    top_k: int = 10,
    output_dir: str = 'data',
) -> list[tuple[str, float]]:
    """
    Convenience function: returns the top-k most similar documents for a
    given document URL, using the pre-built similarity matrix.

    Args:
        url:        The URL of the query document.
        top_k:      Number of neighbours to return.
        output_dir: Directory where the matrix files are stored.

    Returns:
        List of (url, score) tuples sorted by descending similarity.
    """
    doc_ids, matrix = load_similarity_matrix(output_dir)

    if url not in doc_ids:
        print(f"[Warning] URL not found in similarity matrix: {url}")
        return []

    idx = doc_ids.index(url)
    row = matrix[idx]

    neighbours = [
        (doc_ids[j], float(row[j]))
        for j in range(len(doc_ids))
        if j != idx and row[j] > 0.0
    ]
    neighbours.sort(key=lambda x: x[1], reverse=True)
    return neighbours[:top_k]


def main():
    index, publications = load_resources()
    total_docs = len(publications)
    
    print("=== Advanced Search Engine: TF-IDF & Similarity ===")
    
    while True:
        query = input("\nEnter query (or 'exit'): ").strip()
        if query.lower() == 'exit': break
        
        mode = input("Choose Implementation: [1] Custom (Manual) [2] Sklearn [3] Doc Similarity Matrix: ").strip()
        
        if mode == '1':
            results = get_custom_ranking(query, index, total_docs)
            print("\n--- Custom Implementation Results ---")
            if not results:
                print("No relevant documents found.")
            else:
                for i, (url, score) in enumerate(results[:10]):
                    print(f"{i+1}. [{score:.4f}] - {url}")

        elif mode == '2':
            results = get_sklearn_ranking(query, publications)
            print("\n--- Sklearn Implementation Results ---")
            if not results:
                print("No relevant documents found.")
            else:
                for i, (url, score) in enumerate(results[:10]):
                    print(f"{i+1}. [{score:.4f}] - {url}")

        elif mode == '3':
            # REQ-B40: Build (or rebuild) the document similarity matrix
            print("\n--- Building Document-Document Similarity Matrix ---")
            build_similarity_matrix(publications)
            
            # Optionally look up neighbours for a specific document
            lookup = input("\nEnter a document URL to find similar docs (or press Enter to skip): ").strip()
            if lookup:
                similar = get_similar_documents(lookup, top_k=10)
                if similar:
                    print(f"\nTop similar documents for: {lookup}")
                    for i, (url, score) in enumerate(similar, 1):
                        print(f"  {i}. [{score:.4f}] {url}")
                else:
                    print("No similar documents found.")

if __name__ == "__main__":
    main()