import json
import math
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

def main():
    index, publications = load_resources()
    total_docs = len(publications)
    
    print("=== Advanced Search Engine: TF-IDF & Similarity ===")
    
    while True:
        query = input("\nEnter query (or 'exit'): ").strip()
        if query.lower() == 'exit': break
        
        mode = input("Choose Implementation: [1] Custom (Manual) [2] Sklearn: ").strip()
        
        if mode == '1':
            results = get_custom_ranking(query, index, total_docs)
            print("\n--- Custom Implementation Results ---")
        else:
            results = get_sklearn_ranking(query, publications)
            print("\n--- Sklearn Implementation Results ---")

        if not results:
            print("No relevant documents found.")
        else:
            for i, (url, score) in enumerate(results[:10]): # Show top 10
                print(f"{i+1}. [{score:.4f}] - {url}")

if __name__ == "__main__":
    main()