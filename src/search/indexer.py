import json
import os
from src.search.nlp import preprocess

def build_index(json_path='data/scraper_results.json', output_path='data/index.json'):
    """
    Builds or updates an inverted index with incremental support and sorted postings.
    Supports incremental updates and stores term/doc frequencies.
    """
    print(f"--- Starting Indexer ---")
    
    # Initialization and Incremental Support Logic
    inverted_index = {}
    indexed_urls = set()
    
    # Check if an index already exists to perform an incremental update
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                inverted_index = json.load(f)
                # Identify documents already present in the index to avoid duplicates
                for term in inverted_index:
                    for url in inverted_index[term]['postings']:
                        indexed_urls.add(url)
            print(f"[Incremental] Loaded existing index with {len(indexed_urls)} documents.")
        except Exception as e:
            print(f"[Warning] Could not load existing index: {e}")

    # Load and validate Scraper Data
    if not os.path.exists(json_path):
        print(f"[Error] Scraper results not found at: {json_path}")
        print("Make sure your scraper has saved the results in the 'data' folder.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        publications = json.load(f)
    
    print(f"Found {len(publications)} documents in scraper results.")

    # Processing New Documents
    new_docs_count = 0
    for pub in publications:
        url = pub.get('url')
        
        # Skip documents that are already indexed (Core Incremental Requirement)
        if not url or url in indexed_urls:
            continue

        new_docs_count += 1
        # Combine title and abstract for text analysis
        text = f"{pub.get('title', '')} {pub.get('abstract', '')}"
        words = preprocess(text)

        # Calculate Local Term Frequency (TF)
        doc_word_counts = {}
        for word in words:
            doc_word_counts[word] = doc_word_counts.get(word, 0) + 1

        # Integrate local counts into the Global Inverted Index
        for word, freq in doc_word_counts.items():
            if word not in inverted_index:
                inverted_index[word] = {"df": 0, "postings": {}}
            
            # Map the URL to its term frequency
            inverted_index[word]["postings"][url] = freq

    # Sorting and Document Frequency (DF)
    # Postings must be sorted alphabetically for Skip Pointers to function!
    for word in inverted_index:
        sorted_postings = dict(sorted(inverted_index[word]["postings"].items()))
        inverted_index[word]["postings"] = sorted_postings
        # DF is the total number of unique documents containing the term
        inverted_index[word]["df"] = len(sorted_postings)

    # Save the Inverted Index to disk
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(inverted_index, f, ensure_ascii=False, indent=4)

    print(f"Success! {new_docs_count} new documents added to the index.")
    print(f"Final index saved at: {output_path}")

if __name__ == "__main__":
    build_index()