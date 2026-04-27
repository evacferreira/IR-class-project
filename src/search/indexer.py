import json
import os
import numpy as np
from src.search.nlp import preprocess

def build_index(json_path='data/scraper_results.json', output_path='data/index.json'):
    """
    Builds or updates an inverted index with incremental support and sorted postings.
    Supports incremental updates and stores term/doc frequencies.
    Also builds and persists the term-document matrix (REQ-B24).
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

    # REQ-B24 — Build and persist the Term-Document Matrix
    build_term_document_matrix(inverted_index, output_dir=os.path.dirname(output_path))

    from src.database import init_db, insert_publications, save_index

    init_db()
    insert_publications(publications)
    save_index(inverted_index)

if __name__ == "__main__":
    build_index()


# ---------------------------------------------------------------------------
# REQ-B24 — Term-Document Matrix
# ---------------------------------------------------------------------------

def build_term_document_matrix(
    inverted_index: dict,
    output_dir: str = 'data',
) -> dict:
    """
    Builds a Term-Document Matrix from an existing inverted index and persists
    it to disk as a JSON file.

    The matrix maps every term to an ordered vector of raw TF values across
    all documents in the corpus.  The document order is fixed (sorted URLs)
    so that column indices are stable and reproducible.

    Structure of the saved JSON
    ---------------------------
    {
        "doc_ids": ["url_0", "url_1", ...],          # ordered column labels
        "matrix": {
            "term_a": [tf_url0, tf_url1, ...],
            "term_b": [tf_url0, tf_url1, ...],
            ...
        }
    }

    Args:
        inverted_index: The in-memory inverted index produced by build_index().
        output_dir:     Directory where ``term_document_matrix.json`` is saved.

    Returns:
        A dict with keys ``doc_ids`` and ``matrix`` (same shape as the JSON).
    """
    # Collect and sort all document IDs for a stable column ordering
    all_doc_ids = sorted({
        url
        for entry in inverted_index.values()
        for url in entry["postings"]
    })

    doc_index = {url: idx for idx, url in enumerate(all_doc_ids)}
    n_docs = len(all_doc_ids)

    matrix = {}
    for term, entry in inverted_index.items():
        # Initialise the full row with zeros
        row = [0] * n_docs
        for url, tf in entry["postings"].items():
            row[doc_index[url]] = tf
        matrix[term] = row

    result = {"doc_ids": all_doc_ids, "matrix": matrix}

    # Persist to disk
    os.makedirs(output_dir, exist_ok=True)
    matrix_path = os.path.join(output_dir, "term_document_matrix.json")
    with open(matrix_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[REQ-B24] Term-document matrix saved: {len(matrix)} terms × {n_docs} docs → {matrix_path}")
    return result


def load_term_document_matrix(matrix_path: str = 'data/term_document_matrix.json') -> tuple[list, dict]:
    """
    Loads the persisted term-document matrix from disk.

    Returns:
        (doc_ids, matrix) where doc_ids is the ordered list of URLs and
        matrix is a dict mapping each term to its TF row vector.
    """
    with open(matrix_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["doc_ids"], data["matrix"]