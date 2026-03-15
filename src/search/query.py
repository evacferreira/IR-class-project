import json
import math
from src.search.nlp import preprocess

def load_resources(index_path='data/index.json', pubs_path='data/scraper_results.json'):
    """
    Loads necessary IR components from disk.
    - index: The Inverted Index containing terms, DF, and postings (with TF).
    - all_doc_ids: Set of all document URLs, required for the Boolean 'NOT' (complement) operation.
    """
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            index = json.load(f)
        with open(pubs_path, 'r', encoding='utf-8') as f:
            pubs = json.load(f)
            # We create a universal set of document IDs to handle 'NOT' logic 
            all_doc_ids = {p.get('url') for p in pubs if p.get('url')}
        return index, all_doc_ids
    except FileNotFoundError:
        print("Error: Index files not found. Run the indexer first.")
        return None, None

def intersect_with_skips(list1, list2):
    """
    Optimized intersection algorithm for two sorted posting lists using Skip Pointers.
    Improves performance by skipping irrelevant documents.
    
    Time Complexity: O(P1 + P2) where P is the length of the lists, but with a 
    significantly lower constant factor due to skips.
    """
    answer = []
    i, j = 0, 0
    
    # Heuristic: Skip interval is typically the square root of the list length
    skip1 = int(math.sqrt(len(list1)))
    skip2 = int(math.sqrt(len(list2)))

    while i < len(list1) and j < len(list2):
        # Case 1: Match found
        if list1[i] == list2[j]:
            answer.append(list1[i])
            i += 1
            j += 1
        # Case 2: Element in list1 is smaller; try to skip forward
        elif list1[i] < list2[j]:
            # If a skip pointer exists and the target is still <= the other list's current value, skip!
            if i + skip1 < len(list1) and list1[i + skip1] <= list2[j]:
                while i + skip1 < len(list1) and list1[i + skip1] <= list2[j]:
                    i += skip1
            else:
                i += 1
        # Case 3: Element in list2 is smaller; try to skip forward
        else:
            if j + skip2 < len(list2) and list2[j + skip2] <= list1[i]:
                while j + skip2 < len(list2) and list2[j + skip2] <= list1[i]:
                    j += skip2
            else:
                j += 1
    return answer

def execute_boolean_query(query, index, all_doc_ids):
    """
    Main Boolean Retrieval Engine.
    Processes queries using Set Theory (Union, Intersection, Difference).
    Handles implicit AND (spaces) and explicit operators (AND, OR, NOT).
    """
    # Tokenize query and handle basic parenthesis padding
    tokens = query.lower().replace('(', ' ( ').replace(')', ' ) ').split()
    if not tokens:
        return set()

    results = None
    current_op = "AND"  # Defaulting to AND for implicit space support

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Operator Identification
        if token == "and":
            current_op = "AND"
        elif token == "or":
            current_op = "OR"
        elif token == "not":
            current_op = "NOT"
        elif token in ("(", ")"):
            i += 1
            continue
        else:
            # Word Identification & NLP Processing
            processed = preprocess(token)
            if not processed:
                i += 1
                continue
            
            # Retrieve the postings list for the processed term (stem/lemma)
            term = processed[0]
            term_entry = index.get(term, {})
            # Convert postings keys (URLs) into a set for Boolean logic
            term_docs = set(term_entry.get("postings", {}).keys())

            if results is None:
                # Initialization of the result set
                # If first term is 'NOT word', we start with the full collection minus that word
                results = term_docs if current_op != "NOT" else all_doc_ids - term_docs
            else:
                # APPLY BOOLEAN LOGIC 
                if current_op == "AND":
                    # Skip Pointers
                    # We sort lists to ensure the skip pointer algorithm works correctly
                    list_res = sorted(list(results))
                    list_term = sorted(list(term_docs))
                    results = set(intersect_with_skips(list_res, list_term))
                elif current_op == "OR":
                    # Standard Set Union
                    results |= term_docs
                elif current_op == "NOT":
                    # Standard Set Difference
                    results -= term_docs
            
            # Reset to implicit AND for the next token (Space = AND)
            current_op = "AND"
        i += 1

    return results if results is not None else set()

def main():
    """
    Command Line Interface for the Boolean Search System.
    """
    index, all_doc_ids = load_resources()
    if not index: return

    print("\n" + "="*50)
    print("   BILINGUAL BOOLEAN SEARCH ENGINE (v2.0)   ")
    print("="*50)
    print("Features:")
    print(" - NLP: Stemming, Lemmatization, Stopwords")
    print(" - Logic: AND, OR, NOT (and implicit AND)")
    print(" - Optimization: Skip Pointer Intersection")
    
    while True:
        user_input = input("\nEnter Boolean Query (or 'exit'): ").strip()
        if user_input.lower() in ['exit', 'quit', 'sair']:
            break
            
        matches = execute_boolean_query(user_input, index, all_doc_ids)
        
        if matches:
            print(f"\n[Success] Found {len(matches)} document(s):")
            for url in matches:
                print(f" -> {url}")
        else:
            print("\n[Notice] No documents found matching that query.")

if __name__ == "__main__":
    main()