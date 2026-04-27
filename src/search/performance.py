"""
performance.py — Indexing & Search Performance Monitor
Universidade do Minho · PRI

Covers:
  REQ-B56  Measure and log indexing time performance
  REQ-B57  Compare indexing speed: stems vs lemmas
  REQ-B58  Monitor memory usage during indexing
  REQ-B59  Implement batch processing for large collections
  REQ-B60  Measure query response times
  REQ-B61  Evaluate search result relevance (Precision, Recall, F1, MAP)
  REQ-B62  Compare ranking effectiveness across different methods
"""

import json
import os
import time
import tracemalloc

from src.search.nlp import preprocess, ReductionMode
from src.search.tfidf import (
    get_custom_ranking,
    get_bm25_ranking,
    get_tf_ranking,
    get_sklearn_ranking,
    load_resources,
)

LOG_PATH = "data/performance_log.json"


def _append_log(entry: dict) -> None:
    """Append a result entry to the performance log JSON file."""
    log = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            try:
                log = json.load(f)
            except Exception:
                log = []
    log.append(entry)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# REQ-B56 + REQ-B58: Indexing time and memory measurement
# ---------------------------------------------------------------------------

def measure_indexing_performance(
    json_path: str = "data/scraper_results.json",
    output_path: str = "data/index.json",
) -> dict:
    """
    REQ-B56: Measures and logs total indexing time.
    REQ-B58: Monitors peak memory usage during indexing via tracemalloc.
    """
    from src.search.indexer import build_index

    print("\n" + "=" * 55)
    print("  REQ-B56 / B58 — Indexing Time & Memory Benchmark")
    print("=" * 55)

    tracemalloc.start()
    start = time.perf_counter()

    build_index(json_path=json_path, output_path=output_path)

    elapsed = time.perf_counter() - start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak_mem / 1024 / 1024

    result = {
        "benchmark": "indexing",
        "elapsed_seconds": round(elapsed, 4),
        "peak_memory_mb": round(peak_mb, 4),
    }

    print(f"\n[REQ-B56] Indexing time : {elapsed:.4f}s")
    print(f"[REQ-B58] Peak memory   : {peak_mb:.2f} MB")

    _append_log(result)
    return result


# ---------------------------------------------------------------------------
# REQ-B57: Compare indexing speed — stemming vs lemmatization
# ---------------------------------------------------------------------------

def compare_reduction_modes(
    json_path: str = "data/scraper_results.json",
) -> dict:
    """
    REQ-B57: Runs the NLP pipeline on all documents using STEMMING,
    LEMMATIZATION, and BOTH modes, comparing elapsed time and memory.
    """
    print("\n" + "=" * 55)
    print("  REQ-B57 — Stemming vs Lemmatization Speed Comparison")
    print("=" * 55)

    with open(json_path, "r", encoding="utf-8") as f:
        publications = json.load(f)

    modes = [ReductionMode.STEMMING, ReductionMode.LEMMATIZATION, ReductionMode.BOTH]
    results = {}

    for mode in modes:
        tracemalloc.start()
        start = time.perf_counter()

        for pub in publications:
            text = f"{pub.get('title', '')} {pub.get('abstract', '')}"
            preprocess(text, reduction_mode=mode)

        elapsed = time.perf_counter() - start
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_mem / 1024 / 1024
        results[mode.value] = {
            "elapsed_seconds": round(elapsed, 4),
            "peak_memory_mb": round(peak_mb, 4),
        }
        print(f"  [{mode.value:>15}]  time={elapsed:.4f}s   memory={peak_mb:.2f} MB")

    _append_log({"benchmark": "reduction_mode_comparison", "results": results})
    return results


# ---------------------------------------------------------------------------
# REQ-B59: Batch processing for large collections
# ---------------------------------------------------------------------------

def build_index_in_batches(
    json_path: str = "data/scraper_results.json",
    output_path: str = "data/index.json",
    batch_size: int = 10,
) -> dict:
    """
    REQ-B59: Processes documents in batches to limit peak memory usage.
    Each batch is indexed incrementally into the same output file.
    """
    print("\n" + "=" * 55)
    print(f"  REQ-B59 — Batch Indexing (batch_size={batch_size})")
    print("=" * 55)

    with open(json_path, "r", encoding="utf-8") as f:
        publications = json.load(f)

    total = len(publications)
    batches = [publications[i:i + batch_size] for i in range(0, total, batch_size)]
    print(f"  Total docs: {total} → {len(batches)} batches")

    batch_times = []
    overall_start = time.perf_counter()
    tmp_path = "data/_batch_tmp.json"

    for idx, batch in enumerate(batches):
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(batch, f, ensure_ascii=False)

        t0 = time.perf_counter()
        from src.search.indexer import build_index
        build_index(json_path=tmp_path, output_path=output_path)
        elapsed = time.perf_counter() - t0

        batch_times.append(round(elapsed, 4))
        print(f"  Batch {idx + 1}/{len(batches)}: {len(batch)} docs in {elapsed:.4f}s")

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    total_elapsed = time.perf_counter() - overall_start

    result = {
        "benchmark": "batch_indexing",
        "batch_size": batch_size,
        "total_docs": total,
        "total_batches": len(batches),
        "total_elapsed_seconds": round(total_elapsed, 4),
        "batch_times": batch_times,
    }

    print(f"\n  Total time: {total_elapsed:.4f}s")
    _append_log(result)
    return result


# ---------------------------------------------------------------------------
# REQ-B60: Query response time measurement
# ---------------------------------------------------------------------------

def measure_query_times(
    queries: list[str],
    index: dict,
    publications: list[dict],
    total_docs: int,
) -> dict:
    """
    REQ-B60: Runs each query through all ranking methods and measures
    response time per method.
    """
    print("\n" + "=" * 55)
    print("  REQ-B60 — Query Response Time Benchmark")
    print("=" * 55)

    methods = {
        "custom_tfidf": lambda q: get_custom_ranking(q, index, total_docs),
        "bm25":         lambda q: get_bm25_ranking(q, index, total_docs),
        "tf_only":      lambda q: get_tf_ranking(q, index),
        "sklearn":      lambda q: get_sklearn_ranking(q, publications),
    }

    results = {}
    for query in queries:
        results[query] = {}
        for method_name, fn in methods.items():
            start = time.perf_counter()
            fn(query)
            elapsed = time.perf_counter() - start
            results[query][method_name] = round(elapsed, 6)
            print(f"  [{method_name:>15}] '{query}' → {elapsed:.6f}s")

    _append_log({"benchmark": "query_response_times", "results": results})
    return results


# ---------------------------------------------------------------------------
# REQ-B61: Search result relevance evaluation
# ---------------------------------------------------------------------------

def evaluate_relevance(
    query: str,
    retrieved: list[str],
    relevant: list[str],
) -> dict:
    """
    REQ-B61: Computes Precision, Recall, F1, and Average Precision (AP)
    given retrieved and ground-truth relevant document URLs.
    """
    retrieved_set = set(retrieved)
    relevant_set  = set(relevant)

    tp = len(retrieved_set & relevant_set)
    precision = tp / len(retrieved_set) if retrieved_set else 0.0
    recall    = tp / len(relevant_set)  if relevant_set  else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    hits, ap_sum = 0, 0.0
    for k, url in enumerate(retrieved, 1):
        if url in relevant_set:
            hits += 1
            ap_sum += hits / k
    ap = ap_sum / len(relevant_set) if relevant_set else 0.0

    result = {
        "query":          query,
        "precision":      round(precision, 4),
        "recall":         round(recall, 4),
        "f1":             round(f1, 4),
        "ap":             round(ap, 4),
        "retrieved":      len(retrieved),
        "relevant":       len(relevant),
        "true_positives": tp,
    }

    print(f"\n[REQ-B61] Query: '{query}'")
    print(f"  Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}  AP={ap:.4f}")

    _append_log({"benchmark": "relevance_evaluation", **result})
    return result


def mean_average_precision(evaluations: list[dict]) -> float:
    """Computes MAP from a list of evaluate_relevance() results."""
    if not evaluations:
        return 0.0
    map_score = sum(e["ap"] for e in evaluations) / len(evaluations)
    print(f"\n[REQ-B61] MAP across {len(evaluations)} queries: {map_score:.4f}")
    return map_score


# ---------------------------------------------------------------------------
# REQ-B62: Compare ranking effectiveness across methods
# ---------------------------------------------------------------------------

def compare_ranking_methods(
    queries: list[str],
    relevant_docs: dict,
    index: dict,
    publications: list[dict],
    total_docs: int,
) -> dict:
    """
    REQ-B62: Compares Precision, Recall, F1, and AP across all ranking
    methods for a set of queries with known relevant documents.
    """
    print("\n" + "=" * 55)
    print("  REQ-B62 — Ranking Method Comparison")
    print("=" * 55)

    methods = {
        "custom_tfidf": lambda q: [url for url, _ in get_custom_ranking(q, index, total_docs)],
        "bm25":         lambda q: [url for url, _ in get_bm25_ranking(q, index, total_docs)],
        "tf_only":      lambda q: [url for url, _ in get_tf_ranking(q, index)],
        "sklearn":      lambda q: [url for url, _ in get_sklearn_ranking(q, publications)],
    }

    summary = {m: {"precision": [], "recall": [], "f1": [], "ap": []} for m in methods}

    for query in queries:
        relevant = relevant_docs.get(query, [])
        for method_name, fn in methods.items():
            retrieved = fn(query)[:10]  # evaluate top-10
            ev = evaluate_relevance(query, retrieved, relevant)
            for metric in ("precision", "recall", "f1", "ap"):
                summary[method_name][metric].append(ev[metric])

    avg_summary = {}
    print("\n  --- Average Metrics per Method (top-10) ---")
    for method_name, scores in summary.items():
        avg = {k: round(sum(v) / len(v), 4) for k, v in scores.items()}
        avg_summary[method_name] = avg
        print(f"  {method_name:>15}: P={avg['precision']}  R={avg['recall']}  F1={avg['f1']}  MAP={avg['ap']}")

    _append_log({"benchmark": "ranking_comparison", "results": avg_summary})
    return avg_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    index, publications = load_resources()
    total_docs = len(publications)

    print("\nPerformance Benchmarks:")
    print("  [1] Indexing time & memory  (REQ-B56, B58)")
    print("  [2] Stemming vs Lemmatization  (REQ-B57)")
    print("  [3] Batch indexing  (REQ-B59)")
    print("  [4] Query response times  (REQ-B60)")
    print("  [5] Ranking method comparison  (REQ-B62)")
    print("  [6] Run all")

    choice = input("\nChoice: ").strip()

    if choice in ("1", "6"):
        measure_indexing_performance()

    if choice in ("2", "6"):
        compare_reduction_modes()

    if choice in ("3", "6"):
        build_index_in_batches(batch_size=5)

    if choice in ("4", "6"):
        test_queries = ["machine learning", "health", "neural network"]
        measure_query_times(test_queries, index, publications, total_docs)

    if choice in ("5", "6"):
        test_queries = ["machine learning", "health"]
        # Pseudo-relevant: top-5 from custom TF-IDF used as ground truth
        pseudo_relevant = {
            q: [url for url, _ in get_custom_ranking(q, index, total_docs)[:5]]
            for q in test_queries
        }
        compare_ranking_methods(test_queries, pseudo_relevant, index, publications, total_docs)

    print(f"\nResults logged to: {LOG_PATH}")