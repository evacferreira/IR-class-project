"""
API REST — Motor de Pesquisa de Publicações Científicas
Universidade do Minho — Pesquisa e Recuperação de Informação
"""

import json
import os
import re
import unicodedata
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.search.nlp import preprocess
from src.search.query import execute_boolean_query
from src.search.tfidf import get_custom_ranking, get_sklearn_ranking

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RepositóriUM Search Engine",
    description=(
        "Motor de pesquisa de publicações científicas da Universidade do Minho. "
        "Suporta pesquisa por texto livre (TF-IDF), pesquisa booleana (AND/OR/NOT) "
        "e pesquisa por autor."
    ),
    version="1.0.0",
    contact={"name": "PRI — UMinho"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Data loading (loaded once at startup)
# ---------------------------------------------------------------------------

INDEX_PATH = "data/index.json"
PUBS_PATH = "data/scraper_results.json"


def _load_data():
    if not os.path.exists(INDEX_PATH):
        raise RuntimeError(f"Index not found at '{INDEX_PATH}'. Run the indexer first.")

    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        index = json.load(f)

    publications: List[dict] = []
    if os.path.exists(PUBS_PATH):
        with open(PUBS_PATH, "r", encoding="utf-8") as f:
            publications = json.load(f)

    all_doc_ids = {p.get("url") for p in publications if p.get("url")}
    pub_lookup = {p.get("url"): p for p in publications if p.get("url")}

    return index, publications, all_doc_ids, pub_lookup


try:
    INDEX, PUBLICATIONS, ALL_DOC_IDS, PUB_LOOKUP = _load_data()
except Exception as _e:
    INDEX, PUBLICATIONS, ALL_DOC_IDS, PUB_LOOKUP = {}, [], set(), {}
    print(f"[WARNING] Could not load data at startup: {_e}")


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class PublicationResult(BaseModel):
    url: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    # REQ-B50: snippet with highlighted query terms (HTML <mark> tags)
    snippet: Optional[str] = None
    date: Optional[str] = None
    doi: Optional[str] = None
    pdf_link: Optional[str] = None
    score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    total: int
    page: int
    page_size: int
    results: List[PublicationResult]


class AuthorProfile(BaseModel):
    name: str
    total_publications: int
    publications: List[PublicationResult]


# ---------------------------------------------------------------------------
# REQ-B66 — Query sanitization
# ---------------------------------------------------------------------------

# Characters that have no place in a free-text or boolean query
_FORBIDDEN_PATTERN = re.compile(r"[<>{}\[\]\\|`~@#$%^*]")

# Collapse runs of the same boolean operator (e.g. "AND AND") to one
_REPEATED_OPERATOR = re.compile(
    r"\b(AND|OR|NOT)\b(?:\s+\b(?:AND|OR|NOT)\b)+", re.IGNORECASE
)

# Maximum accepted query length (characters)
_MAX_QUERY_LEN = 512


def sanitize_query(raw: str) -> str:
    """
    REQ-B66 — Advanced query sanitization:

    1. Reject (raise 400) if the query exceeds ``_MAX_QUERY_LEN`` characters.
    2. Strip leading / trailing whitespace.
    3. Normalise Unicode to NFC so accented characters are consistent.
    4. Remove characters that are forbidden in a search context
       (angle brackets, curly / square brackets, back-slash, pipe, back-tick,
        and other shell / injection meta-characters).
    5. Collapse multiple consecutive whitespace into a single space.
    6. Collapse repeated boolean operators (e.g. ``AND AND``) to a single one.
    7. Reject (raise 400) if the sanitized query is empty.

    Returns the cleaned query string.
    """
    if len(raw) > _MAX_QUERY_LEN:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Query too long: {len(raw)} characters "
                f"(maximum allowed: {_MAX_QUERY_LEN})."
            ),
        )

    q = raw.strip()

    # Normalise Unicode (NFC: composed form)
    q = unicodedata.normalize("NFC", q)

    # Strip forbidden meta-characters
    q = _FORBIDDEN_PATTERN.sub(" ", q)

    # Collapse whitespace
    q = re.sub(r"\s+", " ", q).strip()

    # Collapse repeated boolean operators: "AND AND" → "AND"
    q = _REPEATED_OPERATOR.sub(lambda m: m.group(1).upper(), q)

    # Strip dangling leading/trailing operators
    q = re.sub(r"^(AND|OR|NOT)\s+", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+(AND|OR|NOT)$", "", q, flags=re.IGNORECASE)

    q = q.strip()
    if not q:
        raise HTTPException(
            status_code=400,
            detail="Query is empty after sanitization. Please provide a valid search term.",
        )

    return q


# ---------------------------------------------------------------------------
# REQ-B50 — Snippet generation with term highlighting
# ---------------------------------------------------------------------------

_SNIPPET_WINDOW = 40   # words on each side of the best-matching sentence
_SNIPPET_MAX_CHARS = 300  # hard cap on final snippet length


def _extract_snippet(text: str, query_tokens: List[str]) -> Optional[str]:
    """
    REQ-B50 — Generate a short, relevant snippet from *text* that:

    * Selects the sentence (or word window) with the highest density of
      query-term matches.
    * Wraps every matched term in ``<mark>…</mark>`` for front-end highlighting.
    * Falls back to the first ``_SNIPPET_MAX_CHARS`` characters when no match
      is found.

    ``query_tokens`` should be the preprocessed (stemmed/lowercased) terms so
    they align with how the index was built. We also do a raw case-insensitive
    match against the original surface forms so the snippet stays readable.
    """
    if not text:
        return None

    # Build a regex that matches any of the query tokens (surface form, since
    # the abstract is not preprocessed).  We try both the raw token and its
    # unstemmed version by just doing a prefix match (word-boundary aware).
    if not query_tokens:
        # No tokens → return the start of the abstract
        snippet = text[:_SNIPPET_MAX_CHARS]
        return snippet + ("…" if len(text) > _SNIPPET_MAX_CHARS else "")

    # Pattern matches any query token at a word boundary (case-insensitive)
    token_pattern = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in query_tokens) + r")\b",
        re.IGNORECASE,
    )

    # Split into sentences (simple heuristic)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    best_sentence_idx = 0
    best_count = -1
    for i, sent in enumerate(sentences):
        count = len(token_pattern.findall(sent))
        if count > best_count:
            best_count = count
            best_sentence_idx = i

    # Take a window of sentences centred on the best one
    start = max(0, best_sentence_idx - 1)
    end = min(len(sentences), best_sentence_idx + 2)
    window_text = " ".join(sentences[start:end])

    # Truncate to character limit (cut at word boundary)
    if len(window_text) > _SNIPPET_MAX_CHARS:
        window_text = window_text[:_SNIPPET_MAX_CHARS]
        # Back-track to last space so we don't cut a word in half
        last_space = window_text.rfind(" ")
        if last_space > _SNIPPET_MAX_CHARS // 2:
            window_text = window_text[:last_space]
        window_text += "…"

    # Highlight matched terms with <mark> tags
    highlighted = token_pattern.sub(r"<mark>\1</mark>", window_text)
    return highlighted


def _query_surface_tokens(q: str) -> List[str]:
    """
    Return tokens suitable for snippet highlighting.

    We combine:
    * Raw words from the query (split on whitespace, boolean operators removed)
      — these match the surface form in the abstract.
    * Preprocessed tokens from the NLP pipeline — catch stemmed variants.
    """
    # Remove boolean operators
    cleaned = re.sub(r"\b(AND|OR|NOT)\b", " ", q, flags=re.IGNORECASE)
    raw_words = [w for w in re.split(r"\s+", cleaned) if len(w) > 1]

    try:
        nlp_tokens = preprocess(q)
    except Exception:
        nlp_tokens = []

    # Deduplicate while preserving order, prefer longer tokens first so the
    # regex alternation matches them greedily.
    seen: set = set()
    combined = []
    for tok in sorted(raw_words + nlp_tokens, key=len, reverse=True):
        tl = tok.lower()
        if tl not in seen and tl not in {"and", "or", "not"}:
            seen.add(tl)
            combined.append(tok)

    return combined


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _build_result(
    url: str,
    score: Optional[float],
    pub: Optional[dict],
    query_tokens: Optional[List[str]] = None,
) -> PublicationResult:
    if pub is None:
        pub = {}
    authors = pub.get("authors", [])
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(";") if a.strip()]

    abstract = pub.get("abstract")

    # REQ-B50: generate snippet only when query context is available
    snippet: Optional[str] = None
    if query_tokens is not None:
        snippet = _extract_snippet(abstract or "", query_tokens)

    return PublicationResult(
        url=url,
        title=pub.get("title"),
        authors=authors or None,
        abstract=abstract,
        snippet=snippet,
        date=pub.get("date") or pub.get("publication_date"),
        doi=pub.get("doi"),
        pdf_link=pub.get("pdf_link") or pub.get("pdf_url"),
        score=round(score, 6) if score is not None else None,
    )


def _paginate(items, page: int, page_size: int):
    start = (page - 1) * page_size
    return items[start: start + page_size]


def _apply_filters(urls: List[str], year: Optional[int], doc_type: Optional[str]) -> List[str]:
    if not year and not doc_type:
        return urls
    filtered = []
    for url in urls:
        pub = PUB_LOOKUP.get(url, {})
        if year:
            pub_date = str(pub.get("date", "") or pub.get("publication_date", ""))
            if str(year) not in pub_date:
                continue
        if doc_type:
            pub_type = str(pub.get("type", "")).lower()
            if doc_type.lower() not in pub_type:
                continue
        filtered.append(url)
    return filtered


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Info"])
def root():
    """Health check / welcome endpoint."""
    return {
        "message": "RepositóriUM Search Engine is running.",
        "docs": "/docs",
        "total_documents_indexed": len(ALL_DOC_IDS),
        "total_terms_indexed": len(INDEX),
    }


# ── 1. Free-text search (TF-IDF) ───────────────────────────────────────────

@app.get("/search", response_model=SearchResponse, tags=["Search"])
def search(
    q: str = Query(..., description="Texto a pesquisar"),
    mode: str = Query(
        "custom",
        description="Implementação TF-IDF: 'custom' (implementação própria) ou 'sklearn'",
        pattern="^(custom|sklearn)$",
    ),
    year: Optional[int] = Query(None, description="Filtrar por ano de publicação"),
    doc_type: Optional[str] = Query(None, description="Filtrar por tipo de documento"),
    page: int = Query(1, ge=1, description="Número da página"),
    page_size: int = Query(10, ge=1, le=100, description="Resultados por página"),
):
    """
    **Pesquisa por texto livre** com ranking TF-IDF e similaridade do cosseno.

    - `mode=custom` — implementação própria (TF × log(N/DF))
    - `mode=sklearn` — implementação via scikit-learn

    Cada resultado inclui um **snippet** do abstract com os termos da query
    destacados em `<mark>…</mark>` (REQ-B50).
    """
    if not INDEX:
        raise HTTPException(status_code=503, detail="Index not loaded. Run the indexer first.")

    # REQ-B66: sanitize before any processing
    q = sanitize_query(q)

    if mode == "custom":
        raw_results = get_custom_ranking(q, INDEX, max(len(PUBLICATIONS), 1))
    else:
        if not PUBLICATIONS:
            raise HTTPException(status_code=503, detail="Publications data not available for sklearn mode.")
        raw_results = get_sklearn_ranking(q, PUBLICATIONS)

    urls_ordered = [url for url, _ in raw_results]
    scores = {url: score for url, score in raw_results}

    filtered_urls = _apply_filters(urls_ordered, year, doc_type)
    paginated = _paginate(filtered_urls, page, page_size)

    # REQ-B50: derive highlight tokens once for all results
    highlight_tokens = _query_surface_tokens(q)
    results = [
        _build_result(url, scores.get(url), PUB_LOOKUP.get(url), highlight_tokens)
        for url in paginated
    ]

    return SearchResponse(
        query=q,
        total=len(filtered_urls),
        page=page,
        page_size=page_size,
        results=results,
    )


# ── 2. Boolean search ───────────────────────────────────────────────────────

@app.get("/search/boolean", response_model=SearchResponse, tags=["Search"])
def search_boolean(
    q: str = Query(..., description="Query booleana (ex: 'machine learning AND health NOT survey')"),
    year: Optional[int] = Query(None, description="Filtrar por ano de publicação"),
    doc_type: Optional[str] = Query(None, description="Filtrar por tipo de documento"),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
):
    """
    **Pesquisa booleana** com suporte a operadores AND, OR, NOT.

    - Termos separados por espaço são tratados como AND implícito.
    - Exemplo: `machine learning AND health NOT survey`

    Cada resultado inclui um **snippet** do abstract com os termos da query
    destacados em `<mark>…</mark>` (REQ-B50).
    """
    if not INDEX:
        raise HTTPException(status_code=503, detail="Index not loaded. Run the indexer first.")

    # REQ-B66: sanitize before any processing
    q = sanitize_query(q)

    matching_urls = execute_boolean_query(q, INDEX, ALL_DOC_IDS)
    urls_list = sorted(list(matching_urls))

    filtered = _apply_filters(urls_list, year, doc_type)
    paginated = _paginate(filtered, page, page_size)

    # REQ-B50: highlight tokens (strip boolean operators for highlighting)
    highlight_tokens = _query_surface_tokens(q)
    results = [
        _build_result(url, None, PUB_LOOKUP.get(url), highlight_tokens)
        for url in paginated
    ]

    return SearchResponse(
        query=q,
        total=len(filtered),
        page=page,
        page_size=page_size,
        results=results,
    )


# ── 3. Author search ────────────────────────────────────────────────────────

@app.get("/search/author", tags=["Search"])
def search_author(
    name: str = Query(..., description="Nome do autor (pesquisa parcial, case-insensitive)"),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
):
    """
    **Pesquisa por autor** com correspondência parcial (case-insensitive).
    """
    if not PUBLICATIONS:
        raise HTTPException(status_code=503, detail="Publications data not available.")

    # REQ-B66: basic sanitization for author name (no boolean operators needed)
    name = sanitize_query(name)

    name_lower = name.lower()
    matched: List[PublicationResult] = []

    for pub in PUBLICATIONS:
        authors = pub.get("authors", [])
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(";")]
        if any(name_lower in (a or "").lower() for a in authors):
            # No snippet for author search (no free-text query)
            matched.append(_build_result(pub.get("url", ""), None, pub))

    paginated = _paginate(matched, page, page_size)

    return {
        "query_author": name,
        "total": len(matched),
        "page": page,
        "page_size": page_size,
        "results": paginated,
    }


# ── 4. Author profile ───────────────────────────────────────────────────────

@app.get("/author/{author_name}", response_model=AuthorProfile, tags=["Authors"])
def author_profile(author_name: str):
    """
    **Perfil de um autor** — lista todas as publicações associadas ao nome.
    """
    if not PUBLICATIONS:
        raise HTTPException(status_code=503, detail="Publications data not available.")

    name_lower = author_name.lower()
    pubs = []
    for pub in PUBLICATIONS:
        authors = pub.get("authors", [])
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(";")]
        if any(name_lower in (a or "").lower() for a in authors):
            pubs.append(_build_result(pub.get("url", ""), None, pub))

    if not pubs:
        raise HTTPException(status_code=404, detail=f"No publications found for author '{author_name}'.")

    return AuthorProfile(name=author_name, total_publications=len(pubs), publications=pubs)


# ── 5. Document detail ──────────────────────────────────────────────────────

@app.get("/document", response_model=PublicationResult, tags=["Documents"])
def get_document(
    url: str = Query(..., description="URL/handle do documento"),
):
    """
    **Detalhes de um documento** a partir do seu URL.
    """
    pub = PUB_LOOKUP.get(url)
    if pub is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return _build_result(url, None, pub)


# ── 6. Index stats ──────────────────────────────────────────────────────────

@app.get("/stats", tags=["Info"])
def stats():
    """
    **Estatísticas do índice** — número de termos, documentos e top 20 termos por DF.
    """
    if not INDEX:
        raise HTTPException(status_code=503, detail="Index not loaded.")

    top_terms = sorted(INDEX.items(), key=lambda x: x[1]["df"], reverse=True)[:20]

    return {
        "total_terms": len(INDEX),
        "total_documents": len(ALL_DOC_IDS),
        "top_20_terms_by_df": [
            {"term": term, "document_frequency": data["df"]}
            for term, data in top_terms
        ],
    }


# ── 7. NLP debug endpoint ───────────────────────────────────────────────────

@app.get("/debug/preprocess", tags=["Debug"])
def debug_preprocess(
    text: str = Query(..., description="Texto a pré-processar"),
):
    """
    **Debug NLP** — mostra os tokens gerados pelo pipeline de pré-processamento.
    Útil para perceber como os termos são indexados.
    """
    tokens = preprocess(text)
    return {
        "input": text,
        "tokens": tokens,
        "token_count": len(tokens),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)