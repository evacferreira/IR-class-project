"""
API REST — Motor de Pesquisa de Publicações Científicas
Universidade do Minho — Pesquisa e Recuperação de Informação
"""

import json
import math
import os
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
# Helper utilities
# ---------------------------------------------------------------------------

def _build_result(url: str, score: Optional[float], pub: Optional[dict]) -> PublicationResult:
    if pub is None:
        pub = {}
    authors = pub.get("authors", [])
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(";") if a.strip()]
    return PublicationResult(
        url=url,
        title=pub.get("title"),
        authors=authors or None,
        abstract=pub.get("abstract"),
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
    """
    if not INDEX:
        raise HTTPException(status_code=503, detail="Index not loaded. Run the indexer first.")

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
    results = [_build_result(url, scores.get(url), PUB_LOOKUP.get(url)) for url in paginated]

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
    """
    if not INDEX:
        raise HTTPException(status_code=503, detail="Index not loaded. Run the indexer first.")

    matching_urls = execute_boolean_query(q, INDEX, ALL_DOC_IDS)
    urls_list = sorted(list(matching_urls))

    filtered = _apply_filters(urls_list, year, doc_type)
    paginated = _paginate(filtered, page, page_size)
    results = [_build_result(url, None, PUB_LOOKUP.get(url)) for url in paginated]

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

    name_lower = name.lower()
    matched: List[PublicationResult] = []

    for pub in PUBLICATIONS:
        authors = pub.get("authors", [])
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(";")]
        if any(name_lower in (a or "").lower() for a in authors):
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