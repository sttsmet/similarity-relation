from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mvsim.schemas import PaperRecord



def _paper_text(paper: PaperRecord) -> str:
    pieces = [paper.title or "", paper.abstract or "", " ".join(paper.keywords or [])]
    return " ".join(x.strip() for x in pieces if x).strip()



def tfidf_similarity(papers: list[PaperRecord], max_features: int = 5000) -> sparse.csr_matrix:
    corpus = [_paper_text(paper) for paper in papers]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    S = cosine_similarity(X, dense_output=False)
    S = S.tocsr()
    S.setdiag(0.0)
    S.eliminate_zeros()
    return S



def sentence_transformer_similarity(papers: list[PaperRecord], model_name: str = "allenai/specter2") -> sparse.csr_matrix:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Run: pip install -e .[text]"
        ) from exc

    model = SentenceTransformer(model_name)
    texts = [_paper_text(paper) for paper in papers]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    S = embeddings @ embeddings.T
    np.fill_diagonal(S, 0.0)
    return sparse.csr_matrix(S)



def semantic_similarity(
    papers: list[PaperRecord],
    backend: Literal["tfidf", "sentence-transformers"] = "tfidf",
    model_name: str = "allenai/specter2",
    max_features: int = 5000,
) -> sparse.csr_matrix:
    if backend == "tfidf":
        return tfidf_similarity(papers=papers, max_features=max_features)
    if backend == "sentence-transformers":
        return sentence_transformer_similarity(papers=papers, model_name=model_name)
    raise ValueError(f"Unsupported semantic backend: {backend}")
