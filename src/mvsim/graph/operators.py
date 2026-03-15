from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from scipy import sparse

from mvsim.schemas import PaperRecord



def build_paper_index(papers: list[PaperRecord]) -> dict[str, int]:
    return {paper.paper_id: idx for idx, paper in enumerate(papers)}



def build_citation_adjacency(papers: list[PaperRecord]) -> tuple[sparse.csr_matrix, dict[str, int]]:
    index = build_paper_index(papers)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for src_idx, paper in enumerate(papers):
        for ref in paper.references:
            dst_idx = index.get(ref)
            if dst_idx is None:
                continue
            rows.append(src_idx)
            cols.append(dst_idx)
            data.append(1.0)

    n = len(papers)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=float)
    return A, index



def cosine_normalize_similarity(S: sparse.csr_matrix) -> sparse.csr_matrix:
    diag = np.asarray(S.diagonal()).astype(float)
    norms = np.sqrt(np.maximum(diag, 1e-12))
    inv = 1.0 / norms
    D_inv = sparse.diags(inv)
    return D_inv @ S @ D_inv



def bibliographic_coupling(A: sparse.csr_matrix, normalize: bool = True) -> sparse.csr_matrix:
    B = A @ A.T
    B = B.tocsr()
    B.setdiag(0.0)
    B.eliminate_zeros()
    return cosine_normalize_similarity(B) if normalize else B



def cocitation(A: sparse.csr_matrix, normalize: bool = True) -> sparse.csr_matrix:
    C = A.T @ A
    C = C.tocsr()
    C.setdiag(0.0)
    C.eliminate_zeros()
    return cosine_normalize_similarity(C) if normalize else C



def top_k_neighbors(sim_matrix: sparse.csr_matrix, labels: list[str], seed_index: int, top_k: int = 10) -> list[tuple[str, float]]:
    row = sim_matrix.getrow(seed_index).toarray().ravel()
    ranked = np.argsort(-row)
    out: list[tuple[str, float]] = []
    for idx in ranked:
        if idx == seed_index:
            continue
        score = float(row[idx])
        if score <= 0:
            continue
        out.append((labels[idx], score))
        if len(out) >= top_k:
            break
    return out



def dense_from_sparse(S: sparse.csr_matrix) -> np.ndarray:
    return S.toarray().astype(float)
