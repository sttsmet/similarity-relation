from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from mvsim.graph.temporal import recency_weight
from mvsim.schemas import PaperRecord


@dataclass
class AuthorPaperLink:
    author_id: str
    paper_id: str
    contribution_weight: float
    recency_weight: float



def build_author_publication_map(
    papers: list[PaperRecord],
    recency_gamma: float = 0.15,
    contribution_mode: str = "inverse_num_authors",
) -> dict[str, list[tuple[int, float]]]:
    max_year = max((paper.year or 0) for paper in papers) if papers else 0
    mapping: dict[str, list[tuple[int, float]]] = defaultdict(list)

    for paper_idx, paper in enumerate(papers):
        num_authors = max(len(paper.authors), 1)
        if contribution_mode == "inverse_num_authors":
            contribution = 1.0 / num_authors
        else:
            contribution = 1.0
        time_w = recency_weight(paper.year, max_year=max_year, gamma=recency_gamma)
        total_w = contribution * time_w
        for author in paper.authors:
            mapping[author.author_id].append((paper_idx, total_w))
    return dict(mapping)



def author_similarity_from_paper_similarity(
    paper_sim: sparse.csr_matrix,
    papers: list[PaperRecord],
    recency_gamma: float = 0.15,
    contribution_mode: str = "inverse_num_authors",
) -> tuple[np.ndarray, list[str]]:
    author_map = build_author_publication_map(
        papers=papers,
        recency_gamma=recency_gamma,
        contribution_mode=contribution_mode,
    )
    author_ids = sorted(author_map)
    n = len(author_ids)
    S = np.zeros((n, n), dtype=float)

    dense_paper = paper_sim.toarray()
    for i, a in enumerate(author_ids):
        papers_a = author_map[a]
        for j, b in enumerate(author_ids):
            if i == j:
                continue
            papers_b = author_map[b]
            num = 0.0
            den = 0.0
            for p_idx, p_w in papers_a:
                for q_idx, q_w in papers_b:
                    w = p_w * q_w
                    num += w * dense_paper[p_idx, q_idx]
                    den += w
            S[i, j] = num / den if den > 0 else 0.0
    return S, author_ids
