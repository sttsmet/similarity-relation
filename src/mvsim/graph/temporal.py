from __future__ import annotations

import math
import numpy as np
from scipy import sparse

from mvsim.schemas import PaperRecord



def temporal_similarity(papers: list[PaperRecord], gamma: float = 0.2) -> sparse.csr_matrix:
    years = np.array([paper.year if paper.year is not None else 0 for paper in papers], dtype=float)
    n = len(years)
    M = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if years[i] == 0 or years[j] == 0:
                M[i, j] = 0.0
            else:
                delta = abs(years[i] - years[j])
                M[i, j] = math.exp(-gamma * delta)

    return sparse.csr_matrix(M)



def recency_weight(year: int | None, max_year: int, gamma: float = 0.15) -> float:
    if year is None:
        return 1.0
    delta = max_year - year
    return math.exp(-gamma * max(delta, 0))
