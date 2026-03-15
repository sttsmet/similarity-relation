from __future__ import annotations

import numpy as np
from scipy import sparse



def normalize_sparse_matrix(S: sparse.csr_matrix) -> sparse.csr_matrix:
    if S.nnz == 0:
        return S.copy()
    data = S.data.astype(float)
    max_value = float(data.max()) if data.size else 1.0
    if max_value <= 0:
        return S.copy()
    return S / max_value



def weighted_fusion(views: dict[str, sparse.csr_matrix], weights: dict[str, float]) -> sparse.csr_matrix:
    active = []
    for name, matrix in views.items():
        weight = weights.get(name, 0.0)
        if weight <= 0:
            continue
        active.append((weight, normalize_sparse_matrix(matrix)))

    if not active:
        raise ValueError("No active views were provided for fusion.")

    total = None
    weight_sum = 0.0
    for weight, matrix in active:
        total = matrix * weight if total is None else total + matrix * weight
        weight_sum += weight
    return total / max(weight_sum, 1e-12)
