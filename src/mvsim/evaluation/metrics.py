from __future__ import annotations

import math



def precision_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for item in top if item in relevant) / len(top)



def recall_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = ranked[:k]
    return sum(1 for item in top if item in relevant) / len(relevant)



def average_precision(relevant: set[str], ranked: list[str], k: int | None = None) -> float:
    if not relevant:
        return 0.0
    items = ranked if k is None else ranked[:k]
    hits = 0
    score = 0.0
    for idx, item in enumerate(items, start=1):
        if item in relevant:
            hits += 1
            score += hits / idx
    return score / len(relevant)



def ndcg_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    def dcg(items: list[str]) -> float:
        score = 0.0
        for idx, item in enumerate(items, start=1):
            gain = 1.0 if item in relevant else 0.0
            score += gain / math.log2(idx + 1)
        return score

    actual = dcg(ranked[:k])
    ideal_count = min(k, len(relevant))
    ideal_items = [f"REL_{i}" for i in range(ideal_count)]
    ideal_relevant = set(ideal_items)

    ideal = 0.0
    for idx, item in enumerate(ideal_items, start=1):
        gain = 1.0 if item in ideal_relevant else 0.0
        ideal += gain / math.log2(idx + 1)

    return actual / ideal if ideal > 0 else 0.0
