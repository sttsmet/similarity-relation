from __future__ import annotations

from pathlib import Path
import json

from scipy import sparse

from mvsim.aggregation import author_similarity_from_paper_similarity
from mvsim.data.io import load_papers
from mvsim.features.text import semantic_similarity
from mvsim.fusion.multiview import weighted_fusion
from mvsim.graph.operators import (
    build_citation_adjacency,
    bibliographic_coupling,
    cocitation,
    top_k_neighbors,
)
from mvsim.graph.temporal import temporal_similarity



def compute_views(config: dict) -> tuple[list, dict[str, sparse.csr_matrix]]:
    papers = load_papers(config["input_path"])
    A, _ = build_citation_adjacency(papers)

    structural_cfg = config["views"]["structural"]
    bc = bibliographic_coupling(A)
    cc = cocitation(A)
    structural = (
        bc * structural_cfg.get("bibliographic_coupling_weight", 0.5)
        + cc * structural_cfg.get("cocitation_weight", 0.5)
    )

    semantic_cfg = config["views"]["semantic"]
    sem = semantic_similarity(
        papers,
        backend=semantic_cfg.get("backend", "tfidf"),
        model_name=semantic_cfg.get("model_name", "allenai/specter2"),
        max_features=int(semantic_cfg.get("max_features", 5000)),
    )

    temp_cfg = config["views"]["temporal"]
    temp = temporal_similarity(papers, gamma=float(temp_cfg.get("gamma", 0.2)))

    return papers, {"structural": structural, "semantic": sem, "temporal": temp}



def run_paper_similarity_experiment(config: dict) -> dict:
    papers, views = compute_views(config)
    fused = weighted_fusion(
        views=views,
        weights={
            "structural": float(config["fusion"].get("structural_weight", 0.45)),
            "semantic": float(config["fusion"].get("semantic_weight", 0.40)),
            "temporal": float(config["fusion"].get("temporal_weight", 0.15)),
        },
    )

    labels = [paper.paper_id for paper in papers]
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    results: dict[str, list[dict]] = {}
    top_k = int(config["paper_experiment"].get("top_k", 5))

    for seed in config["paper_experiment"].get("seed_paper_ids", []):
        if seed not in label_to_index:
            continue
        neighbors = top_k_neighbors(fused, labels, label_to_index[seed], top_k=top_k)
        results[seed] = [{"paper_id": pid, "score": float(score)} for pid, score in neighbors]

    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "paper_similarity_results.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results



def run_author_similarity_experiment(config: dict) -> dict:
    papers, views = compute_views(config)
    fused = weighted_fusion(
        views=views,
        weights={
            "structural": float(config["fusion"].get("structural_weight", 0.45)),
            "semantic": float(config["fusion"].get("semantic_weight", 0.40)),
            "temporal": float(config["fusion"].get("temporal_weight", 0.15)),
        },
    )

    author_cfg = config.get("author_experiment", {}).get("publication_weighting", {})
    author_sim, author_ids = author_similarity_from_paper_similarity(
        fused,
        papers,
        recency_gamma=float(author_cfg.get("recency_gamma", 0.15)),
        contribution_mode=str(author_cfg.get("contribution_mode", "inverse_num_authors")),
    )

    top_k = int(config["author_experiment"].get("top_k", 5))
    results: dict[str, list[dict]] = {}
    for i, author_id in enumerate(author_ids):
        scores = author_sim[i]
        ranked = sorted(
            [(author_ids[j], float(scores[j])) for j in range(len(author_ids)) if j != i and scores[j] > 0],
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
        results[author_id] = [{"author_id": aid, "score": score} for aid, score in ranked]

    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "author_similarity_results.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results
