# Similarity-Relation

A VS Code ready project for building and testing a **multi-view similarity framework** for scientific papers and researchers.

The project is designed around four views:

1. **Structural view**: citation graph, bibliographic coupling, co-citation.
2. **Semantic view**: title and abstract similarity.
3. **Temporal view**: recency and age-aware weighting.
4. **Aggregation view**: from paper similarity to researcher similarity.

The goal is not to lock you into one formula from day one. The goal is to make it easy to:

- collect and clean data,
- build several similarity operators,
- fuse them in a controlled way,
- run ablation studies,
- move from paper-level to author-level experiments.

## Suggested research workflow

### Phase 1: Minimal reproducible pipeline
- Load a focused corpus from AMiner / OAG.
- Build the paper citation graph.
- Compute simple baselines:
  - bibliographic coupling,
  - co-citation,
  - TF-IDF cosine similarity,
  - time decay.
- Fuse them with a weighted sum.
- Inspect nearest neighbors for a few seed papers.

### Phase 2: Author similarity
- Build the author-paper incidence matrix.
- Aggregate paper similarity into author similarity.
- Try different contribution rules:
  - equal credit,
  - first-author boost,
  - 1 / number of authors.

### Phase 3: Better semantics and better structure
- Replace TF-IDF with a scientific embedding model.
- Add random-walk or path-based structural signals.
- Add venue or topic as another view.

### Phase 4: Evaluation
- Compare single-view baselines vs fused similarity.
- Run ablation studies.
- Evaluate paper retrieval and author retrieval separately.

## Internal paper schema

All experiments use a simple internal schema:

```json
{
  "paper_id": "P1",
  "title": "...",
  "abstract": "...",
  "year": 2024,
  "venue": "Journal or Conference",
  "authors": [
    {"author_id": "A1", "name": "Author Name"}
  ],
  "references": ["P0", "P7"],
  "keywords": ["citation graph", "similarity"]
}
```

You can adapt AMiner / OAG fields into this schema in `src/mvsim/data/aminer_adapter.py`.

## Quick start

### 1. Create environment

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

If `py` is not available, install Python 3.10 or newer first and rerun the commands above.

Optional, if you want transformer embeddings later:

```powershell
pip install -e .[text]
```

### 2. Build a small processed corpus from the sample data

```powershell
.\.venv\Scripts\python.exe scripts\prepare_aminer_dump.py --input data\external\sample\sample_papers.jsonl --output data\processed\sample_papers.jsonl
```

### 3. Run paper similarity experiment

```powershell
.\.venv\Scripts\python.exe scripts\run_paper_similarity.py --config configs\paper_similarity.yaml
```

### 4. Run author similarity experiment

```powershell
.\.venv\Scripts\python.exe scripts\run_author_similarity.py --config configs\author_similarity.yaml
```

## VS Code

Open the `Similarity-Relation` folder directly in VS Code. The bundled workspace settings and tasks expect a local `.venv` and provide one-command runs for:

- preparing the sample AMiner-style dataset,
- generating paper similarity results,
- generating author similarity results.

## Recommended directory strategy for AMiner data

Keep the raw downloaded files unchanged:

- `data/external/aminer/` for original AMiner or OAG files
- `data/raw/` for lightly normalized files
- `data/processed/` for the internal schema used in experiments

This separation makes your pipeline reproducible and easier to audit.

## Default similarity idea

For papers:

```text
S_paper = w_struct * S_struct + w_sem * S_sem + w_temp * S_temp
```

For researchers:

```text
S_author(a, b) = weighted average of S_paper(p, q)
for p in papers(a), q in papers(b)
```

with publication weights based on recency, contribution share, or influence.

## Important note about AMiner integration

This project includes:

- a **generic AMiner adapter** for local JSONL / JSON dumps,
- a **clean internal schema**,
- a **place to plug in API logic** if you decide to use the AMiner API account.

Because AMiner data exports can vary by source and version, the adapter is intentionally easy to edit.

## Good first experiment

Use a narrow field first, for example:

- scientometrics,
- citation analysis,
- information retrieval,
- network science.

Do **not** start with the full graph. Start with a focused slice of a few thousand papers.

That will make debugging, feature design, and evaluation much easier.
