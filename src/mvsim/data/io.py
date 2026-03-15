from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable

from mvsim.schemas import PaperRecord



def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows



def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")



def load_papers(path: str | Path) -> list[PaperRecord]:
    return [PaperRecord.from_dict(row) for row in read_jsonl(path)]



def dump_papers(path: str | Path, papers: list[PaperRecord]) -> None:
    rows = []
    for p in papers:
        rows.append(
            {
                "paper_id": p.paper_id,
                "title": p.title,
                "abstract": p.abstract,
                "year": p.year,
                "venue": p.venue,
                "authors": [{"author_id": a.author_id, "name": a.name} for a in p.authors],
                "references": p.references,
                "keywords": p.keywords,
            }
        )
    write_jsonl(path, rows)
