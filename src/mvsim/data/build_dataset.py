from __future__ import annotations

from pathlib import Path
import json

from mvsim.data.aminer_adapter import adapt_aminer_row
from mvsim.data.io import dump_papers



def build_internal_dataset(input_path: str | Path, output_path: str | Path) -> None:
    papers = []
    with Path(input_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            papers.append(adapt_aminer_row(row))
    dump_papers(output_path, papers)
