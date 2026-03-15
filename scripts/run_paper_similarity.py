from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mvsim.experiments import run_paper_similarity_experiment
from mvsim.settings import load_yaml_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run paper similarity experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)
    results = run_paper_similarity_experiment(cfg)
    print(json.dumps(results, indent=2, ensure_ascii=False))
