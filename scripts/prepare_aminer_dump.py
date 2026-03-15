from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mvsim.data.build_dataset import build_internal_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare AMiner/OAG dump into internal JSONL schema")
    parser.add_argument("--input", required=True, help="Path to source JSONL file")
    parser.add_argument("--output", required=True, help="Path to processed JSONL file")
    args = parser.parse_args()
    build_internal_dataset(input_path=args.input, output_path=args.output)
