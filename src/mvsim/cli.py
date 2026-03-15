from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print

from mvsim.data.build_dataset import build_internal_dataset
from mvsim.experiments import run_author_similarity_experiment, run_paper_similarity_experiment
from mvsim.settings import load_yaml_config

app = typer.Typer(help="Multi-view similarity experiments for citation graphs")


@app.command()
def prepare(input: str, output: str) -> None:
    build_internal_dataset(input_path=input, output_path=output)
    print(f"[green]Prepared dataset:[/green] {output}")


@app.command("paper")
def paper_similarity(config: str) -> None:
    cfg = load_yaml_config(config)
    results = run_paper_similarity_experiment(cfg)
    print(json.dumps(results, indent=2, ensure_ascii=False))


@app.command("author")
def author_similarity(config: str) -> None:
    cfg = load_yaml_config(config)
    results = run_author_similarity_experiment(cfg)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
