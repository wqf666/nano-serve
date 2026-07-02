"""Result writer for offline batch inference.

Writes request-level results and summary statistics to JSON/JSONL files.
"""
import json
import os
from typing import Optional


class ResultWriter:
    """Writes offline batch inference results to disk."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_results(self, results: list[dict], filename: str = "results.jsonl"):
        """Write per-request results as JSONL."""
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return path

    def write_summary(self, summary: dict, filename: str = "summary.json"):
        """Write summary statistics as JSON."""
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return path

    def write_all(self, results: list[dict], summary: dict,
                  results_file: str = "results.jsonl",
                  summary_file: str = "summary.json"):
        """Write both results and summary, return paths."""
        r_path = self.write_results(results, results_file)
        s_path = self.write_summary(summary, summary_file)
        return r_path, s_path


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items: list[dict], path: str):
    """Save a list of dicts to a JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
