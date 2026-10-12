from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


TOPICS = [
    "quantum computing",
    "machine learning",
    "cloud security",
    "data structures",
    "software testing",
    "python programming",
    "api design",
    "docker basics",
]

TASKS = [
    "Explain in one sentence",
    "Summarize for a beginner",
    "Give 3 key points about",
    "Describe real-world use of",
    "List common pitfalls in",
]


def build_row(index: int) -> dict[str, str]:
    topic = random.choice(TOPICS)
    task = random.choice(TASKS)
    prompt = f"{task} {topic}."
    answer = (
        f"Sample answer {index}: {topic.title()} involves core concepts, practical tradeoffs, "
        "and common implementation patterns."
    )
    return {
        "id": str(index),
        "topic": topic,
        "prompt": prompt,
        "answer": answer,
    }


def generate_dataset(rows: int = 100, seed: int = 42) -> list[dict[str, str]]:
    if rows <= 0:
        raise ValueError("rows must be greater than 0")

    random.seed(seed)
    return [build_row(index=i) for i in range(1, rows + 1)]


def write_jsonl(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "topic", "prompt", "answer"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic conversational dataset.")
    parser.add_argument("--rows", type=int, default=100, help="Number of rows to generate")
    parser.add_argument(
        "--format",
        choices=["jsonl", "csv"],
        default="jsonl",
        help="Output dataset format",
    )
    parser.add_argument(
        "--output",
        default="data/synthetic_dataset.jsonl",
        help="Output file path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = generate_dataset(rows=args.rows, seed=args.seed)
    output_path = Path(args.output)

    if args.format == "csv":
        write_csv(rows, output_path)
    else:
        write_jsonl(rows, output_path)

    print(f"Generated {len(rows)} rows at {output_path}")


if __name__ == "__main__":
    main()