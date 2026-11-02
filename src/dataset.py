from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


JOB_ROLES = [
    "Full Stack Engineer",
    "Full Stack Developer",
    "React Engineer",
    "Node.js Engineer",
    ".NET Engineer",
    "Fullstack Developer",
]

COMPANIES = [
    "Northstar Labs",
    "Helio Systems",
    "CrestPoint AI",
    "BluePeak Data",
    "NovaGrid Cloud",
    "Signal Forge",
    "KiteWorks",
    "Summit Logic",
    "ApexForge",
    "CloudHarbor",
    "DevCore Studio",
    "StackPilot",
]

LOCATIONS = [
    "Remote",
    "New York, NY",
    "Austin, TX",
    "Seattle, WA",
    "Boston, MA",
    "Chicago, IL",
    "London, UK",
    "Berlin, Germany",
    "Toronto, ON",
    "Denver, CO",
]

PROMPT_TEMPLATES = [
    "List the fullstack React, Node.js, and .NET job postings from the last 24 hours for {role} roles.",
    "Give me the recent fullstack job postings in the last 24 hours for {role} positions with React, Node.js, and .NET.",
    "What fullstack React, Node.js, and .NET job postings were published in the last 24 hours for {role}?",
    "Show all {role} fullstack job postings from the past 24 hours, including React, Node.js, and .NET.",
]


def build_row(index: int) -> dict[str, str]:
    role = random.choice(JOB_ROLES)
    company = random.choice(COMPANIES)
    location = random.choice(LOCATIONS)
    prompt = random.choice(PROMPT_TEMPLATES).format(role=role)
    hours_ago = random.randint(1, 24)
    answer = (
        f"[{index}] {company} — {role} ({location}) — React, Node.js, .NET — posted {hours_ago} hours ago. "
        f"Apply via the company careers page or recruiter contact."
    )
    return {
        "id": str(index),
        "role": role,
        "company": company,
        "location": location,
        "hours_ago": str(hours_ago),
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
        writer = csv.DictWriter(file, fieldnames=["id", "role", "company", "location", "hours_ago", "prompt", "answer"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic full-stack job-posting dataset for the last 24 hours.")
    parser.add_argument("--rows", type=int, default=100, help="Number of rows to generate")
    parser.add_argument(
        "--format",
        choices=["jsonl", "csv"],
        default="jsonl",
        help="Output dataset format",
    )
    parser.add_argument(
        "--output",
        default="data/datasets/job_postings_last_24h.jsonl",
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

    print(f"Generated {len(rows)} job-posting rows at {output_path}")


if __name__ == "__main__":
    main()