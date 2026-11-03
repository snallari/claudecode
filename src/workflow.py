from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.dataset import generate_dataset, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end dataset + evaluation workflow.")
    parser.add_argument(
        "--dataset",
        default="data/datasets/workflow_dataset.jsonl",
        help="Path to workflow dataset file",
    )
    parser.add_argument("--rows", type=int, default=25, help="Rows to generate when dataset file does not exist")
    parser.add_argument("--seed", type=int, default=42, help="Seed for dataset generation")
    parser.add_argument(
        "--regenerate-dataset",
        action="store_true",
        help="Regenerate dataset even if --dataset already exists",
    )

    parser.add_argument(
        "--mode",
        choices=["auto", "api", "demo"],
        default="auto",
        help="Execution mode passed to src.eval",
    )
    parser.add_argument("--model", default=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"), help="Prediction model")
    parser.add_argument("--fallback-model", default=os.getenv("ANTHROPIC_FALLBACK_MODEL"), help="Prediction fallback model")
    parser.add_argument("--max-tokens", type=int, default=256, help="Prediction max tokens")

    parser.add_argument(
        "--grader",
        choices=["lexical", "model", "code", "coding"],
        default=os.getenv("EVAL_GRADER", "lexical"),
        help="Grading strategy passed to src.eval (lexical|model|code|coding)",
    )
    parser.add_argument(
        "--grader-model",
        default=os.getenv("ANTHROPIC_GRADER_MODEL"),
        help="Grader model when --grader model",
    )
    parser.add_argument(
        "--grader-fallback-model",
        default=os.getenv("ANTHROPIC_GRADER_FALLBACK_MODEL"),
        help="Grader fallback model",
    )
    parser.add_argument("--grader-max-tokens", type=int, default=300, help="Grader max tokens")
    parser.add_argument(
        "--code-timeout-seconds",
        type=float,
        default=3.0,
        help="Timeout in seconds for per-row code execution when --grader code",
    )

    parser.add_argument("--output-dir", default="data/eval_runs", help="Output directory for eval artifacts")
    parser.add_argument("--run-name", default=None, help="Optional run name")
    parser.add_argument("--extra-prompt", default=None, help="Optional inline extra prompt")
    parser.add_argument("--extra-prompt-file", default=None, help="Optional extra prompt file")
    parser.add_argument("--fail-fast", action="store_true", help="Pass --fail-fast to src.eval")
    return parser.parse_args()


def ensure_dataset(path: Path, rows: int, seed: int, regenerate: bool) -> None:
    if path.exists() and not regenerate:
        print(f"Using existing dataset: {path}")
        return

    generated = generate_dataset(rows=rows, seed=seed)
    write_jsonl(generated, path)
    print(f"Generated dataset: {path} (rows={len(generated)}, seed={seed})")


def build_eval_command(args: argparse.Namespace) -> list[str]:
    command: list[str] = [
        sys.executable,
        "-m",
        "src.eval",
        "--dataset",
        args.dataset,
        "--mode",
        args.mode,
        "--model",
        args.model,
        "--max-tokens",
        str(args.max_tokens),
        "--grader",
        args.grader,
        "--grader-max-tokens",
        str(args.grader_max_tokens),
        "--code-timeout-seconds",
        str(args.code_timeout_seconds),
        "--output-dir",
        args.output_dir,
    ]

    if args.fallback_model:
        command.extend(["--fallback-model", args.fallback_model])
    if args.grader_model:
        command.extend(["--grader-model", args.grader_model])
    if args.grader_fallback_model:
        command.extend(["--grader-fallback-model", args.grader_fallback_model])
    if args.run_name:
        command.extend(["--run-name", args.run_name])
    if args.extra_prompt:
        command.extend(["--extra-prompt", args.extra_prompt])
    if args.extra_prompt_file:
        command.extend(["--extra-prompt-file", args.extra_prompt_file])
    if args.fail_fast:
        command.append("--fail-fast")

    return command


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    ensure_dataset(dataset_path, rows=args.rows, seed=args.seed, regenerate=args.regenerate_dataset)
    eval_command = build_eval_command(args)

    print("Running evaluation workflow:")
    print(" ".join(eval_command))

    completed = subprocess.run(eval_command, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
