from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.workflow_classes import ClaudeRunner, EvalDataset, EvalGrader, PromptDraft, WorkflowRunner, resolve_prompt_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end dataset + evaluation workflow.")
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=[],
        help="Path to a workflow dataset file. Repeat to pass multiple datasets.",
    )
    parser.add_argument("--rows", type=int, default=25, help="Rows to generate when a dataset file does not exist")
    parser.add_argument("--seed", type=int, default=42, help="Seed for dataset generation")
    parser.add_argument(
        "--regenerate-dataset",
        action="store_true",
        help="Regenerate generated dataset file(s) even if already present",
    )
    parser.add_argument("--eval-rows", type=int, default=None, help="Optional max rows to evaluate after loading datasets")

    parser.add_argument(
        "--mode",
        choices=["auto", "api", "demo"],
        default="auto",
        help="Execution mode for Claude runner",
    )
    parser.add_argument("--model", default=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"), help="Prediction model")
    parser.add_argument("--fallback-model", default=os.getenv("ANTHROPIC_FALLBACK_MODEL"), help="Prediction fallback model")
    parser.add_argument("--max-tokens", type=int, default=256, help="Prediction max tokens")

    parser.add_argument(
        "--grader",
        choices=["lexical", "model", "code", "coding"],
        default=os.getenv("EVAL_GRADER", "lexical"),
        help="Grading strategy (lexical|model|code|coding)",
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
    parser.add_argument(
        "--prompt-template",
        default=None,
        help="Optional prompt drafting template. Supported placeholders: {question}, {prompt}, {extra_context}",
    )
    parser.add_argument("--extra-prompt", default=None, help="Optional inline extra prompt")
    parser.add_argument("--extra-prompt-file", default=None, help="Optional extra prompt file")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately if one row fails")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    dataset_paths = [Path(path) for path in (args.datasets or ["data/datasets/workflow_dataset.jsonl"])]
    for dataset_path in dataset_paths:
        dataset_path.parent.mkdir(parents=True, exist_ok=True)

    extra_context = resolve_prompt_context(args.extra_prompt, args.extra_prompt_file)

    dataset = EvalDataset(
        dataset_paths=dataset_paths,
        rows_when_generating=args.rows,
        seed=args.seed,
        regenerate=args.regenerate_dataset,
    )
    prompt_draft = PromptDraft(template=args.prompt_template, extra_context=extra_context)
    claude_runner = ClaudeRunner(
        mode=args.mode,
        model=args.model,
        fallback_model=args.fallback_model,
        max_tokens=args.max_tokens,
    )
    grader = EvalGrader(
        method=args.grader,
        grader_model=args.grader_model,
        grader_fallback_model=args.grader_fallback_model,
        grader_max_tokens=args.grader_max_tokens,
        code_timeout_seconds=args.code_timeout_seconds,
    )

    workflow = WorkflowRunner(
        dataset=dataset,
        prompt_draft=prompt_draft,
        claude=claude_runner,
        grader=grader,
        output_dir=Path(args.output_dir),
        run_name=args.run_name,
        eval_rows=args.eval_rows,
        fail_fast=args.fail_fast,
    )

    exit_code = workflow.run()
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
