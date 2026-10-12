from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GOALS = [
    "build lean muscle",
    "improve endurance",
    "maintain performance",
    "reduce body fat",
]

RESTRICTIONS = [
    "none",
    "vegetarian",
    "lactose-free",
    "gluten-free",
    "nut-free",
]

DEFAULT_EVAL_CRITERIA = [
    "Includes breakfast, lunch, and dinner",
    "Aligns with the athlete goal",
    "Respects dietary restrictions",
    "Keeps the plan concise and practical",
]

DEFAULT_TASK_DESCRIPTION = (
    "Generate a compact one-day meal plan for a single athlete. "
    "Include breakfast, lunch, dinner, and 1-2 snacks. "
    "Ensure the plan strictly follows the athlete's dietary restrictions, "
    "supports their stated goal, and uses practical foods with brief portion guidance."
)


@dataclass
class Evaluator:
    seed: int = 42

    def _sample_input(self, index: int) -> dict[str, Any]:
        random.seed(self.seed + index)
        return {
            "height": random.randint(155, 200),
            "weight": random.randint(50, 110),
            "goal": random.choice(GOALS),
            "restrictions": random.choice(RESTRICTIONS),
        }

    def _build_prompt(
        self,
        task_description: str,
        prompt_inputs_spec: dict[str, str],
        sample_input: dict[str, Any],
        extra_prompt: str | None = None,
        ideal_output: str | None = None,
    ) -> str:
        lines = [task_description.strip(), "", "Inputs:"]
        for key, description in prompt_inputs_spec.items():
            value = sample_input.get(key, "")
            lines.append(f"- {key} ({description}): {value}")

        if ideal_output and ideal_output.strip():
            lines.append("")
            lines.append("Use the following ideal output style as a quality reference:")
            lines.append("<ideal_output>")
            lines.append(ideal_output.strip())
            lines.append("</ideal_output>")
            lines.append(
                "This example is well-structured, provides detailed food choices and quantities, "
                "and aligns with the athlete's goals and restrictions."
            )

        if extra_prompt and extra_prompt.strip():
            lines.append("")
            lines.append("Additional context:")
            lines.append(extra_prompt.strip())

        return "\n".join(lines).strip()

    def _build_expected_output(self, sample_input: dict[str, Any]) -> str:
        return (
            "Breakfast: Oats + fruit + protein. "
            "Lunch: Lean protein, whole grains, vegetables. "
            "Dinner: Protein + complex carbs + salad. "
            "Snacks: Yogurt/fruit/nuts as allowed. "
            f"Target aligns with {sample_input['goal']} and respects {sample_input['restrictions']}."
        )

    def generate_dataset(
        self,
        task_description: str = DEFAULT_TASK_DESCRIPTION,
        *,
        prompt_inputs_spec: dict[str, str],
        output_file: str,
        num_cases: int,
        eval_criteria: list[str] | None = None,
        extra_prompt: str | None = None,
        ideal_output: str | None = None,
    ) -> list[dict[str, Any]]:
        if num_cases <= 0:
            raise ValueError("num_cases must be greater than 0")
        if not task_description.strip():
            raise ValueError("task_description must not be empty")
        if not prompt_inputs_spec:
            raise ValueError("prompt_inputs_spec must not be empty")

        criteria = [criterion.strip() for criterion in (eval_criteria or DEFAULT_EVAL_CRITERIA) if criterion.strip()]
        if not criteria:
            raise ValueError("eval_criteria must contain at least one non-empty criterion")

        dataset: list[dict[str, Any]] = []
        for case_index in range(1, num_cases + 1):
            sample_input = self._sample_input(case_index)
            prompt = self._build_prompt(
                task_description=task_description,
                prompt_inputs_spec=prompt_inputs_spec,
                sample_input=sample_input,
                extra_prompt=extra_prompt,
                ideal_output=ideal_output,
            )
            expected_output = self._build_expected_output(sample_input)

            dataset.append(
                {
                    "id": case_index,
                    "task_description": task_description,
                    "input": sample_input,
                    "prompt": prompt,
                    "expected_output": expected_output,
                    "eval_criteria": criteria,
                }
            )

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(dataset, file, ensure_ascii=False, indent=2)

        return dataset


evaluator = Evaluator()
