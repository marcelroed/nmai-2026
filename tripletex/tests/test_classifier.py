import csv
import importlib.util
import os

import pytest

from tripletex.classifier import classify

PARSER_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "tripletex", "parsers")


def _load_parser_prompts() -> dict[int, set[str]]:
    result = {}
    for f in os.listdir(PARSER_DIR):
        if not f.startswith("task") or not f.endswith(".py"):
            continue
        num = int(f.split("_")[0].replace("task", ""))
        spec = importlib.util.spec_from_file_location(f[:-3], os.path.join(PARSER_DIR, f))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result[num] = set(getattr(module, "prompts", []))
    return result


def _load_csv_prompts() -> list[str]:
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "tasks", "task_1_2_3.csv")
    seen = set()
    unique = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row["prompt"]
            if p not in seen:
                seen.add(p)
                unique.append(p)
    return unique


PARSER_PROMPTS = _load_parser_prompts()
CSV_PROMPTS = _load_csv_prompts()


@pytest.mark.parametrize("prompt", CSV_PROMPTS, ids=[f"prompt_{i}" for i in range(len(CSV_PROMPTS))])
def test_classifier_maps_prompt_to_correct_parser(prompt: str):
    task_num = classify(prompt)
    assert task_num in PARSER_PROMPTS, f"classify returned {task_num}, but no parser file for that task"
    assert prompt in PARSER_PROMPTS[task_num], (
        f"classify returned task {task_num}, but prompt not in that parser's prompts list"
    )
