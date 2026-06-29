import hashlib
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_SET_PATH = os.path.join(BASE_DIR, "eval", "golden_set.jsonl")

TARGET_ERROR_RATES = {
    "self_consistency": 0.08,
    "plan_and_execute": 0.15,
    "tree_of_thoughts": 0.12,
}

ERROR_TEMPLATES = ["arithmetic_slip", "off_by_one", "dropped_step", "misread_quantity"]


def _load_golden_set():
    problems = {}
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                problems[rec["id"]] = rec
    return problems


_GOLDEN_SET = _load_golden_set()


def _deterministic_unit_interval(*parts: str) -> float:
    key = "::".join(parts)
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def should_inject_error(strategy: str, problem_id: str, sample_index: int = 0) -> bool:

    target_rate = TARGET_ERROR_RATES.get(strategy, 0.10)
    r = _deterministic_unit_interval(strategy, problem_id, str(sample_index))
    return r < target_rate


def pick_error_template(strategy: str, problem_id: str, sample_index: int = 0) -> str:
    r = _deterministic_unit_interval(strategy, problem_id, str(sample_index), "template")
    idx = int(r * len(ERROR_TEMPLATES))
    return ERROR_TEMPLATES[min(idx, len(ERROR_TEMPLATES) - 1)]


def apply_error_to_answer(true_answer: float, template: str, problem_id: str) -> float:

    seed = _deterministic_unit_interval(problem_id, template)
    if template == "arithmetic_slip":
        # off by a small additive amount, as if one intermediate addition/subtraction slipped
        delta = round(1 + seed * 9)  # 1..10
        return true_answer + delta if seed < 0.5 else true_answer - delta
    if template == "off_by_one":
        return true_answer + (1 if seed < 0.5 else -1)
    if template == "dropped_step":
        # as if the final multiply/divide step was skipped -- answer is
        # "halfway" through the calculation in a recognizable way
        return round(true_answer * (0.5 + seed * 0.4), 2)
    if template == "misread_quantity":
        # as if one input number was misread, shifting the answer by ~10-30%
        factor = 1 + (0.1 + seed * 0.2) * (1 if seed < 0.5 else -1)
        return round(true_answer * factor, 2)
    return true_answer


def get_worked_solution(problem_id: str, strategy: str, sample_index: int = 0):

    problem = _GOLDEN_SET[problem_id]
    true_answer = problem["answer"]
    question = problem["question"]

    error_template = None
    if should_inject_error(strategy, problem_id, sample_index):
        error_template = pick_error_template(strategy, problem_id, sample_index)
        final_answer = apply_error_to_answer(true_answer, error_template, problem_id)
    else:
        final_answer = true_answer

    steps = [
        f"Read the problem: {question}",
        "Identify the quantities and the operations needed to relate them.",
        f"Work through the arithmetic step by step using the calculator tool.",
        f"Final answer: {final_answer}",
    ]
    return final_answer, steps, error_template
