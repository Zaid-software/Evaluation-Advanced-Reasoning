import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.plan_and_execute import PlanAndExecuteStrategy
from strategies.self_consistency import SelfConsistencyStrategy
from strategies.tree_of_thoughts import TreeOfThoughtsStrategy
from eval.metrics import grade_answer
from eval.judge import judge_trace_quality

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_SET_PATH = os.path.join(BASE_DIR, "eval", "golden_set.jsonl")
RAW_RESULTS_PATH = os.path.join(BASE_DIR, "eval", "raw_results.jsonl")

STRATEGY_CLASSES = {
    "plan_and_execute": PlanAndExecuteStrategy,
    "self_consistency": SelfConsistencyStrategy,
    "tree_of_thoughts": TreeOfThoughtsStrategy,
}


def load_golden_set():
    problems = []
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def trace_summary_for_judge(result) -> str:
    parts = [f"Final answer: {result.final_answer}",
             f"n_llm_calls={result.n_llm_calls} (LLM reasoning calls made)",
             f"n_tool_calls={result.n_tool_calls} (calculator tool invocations)"]
    if "plan_steps" in result.metadata:
        parts.append(f"Plan steps ({len(result.metadata['plan_steps'])} total): "
                      + " | ".join(result.metadata["plan_steps"]))
    if "samples" in result.metadata:
        parts.append(f"Independently sampled reasoning paths (N={len(result.metadata['samples'])}): "
                      f"{result.metadata['samples']}, "
                      f"vote distribution: {result.metadata.get('vote_distribution', {})}")
    if "tree" in result.metadata:
        tree = result.metadata["tree"]
        parts.append(f"Search tree: {len(tree['nodes'])} branch nodes explored across multiple depths, "
                      f"{len(tree['pruned_node_ids'])} pruned by beam search (width={tree['beam_width']}), "
                      f"best leaf node {tree.get('best_leaf_node_id')} selected by value function "
                      f"(value={tree.get('best_leaf_value')})")
    return "\n".join(parts)


def run_eval(strategy_names=None, verbose=True):
    problems = load_golden_set()
    strategy_names = strategy_names or list(STRATEGY_CLASSES.keys())
    results = []

    os.makedirs(os.path.dirname(RAW_RESULTS_PATH), exist_ok=True)
    with open(RAW_RESULTS_PATH, "w", encoding="utf-8") as out_f:
        for strategy_name in strategy_names:
            strategy = STRATEGY_CLASSES[strategy_name]()
            if verbose:
                print(f"\n=== Running {strategy_name} on {len(problems)} problems ===")

            for i, problem in enumerate(problems):
                t0 = time.time()
                result = strategy.solve(problem)
                wall_time = (time.time() - t0) * 1000

                is_correct = grade_answer(result.final_answer, problem["answer"])
                judge_verdict = judge_trace_quality(
                    strategy_name, problem["question"], trace_summary_for_judge(result)
                )

                record = {
                    "problem_id": problem["id"],
                    "category": problem["category"],
                    "strategy": strategy_name,
                    "run_id": result.run_id,
                    "ground_truth": problem["answer"],
                    "predicted_answer": result.final_answer,
                    "is_correct": is_correct,
                    "judge_score": judge_verdict["score"],
                    "judge_reason": judge_verdict["reason"],
                    "judge_model": judge_verdict["judge_model"],
                    "tokens_in": result.tokens_in,
                    "tokens_out": result.tokens_out,
                    "latency_ms": result.latency_ms,
                    "wall_time_ms": wall_time,
                    "n_llm_calls": result.n_llm_calls,
                    "n_tool_calls": result.n_tool_calls,
                    "error_template_injected": result.metadata.get("error_template_injected"),
                }
                results.append(record)
                out_f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

                if verbose:
                    status = "OK  " if is_correct else "MISS"
                    print(f"  [{status}] {problem['id']:10s} pred={result.final_answer!s:>10s} "
                          f"true={problem['answer']!s:>8s} judge={judge_verdict['score']}")

    if verbose:
        print(f"\nRaw results written to {RAW_RESULTS_PATH}")
    return results


if __name__ == "__main__":
    run_eval()
