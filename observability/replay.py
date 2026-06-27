import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from observability.tracing import load_trace_log, get_run_events

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_SET_PATH = os.path.join(BASE_DIR, "eval", "golden_set.jsonl")

STRATEGY_CLASSES = {}


def _lazy_load_strategy_classes():
    global STRATEGY_CLASSES
    if not STRATEGY_CLASSES:
        from strategies.plan_and_execute import PlanAndExecuteStrategy
        from strategies.self_consistency import SelfConsistencyStrategy
        from strategies.tree_of_thoughts import TreeOfThoughtsStrategy
        STRATEGY_CLASSES = {
            "plan_and_execute": PlanAndExecuteStrategy,
            "self_consistency": SelfConsistencyStrategy,
            "tree_of_thoughts": TreeOfThoughtsStrategy,
        }
    return STRATEGY_CLASSES


def load_golden_set():
    problems = {}
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                problems[rec["id"]] = rec
    return problems


def find_run_metadata(run_id: str):
    """Looks up the original (strategy, problem_id) for a run_id from the
    trace log."""
    events = get_run_events(run_id)
    if not events:
        return None
    return {"strategy": events[0]["strategy"], "problem_id": events[0]["problem_id"], "n_events": len(events)}


def print_original_trace(run_id: str):
    events = get_run_events(run_id)
    print(f"--- ORIGINAL TRACE for run_id={run_id} ({len(events)} events) ---")
    for e in events:
        print(f"  [{e['event_index']}] {e['step_type']:15s} {json.dumps(e['outputs'])[:120]}")


def replay(run_id: str = None, problem_id: str = None, strategy_override: str = None):
    golden_set = load_golden_set()
    strategy_classes = _lazy_load_strategy_classes()

    if run_id:
        meta = find_run_metadata(run_id)
        if meta is None:
            print(f"No trace found for run_id={run_id}. Is logs/trace_log.jsonl present and not cleared?")
            return
        print_original_trace(run_id)
        problem_id = meta["problem_id"]
        strategy_name = strategy_override or meta["strategy"]
    else:
        if not problem_id:
            print("Must specify --run-id or --problem-id.")
            return
        strategy_name = strategy_override or "plan_and_execute"

    if problem_id not in golden_set:
        print(f"Problem id {problem_id} not found in golden set.")
        return
    problem = golden_set[problem_id]

    print(f"\n--- REPLAYING problem={problem_id} with strategy={strategy_name} ---")
    print(f"Question: {problem['question']}")
    print(f"Ground truth: {problem['answer']}")

    strategy = strategy_classes[strategy_name]()
    result = strategy.solve(problem)

    print(f"\nNew run_id: {result.run_id}")
    print(f"Predicted answer: {result.final_answer}")
    print(f"Correct: {abs((result.final_answer or float('nan')) - problem['answer']) < 1e-4}")
    print(f"n_llm_calls={result.n_llm_calls}, n_tool_calls={result.n_tool_calls}, "
          f"latency_ms={result.latency_ms:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--problem-id", type=str, default=None)
    ap.add_argument("--strategy", type=str, default=None,
                     choices=["plan_and_execute", "self_consistency", "tree_of_thoughts"])
    args = ap.parse_args()

    if not args.run_id and not args.problem_id:
        print("Specify --run-id <id> (replay exact original) or --problem-id <id> [--strategy <name>].")
        sys.exit(1)

    replay(run_id=args.run_id, problem_id=args.problem_id, strategy_override=args.strategy)
