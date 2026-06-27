import argparse
import json
import os

from strategies.plan_and_execute import PlanAndExecuteStrategy
from strategies.self_consistency import SelfConsistencyStrategy
from strategies.tree_of_thoughts import TreeOfThoughtsStrategy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GOLDEN_SET_PATH = os.path.join(BASE_DIR, "eval", "golden_set.jsonl")

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


def print_tree(result):
    tree = result.metadata.get("tree")
    if not tree:
        return
    print("\n  --- Tree-of-Thoughts search tree ---")
    for node in tree["nodes"]:
        status = "[PRUNED]" if node["node_id"] in tree["pruned_node_ids"] else "[KEPT]  "
        print(f"  {status} {node['node_id']:5s} parent={node['parent_id']:5s} "
              f"depth={node['depth']} value={node['value']:.3f}  {node['step'][:60]}")
    print(f"  Best leaf: {tree['best_leaf_node_id']} (value={tree['best_leaf_value']})")
    print(f"  Pruned: {len(tree['pruned_node_ids'])}/{len(tree['nodes'])} branches")


def run_demo():
    problems = load_golden_set()
    # pick a problem with enough structure to show real strategy differences
    problem = next(p for p in problems if p["id"] == "gsm_09")

    print("=" * 80)
    print(f"DEMO: same problem, all 3 strategies")
    print(f"Question: {problem['question']}")
    print(f"Ground truth: {problem['answer']}")
    print("=" * 80)

    for name, cls in STRATEGY_CLASSES.items():
        strategy = cls()
        result = strategy.solve(problem)
        correct = result.final_answer is not None and abs(result.final_answer - problem["answer"]) < 1e-4
        print(f"\n--- {name} ---")
        print(f"  Predicted: {result.final_answer}  (correct: {correct})")
        print(f"  n_llm_calls={result.n_llm_calls}, n_tool_calls={result.n_tool_calls}, "
              f"tokens_in={result.tokens_in}, tokens_out={result.tokens_out}")
        print(f"  run_id={result.run_id}")
        if name == "plan_and_execute":
            print(f"  Plan: {result.metadata['plan_steps']}")
        if name == "self_consistency":
            print(f"  Samples: {result.metadata['samples']}  votes: {result.metadata['vote_distribution']}")
        if name == "tree_of_thoughts":
            print_tree(result)

    print("\n" + "=" * 80)
    print("Run `python -m eval.run_eval` for the full 24-problem eval, or")
    print("`make eval` for eval + stats + baseline diff.")


def run_solve(question_id_or_text: str, strategy_name: str):
    problems = load_golden_set()
    problem = next((p for p in problems if p["id"] == question_id_or_text), None)
    if problem is None:
        # treat as raw question text with no ground truth (can't grade, but can still run)
        problem = {"id": "adhoc", "question": question_id_or_text, "answer": None, "category": "adhoc"}

    strategy = STRATEGY_CLASSES[strategy_name]()
    result = strategy.solve(problem)
    print(f"Question: {problem['question']}")
    if problem["answer"] is not None:
        print(f"Ground truth: {problem['answer']}")
    print(f"Predicted: {result.final_answer}")
    print(f"run_id: {result.run_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--solve", type=str, help="golden-set problem id (e.g. gsm_01) or raw question text")
    ap.add_argument("--strategy", type=str, default="plan_and_execute", choices=list(STRATEGY_CLASSES.keys()))
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--run-id", type=str, default=None)
    args = ap.parse_args()

    if args.demo:
        run_demo()
    elif args.solve:
        run_solve(args.solve, args.strategy)
    elif args.replay:
        from observability.replay import replay
        replay(run_id=args.run_id, strategy_override=args.strategy if args.strategy != "plan_and_execute" else None)
    else:
        print("Specify --demo, --solve \"...\" [--strategy ...], or --replay --run-id <id>. See --help.")
