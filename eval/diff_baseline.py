import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_PATH = os.path.join(BASE_DIR, "eval", "baseline.json")
STATS_PATH = os.path.join(BASE_DIR, "eval", "stats_results.json")


def main():
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    with open(STATS_PATH) as f:
        current = json.load(f)

    print(f"Baseline recorded: {baseline['recorded_at']} ({baseline['recorded_against']})\n")
    print(f"{'Strategy':20s} {'Baseline':10s} {'Current':10s} {'Delta'}")
    print("-" * 55)
    for strat, base_row in baseline["accuracy"].items():
        current_row = current["accuracy"].get(strat)
        if current_row is None:
            print(f"{strat:20s} {base_row['accuracy']:<10.4f} (missing from current run)")
            continue
        delta = current_row["accuracy"] - base_row["accuracy"]
        flag = "  <-- CHANGED" if abs(delta) > 1e-9 else ""
        print(f"{strat:20s} {base_row['accuracy']:<10.4f} {current_row['accuracy']:<10.4f} {delta:+.4f}{flag}")


if __name__ == "__main__":
    main()
