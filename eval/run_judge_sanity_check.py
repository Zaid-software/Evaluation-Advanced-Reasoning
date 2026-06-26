import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.judge import judge_trace_quality

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SANITY_SET_PATH = os.path.join(BASE_DIR, "eval", "judge_sanity_check.jsonl")
RESULTS_PATH = os.path.join(BASE_DIR, "eval", "judge_sanity_check_results.json")

AGREEMENT_THRESHOLD = 0.70


def load_sanity_set():
    items = []
    with open(SANITY_SET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_sanity_check():
    samples = load_sanity_set()
    rows = []
    n_agree = 0

    for sample in samples:
        verdict = judge_trace_quality(sample["strategy"], sample["question"], sample["trace_summary"])
        agrees = verdict["score"] == sample["human_label"]
        n_agree += int(agrees)
        rows.append({
            "sample_id": sample["sample_id"],
            "strategy": sample["strategy"],
            "human_label": sample["human_label"],
            "human_reason": sample["human_reason"],
            "judge_score": verdict["score"],
            "judge_reason": verdict["reason"],
            "judge_model": verdict["judge_model"],
            "agrees": agrees,
        })

    accuracy = n_agree / len(samples) if samples else 0.0
    return rows, accuracy


if __name__ == "__main__":
    rows, accuracy = run_sanity_check()

    print(f"Judge-human agreement: {accuracy:.2%} ({sum(r['agrees'] for r in rows)}/{len(rows)})")
    print(f"Threshold for trusting judge: {AGREEMENT_THRESHOLD:.0%}")
    print()
    print(f"{'ID':8s} {'Strategy':18s} {'Human':6s} {'Judge':6s} {'Agree?'}")
    print("-" * 60)
    for r in rows:
        print(f"{r['sample_id']:8s} {r['strategy']:18s} {r['human_label']:<6d} {r['judge_score']:<6d} {r['agrees']}")

    if accuracy < AGREEMENT_THRESHOLD:
        print(f"\nWARNING: judge agreement {accuracy:.2%} is BELOW the {AGREEMENT_THRESHOLD:.0%} threshold. "
              "Per the assignment's own guidance, the rubric should be fixed before trusting judge scores "
              "on the full eval set.")
    else:
        print(f"\nPASS: judge agreement {accuracy:.2%} meets the {AGREEMENT_THRESHOLD:.0%} threshold.")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy, "threshold": AGREEMENT_THRESHOLD, "rows": rows}, f, indent=2)
    print(f"\nFull results written to {RESULTS_PATH}")
