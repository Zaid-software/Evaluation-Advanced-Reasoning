
import json
import os
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_RESULTS_PATH = os.path.join(BASE_DIR, "eval", "raw_results.jsonl")
GOLDEN_SET_PATH = os.path.join(BASE_DIR, "eval", "golden_set.jsonl")

# Hand-classified failures with worked reasoning. Each entry: (strategy,
# problem_id) -> {category, reasoning}
HAND_CLASSIFICATIONS = {
    ("plan_and_execute", "gsm_09"): {
        "category": "misread_quantity",
        "reasoning": "True answer is 300 (60% of 500L tank remaining). Predicted "
                     "221.36 doesn't correspond to any clean variant of the 40%/60% "
                     "split -- consistent with the executor having substituted a "
                     "different percentage or base quantity than the one stated, "
                     "then computing correctly from that wrong number.",
    },
    ("plan_and_execute", "gsm_10"): {
        "category": "dropped_step",
        "reasoning": "True answer is 14 full days (300/23, floor). Predicted 11.44 "
                     "looks like the raw division result before the 'how many FULL "
                     "days' step (flooring to an integer) was applied -- the plan "
                     "likely treated the division as the final step instead of an "
                     "intermediate one.",
    },
    ("plan_and_execute", "gsm_19"): {
        "category": "arithmetic_slip",
        "reasoning": "True answer 75 (Jake spends 1/4 of $200=$50, left with $150, "
                     "then spends 1/2 of that = $75 left). Predicted 80 is close in "
                     "magnitude and structure -- consistent with a small slip in one "
                     "of the two sequential fraction calculations rather than a "
                     "wrong overall approach.",
    },
    ("plan_and_execute", "gsm_20"): {
        "category": "off_by_one",
        "reasoning": "True answer 300 (480/8*5). Predicted 299 is off by exactly 1 -- "
                     "classic rounding/truncation slip in a proportional-rate "
                     "calculation rather than a structural error.",
    },
    ("tree_of_thoughts", "gsm_07"): {
        "category": "off_by_one",
        "reasoning": "True answer 125 (5*28-15). Predicted 124, off by exactly 1. "
                     "The search tree's best-value leaf was very close to correct, "
                     "suggesting the value function correctly favored a "
                     "nearly-right branch but a final-step subtraction was off by one.",
    },
    ("tree_of_thoughts", "gsm_08"): {
        "category": "misread_quantity",
        "reasoning": "True answer 2160 ($18*6*5*4). Predicted 2542.73 is far enough "
                     "off (~18% high) that it's not a small slip -- looks like one "
                     "of the four multiplied factors (hours, days, weeks, or rate) "
                     "was substituted with a wrong value partway through the tree.",
    },
    ("tree_of_thoughts", "gsm_09"): {
        "category": "misread_quantity",
        "reasoning": "Same predicted value (221.36) as plan_and_execute's gsm_09 "
                     "failure -- both strategies converged on the same wrong "
                     "intermediate quantity for this problem, suggesting the "
                     "question's '40% full' framing is a genuinely confusable "
                     "phrasing (easy to subtract the wrong percentage) rather than "
                     "a strategy-specific bug.",
    },
    ("tree_of_thoughts", "gsm_14"): {
        "category": "arithmetic_slip",
        "reasoning": "True answer 45 ($40*1.25=$50, then $50*0.9=$45, a markup-then-"
                     "discount chain). Predicted 49 is close, consistent with a slip "
                     "in one of the two percentage operations (e.g. applying both "
                     "percentages to the original $40 instead of chaining them).",
    },
    ("tree_of_thoughts", "gsm_18"): {
        "category": "dropped_step",
        "reasoning": "True answer 6.25 cups (proportion: 2.5/20*50). Predicted 3.38 "
                     "is roughly half the correct answer -- consistent with the "
                     "search tree terminating on a branch that computed only part "
                     "of the proportional-scaling chain (e.g. scaled by 25 "
                     "cookies-worth instead of 50).",
    },
    ("tree_of_thoughts", "gsm_19"): {
        "category": "misread_quantity",
        "reasoning": "True answer 75 (same problem as plan_and_execute's gsm_19 "
                     "failure above). Predicted 85.33 is a different wrong value than "
                     "plan_and_execute's 80 -- here the magnitude (~14% off, non-"
                     "integer) suggests a wrong fraction was used in one of the two "
                     "sequential 1/4, 1/2 reductions rather than a slip in a clean "
                     "integer step.",
    },
    ("tree_of_thoughts", "gsm_23"): {
        "category": "off_by_one",
        "reasoning": "True answer 43 (38-9+14). Predicted 42, off by exactly 1 -- "
                     "same pattern as gsm_07: a near-correct branch with a final "
                     "single-unit slip, the kind of error self-consistency's "
                     "majority vote is specifically designed to catch.",
    },
}


def load_failures():
    failures = []
    with open(RAW_RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if not rec["is_correct"]:
                failures.append(rec)
    return failures


def run_taxonomy():
    failures = load_failures()
    classified = []
    unclassified = []

    for failure in failures:
        key = (failure["strategy"], failure["problem_id"])
        if key in HAND_CLASSIFICATIONS:
            entry = dict(failure)
            entry.update(HAND_CLASSIFICATIONS[key])
            classified.append(entry)
        else:
            unclassified.append(failure)

    distribution = Counter(c["category"] for c in classified)
    return classified, unclassified, distribution


if __name__ == "__main__":
    classified, unclassified, distribution = run_taxonomy()

    print(f"Classified {len(classified)}/{len(classified) + len(unclassified)} failures.\n")
    print("=== Failure category distribution ===")
    for category, count in distribution.most_common():
        print(f"  {category:20s} {count}")

    print("\n=== Worked examples ===")
    for entry in classified:
        print(f"\n[{entry['strategy']} / {entry['problem_id']}] category={entry['category']}")
        print(f"  predicted={entry['predicted_answer']}  true={entry['ground_truth']}")
        print(f"  reasoning: {entry['reasoning']}")

    if unclassified:
        print(f"\nWARNING: {len(unclassified)} failures have no hand classification yet:")
        for u in unclassified:
            print(f"  {u['strategy']} / {u['problem_id']}")
