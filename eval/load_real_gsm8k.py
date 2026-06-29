import argparse
import json
import os
import re


def load_gsm8k(n: int):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        m = re.search(r"####\s*([\-\d,.]+)", row["answer"])
        if not m:
            continue
        ans = float(m.group(1).replace(",", ""))
        problems.append({
            "id": f"gsm_{i+1:02d}",
            "question": row["question"],
            "answer": ans,
            "category": "real_gsm8k",
        })
    return problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    problems = load_gsm8k(args.n)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_set.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for p in problems:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {len(problems)} real GSM8K problems to {out_path}")
    print("Remember to redo eval/judge_sanity_check.jsonl hand-grades for the new set.")
