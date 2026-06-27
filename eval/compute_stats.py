import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats as scipy_stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_RESULTS_PATH = os.path.join(BASE_DIR, "eval", "raw_results.jsonl")
STATS_OUTPUT_PATH = os.path.join(BASE_DIR, "eval", "stats_results.json")


def load_raw_results():
    records = []
    with open(RAW_RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def wilson_ci(n_correct: int, n_total: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion. More robust than the
    normal approximation at small N or extreme proportions (e.g. 24/24 or
    0/24), both of which can occur in our 24-problem eval set."""
    if n_total == 0:
        return 0.0, 0.0, 0.0
    p_hat = n_correct / n_total
    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    half_width = (z * math.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2))) / denom
    return p_hat, max(0.0, center - half_width), min(1.0, center + half_width)


def per_strategy_accuracy(records):
    strategies = sorted(set(r["strategy"] for r in records))
    results = {}
    for strat in strategies:
        strat_records = [r for r in records if r["strategy"] == strat]
        n_correct = sum(r["is_correct"] for r in strat_records)
        n_total = len(strat_records)
        p_hat, ci_low, ci_high = wilson_ci(n_correct, n_total)
        results[strat] = {
            "n_correct": n_correct, "n_total": n_total,
            "accuracy": round(p_hat, 4), "ci_95_low": round(ci_low, 4), "ci_95_high": round(ci_high, 4),
        }
    return results


def build_outcome_matrix(records):
    """Returns {strategy: {problem_id: is_correct}}."""
    matrix = {}
    for r in records:
        matrix.setdefault(r["strategy"], {})[r["problem_id"]] = r["is_correct"]
    return matrix


def mcnemar_test(outcomes_a: dict, outcomes_b: dict):
    """outcomes_a/b: {problem_id: bool}. Returns (b01, b10, statistic, p_value)
    where b01 = A wrong & B right, b10 = A right & B wrong (standard McNemar
    notation). Uses the exact binomial McNemar test (more appropriate than
    the chi-square approximation at our sample size, where the chi-square
    approx with continuity correction can be unreliable for small
    discordant-pair counts)."""
    problem_ids = set(outcomes_a) & set(outcomes_b)
    b01 = sum(1 for pid in problem_ids if not outcomes_a[pid] and outcomes_b[pid])
    b10 = sum(1 for pid in problem_ids if outcomes_a[pid] and not outcomes_b[pid])
    n_discordant = b01 + b10
    if n_discordant == 0:
        return b01, b10, None, 1.0
    # exact binomial test: under H0, b10 ~ Binomial(n_discordant, 0.5)
    result = scipy_stats.binomtest(min(b01, b10), n_discordant, 0.5, alternative="two-sided")
    return b01, b10, n_discordant, result.pvalue


def build_win_matrix(records):
    matrix = build_outcome_matrix(records)
    strategies = sorted(matrix.keys())
    win_matrix = {}
    mcnemar_results = {}

    for a in strategies:
        win_matrix[a] = {}
        for b in strategies:
            if a == b:
                win_matrix[a][b] = None
                continue
            problem_ids = set(matrix[a]) & set(matrix[b])
            a_wins = sum(1 for pid in problem_ids if matrix[a][pid] and not matrix[b][pid])
            b_wins = sum(1 for pid in problem_ids if matrix[b][pid] and not matrix[a][pid])
            both = sum(1 for pid in problem_ids if matrix[a][pid] and matrix[b][pid])
            neither = sum(1 for pid in problem_ids if not matrix[a][pid] and not matrix[b][pid])
            win_matrix[a][b] = {
                f"{a}_wins": a_wins, f"{b}_wins": b_wins, "both_correct": both, "both_wrong": neither,
            }

            pair_key = tuple(sorted([a, b]))
            if pair_key not in mcnemar_results:
                b01, b10, n_disc, p_value = mcnemar_test(matrix[a], matrix[b])
                mcnemar_results[pair_key] = {
                    "n_discordant_pairs": n_disc, "p_value": round(p_value, 4) if p_value is not None else None,
                    "significant_at_0.05": (p_value is not None and p_value < 0.05),
                }

    return win_matrix, mcnemar_results


def cost_latency_table(records, cost_per_1k_tokens_in=0.0002, cost_per_1k_tokens_out=0.0002):
    strategies = sorted(set(r["strategy"] for r in records))
    table = {}
    for strat in strategies:
        strat_records = [r for r in records if r["strategy"] == strat]
        tokens_in = sum(r["tokens_in"] for r in strat_records)
        tokens_out = sum(r["tokens_out"] for r in strat_records)
        latency_ms = sum(r["latency_ms"] for r in strat_records)
        n_correct = sum(r["is_correct"] for r in strat_records)
        cost = (tokens_in / 1000) * cost_per_1k_tokens_in + (tokens_out / 1000) * cost_per_1k_tokens_out
        table[strat] = {
            "tokens_in_total": tokens_in, "tokens_out_total": tokens_out,
            "wall_clock_ms_total": round(latency_ms, 1),
            "n_llm_calls_total": sum(r["n_llm_calls"] for r in strat_records),
            "n_tool_calls_total": sum(r["n_tool_calls"] for r in strat_records),
            "estimated_cost_usd": round(cost, 6),
            "cost_per_correct_answer_usd": round(cost / n_correct, 6) if n_correct else None,
        }
    return table


def _json_default(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def run_stats():
    records = load_raw_results()
    accuracy = per_strategy_accuracy(records)
    win_matrix, mcnemar_results = build_win_matrix(records)
    cost_table = cost_latency_table(records)

    output = {
        "accuracy": accuracy,
        "win_matrix": win_matrix,
        "mcnemar": {f"{a} vs {b}": v for (a, b), v in mcnemar_results.items()},
        "cost_latency": cost_table,
    }
    with open(STATS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=_json_default)
    return output


if __name__ == "__main__":
    output = run_stats()

    print("=== Per-strategy accuracy (95% Wilson CI) ===")
    for strat, stats_row in output["accuracy"].items():
        print(f"  {strat:20s} {stats_row['n_correct']}/{stats_row['n_total']} = "
              f"{stats_row['accuracy']:.3f}  CI=[{stats_row['ci_95_low']:.3f}, {stats_row['ci_95_high']:.3f}]")

    print("\n=== McNemar pairwise significance ===")
    for pair, result in output["mcnemar"].items():
        sig = "SIGNIFICANT (p<0.05)" if result["significant_at_0.05"] else "not significant"
        print(f"  {pair:45s} n_discordant={result['n_discordant_pairs']:<3} "
              f"p={result['p_value']}  {sig}")

    print("\n=== Cost / latency ===")
    for strat, row in output["cost_latency"].items():
        print(f"  {strat:20s} tokens_in={row['tokens_in_total']:<6} tokens_out={row['tokens_out_total']:<6} "
              f"wall_ms={row['wall_clock_ms_total']:<8} cost/correct=${row['cost_per_correct_answer_usd']}")

    print(f"\nFull stats written to {STATS_OUTPUT_PATH}")
