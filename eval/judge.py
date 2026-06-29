import re

from tools.llm_client import get_judge_llm, is_offline

JUDGE_SYSTEM_PROMPT = """You are grading the QUALITY of a reasoning trace for a math word problem,
not whether the final number is correct (that's checked separately).

Question: does this trace show a clear, traceable chain of reasoning steps
that plausibly leads to its stated final answer, using explicit arithmetic
(via a calculator tool or shown expressions) rather than just asserting a
number with no derivation shown?

Respond with EXACTLY one line:
SCORE: <0 or 1> | REASON: <short reason>
"""


def _format_trace_for_judge(strategy: str, question: str, trace_summary: str) -> str:
    return f"QUESTION: {question}\n\nSTRATEGY: {strategy}\n\nTRACE SUMMARY:\n{trace_summary}"


def judge_trace_quality(strategy: str, question: str, trace_summary: str) -> dict:
    """Returns {"score": 0 or 1, "reason": str, "judge_model": str}."""
    if is_offline():
        return _judge_offline(strategy, question, trace_summary)

    judge = get_judge_llm()
    prompt = _format_trace_for_judge(strategy, question, trace_summary)
    raw_output, _, _ = judge.call(JUDGE_SYSTEM_PROMPT, prompt, max_tokens=80)
    return _parse_judge_output(raw_output, judge.name)


def _judge_offline(strategy: str, question: str, trace_summary: str) -> dict:
    has_arithmetic_expression = bool(re.search(r"\d+\s*[\+\-\*/]\s*\d+", trace_summary))
    has_multiple_steps = bool(re.search(r"step", trace_summary, re.IGNORECASE)) and \
        len(re.findall(r"step", trace_summary, re.IGNORECASE)) >= 2
    has_tool_call_evidence = bool(re.search(r"tool.{0,15}(call|invocation)", trace_summary, re.IGNORECASE)) \
        and "n_tool_calls=0" not in trace_summary
    has_multi_sample_evidence = bool(re.search(r"sampled reasoning paths.*N=([3-9]|\d{2,})", trace_summary))
    has_search_tree_evidence = bool(re.search(r"branch nodes explored", trace_summary)) and \
        bool(re.search(r"pruned by beam search", trace_summary))

    signals_present = sum([
        has_arithmetic_expression, has_multiple_steps, has_tool_call_evidence,
        has_multi_sample_evidence, has_search_tree_evidence,
    ])
    score = 1 if signals_present >= 1 else 0
    reason = (f"signals: arithmetic_expr={has_arithmetic_expression}, multi_step={has_multiple_steps}, "
              f"tool_call={has_tool_call_evidence}, multi_sample={has_multi_sample_evidence}, "
              f"search_tree={has_search_tree_evidence}")
    return {"score": score, "reason": reason, "judge_model": "stub-rule-based-judge"}


def _parse_judge_output(raw_output: str, judge_model: str) -> dict:
    if not raw_output:
        return {"score": 0, "reason": "empty_judge_output", "judge_model": judge_model}
    m = re.search(r"SCORE:\s*([01])\s*\|\s*REASON:\s*(.+)", raw_output, re.IGNORECASE)
    if not m:
        return {"score": 0, "reason": f"unparseable_judge_output: {raw_output[:150]}", "judge_model": judge_model}
    return {"score": int(m.group(1)), "reason": m.group(2).strip(), "judge_model": judge_model}
