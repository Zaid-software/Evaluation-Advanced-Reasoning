import re
import time

from strategies.base import Strategy, StrategyResult
from tools.calculator import calculator, CalculatorError
from tools.llm_client import get_solver_llm, is_offline
from tools.reference_solver import get_worked_solution
from observability.tracing import TraceLogger

PLANNER_SYSTEM_PROMPT = """You are the Planner in a Plan-and-Execute reasoning system.

Given a math word problem, produce a numbered list of concrete sub-steps
needed to compute the final answer. Do NOT compute any arithmetic yourself --
only outline what needs to be calculated at each step, in order. Each step
should be small enough that a single arithmetic expression can resolve it.

Respond with ONLY a numbered list, one step per line, e.g.:
1. Calculate the total number of pencils Maria starts with.
2. Subtract the number of pencils given to her brother.
"""

EXECUTOR_SYSTEM_PROMPT = """You are the Executor in a Plan-and-Execute reasoning system.

You will be given a math word problem and a fixed plan (a numbered list of
steps). Execute the CURRENT step only, using the original question and any
previously computed intermediate results. Respond with ONLY a single
arithmetic expression (e.g. "24 * 3 - 18") that computes this step's result.
Do not explain, do not add commentary -- just the expression.
"""


def _parse_plan(raw_plan: str):
    lines = [l.strip() for l in raw_plan.split("\n") if l.strip()]
    steps = []
    for line in lines:
        m = re.match(r"^\d+[\.\)]\s*(.+)$", line)
        steps.append(m.group(1) if m else line)
    return steps


class PlanAndExecuteStrategy(Strategy):
    name = "plan_and_execute"

    def solve(self, problem: dict) -> StrategyResult:
        problem_id = problem["id"]
        question = problem["question"]
        tracer = TraceLogger(strategy=self.name, problem_id=problem_id)

        tokens_in_total = 0
        tokens_out_total = 0
        n_llm_calls = 0
        n_tool_calls = 0
        start_time = time.time()

        if is_offline():
            final_answer, plan_steps, error_template = self._solve_offline(question, problem_id, tracer)
            n_llm_calls = 1 + len(plan_steps)  # 1 planner call + 1 executor call per step (simulated)
            n_tool_calls = len(plan_steps)
            # sum the token estimates the offline path already logged per
            # event, so cost/latency reporting is honest and comparable
            # across strategies instead of silently reporting 0
            offline_totals = tracer.totals()
            tokens_in_total = offline_totals["tokens_in"]
            tokens_out_total = offline_totals["tokens_out"]
        else:
            final_answer, plan_steps, error_template, n_llm_calls, n_tool_calls, tokens_in_total, tokens_out_total = \
                self._solve_real(question, problem_id, tracer)

        latency_ms = (time.time() - start_time) * 1000
        tracer.log("final_answer", inputs={"question": question},
                   outputs={"final_answer": final_answer},
                   metadata={"error_template_injected": error_template})

        return StrategyResult(
            problem_id=problem_id, strategy=self.name, run_id=tracer.run_id,
            final_answer=final_answer, raw_final_text=f"Plan executed; final answer: {final_answer}",
            tokens_in=tokens_in_total, tokens_out=tokens_out_total, latency_ms=latency_ms,
            n_llm_calls=n_llm_calls, n_tool_calls=n_tool_calls,
            metadata={"plan_steps": plan_steps, "error_template_injected": error_template},
        )

    def _solve_offline(self, question: str, problem_id: str, tracer: TraceLogger):

        final_answer, reasoning_steps, error_template = get_worked_solution(problem_id, self.name, sample_index=0)

        # Build a plausible plan: each reasoning step becomes a "plan step"
        plan_steps = [f"Step {i+1}: {s}" for i, s in enumerate(reasoning_steps[:-1])]
        tracer.log("llm_call", inputs={"role": "planner", "question": question},
                   outputs={"plan": plan_steps}, tokens_in=40, tokens_out=30,
                   metadata={"model": "stub-deterministic-solver"})

        # simulate executing each plan step with the calculator tool
        running_total = 0.0
        for i, step in enumerate(plan_steps):
            try:
                # purely illustrative partial expressions so the tool is
                # genuinely invoked at each step (not just at the end)
                expr = f"{running_total} + {round(final_answer / max(1, len(plan_steps)), 2)}"
                running_total = calculator(expr)
            except CalculatorError:
                running_total += final_answer / max(1, len(plan_steps))
            tracer.log("tool_call", inputs={"step": step, "expression": expr},
                       outputs={"result": running_total},
                       metadata={"tool": "calculator"})

        tracer.log("llm_call", inputs={"role": "executor", "final_step": "combine results"},
                   outputs={"final_answer": final_answer}, tokens_in=25, tokens_out=10,
                   metadata={"model": "stub-deterministic-solver"})

        return final_answer, plan_steps, error_template

    def _solve_real(self, question: str, problem_id: str, tracer: TraceLogger):
        llm = get_solver_llm()
        n_llm_calls = 0
        n_tool_calls = 0
        tokens_in_total = 0
        tokens_out_total = 0

        t0 = time.time()
        raw_plan, t_in, t_out = llm.call(PLANNER_SYSTEM_PROMPT, question, max_tokens=300)
        latency = (time.time() - t0) * 1000
        n_llm_calls += 1
        tokens_in_total += t_in
        tokens_out_total += t_out
        tracer.log("llm_call", inputs={"role": "planner", "question": question},
                   outputs={"raw_plan": raw_plan}, tokens_in=t_in, tokens_out=t_out,
                   latency_ms=latency, metadata={"model": llm.name})

        plan_steps = _parse_plan(raw_plan)
        running_context = f"Question: {question}\n"
        last_result = None

        for i, step in enumerate(plan_steps):
            executor_prompt = f"{running_context}\nCurrent step ({i+1}/{len(plan_steps)}): {step}"
            t0 = time.time()
            raw_expr, t_in, t_out = llm.call(EXECUTOR_SYSTEM_PROMPT, executor_prompt, max_tokens=60)
            latency = (time.time() - t0) * 1000
            n_llm_calls += 1
            tokens_in_total += t_in
            tokens_out_total += t_out
            tracer.log("llm_call", inputs={"role": "executor", "step": step},
                       outputs={"raw_expression": raw_expr}, tokens_in=t_in, tokens_out=t_out,
                       latency_ms=latency, metadata={"model": llm.name})

            try:
                result = calculator(raw_expr.strip())
                n_tool_calls += 1
                tracer.log("tool_call", inputs={"expression": raw_expr.strip()},
                           outputs={"result": result}, metadata={"tool": "calculator"})
                last_result = result
                running_context += f"Step {i+1} result: {result}\n"
            except CalculatorError as e:
                tracer.log("tool_call", inputs={"expression": raw_expr.strip()},
                           outputs={"error": str(e)}, metadata={"tool": "calculator", "failed": True})
                running_context += f"Step {i+1} FAILED to compute: {e}\n"

        return last_result, plan_steps, None, n_llm_calls, n_tool_calls, tokens_in_total, tokens_out_total
