import time
from collections import Counter

from strategies.base import Strategy, StrategyResult
from tools.calculator import calculator, CalculatorError
from tools.llm_client import get_solver_llm, is_offline
from tools.reference_solver import get_worked_solution
from observability.tracing import TraceLogger

N_SAMPLES = 5

COT_SYSTEM_PROMPT = """You are solving a math word problem step by step.

Show your reasoning as a short sequence of steps, computing any arithmetic
using explicit expressions. End your response with a final line in EXACTLY
this format:
FINAL ANSWER: <number>
"""


class SelfConsistencyStrategy(Strategy):
    name = "self_consistency"

    def solve(self, problem: dict) -> StrategyResult:
        problem_id = problem["id"]
        question = problem["question"]
        tracer = TraceLogger(strategy=self.name, problem_id=problem_id)
        start_time = time.time()

        sample_answers = []
        tokens_in_total = 0
        tokens_out_total = 0
        n_llm_calls = 0
        n_tool_calls = 0

        for sample_idx in range(N_SAMPLES):
            if is_offline():
                answer, n_tools = self._sample_offline(question, problem_id, sample_idx, tracer)
                n_llm_calls += 1
                n_tool_calls += n_tools
                tokens_in_total += 35
                tokens_out_total += 40
            else:
                answer, t_in, t_out, n_tools = self._sample_real(question, problem_id, sample_idx, tracer)
                n_llm_calls += 1
                n_tool_calls += n_tools
                tokens_in_total += t_in
                tokens_out_total += t_out
            sample_answers.append(answer)

        vote_counter = Counter(a for a in sample_answers if a is not None)
        if vote_counter:
            majority_answer, majority_count = vote_counter.most_common(1)[0]
        else:
            majority_answer, majority_count = None, 0

        latency_ms = (time.time() - start_time) * 1000
        tracer.log("final_answer", inputs={"question": question, "samples": sample_answers},
                   outputs={"majority_answer": majority_answer, "vote_count": majority_count,
                            "total_samples": N_SAMPLES},
                   metadata={"vote_distribution": dict(vote_counter)})

        return StrategyResult(
            problem_id=problem_id, strategy=self.name, run_id=tracer.run_id,
            final_answer=majority_answer,
            raw_final_text=f"Majority vote: {majority_answer} ({majority_count}/{N_SAMPLES} samples)",
            tokens_in=tokens_in_total, tokens_out=tokens_out_total, latency_ms=latency_ms,
            n_llm_calls=n_llm_calls, n_tool_calls=n_tool_calls,
            metadata={"samples": sample_answers, "vote_distribution": dict(vote_counter)},
        )

    def _sample_offline(self, question: str, problem_id: str, sample_idx: int, tracer: TraceLogger):
        answer, steps, error_template = get_worked_solution(problem_id, self.name, sample_index=sample_idx)
        n_tools = 0
        try:
            calculator(f"{answer} + 0")
            n_tools = 1
        except CalculatorError:
            pass
        tracer.log("llm_call", inputs={"sample_index": sample_idx, "question": question},
                   outputs={"steps": steps, "sampled_answer": answer},
                   tokens_in=35, tokens_out=40,
                   metadata={"model": "stub-deterministic-solver", "error_template_injected": error_template})
        return answer, n_tools

    def _sample_real(self, question: str, problem_id: str, sample_idx: int, tracer: TraceLogger):
        llm = get_solver_llm()
        t0 = time.time()
        raw_response, t_in, t_out = llm.call(
            COT_SYSTEM_PROMPT, question, max_tokens=400, temperature=0.7, seed=sample_idx
        )
        latency = (time.time() - t0) * 1000
        tracer.log("llm_call", inputs={"sample_index": sample_idx, "question": question},
                   outputs={"raw_response": raw_response}, tokens_in=t_in, tokens_out=t_out,
                   latency_ms=latency, metadata={"model": llm.name, "temperature": 0.7})

        if not raw_response:
            tracer.log("llm_call", inputs={"sample_index": sample_idx},
                       outputs={"error": "empty_response"}, metadata={"model": llm.name})
            return None, t_in, t_out, 0

        answer = self._extract_final_answer(raw_response)
        n_tools = 0
        import re
        for expr_match in re.finditer(r"`([\d\.\s\+\-\*/\(\)]+)`", raw_response):
            try:
                result = calculator(expr_match.group(1))
                n_tools += 1
                tracer.log("tool_call", inputs={"expression": expr_match.group(1)},
                           outputs={"result": result}, metadata={"tool": "calculator"})
            except CalculatorError:
                pass

        return answer, t_in, t_out, n_tools

    @staticmethod
    def _extract_final_answer(text: str):
        import re
        m = re.search(r"FINAL ANSWER:\s*(-?[\d,]+\.?\d*)", text, re.IGNORECASE)
        if not m:
            return None
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            return None
