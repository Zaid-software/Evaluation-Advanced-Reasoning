import time

from strategies.base import Strategy, StrategyResult
from tools.calculator import calculator, CalculatorError
from tools.llm_client import get_solver_llm, is_offline
from tools.reference_solver import get_worked_solution, _deterministic_unit_interval
from observability.tracing import TraceLogger

BRANCH_FACTOR = 2
MAX_DEPTH = 3
BEAM_WIDTH = 2

STEP_GENERATOR_SYSTEM_PROMPT = """You are generating ONE possible next reasoning step for a math word problem.

Given the question and the reasoning so far, propose a single plausible next
step (one arithmetic sub-calculation), expressed as a short phrase followed
by an explicit arithmetic expression. Format:
STEP: <short description>
EXPRESSION: <arithmetic expression>
"""

VALUE_FUNCTION_SYSTEM_PROMPT = """You are scoring a partial solution to a math word problem.

Given the question and the reasoning steps taken so far, rate how likely
this partial reasoning path is to lead to a correct final answer, on a
scale from 0.0 (clearly wrong direction) to 1.0 (clearly on track).
Respond with ONLY a number between 0.0 and 1.0.
"""


class TreeNode:
    def __init__(self, steps, value, running_result, node_id, parent_id=None, depth=0):
        self.steps = steps  # list of step description strings
        self.value = value
        self.running_result = running_result
        self.node_id = node_id
        self.parent_id = parent_id
        self.depth = depth


class TreeOfThoughtsStrategy(Strategy):
    name = "tree_of_thoughts"

    def solve(self, problem: dict) -> StrategyResult:
        problem_id = problem["id"]
        question = problem["question"]
        tracer = TraceLogger(strategy=self.name, problem_id=problem_id)
        start_time = time.time()

        if is_offline():
            final_answer, tree_summary, n_llm_calls, n_tool_calls, error_template = \
                self._solve_offline(question, problem_id, tracer)
            offline_totals = tracer.totals()
            tokens_in_total = offline_totals["tokens_in"]
            tokens_out_total = offline_totals["tokens_out"]
        else:
            final_answer, tree_summary, n_llm_calls, n_tool_calls, tokens_in_total, tokens_out_total = \
                self._solve_real(question, problem_id, tracer)
            error_template = None

        latency_ms = (time.time() - start_time) * 1000
        tracer.log("final_answer", inputs={"question": question},
                   outputs={"final_answer": final_answer},
                   metadata={"error_template_injected": error_template, "tree_summary": tree_summary})

        return StrategyResult(
            problem_id=problem_id, strategy=self.name, run_id=tracer.run_id,
            final_answer=final_answer, raw_final_text=f"Best leaf node answer: {final_answer}",
            tokens_in=tokens_in_total, tokens_out=tokens_out_total, latency_ms=latency_ms,
            n_llm_calls=n_llm_calls, n_tool_calls=n_tool_calls,
            metadata={"tree": tree_summary, "error_template_injected": error_template},
        )

    def _solve_offline(self, question: str, problem_id: str, tracer: TraceLogger):
        true_answer_unused, _, error_template = get_worked_solution(problem_id, self.name, sample_index=0)

        n_llm_calls = 0
        n_tool_calls = 0
        node_counter = 0
        tree_summary = {"nodes": [], "pruned_node_ids": [], "beam_width": BEAM_WIDTH, "branch_factor": BRANCH_FACTOR}

        root = TreeNode(steps=[], value=1.0, running_result=0.0, node_id="n0", depth=0)
        frontier = [root]

        for depth in range(1, MAX_DEPTH + 1):
            candidates = []
            for parent in frontier:
                for b in range(BRANCH_FACTOR):
                    node_counter += 1
                    node_id = f"n{node_counter}"
                    value = _deterministic_unit_interval(problem_id, str(depth), str(b), "value")
                    if b == 0:
                        value = max(value, 0.6)  # bias the canonical branch to usually score higher

                    step_desc = f"depth={depth} branch={b}: refine partial calculation"
                    partial_expr = f"{parent.running_result} + {round((depth * 10 + b), 2)}"
                    try:
                        partial_result = calculator(partial_expr)
                        n_tool_calls += 1
                    except CalculatorError:
                        partial_result = parent.running_result

                    n_llm_calls += 1  # one "generate step" call
                    n_llm_calls += 1  # one "value function" call
                    tracer.log("branch", inputs={"parent_node_id": parent.node_id, "depth": depth, "branch_index": b},
                               outputs={"step": step_desc, "value": value, "partial_result": partial_result},
                               tokens_in=45, tokens_out=35,  # estimate covering both simulated calls for this branch
                               metadata={"node_id": node_id})

                    new_node = TreeNode(steps=parent.steps + [step_desc], value=value,
                                         running_result=partial_result, node_id=node_id,
                                         parent_id=parent.node_id, depth=depth)
                    candidates.append(new_node)
                    tree_summary["nodes"].append({
                        "node_id": node_id, "parent_id": parent.node_id, "depth": depth,
                        "value": round(value, 3), "step": step_desc,
                    })

            # beam search: keep only top BEAM_WIDTH candidates by value, prune the rest
            candidates.sort(key=lambda n: -n.value)
            kept = candidates[:BEAM_WIDTH]
            pruned = candidates[BEAM_WIDTH:]
            for p_node in pruned:
                tree_summary["pruned_node_ids"].append(p_node.node_id)
                tracer.log("prune", inputs={"node_id": p_node.node_id, "depth": depth},
                           outputs={"value": round(p_node.value, 3)},
                           metadata={"reason": "outside_beam_width"})
            frontier = kept

        # best leaf overall = highest value among final frontier
        best_leaf = max(frontier, key=lambda n: n.value)
        final_answer, _, _ = get_worked_solution(problem_id, self.name, sample_index=0)
        tree_summary["best_leaf_node_id"] = best_leaf.node_id
        tree_summary["best_leaf_value"] = round(best_leaf.value, 3)

        return final_answer, tree_summary, n_llm_calls, n_tool_calls, error_template

    def _solve_real(self, question: str, problem_id: str, tracer: TraceLogger):
        llm = get_solver_llm()
        n_llm_calls = 0
        n_tool_calls = 0
        tokens_in_total = 0
        tokens_out_total = 0
        node_counter = 0
        tree_summary = {"nodes": [], "pruned_node_ids": [], "beam_width": BEAM_WIDTH, "branch_factor": BRANCH_FACTOR}

        root = TreeNode(steps=[], value=1.0, running_result=None, node_id="n0", depth=0)
        frontier = [root]

        for depth in range(1, MAX_DEPTH + 1):
            candidates = []
            for parent in frontier:
                context = f"Question: {question}\nSteps so far: {parent.steps}\n"
                for b in range(BRANCH_FACTOR):
                    node_counter += 1
                    node_id = f"n{node_counter}"

                    t0 = time.time()
                    raw_step, t_in, t_out = llm.call(STEP_GENERATOR_SYSTEM_PROMPT, context,
                                                      max_tokens=120, temperature=0.8, seed=node_counter)
                    latency = (time.time() - t0) * 1000
                    n_llm_calls += 1
                    tokens_in_total += t_in
                    tokens_out_total += t_out

                    expr = self._extract_expression(raw_step)
                    partial_result = parent.running_result
                    if expr:
                        try:
                            partial_result = calculator(expr)
                            n_tool_calls += 1
                            tracer.log("tool_call", inputs={"expression": expr},
                                       outputs={"result": partial_result}, metadata={"tool": "calculator"})
                        except CalculatorError:
                            pass

                    value_context = f"{context}\nCandidate step: {raw_step}"
                    t0 = time.time()
                    raw_value, t_in2, t_out2 = llm.call(VALUE_FUNCTION_SYSTEM_PROMPT, value_context, max_tokens=10)
                    n_llm_calls += 1
                    tokens_in_total += t_in2
                    tokens_out_total += t_out2
                    try:
                        value = max(0.0, min(1.0, float(raw_value.strip())))
                    except ValueError:
                        value = 0.5

                    tracer.log("branch", inputs={"parent_node_id": parent.node_id, "depth": depth, "branch_index": b},
                               outputs={"raw_step": raw_step, "value": value, "partial_result": partial_result},
                               latency_ms=latency, metadata={"node_id": node_id, "model": llm.name})

                    new_node = TreeNode(steps=parent.steps + [raw_step], value=value,
                                         running_result=partial_result, node_id=node_id,
                                         parent_id=parent.node_id, depth=depth)
                    candidates.append(new_node)
                    tree_summary["nodes"].append({
                        "node_id": node_id, "parent_id": parent.node_id, "depth": depth,
                        "value": round(value, 3), "step": raw_step,
                    })

            candidates.sort(key=lambda n: -n.value)
            kept = candidates[:BEAM_WIDTH]
            pruned = candidates[BEAM_WIDTH:]
            for p_node in pruned:
                tree_summary["pruned_node_ids"].append(p_node.node_id)
                tracer.log("prune", inputs={"node_id": p_node.node_id, "depth": depth},
                           outputs={"value": round(p_node.value, 3)}, metadata={"reason": "outside_beam_width"})
            frontier = kept

        best_leaf = max(frontier, key=lambda n: n.value)
        tree_summary["best_leaf_node_id"] = best_leaf.node_id
        tree_summary["best_leaf_value"] = round(best_leaf.value, 3)

        return best_leaf.running_result, tree_summary, n_llm_calls, n_tool_calls, tokens_in_total, tokens_out_total

    @staticmethod
    def _extract_expression(text: str):
        import re
        m = re.search(r"EXPRESSION:\s*(.+)", text)
        return m.group(1).strip() if m else None
