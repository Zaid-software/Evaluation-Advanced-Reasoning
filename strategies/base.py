from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StrategyResult:
    problem_id: str
    strategy: str
    run_id: str
    final_answer: Optional[float]
    raw_final_text: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    n_llm_calls: int
    n_tool_calls: int
    metadata: Dict[str, Any] = field(default_factory=dict)  # strategy-specific extras
                                                              # (e.g. tree structure, votes, plan steps)


class Strategy(ABC):
    name: str = "base_strategy"

    @abstractmethod
    def solve(self, problem: dict) -> StrategyResult:
        raise NotImplementedError
