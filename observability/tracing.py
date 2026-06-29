import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_LOG_PATH = os.path.join(BASE_DIR, "logs", "trace_log.jsonl")


@dataclass
class TokenUsage:
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class TraceEvent:
    run_id: str
    event_index: int
    timestamp: str
    strategy: str
    problem_id: str
    step_type: str  # "llm_call" | "tool_call" | "reasoning_step" | "final_answer" | "branch" | "prune" | "reflection"
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        return d


class TraceLogger:


    def __init__(self, strategy: str, problem_id: str, log_path: str = TRACE_LOG_PATH):
        self.run_id = uuid.uuid4().hex[:12]
        self.strategy = strategy
        self.problem_id = problem_id
        self.log_path = log_path
        self.events = []
        self._event_index = 0
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(self, step_type: str, inputs: dict = None, outputs: dict = None,
            tokens_in: int = 0, tokens_out: int = 0, latency_ms: float = 0.0,
            metadata: dict = None) -> TraceEvent:
        event = TraceEvent(
            run_id=self.run_id,
            event_index=self._event_index,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
            strategy=self.strategy,
            problem_id=self.problem_id,
            step_type=step_type,
            inputs=inputs or {},
            outputs=outputs or {},
            token_usage=TokenUsage(tokens_in=tokens_in, tokens_out=tokens_out),
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
        self._event_index += 1
        self.events.append(event)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False, default=str) + "\n")
        return event

    def totals(self):
        tokens_in = sum(e.token_usage.tokens_in for e in self.events)
        tokens_out = sum(e.token_usage.tokens_out for e in self.events)
        latency_ms = sum(e.latency_ms for e in self.events)
        return {"tokens_in": tokens_in, "tokens_out": tokens_out, "latency_ms": latency_ms}


def load_trace_log(log_path: str = TRACE_LOG_PATH):
    events = []
    if not os.path.exists(log_path):
        return events
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def get_run_events(run_id: str, log_path: str = TRACE_LOG_PATH):
    return [e for e in load_trace_log(log_path) if e["run_id"] == run_id]
