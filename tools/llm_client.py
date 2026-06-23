import os
import requests

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SOLVER_MODEL = os.environ.get("SOLVER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "mistralai/mistral-7b-instruct:free")


def _call_openrouter(model: str, system_prompt: str, user_prompt: str, max_tokens: int = 600,
                      temperature: float = 0.2, seed: int = None):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if seed is not None:
        body["seed"] = seed
    resp = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


class RealLLM:
    def __init__(self, model: str):
        self.model = model
        self.name = model

    def call(self, system_prompt: str, user_prompt: str, max_tokens: int = 600,
              temperature: float = 0.2, seed: int = None):
        return _call_openrouter(self.model, system_prompt, user_prompt, max_tokens, temperature, seed)


class StubLLM:

    name = "stub-deterministic-solver"

    def call(self, system_prompt: str, user_prompt: str, max_tokens: int = 600,
              temperature: float = 0.2, seed: int = None):
        raise NotImplementedError("StubLLM is used via tools.reference_solver helpers, not raw .call()")


def get_solver_llm():
    if OPENROUTER_API_KEY:
        return RealLLM(SOLVER_MODEL)
    return StubLLM()


def get_judge_llm():
    if OPENROUTER_API_KEY:
        return RealLLM(JUDGE_MODEL)
    return StubLLM()


def is_offline() -> bool:
    return not bool(OPENROUTER_API_KEY)
