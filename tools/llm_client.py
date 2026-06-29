import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()  

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


RATE_LIMIT_DELAY_SECONDS = float(os.environ.get("RATE_LIMIT_DELAY_SECONDS", "3.0"))
JUDGE_RATE_LIMIT_DELAY_SECONDS = float(os.environ.get("JUDGE_RATE_LIMIT_DELAY_SECONDS", "6.0"))

SOLVER_MODEL = os.environ.get("SOLVER_MODEL", "openrouter/free")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "nvidia/nemotron-nano-9b-v2:free")



def _call_openrouter(model: str, system_prompt: str, user_prompt: str, max_tokens: int = 600,
                      temperature: float = 0.2, seed: int = None, max_retries: int = 6,
                      per_call_delay: float = None):
    delay = per_call_delay if per_call_delay is not None else RATE_LIMIT_DELAY_SECONDS
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

    for attempt in range(max_retries):
        time.sleep(delay)
        resp = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=30)

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait_seconds = float(retry_after) if retry_after else (2 ** attempt) * 2
            print(f"[rate limit] 429 on {model}, attempt {attempt + 1}/{max_retries}, "
                  f"waiting {wait_seconds:.1f}s before retry...")
            time.sleep(wait_seconds)
            continue

        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        if not content:
            print(f"[empty content] {model} returned no content, attempt {attempt + 1}/{max_retries}, retrying...")
            time.sleep((2 ** attempt) * 1.5)
            continue

        return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    raise RuntimeError(f"Exceeded max_retries ({max_retries}) on {model} due to repeated 429 rate limits "
                        f"or empty responses. Wait a minute and re-run -- free-tier issues usually clear up.")


class RealLLM:
    def __init__(self, model: str):
        self.model = model
        self.name = model
        self._delay = JUDGE_RATE_LIMIT_DELAY_SECONDS if model == JUDGE_MODEL else RATE_LIMIT_DELAY_SECONDS

    def call(self, system_prompt: str, user_prompt: str, max_tokens: int = 600,
              temperature: float = 0.2, seed: int = None):
        return _call_openrouter(self.model, system_prompt, user_prompt, max_tokens, temperature, seed,
                                 per_call_delay=self._delay)


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
