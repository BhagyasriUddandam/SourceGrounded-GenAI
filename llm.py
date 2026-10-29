"""GroqCloud chat client using Groq's officially supported Python SDK."""

from __future__ import annotations

from dataclasses import dataclass
import os
from time import monotonic
from urllib.parse import urlparse


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResult:
    text: str
    usage: dict
    latency_seconds: float
    model: str


def load_local_env(path: str = ".env") -> None:
    """Small .env loader; never overwrite an already-configured environment."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key.isidentifier():
                continue
            os.environ.setdefault(key, value.strip().strip("'\""))


class ChatClient:
    def __init__(
        self, model: str | None = None, base_url: str | None = None,
        api_key_env: str = "GROQ_API_KEY", timeout: int = 60,
        temperature: float = 0.1, max_tokens: int = 2500, retries: int = 2,
    ) -> None:
        self.model = model or os.getenv("GENAI_MODEL", "openai/gpt-oss-120b")
        self.base_url = (base_url or os.getenv("GENAI_BASE_URL", "https://api.groq.com/openai/v1")).rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        parsed = urlparse(self.base_url)
        if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}):
            raise ValueError("API base URL must be HTTPS, except for local development")

    def chat(self, messages: list[dict[str, str]]) -> LLMResult:
        key = os.getenv(self.api_key_env)
        if not key:
            raise ProviderError(
                f"Missing {self.api_key_env}. Set it in the environment or in a local .env file; "
                "see README.md. No request was sent."
            )
        try:
            from groq import APIError, Groq
        except ImportError as exc:
            raise ProviderError(
                "Groq SDK is not installed. From this project's folder run: "
                "python3 -m pip install -e '.[ui,pdf]'"
            ) from exc
        # Groq's Python SDK appends /openai/v1/chat/completions internally;
        # its base URL is the server root, not an OpenAI-compatible /v1 path.
        sdk_base_url = self.base_url.removesuffix("/openai/v1")
        client = Groq(
            api_key=key,
            base_url=sdk_base_url,
            timeout=float(self.timeout),
            max_retries=self.retries,
        )
        params = {
            "model": self.model, "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
        }
        if self.model.startswith("openai/gpt-oss-"):
            params["reasoning_effort"] = "low"
        start = monotonic()
        # GPT-OSS sometimes spends its output budget on reasoning and returns
        # an empty visible answer. Retry once with a larger completion budget.
        for attempt in range(2):
            try:
                completion = client.chat.completions.create(**params)
            except APIError as exc:
                status = getattr(exc, "status_code", None)
                prefix = f"HTTP {status}: " if status else ""
                raise ProviderError(f"Groq chat request failed: {prefix}{str(exc)[:320]}") from exc
            except (OSError, TimeoutError) as exc:
                raise ProviderError(f"Groq chat connection failed: {exc}") from exc
            try:
                answer_text = completion.choices[0].message.content
                if isinstance(answer_text, str) and answer_text.strip():
                    usage = completion.usage.model_dump() if completion.usage is not None else {}
                    return LLMResult(
                        text=answer_text,
                        usage=usage,
                        latency_seconds=round(monotonic() - start, 3),
                        model=str(completion.model or self.model),
                    )
            except (IndexError, AttributeError, ValueError, TypeError) as exc:
                raise ProviderError("Unexpected Groq SDK response structure") from exc
            if attempt == 0:
                params["max_completion_tokens"] = max(4000, self.max_tokens * 2)
                continue
            raise ProviderError(
                "Groq returned no visible answer text after two attempts, "
                "even with a larger completion-token budget"
            )
        raise AssertionError("Unreachable answer retry state")

