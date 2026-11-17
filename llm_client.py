"""A chat client for any OpenAI-compatible endpoint.

Groq, OpenRouter, LM Studio and a local Ollama server all expose the same
`/chat/completions` shape, so the provider is configuration rather than code:

    LLM_BASE_URL   default https://api.groq.com/openai/v1
    LLM_MODEL      default openai/gpt-oss-120b
    LLM_API_KEY    falls back to GROQ_API_KEY, then to a placeholder for
                   local servers that do not check the key

Two kinds of failure are handled here, separately from the malformed-JSON
retry in summarise_alerts.py:

  * rate limits (429) and server errors (5xx) are retried with exponential
    backoff, honouring Retry-After when the server sends one;
  * authentication and model-not-found errors are not retried, because they
    will fail identically every time.

Free tiers are token-per-minute limited, not just request-per-minute limited.
Groq's free plan allows 8,000 tokens/minute and one short alert call is about
350 tokens, so pacing defaults to one call every 3 seconds (~20/minute).
"""

from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    NotFoundError,
    OpenAI,
    RateLimitError,
)

load_dotenv()  # .env is gitignored; env vars win if both are set

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_MIN_INTERVAL = 3.0
MAX_API_ATTEMPTS = 4


class LLMError(RuntimeError):
    """The endpoint could not produce a reply, after retrying where sensible."""


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"warning: {name}={raw!r} is not a number, using {default}")
        return default


class ChatClient:
    """Thin wrapper that adds pacing and retry around the SDK call."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        min_interval: float | None = None,
        max_attempts: int = MAX_API_ATTEMPTS,
    ) -> None:
        self.base_url = base_url or os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL
        self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODEL
        self.min_interval = (
            min_interval if min_interval is not None
            else _env_float("LLM_MIN_INTERVAL", DEFAULT_MIN_INTERVAL)
        )
        self.max_attempts = max_attempts
        self._last_call = 0.0

        key = api_key or os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
        local = "localhost" in self.base_url or "127.0.0.1" in self.base_url
        if not key and not local:
            raise LLMError(
                "No API key found. Put GROQ_API_KEY=... in .env (gitignored) "
                "or set it in the environment."
            )
        self._client = OpenAI(api_key=key or "not-needed", base_url=self.base_url)

    @property
    def endpoint_host(self) -> str:
        """Host only, so a run summary can name the provider without a secret."""
        return self.base_url.split("://", 1)[-1].split("/", 1)[0]

    def _pace(self) -> None:
        if self.min_interval <= 0:
            return
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def chat(self, system: str, messages: list[dict], max_tokens: int = 300) -> str:
        """Return the assistant text, or raise LLMError."""
        payload = [{"role": "system", "content": system}, *messages]

        for attempt in range(1, self.max_attempts + 1):
            self._pace()
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=payload,
                    max_tokens=max_tokens,
                    temperature=0,
                )
            except AuthenticationError as exc:
                raise LLMError(
                    f"Authentication failed at {self.endpoint_host}. "
                    "The API key is wrong, revoked, or has a stray space."
                ) from exc
            except NotFoundError as exc:
                raise LLMError(
                    f"Model {self.model!r} is not available at {self.endpoint_host}. "
                    "Free-tier catalogues change; run "
                    "`python llm_client.py --list-models` and set LLM_MODEL."
                ) from exc
            except RateLimitError as exc:
                if attempt == self.max_attempts:
                    raise LLMError(
                        f"Rate limited by {self.endpoint_host} after "
                        f"{attempt} attempts. Raise LLM_MIN_INTERVAL."
                    ) from exc
                delay = self._retry_after(exc) or min(2 ** attempt, 30)
                print(f"    rate limited, sleeping {delay:.0f}s (attempt {attempt})")
                time.sleep(delay)
            except APIConnectionError as exc:
                if attempt == self.max_attempts:
                    raise LLMError(f"Cannot reach {self.endpoint_host}.") from exc
                delay = min(2 ** attempt, 15)
                print(f"    connection error, retrying in {delay}s")
                time.sleep(delay)
            except APIStatusError as exc:
                if exc.status_code < 500 or attempt == self.max_attempts:
                    raise LLMError(
                        f"HTTP {exc.status_code} from {self.endpoint_host}: "
                        f"{_short(exc)}"
                    ) from exc
                delay = min(2 ** attempt, 30)
                print(f"    server error {exc.status_code}, retrying in {delay}s")
                time.sleep(delay)
            else:
                if not resp.choices:
                    raise LLMError("Endpoint returned no choices.")
                return (resp.choices[0].message.content or "").strip()

        raise LLMError("Exhausted retries without a reply.")

    @staticmethod
    def _retry_after(exc: RateLimitError) -> float | None:
        response = getattr(exc, "response", None)
        if response is None:
            return None
        raw = response.headers.get("retry-after")
        if not raw:
            return None
        try:
            return min(float(raw), 60.0)
        except ValueError:
            return None

    def list_models(self) -> list[str]:
        """Model IDs the endpoint will accept, for when a default goes stale."""
        return sorted(m.id for m in self._client.models.list().data)


def _short(exc: Exception, limit: int = 200) -> str:
    text = " ".join(str(exc).split())
    return text[:limit] + ("..." if len(text) > limit else "")


if __name__ == "__main__":
    if "--list-models" not in sys.argv:
        print(__doc__)
        raise SystemExit(0)
    try:
        client = ChatClient()
    except LLMError as exc:
        raise SystemExit(f"error: {exc}")
    print(f"models available at {client.endpoint_host}:")
    for model_id in client.list_models():
        print(f"  {model_id}")
