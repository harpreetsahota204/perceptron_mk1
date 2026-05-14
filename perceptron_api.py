"""Perceptron API client wrapper.

Thin OpenAI-compatible client targeted at ``api.perceptron.inc/v1``:

* `PerceptronClient.chat_completion(...)` -- one shot of `/chat/completions`
  with retry on 429. The ONE place in the codebase where exceptions are
  caught (fail-fast philosophy elsewhere).
* `to_video_data_uri(filepath)` / `to_image_data_uri(data, mime=...)` --
  base64-encode local media into the ``data:...;base64,...`` URI the API
  requires.

Every notable action emits a ``[perceptron]`` log line.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Final, Self

from openai import OpenAI, RateLimitError

logger = logging.getLogger("perceptron")


DEFAULT_BASE_URL: Final[str] = "https://api.perceptron.inc/v1"

# Long timeout: video data URIs are large and the API can take tens of
# seconds for video reasoning.
DEFAULT_TIMEOUT_SECONDS: Final[int] = 300

# Three attempts total (1 initial + 2 retries) with `Retry-After` preferred,
# exponential backoff as fallback. Per https://docs.perceptron.inc/guides/batch.
MAX_RETRY_ATTEMPTS: Final[int] = 3


def to_video_data_uri(filepath: str | Path) -> str:
    """Encode a local video as a ``data:video/<subtype>;base64,...`` URI.

    Uses `mimetypes.guess_type`; falls back to ``video/mp4`` on unusual
    extensions.
    """
    path = Path(filepath)
    mime = mimetypes.guess_type(str(path))[0] or "video/mp4"
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    logger.info(
        "[perceptron] encoded video %s -> data URI: %s, %.1f KB raw, %.1f KB b64",
        path.name,
        mime,
        len(data) / 1024,
        len(b64) / 1024,
    )
    return f"data:{mime};base64,{b64}"


def to_image_data_uri(data: bytes, *, mime: str = "image/jpeg") -> str:
    """Encode raw image bytes as a ``data:<mime>;base64,...`` URI.

    Used by dense per-frame tasks (TRACK, KEYPOINTS) that extract frames
    from a video and send them as image-mode requests. The caller is
    responsible for encoding the frame to bytes (typically JPEG via cv2 or
    PIL) before handing it here.
    """
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _truncate(text: str, limit: int = 200) -> str:
    """Truncate a string for log previews, appending '...' if cut."""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


class PerceptronClient:
    """OpenAI client targeted at the Perceptron API.

    Handles base URL / auth / timeout, 429 retry with backoff, and request /
    response logging. Higher-level concerns (prompt construction, parsing,
    FiftyOne writeback) live in `perceptron_model.PerceptronModel`.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self._api_key = api_key
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        # Running totals across every successful chat_completion call. Reset
        # between operator runs via `reset_usage_totals()` so progress labels
        # don't carry stale numbers across invocations.
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_calls: int = 0
        logger.info(
            "[perceptron] PerceptronClient ready: base_url=%s timeout=%ds",
            base_url,
            timeout,
        )

    @property
    def usage_totals(self) -> dict[str, int]:
        """Running token / call totals: ``{prompt_tokens, completion_tokens,
        total_tokens, calls}``."""
        return {
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "calls": self._total_calls,
        }

    def reset_usage_totals(self) -> None:
        """Zero the token / call totals. Idempotent."""
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_calls = 0

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        vision_config: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_completion_tokens: int = 4096,
    ) -> Any:
        """Call ``/v1/chat/completions`` once, retrying on 429.

        Args:
            model: Model id (e.g. ``"mk1"``, ``"isaac-0.2-2b-preview"``).
            messages: OpenAI-style message list (caller builds it).
            vision_config: Perceptron-specific extension sent via OpenAI's
                ``extra_body``. The documented way to steer output format
                and toggle thinking. Typical contents:
                ``{"annotation_format": "box" | "point" | "polygon" | "clip",
                "enable_thinking": True | False}``.
            response_format: Standard OpenAI ``response_format`` (json_schema).
                Compatible with ``vision_config``.
            temperature: Sampling temperature. Perceptron docs recommend 0.0
                for grounded tasks.
            max_completion_tokens: Output token ceiling.

        Returns:
            The raw OpenAI ``ChatCompletion`` response.

        Raises:
            openai.APIError: For any non-retried error (4xx other than 429,
                5xx, network failures after exhausting retries). We let it
                propagate so the operator framework surfaces the traceback.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
        }
        if vision_config is not None:
            kwargs["extra_body"] = {"vision_config": vision_config}
        if response_format is not None:
            kwargs["response_format"] = response_format

        text_preview = _last_user_text_preview(messages)
        logger.info(
            "[perceptron] POST /chat/completions model=%s vision_config=%s response_format=%s "
            "max_completion_tokens=%d user_text=%r",
            model,
            vision_config,
            "set" if response_format else None,
            max_completion_tokens,
            text_preview,
        )

        # The retry loop is the ONLY try/except in v1. Every iteration either
        # returns or raises, so there's no code path after the loop.
        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                self._log_response_summary(response, attempt)
                self._accumulate_usage(response)
                return response
            except RateLimitError as exc:
                if attempt == MAX_RETRY_ATTEMPTS:
                    logger.error(
                        "[perceptron] rate-limited after %d attempts; giving up", attempt
                    )
                    raise
                wait = self._compute_backoff(exc, attempt)
                logger.warning(
                    "[perceptron] 429 on attempt %d/%d; sleeping %.1fs before retry",
                    attempt,
                    MAX_RETRY_ATTEMPTS,
                    wait,
                )
                time.sleep(wait)

    def _accumulate_usage(self, response: Any) -> None:
        """Add this response's usage to the running totals."""
        # Defensive getattrs: rare 200s have lacked `.usage` in the past.
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self._total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        self._total_calls += 1

    @classmethod
    def from_env(cls, env_var: str = "PERCEPTRON_API_KEY", **kwargs: Any) -> Self:
        """Build a client from `os.environ[env_var]`.

        Raises:
            RuntimeError: If the env var is unset or empty. We surface a
                clear error rather than let a downstream call produce a 401.
        """
        value = os.environ.get(env_var, "")
        if not value:
            raise RuntimeError(
                f"environment variable {env_var!r} is not set or is empty; "
                f"export it before launching FiftyOne or the script that "
                f"instantiates the model"
            )
        return cls(api_key=value, **kwargs)

    @staticmethod
    def _compute_backoff(exc: RateLimitError, attempt: int) -> float:
        """Pick a sleep duration before the next retry.

        Prefers the server's ``Retry-After`` hint when numeric, else falls
        back to ``2 ** attempt`` seconds. Mirrors
        https://docs.perceptron.inc/guides/batch.
        """
        response = getattr(exc, "response", None)
        if response is not None and hasattr(response, "headers"):
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    # Non-numeric (HTTP-date) header; exponential fallback below.
                    pass
        return float(2**attempt)

    @staticmethod
    def _log_response_summary(response: Any, attempt: int) -> None:
        """Log id / finish_reason / usage from a chat-completions response."""
        choice = response.choices[0]
        usage = response.usage
        content_preview = _truncate((choice.message.content or "").replace("\n", " "))
        logger.info(
            "[perceptron] response id=%s attempt=%d finish_reason=%s "
            "tokens={prompt=%d completion=%d total=%d} content=%r",
            response.id,
            attempt,
            choice.finish_reason,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
            content_preview,
        )


def _last_user_text_preview(messages: list[dict[str, Any]]) -> str:
    """Return a truncated text preview from the last user message, or ``""``."""
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return ""
    for part in user_msgs[-1].get("content", []):
        if isinstance(part, dict) and part.get("type") == "text":
            return _truncate(part.get("text", ""))
    return ""
