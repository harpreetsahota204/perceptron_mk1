"""Cross-cutting helpers shared by the operator and zoo registration."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from .prompts import TASKS_DENSE_IMAGE_MODE, Task

# Fixed-name logger so `fiftyone.operators.ProgressHandler` can capture every
# `[perceptron]` line from every module by attaching to a single logger.
# Keep this name in sync across all plugin modules.
logger = logging.getLogger("perceptron")


def _ensure_stream_handler() -> None:
    """Attach a stdout `StreamHandler` to the ``"perceptron"`` logger.

    Idempotent. Without this, our `logger.info(...)` calls vanish silently
    when the plugin is loaded by `python -m fiftyone.server.main`: FiftyOne's
    server config attaches handlers to its own loggers but not to ours, and
    Python's default behavior is to swallow INFO records that have no
    handler. ``propagate=False`` so we don't double-emit through any handler
    a host application may have attached to the root logger.
    """
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s -- %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


_ensure_stream_handler()


# Bumped when stored data (custom runs, secrets schema, etc.) changes shape
# in a way that's incompatible with prior versions.
PLUGIN_VERSION: str = "v1"

SECRET_NAME: str = "PERCEPTRON_API_KEY"


def notify(ctx: Any, message: str, *, variant: str = "info") -> None:
    """Fire an App-side toast prefixed with `[perceptron]`.

    Used for start / success events. Failures bubble up through the operator
    framework on their own; we don't catch-and-notify.
    """
    ctx.ops.notify(f"[perceptron] {message}", variant=variant)


def get_api_key(ctx: Any) -> str | None:
    """Return the Perceptron API key from FiftyOne secrets, or `None`.

    `ctx.secrets` is hydrated at runtime from environment variables matching
    the names declared in `fiftyone.yml`. See
    https://docs.voxel51.com/plugins/developing_plugins.html#accessing-secrets.
    """
    return ctx.secrets.get(SECRET_NAME)


def has_api_key(ctx: Any) -> bool:
    """Return whether `ctx.secrets` has our API key set.

    Used by `resolve_input` to gate the form so a 401 can't happen mid-run.
    """
    return bool(get_api_key(ctx))


def make_run_key(operation: str) -> str:
    """Build a Custom-Run key for auditing.

    Format: ``perceptron_<operation>_<version>_<UTC timestamp>``. Per FiftyOne
    conventions the key must be a valid Python identifier -- no slashes,
    only letters / digits / underscores.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"perceptron_{operation}_{PLUGIN_VERSION}_{timestamp}"


def task_supports_per_frame(task: Task) -> bool:
    """Whether ``task`` writes labels per video frame rather than at sample level.

    True only for `TASKS_DENSE_IMAGE_MODE` (currently TRACK), which decomposes
    a video into per-frame image requests and stamps each result with a ``t``
    (seconds) attribute for routing back to ``sample.frames[i]``.

    All other tasks -- including image grounding tasks like DETECT and KEYPOINTS
    -- write at sample level and return False here.
    """
    return task in TASKS_DENSE_IMAGE_MODE
