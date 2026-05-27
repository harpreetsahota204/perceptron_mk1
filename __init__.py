"""FiftyOne plugin entry point for the Perceptron integration.

This module is intentionally thin: it registers a single operator and exposes
the two hooks the FiftyOne zoo uses when this directory is treated as a
remote zoo model source.

Module layout:

    perceptron_api.py     -- HTTP client + retry/backoff
    perceptron_parser.py  -- raw tag/JSON output -> FiftyOne label types
    perceptron_model.py   -- `PerceptronModel` (SamplesMixin + Model)
    prompts.py            -- canonical prompt templates and task enum
    _shared.py            -- helpers shared by the operator and zoo registration
    operators.py          -- `RunPerceptron`: the single operator, with a
                             conditional-input form for Event Search /
                             Semantic Search / Bootstrap Labels modes.
                             Accessible from the operator browser (backtick)
                             or the grid-action button.

Dual distribution:
    * As a plugin: drop this directory into FiftyOne's plugins dir; the
      plugin loader calls `register(plugin)`.
    * As a zoo model source: register the repo with
      `foz.register_zoo_model_source(...)` then `foz.load_zoo_model(
      "perceptron/mk1", task=..., target=...)`. The zoo loader calls
      `download_model` (a no-op marker file -- nothing to actually download
      for a remote API model) then `load_model`, which returns a configured
      `PerceptronModel`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .chat_panel import PerceptronChatPanel
from .operators import RunPerceptron

logger = logging.getLogger("perceptron")


def register(plugin) -> None:
    """Register the RunPerceptron operator and PerceptronChatPanel."""
    plugin.register(RunPerceptron)
    plugin.register(PerceptronChatPanel)


def download_model(model_name: str, model_path: str) -> None:
    """Marker-file stand-in for a real downloader.

    Mk1 is a remote API, so there's nothing to actually download. The
    convention (matched by gemini-vision-plugin) is to touch an empty file
    at `model_path` so FiftyOne's existence check passes. The real API call
    happens later inside `PerceptronModel.predict(...)`.
    """
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    logger.info("[perceptron] download_model: %s -> marker file at %s", model_name, path)


def load_model(model_name: str, model_path: str, **kwargs: Any):
    """Construct a `PerceptronModel` for the FiftyOne zoo loader.

    `model_name` and `model_path` are ignored -- the actual model variant
    comes from ``kwargs["model"]`` so callers can pick at load time
    (``foz.load_zoo_model(..., model="isaac-0.1")``). `**kwargs` is forwarded
    to `PerceptronModel(config=...)`.
    """
    # Lazy import so listing plugins doesn't pull in the OpenAI client.
    from .perceptron_model import PerceptronModel

    logger.info(
        "[perceptron] load_model: name=%s path=%s kwargs=%s",
        model_name,
        model_path,
        kwargs,
    )
    return PerceptronModel(config=kwargs)
