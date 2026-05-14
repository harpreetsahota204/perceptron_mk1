"""`PerceptronModel` -- the FiftyOne `Model` wrapper.

Owns configuration, per-sample request construction, the call to
`PerceptronClient.chat_completion(...)`, and the handoff to
`perceptron_parser.to_fiftyone(...)`.

Three dispatch paths inside `predict()`:

    * Dense image-mode (TRACK only) -- decompose the video into frames via cv2
      at the configured stride and send each as an ``image_url`` request.
      Returns a single label container with per-item ``t`` (seconds) attributes
      that `write_per_frame_labels` routes to ``sample.frames[i]``.
    * Image mode (DETECT, KEYPOINTS, POLYGON, and shared tasks on image datasets)
      -- single ``image_url`` request per sample, returns a sample-level label.
    * Video mode (FIND_EVENT, KEY_MOMENTS, and shared tasks on video datasets)
      -- single ``video_url`` request per sample.

Implements FiftyOne's `Model` interface so it works with
`dataset.apply_model(...)` and the zoo loader.
"""

from __future__ import annotations

import logging
import mimetypes
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import fiftyone as fo
from fiftyone import Model

from .perceptron_api import PerceptronClient, to_image_data_uri, to_video_data_uri
from .perceptron_parser import FOLabel, to_fiftyone
from .prompts import (
    TASK_TO_API_FORMAT,
    TASK_TO_PARSER_FORMAT,
    TASKS_DENSE_IMAGE_MODE,
    TASKS_IMAGE_GROUNDING,
    Task,
    default_user_prompt,
)

logger = logging.getLogger("perceptron")


# This plugin targets Mk1 exclusively. The usage patterns (grounding, clipping,
# thinking, focus) are specific to Mk1's capabilities.
DEFAULT_MODEL_NAME: str = "perceptron-mk1"

# 4096 tokens is generous for tracking (4 boxes ~200 tokens); captioning and
# classification land well under this.
DEFAULT_MAX_COMPLETION_TOKENS: int = 4096

# Perceptron docs recommend 0.0 for grounded tasks; we use it across the board.
DEFAULT_TEMPERATURE: float = 0.0

# Default stride for TRACK (dense path). Every Nth frame is sent as a separate
# image-mode API call. Empirical miss rate at stride=3 is ~35%; smaller strides
# improve coverage at proportional cost.
DEFAULT_STRIDE: int = 3


# json_schema for CLASSIFY_SINGLE / CLASSIFY_MULTI. Strict mode forces the
# model to return raw JSON matching this shape -- no fenced-block extraction
# needed downstream.
_CLASSIFY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["classifications"],
    "properties": {
        "classifications": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["label", "confidence"],
                "properties": {
                    "label": {"type": "string"},
                    "confidence": {"type": "number"},
                },
            },
        },
    },
}


@dataclass(slots=True, kw_only=True, frozen=True)
class PerceptronConfig:
    """Immutable configuration for one `PerceptronModel` instance.

    Frozen so the task can't change mid-run (which would invalidate the
    parser dispatch).

    Attributes:
        model: Perceptron model id (e.g. ``"perceptron-mk1"``).
        task: Drives prompt selection, vision_config, and predict dispatch.
        media_type: ``"image"`` or ``"video"``. Set by the operator from
            ``ctx.dataset.media_type``. Controls both the FiftyOne
            ``Model.media_type`` property and the predict dispatch for shared
            tasks (CLASSIFY, CAPTION, VQA) that work on both media types.
        target: User-supplied target -- class name, event description,
            classify aspect, or VQA question, depending on ``task``.
            Validated by `default_user_prompt`.
        prompt: Override for the auto-generated user prompt. ``None`` -> use
            `prompts.default_user_prompt`.
        enable_thinking: Sets ``vision_config.enable_thinking``. Off by
            default -- thinking is expensive and can demote structured
            output to prose for weakly-prompted clip tasks.
        focus: Sets ``vision_config.internal_tools.focus``. Off by default;
            when ``True`` the model zooms into regions and re-runs inference
            on crops, sharpening grounding results.
        max_completion_tokens: Output ceiling per call.
        temperature: Sampling temperature.
        stride: For TRACK (dense path), send every Nth frame. Ignored for
            image-mode and video-mode paths.
        max_frames: For TRACK, cap the total frames sent. ``None`` = no cap.
        response_format: Caller-supplied ``response_format`` override (e.g.
            Semantic Search passes a yes/no-constrained json_schema). When
            ``None`` and task is a CLASSIFY_* task, the default
            `_CLASSIFY_JSON_SCHEMA` is applied.
        api_key: Explicit API key. The operator passes
            ``ctx.secrets["PERCEPTRON_API_KEY"]``; the zoo loader and
            standalone scripts leave it ``None`` to fall back to
            ``os.environ["PERCEPTRON_API_KEY"]``.
    """

    model: str = DEFAULT_MODEL_NAME
    task: Task = Task.TRACK
    media_type: str = "video"
    target: str | None = None
    prompt: str | None = None
    enable_thinking: bool = False
    focus: bool = False
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    stride: int = DEFAULT_STRIDE
    max_frames: int | None = None
    response_format: dict[str, Any] | None = field(default=None)
    api_key: str | None = None


class PerceptronModel(Model):
    """FiftyOne `Model` for the Perceptron API.

    Lifecycle: construct with a config dict; FiftyOne (or our operator) calls
    `predict(filepath, sample=sample)` per sample; we dispatch to the dense,
    image, or video-mode path based on task and media type; we parse and
    return a FiftyOne label container or string.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize from a (possibly empty) config dict.

        We accept a plain dict (not a `PerceptronConfig`) because that's what
        the zoo `load_model(**kwargs)` path passes. Unknown keys are ignored.

        API-key resolution:
            * ``config["api_key"]`` if provided (operator passes
              ``ctx.secrets["PERCEPTRON_API_KEY"]``).
            * Otherwise `PerceptronClient.from_env()` reads
              ``os.environ["PERCEPTRON_API_KEY"]``.

        Raises:
            ValueError: If ``config["task"]`` isn't a known `Task` member.
            RuntimeError: If no api_key was provided and the env var is unset.
        """
        cfg_dict = config or {}
        task_value = cfg_dict.get("task", Task.TRACK)
        task = task_value if isinstance(task_value, Task) else Task(task_value)

        self._config = PerceptronConfig(
            model=str(cfg_dict.get("model", DEFAULT_MODEL_NAME)),
            task=task,
            media_type=str(cfg_dict.get("media_type", "video")),
            target=cfg_dict.get("target"),
            prompt=cfg_dict.get("prompt"),
            enable_thinking=bool(cfg_dict.get("enable_thinking", False)),
            focus=bool(cfg_dict.get("focus", False)),
            max_completion_tokens=int(
                cfg_dict.get("max_completion_tokens", DEFAULT_MAX_COMPLETION_TOKENS)
            ),
            temperature=float(cfg_dict.get("temperature", DEFAULT_TEMPERATURE)),
            stride=int(cfg_dict.get("stride", DEFAULT_STRIDE)),
            max_frames=cfg_dict.get("max_frames"),
            response_format=cfg_dict.get("response_format"),
            api_key=cfg_dict.get("api_key"),
        )

        # Explicit api_key (App-side via ctx.secrets) wins over the env var.
        if self._config.api_key:
            self._client = PerceptronClient(api_key=self._config.api_key)
            key_source = "config (ctx.secrets)"
        else:
            self._client = PerceptronClient.from_env()
            key_source = "os.environ"

        # Stores the raw API response content after each predict() call.
        # Read by the operator layer to surface it to the browser console.
        self._last_raw_content: str = ""

        logger.info(
            "[perceptron] PerceptronModel ready: model=%s task=%s target=%r "
            "enable_thinking=%s focus=%s max_completion_tokens=%d stride=%d "
            "max_frames=%s api_key_source=%s",
            self._config.model,
            self._config.task.value,
            self._config.target,
            self._config.enable_thinking,
            self._config.focus,
            self._config.max_completion_tokens,
            self._config.stride,
            self._config.max_frames,
            key_source,
        )

    # -- FiftyOne Model interface ------------------------------------------------

    @property
    def media_type(self) -> str:
        # FiftyOne uses this to route dataset.apply_model() to the right samples.
        # The operator sets this from ctx.dataset.media_type at model-build time.
        return self._config.media_type

    # -- Public access for callers ----------------------------------------------

    @property
    def config(self) -> PerceptronConfig:
        return self._config

    @property
    def annotation_format(self) -> str | None:
        """``vision_config.annotation_format`` for this task.

        ``None`` for free-text and classify tasks, which send no
        ``vision_config`` extension at all.
        """
        return TASK_TO_API_FORMAT[self._config.task]

    @property
    def parser_format(self) -> str:
        """Dispatch key for `perceptron_parser.to_fiftyone(...)`."""
        return TASK_TO_PARSER_FORMAT[self._config.task]

    @property
    def usage_totals(self) -> dict[str, int]:
        """Pass-through to `PerceptronClient.usage_totals`."""
        return self._client.usage_totals

    def reset_usage_totals(self) -> None:
        """Zero the client's token / call totals (call at the start of each run)."""
        self._client.reset_usage_totals()

    def predict(self, filepath: str, sample: fo.Sample | None = None) -> FOLabel:
        """Run inference on one media file (image or video).

        Dispatch order:
            1. TASKS_DENSE_IMAGE_MODE (TRACK) -> ``_predict_dense``:
               decompose the video into per-frame image requests.
            2. TASKS_IMAGE_GROUNDING or media_type == "image" -> ``_predict_image``:
               single image_url request per sample.
            3. Everything else -> ``_predict_video``:
               single video_url request per sample.

        Args:
            filepath: Absolute local path to the image or video file.
            sample: The current FiftyOne sample. Used to read
                ``metadata.frame_rate`` / ``metadata.total_frame_count`` for
                the dense path. Pass ``None`` from standalone scripts to fall
                back to cv2 probing.

        Returns:
            See `perceptron_parser.to_fiftyone` for per-task return shapes.
        """
        if self._config.task in TASKS_DENSE_IMAGE_MODE:
            return self._predict_dense(filepath, sample)
        if self._config.task in TASKS_IMAGE_GROUNDING or self._config.media_type == "image":
            return self._predict_image(filepath)
        return self._predict_video(filepath, sample)

    # -- Image-mode predict (one ``image_url`` request) --------------------------

    def _predict_image(self, filepath: str) -> FOLabel:
        """Single-shot inference on one image file.

        Reads the image bytes from disk, infers the MIME type, and sends a
        single ``image_url`` request. Returns a sample-level label container.
        """
        image_bytes = Path(filepath).read_bytes()
        mime = mimetypes.guess_type(filepath)[0] or "image/jpeg"
        prompt = self._resolve_prompt()
        messages = self._build_image_messages(jpeg_bytes=image_bytes, prompt=prompt, mime=mime)
        vision_config = self._build_vision_config()
        response_format = self._build_response_format()

        logger.info(
            "[perceptron] predict (image-mode): task=%s target=%r path=%s prompt=%r",
            self._config.task.value,
            self._config.target,
            filepath,
            prompt[:200] + ("..." if len(prompt) > 200 else ""),
        )

        response = self._client.chat_completion(
            model=self._config.model,
            messages=messages,
            vision_config=vision_config,
            response_format=response_format,
            temperature=self._config.temperature,
            max_completion_tokens=self._config.max_completion_tokens,
        )
        content = response.choices[0].message.content or ""
        self._last_raw_content = content
        label = to_fiftyone(content, self.parser_format, target=self._config.target)
        logger.info(
            "[perceptron] predict produced %s",
            type(label).__name__ if label is not None else "None",
        )
        return label

    # -- Video-mode predict (one ``video_url`` request) --------------------------

    def _predict_video(self, filepath: str, sample: fo.Sample | None) -> FOLabel:
        """Single-shot inference on one video file.

        Encodes the video as a base64 data URI and sends a single ``video_url``
        request. Returns a sample-level label container (TemporalDetections for
        clip tasks, Classification/str for shared tasks).
        """
        prompt = self._resolve_prompt()
        messages = self._build_video_messages(video_path=filepath, prompt=prompt)
        vision_config = self._build_vision_config()
        response_format = self._build_response_format()

        logger.info(
            "[perceptron] predict (video-mode): task=%s target=%r path=%s prompt=%r",
            self._config.task.value,
            self._config.target,
            filepath,
            prompt[:200] + ("..." if len(prompt) > 200 else ""),
        )

        response = self._client.chat_completion(
            model=self._config.model,
            messages=messages,
            vision_config=vision_config,
            response_format=response_format,
            temperature=self._config.temperature,
            max_completion_tokens=self._config.max_completion_tokens,
        )
        content = response.choices[0].message.content or ""
        self._last_raw_content = content
        label = to_fiftyone(
            content,
            self.parser_format,
            target=self._config.target,
            frame_rate=self._extract_frame_rate(sample),
        )
        logger.info(
            "[perceptron] predict produced %s",
            type(label).__name__ if label is not None else "None",
        )
        return label

    # -- Dense image-mode predict (N per-frame requests) -------------------------

    def _predict_dense(self, video_path: str, sample: fo.Sample | None) -> FOLabel:
        """Decompose the video into frames and run per-frame image-mode inference.

        Used only for TRACK. Extracts frames at the configured stride, sends
        each as an ``image_url`` request, and accumulates all detections into
        a single ``fo.Detections`` container with ``t`` (seconds) attributes.
        ``write_per_frame_labels`` uses those ``t`` values to route each
        detection to the correct ``sample.frames[i]``.
        """
        frame_rate, frame_count = self._read_video_dimensions(video_path, sample)
        if frame_rate is None or frame_rate <= 0:
            raise RuntimeError(
                f"task={self._config.task.value} (dense) requires a valid "
                f"video frame rate. Run dataset.compute_metadata() and "
                f"re-launch, or pass a sample with metadata.frame_rate set."
            )
        if frame_count is None or frame_count <= 0:
            raise RuntimeError(
                f"task={self._config.task.value} (dense) could not determine "
                f"frame count from {video_path}; cv2.VideoCapture returned "
                f"frame_count={frame_count}."
            )

        stride = max(1, self._config.stride)
        indices = list(range(0, frame_count, stride))
        if self._config.max_frames is not None and self._config.max_frames > 0:
            indices = indices[: self._config.max_frames]

        prompt = self._resolve_prompt()
        vision_config = self._build_vision_config()
        parser_format = self.parser_format

        logger.info(
            "[perceptron] predict (dense): task=%s target=%r path=%s "
            "frame_count=%d stride=%d -> %d API calls; prompt=%r",
            self._config.task.value,
            self._config.target,
            video_path,
            frame_count,
            stride,
            len(indices),
            prompt[:200] + ("..." if len(prompt) > 200 else ""),
        )

        accumulated: list[Any] = []
        for frame_idx, jpeg_bytes in _iter_frames(video_path, indices):
            # Dense path always sends JPEG frames extracted by cv2.
            messages = self._build_image_messages(
                jpeg_bytes=jpeg_bytes, prompt=prompt, mime="image/jpeg"
            )
            response = self._client.chat_completion(
                model=self._config.model,
                messages=messages,
                vision_config=vision_config,
                temperature=self._config.temperature,
                max_completion_tokens=self._config.max_completion_tokens,
            )
            content = response.choices[0].message.content or ""
            self._last_raw_content = content  # last frame wins; useful for spot-checking
            label = to_fiftyone(
                content,
                parser_format,
                target=self._config.target,
                frame_rate=frame_rate,
            )
            items = _items_from_container(label)
            t_seconds = frame_idx / frame_rate
            for item in items:
                item["t"] = t_seconds
            accumulated.extend(items)

        # TRACK is the only dense task now (KEYPOINTS moved to image-grounding),
        # so the container is always fo.Detections.
        container: FOLabel = fo.Detections(detections=accumulated)
        logger.info(
            "[perceptron] dense predict produced %s with %d item(s) across %d frame(s)",
            type(container).__name__,
            len(accumulated),
            len(indices),
        )
        return container

    # -- Internal helpers --------------------------------------------------------

    def _resolve_prompt(self) -> str:
        """Return the configured prompt override, or the default template."""
        if self._config.prompt:
            return self._config.prompt
        return default_user_prompt(
            self._config.task,
            self._config.target,
            media_type=self._config.media_type,
        )

    def _build_vision_config(self) -> dict[str, Any] | None:
        """Compose the ``vision_config`` extension for this task.

        Sent via OpenAI's ``extra_body``. Supported ``annotation_format``
        values: ``"box"``, ``"point"``, ``"polygon"``, ``"clip"``.
        ``annotation_format`` and ``enable_thinking`` are only added when
        relevant. ``internal_tools.focus`` is always included so the API
        receives the user's preference regardless of task type.

        Note: Mk1 currently returns bbox-shaped polygons for polygon tasks
        rather than true contour polygons. The parser handles both shapes.
        """
        cfg: dict[str, Any] = {}
        fmt = TASK_TO_API_FORMAT[self._config.task]
        if fmt is not None:
            cfg["annotation_format"] = fmt
        if self._config.enable_thinking:
            cfg["enable_thinking"] = True
        cfg["internal_tools"] = {"focus": self._config.focus}
        return cfg

    def _build_response_format(self) -> dict[str, Any] | None:
        """Compose the ``response_format`` for this task.

        Caller-supplied override (Semantic Search) wins; otherwise CLASSIFY
        tasks get the strict json_schema, others get ``None``.
        """
        if self._config.response_format is not None:
            return self._config.response_format
        if self._config.task in (Task.CLASSIFY_SINGLE, Task.CLASSIFY_MULTI):
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "Classifications",
                    "strict": True,
                    "schema": _CLASSIFY_JSON_SCHEMA,
                },
            }
        return None

    def _build_video_messages(
        self,
        *,
        video_path: str,
        prompt: str,
    ) -> list[dict[str, Any]]:
        """Assemble messages for a video-mode call (one ``video_url`` part).

        Output format and thinking toggle are conveyed via the request-level
        ``vision_config`` (see `_build_vision_config`), not the message list.
        """
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": to_video_data_uri(video_path)},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _build_image_messages(
        self,
        *,
        jpeg_bytes: bytes,
        prompt: str,
        mime: str = "image/jpeg",
    ) -> list[dict[str, Any]]:
        """Assemble messages for an image-mode call (one ``image_url`` part).

        Args:
            jpeg_bytes: Raw image bytes (any format; ``mime`` describes it).
            prompt: User prompt text.
            mime: MIME type of the image (e.g. ``"image/jpeg"``, ``"image/png"``).
                Defaults to ``"image/jpeg"`` for frames extracted by cv2.
        """
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": to_image_data_uri(jpeg_bytes, mime=mime)},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    @staticmethod
    def _extract_frame_rate(sample: fo.Sample | None) -> float | None:
        """``sample.metadata.frame_rate`` if available, else ``None``."""
        if sample is None:
            return None
        metadata = sample.metadata
        if metadata is None:
            return None
        frame_rate = getattr(metadata, "frame_rate", None)
        return float(frame_rate) if frame_rate else None

    @staticmethod
    def _extract_frame_count(sample: fo.Sample | None) -> int | None:
        """``sample.metadata.total_frame_count`` if available, else ``None``."""
        if sample is None:
            return None
        metadata = sample.metadata
        if metadata is None:
            return None
        n = getattr(metadata, "total_frame_count", None)
        return int(n) if n else None

    def _read_video_dimensions(
        self, video_path: str, sample: fo.Sample | None
    ) -> tuple[float | None, int | None]:
        """Resolve (frame_rate, frame_count) from sample metadata, else cv2.

        Prefers FiftyOne metadata (cheap, cached). Falls back to opening the
        file with cv2 for standalone-script usage where no sample is passed.
        """
        frame_rate = self._extract_frame_rate(sample)
        frame_count = self._extract_frame_count(sample)
        if frame_rate is not None and frame_count is not None:
            return frame_rate, frame_count

        # Fallback: probe the file directly.
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                return frame_rate, frame_count
            if frame_rate is None:
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_rate = float(fps) if fps and fps > 0 else None
            if frame_count is None:
                n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                frame_count = int(n) if n and n > 0 else None
        finally:
            cap.release()
        return frame_rate, frame_count


def _iter_frames(video_path: str, indices: list[int]) -> Iterator[tuple[int, bytes]]:
    """Yield ``(frame_idx, jpeg_bytes)`` for every readable frame in ``indices``.

    Frames that fail to read or encode are skipped with a WARNING. The caller
    iterates the result and accumulates per-frame predictions.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            logger.error(
                "[perceptron] cv2 could not open %s; dense path returns 0 frames",
                video_path,
            )
            return
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.warning(
                    "[perceptron] could not read frame %d from %s; skipping",
                    idx,
                    video_path,
                )
                continue
            ok, jpg = cv2.imencode(".jpg", frame)
            if not ok:
                logger.warning(
                    "[perceptron] could not encode frame %d as JPEG; skipping",
                    idx,
                )
                continue
            yield idx, jpg.tobytes()
    finally:
        cap.release()


def _items_from_container(label: FOLabel) -> list[Any]:
    """Extract the detections list from a `fo.Detections` container.

    The dense path (TRACK) always produces ``fo.Detections``; any other shape
    returns an empty list. Items are stamped with ``t`` (seconds) by the caller.
    """
    if isinstance(label, fo.Detections):
        return list(label.detections)
    return []
