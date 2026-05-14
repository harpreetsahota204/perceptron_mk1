"""Single source of truth for Perceptron task definitions and prompt templates.

Defines the `Task` enum, the API/parser format mappings, task-set frozensets,
and the default user-prompt templates.

Task -> media type -> output shape:

    Image-only tasks (single-shot, one API call per image):
        DETECT    -> fo.Detections          (box grounding)
        KEYPOINTS -> fo.Keypoints           (point grounding)
        POLYGON   -> fo.Polylines           (polygon grounding)

    Video-only tasks:
        TRACK       -> per-frame fo.Detections via dense frame decomposition
                       (N image-mode API calls at chosen stride).
                       Note: model is not explicitly trained for tracking;
                       results vary. Run a downstream tracker for ID continuity.
        FIND_EVENT  -> fo.TemporalDetections, targeted moment search.
                       Video-mode, one API call per sample.
        KEY_MOMENTS -> fo.TemporalDetections, unconstrained event summary.
                       Video-mode, one API call per sample. No target required.

    Shared tasks (image or video, one API call per sample):
        CAPTION_CONCISE  -> free text on sample.
        CAPTION_DETAILED -> free text on sample (also implicitly OCRs signage).
        CLASSIFY_SINGLE  -> fo.Classification w/ confidence via json_schema.
        CLASSIFY_MULTI   -> fo.Classifications via json_schema.
        VQA              -> free text on sample (user-supplied question).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class Task(StrEnum):
    """Every task the plugin exposes.

    Members are plain strings (``Task.TRACK == "track"``) so they round-trip
    cleanly through JSON ``ctx.params`` values while still giving us type-safe
    dispatch in ``match`` statements.
    """

    # Image-only grounding tasks (single-shot per image)
    DETECT = "detect"
    KEYPOINTS = "keypoints"
    POLYGON = "polygon"

    # Video-only tasks
    TRACK = "track"
    FIND_EVENT = "find_event"
    KEY_MOMENTS = "key_moments"

    # Shared tasks (work on both image and video samples)
    CAPTION_CONCISE = "caption_concise"
    CAPTION_DETAILED = "caption_detailed"
    CLASSIFY_SINGLE = "classify_single"
    CLASSIFY_MULTI = "classify_multi"
    VQA = "vqa"


# ---------------------------------------------------------------------------
# Format mappings. Two parallel dicts so each layer gets exactly what it needs:
#
#   TASK_TO_API_FORMAT     -> vision_config.annotation_format sent on the wire.
#                             Supported values: box / point / polygon / clip.
#                             None means no vision_config extension is sent.
#   TASK_TO_PARSER_FORMAT  -> dispatch key for perceptron_parser.to_fiftyone().
#                             Adds free-text / JSON variants absent from the API.
# ---------------------------------------------------------------------------

TASK_TO_API_FORMAT: Final[dict[Task, str | None]] = {
    Task.DETECT: "box",
    Task.KEYPOINTS: "point",
    Task.POLYGON: "polygon",
    Task.TRACK: "box",
    Task.FIND_EVENT: "clip",
    Task.KEY_MOMENTS: "clip",
    Task.CAPTION_CONCISE: None,
    Task.CAPTION_DETAILED: None,
    Task.CLASSIFY_SINGLE: None,
    Task.CLASSIFY_MULTI: None,
    Task.VQA: None,
}

TASK_TO_PARSER_FORMAT: Final[dict[Task, str]] = {
    Task.DETECT: "box",
    Task.KEYPOINTS: "point",
    Task.POLYGON: "polygon",
    Task.TRACK: "box",
    Task.FIND_EVENT: "clip",
    Task.KEY_MOMENTS: "clip",
    Task.CAPTION_CONCISE: "caption",
    Task.CAPTION_DETAILED: "caption",
    Task.CLASSIFY_SINGLE: "classify_single",
    Task.CLASSIFY_MULTI: "classify_multi",
    Task.VQA: "vqa",
}


# ---------------------------------------------------------------------------
# Task-set constants. Used by the operator (to build per-media-type form lists)
# and the model (to dispatch between image / video / dense predict paths).
# ---------------------------------------------------------------------------

# Single-shot image grounding; go through _predict_image (one image_url call).
TASKS_IMAGE_GROUNDING: Final[frozenset[Task]] = frozenset(
    {Task.DETECT, Task.KEYPOINTS, Task.POLYGON}
)

# Video-only; not shown on image datasets.
TASKS_VIDEO_ONLY: Final[frozenset[Task]] = frozenset(
    {Task.TRACK, Task.FIND_EVENT, Task.KEY_MOMENTS}
)

# Available on both image and video datasets.
TASKS_SHARED: Final[frozenset[Task]] = frozenset(
    {Task.CAPTION_CONCISE, Task.CAPTION_DETAILED, Task.CLASSIFY_SINGLE, Task.CLASSIFY_MULTI, Task.VQA}
)

# Dense video path: decompose into per-frame image-mode requests.
# Only TRACK remains here; KEYPOINTS was moved to TASKS_IMAGE_GROUNDING.
TASKS_DENSE_IMAGE_MODE: Final[frozenset[Task]] = frozenset({Task.TRACK})


# ---------------------------------------------------------------------------
# Default user-prompt templates.
# ---------------------------------------------------------------------------

# Image grounding: class-restricted detection / segmentation.
_DETECT_TEMPLATE: Final[str] = (
    "Your goal is to segment out the following categories: {target}"
)

# Image grounding: point to each instance of the target class.
_KEYPOINTS_TEMPLATE: Final[str] = "Point to each {target}."

# Image grounding: polygon outlines around each instance.
_POLYGON_TEMPLATE: Final[str] = (
    "Your goal is to segment out the following categories with polygon outlines: {target}"
)

# Moment-pinpoint phrasing. Empirically produces a single-frame <clip ... />
# tag (support=[N, N] in the resulting fo.TemporalDetection). The previous
# "Identify when {target}." phrasing emitted prose with no <clip> tag.
_FIND_EVENT_TEMPLATE: Final[str] = "Clip the exact moment {target}."

# Unconstrained clip extraction -- no target, model decides what's noteworthy.
_KEY_MOMENTS_TEMPLATE: Final[str] = (
    "Find all the distinct moments in this video."
)

_CAPTION_CONCISE_TEMPLATE: Final[str] = (
    "Provide a concise, human-friendly caption for this {media_type}."
)
_CAPTION_DETAILED_TEMPLATE: Final[str] = (
    "Provide a detailed caption describing key objects, relationships, and "
    "context in this {media_type}."
)

_CLASSIFY_SINGLE_TEMPLATE: Final[str] = (
    "What is the primary {target} shown in this {media_type}?"
)
_CLASSIFY_MULTI_TEMPLATE: Final[str] = (
    "List every relevant category that describes what is visible in this "
    "{media_type} -- {target}."
)


def default_user_prompt(
    task: Task,
    target: str | None = None,
    *,
    media_type: str = "video",
) -> str:
    """Return the default user prompt for ``task``, formatted with ``target``.

    Args:
        task: The task being run.
        target: Semantics vary by task:
            - DETECT / KEYPOINTS / POLYGON / TRACK / FIND_EVENT / VQA:
              required; raises ``ValueError`` when empty.
            - CLASSIFY_SINGLE / CLASSIFY_MULTI: optional aspect hint
              (e.g. ``"scene type"``); falls back to a sensible default.
            - CAPTION_* / KEY_MOMENTS: ignored.
        media_type: ``"image"`` or ``"video"``. Substituted into templates
            that mention the medium so prompts read naturally for both.
    """
    match task:
        case Task.DETECT:
            if not target:
                raise ValueError(
                    "task=detect requires a non-empty target (object class to detect)"
                )
            return _DETECT_TEMPLATE.format(target=target)

        case Task.KEYPOINTS:
            if not target:
                raise ValueError(
                    "task=keypoints requires a non-empty target (object to point at)"
                )
            return _KEYPOINTS_TEMPLATE.format(target=target)

        case Task.POLYGON:
            if not target:
                raise ValueError(
                    "task=polygon requires a non-empty target (object class to outline)"
                )
            return _POLYGON_TEMPLATE.format(target=target)

        case Task.TRACK:
            if not target:
                raise ValueError(
                    "task=track requires a non-empty target (the object to detect per frame)"
                )
            return _DETECT_TEMPLATE.format(target=target)

        case Task.FIND_EVENT:
            if not target:
                raise ValueError(
                    "task=find_event requires a non-empty target "
                    "(event description, e.g. 'a pedestrian crosses the street')"
                )
            return _FIND_EVENT_TEMPLATE.format(target=target)

        case Task.KEY_MOMENTS:
            # No target needed -- model freely identifies what matters.
            return _KEY_MOMENTS_TEMPLATE

        case Task.CAPTION_CONCISE:
            return _CAPTION_CONCISE_TEMPLATE.format(media_type=media_type)

        case Task.CAPTION_DETAILED:
            return _CAPTION_DETAILED_TEMPLATE.format(media_type=media_type)

        case Task.CLASSIFY_SINGLE:
            aspect = target or "scene type"
            return _CLASSIFY_SINGLE_TEMPLATE.format(target=aspect, media_type=media_type)

        case Task.CLASSIFY_MULTI:
            aspects = target or (
                "scene type, vehicles present, pedestrians, lighting conditions, "
                "and any other salient attributes"
            )
            return _CLASSIFY_MULTI_TEMPLATE.format(target=aspects, media_type=media_type)

        case Task.VQA:
            if not target:
                raise ValueError(
                    "task=vqa requires a non-empty target (the question to ask)"
                )
            return target
