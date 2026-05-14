"""Parse the Perceptron API's raw output into FiftyOne label objects.

The API emits an XML-ish tag grammar for grounded outputs (``<point>``,
``<point_box>``, ``<clip>``) and raw JSON for classification.
`to_fiftyone(content, annotation_format, ...)` is the single dispatcher.

Three concerns live here:

    1. Vendored SDK -> FiftyOne converters (`BoundingBox` -> `fo.Detection`,
       `SinglePoint` -> `fo.Keypoint`). Vendored rather than imported so the
       plugin is self-contained when installed standalone.
    2. Tag parsing with wrapper-mention propagation and `instance_id`
       extraction for `<track>` wrappers (multi-object tracking IDs).
    3. JSON parsing for classification tasks.

Conventions:
    * Coordinates are 0-1000 normalized; divide by `COORD_MAX` to get
      FiftyOne's [0, 1] relative coords.
    * The Perceptron SDK's `extract_points()` drops the ``t=`` attribute from
      `<point>` / `<point_box>` tags. We recover timestamps via regex on the
      raw text and re-attach positionally.
    * `<clip>` tags are NOT supported by `extract_points`; we parse them with
      our own regex (with a prose fallback for ambiguous events).
    * Polygon output: the API emits ``<polygon mention="LABEL"> (x,y) ... </polygon>``
      with true polygon coordinates in the 0-1000 space. A ``<collection>`` wrapper
      may group multiple polygon tags under one class label. See `_parse_polygons`
      and the ``_POLYGON_*`` regex constants.

"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, TypeAlias

import fiftyone as fo
from perceptron import BoundingBox, SinglePoint, extract_points, pt, strip_tags

logger = logging.getLogger("perceptron")


# Perceptron's normalized coordinate ceiling. Divide by this to convert to
# FiftyOne's [0, 1] relative coords.
COORD_MAX: float = 1000.0


# Return shape of `to_fiftyone`. `TypeAlias` (rather than PEP 695 ``type``)
# because the project targets Python 3.11.
FOLabel: TypeAlias = (
    fo.Detections
    | fo.Keypoints
    | fo.Polylines
    | fo.TemporalDetections
    | fo.Classification
    | fo.Classifications
    | str
    | None
)


# ---------------------------------------------------------------------------
# Vendored SDK-object -> FiftyOne converters (vendored so the plugin works
# standalone, without the `prev_integration/` research directory on sys.path).
#
# Three small helpers are shared across converters:
#   _box_coords       -- normalise box corners from [0,1000] to [0,1]
#   _mot_kwargs       -- build optional MOT constructor kwargs
#   _attach_tracking_id -- set the raw mention as a post-construction attr
# ---------------------------------------------------------------------------


def _box_coords(box: BoundingBox) -> tuple[float, float, float, float]:
    """Return ``(x1, y1, x2, y2)`` for ``box``, normalised to [0, 1] space.

    Perceptron uses a 0-1000 coordinate space; FiftyOne expects [0, 1]
    relative coords. Centralises the divide so callers don't repeat it.
    """
    return (
        box.top_left.x / COORD_MAX,
        box.top_left.y / COORD_MAX,
        box.bottom_right.x / COORD_MAX,
        box.bottom_right.y / COORD_MAX,
    )


def _mot_kwargs(
    instance: fo.Instance | None,
    index: int | None,
) -> dict[str, Any]:
    """Build optional MOT constructor kwargs for ``fo.Detection`` / ``fo.Keypoint``.

    ``instance`` and ``index`` must go into the FiftyOne constructor; they
    can't be set as attributes afterwards. ``tracking_id`` is handled
    separately by `_attach_tracking_id` because it's a custom attribute, not
    a reserved field. Returns an empty dict when both are ``None``.
    """
    kwargs: dict[str, Any] = {}
    if instance is not None:
        kwargs["instance"] = instance
    if index is not None:
        kwargs["index"] = index
    return kwargs


def _attach_tracking_id(label_obj: Any, tracking_id: str | None) -> None:
    """Store the raw ``<track>`` mention on ``label_obj`` as a custom attribute.

    ``tracking_id`` is the model's raw mention string (e.g. ``"vehicle_1"``),
    kept for debugging. Unlike ``instance_id`` it is not a reserved field, so
    it can be set freely after construction.
    """
    if tracking_id is not None:
        label_obj["tracking_id"] = tracking_id


def _box_to_detection(
    box: BoundingBox,
    *,
    instance: fo.Instance | None = None,
    index: int | None = None,
    tracking_id: str | None = None,
) -> fo.Detection:
    """Convert a Perceptron `BoundingBox` to an `fo.Detection` in [0, 1] coords.

    MOT fields: ``instance`` and ``index`` go into the constructor via
    `_mot_kwargs`; ``tracking_id`` is attached post-construction via
    `_attach_tracking_id`. See those helpers for details.
    """
    x1, y1, x2, y2 = _box_coords(box)
    detection = fo.Detection(
        label=box.mention or "object",
        bounding_box=[x1, y1, x2 - x1, y2 - y1],
        **_mot_kwargs(instance, index),
    )
    _attach_tracking_id(detection, tracking_id)
    return detection


def _point_to_keypoint(
    point: SinglePoint,
    *,
    instance: fo.Instance | None = None,
    index: int | None = None,
    tracking_id: str | None = None,
) -> fo.Keypoint:
    """Convert a Perceptron `SinglePoint` to an `fo.Keypoint`.

    Same MOT plumbing as `_box_to_detection` — see `_mot_kwargs` and
    `_attach_tracking_id` for details.
    """
    kp = fo.Keypoint(
        label=point.mention or "point",
        points=[(point.x / COORD_MAX, point.y / COORD_MAX)],
        **_mot_kwargs(instance, index),
    )
    _attach_tracking_id(kp, tracking_id)
    return kp


def _box_center_point(box: BoundingBox) -> SinglePoint:
    """Center of `box` as a `SinglePoint`, preserving its mention.

    Used as a fallback when the model returns boxes for a ``point``-format
    request -- we treat them as points by taking the center.
    """
    cx = (box.top_left.x + box.bottom_right.x) // 2
    cy = (box.top_left.y + box.bottom_right.y) // 2
    return pt(cx, cy, mention=box.mention)


# ---------------------------------------------------------------------------
# Wrapper tag handling. The API uses three wrapper variants:
#   * <collection mention="LABEL">  -- class detection; mention is the class
#   * <track mention="ID">          -- MOT; mention is a tracking id, not a class
#   * (no wrapper)                  -- bare tag form
# `extract_points()` doesn't expose wrapper info, so we re-scan the raw text.
# ---------------------------------------------------------------------------


# Only the first wrapper is considered authoritative; multi-wrapper output is
# rare and we don't try to be clever about it in v1.
_WRAPPER_PATTERN = re.compile(
    r'<(?P<kind>collection|track)\s+[^>]*?\bmention\s*=\s*"(?P<value>[^"]*)"',
    re.IGNORECASE,
)


# Trailing integer in a tracking-id mention (e.g. ``vehicle_1`` -> 1) used as
# the Detection/Keypoint ``index`` for human-friendly track numbering.
_TRAILING_INT_RE = re.compile(r"(\d+)\s*$")


def _detect_wrapper(content: str) -> tuple[str, str] | None:
    """``(kind, value)`` of the outer wrapper, or ``None`` if bare."""
    m = _WRAPPER_PATTERN.search(content)
    if m is None:
        return None
    return m.group("kind").lower(), m.group("value")


# ---------------------------------------------------------------------------
# Polygon tag parsing. The API emits ``<polygon mention="LABEL"> (x,y) ... </polygon>``
# with coordinates as space-separated integer pairs in the 0-1000 space.
# A ``<collection mention="LABEL">`` wrapper may group multiple polygon tags.
# ---------------------------------------------------------------------------

# Matches a full <polygon ...> ... </polygon> block; captures attrs and body separately.
_POLYGON_TAG_RE = re.compile(
    r"<polygon\b(?P<attrs>[^>]*)>(?P<body>.*?)</polygon>",
    re.DOTALL | re.IGNORECASE,
)

# Extracts mention="VALUE" from a polygon tag's attribute string.
_POLYGON_MENTION_RE = re.compile(r'\bmention\s*=\s*"([^"]*)"', re.IGNORECASE)

# Extracts integer (x,y) coordinate pairs from a polygon tag's body.
_POLYGON_COORD_RE = re.compile(r"\((\d+),\s*(\d+)\)")


# ---------------------------------------------------------------------------
# Per-tag timestamp recovery. `extract_points()` returns objects with
# ``t=None`` even when the source tag had ``t="X seconds"``. We re-scan the
# raw text and pair timestamps to parsed objects positionally.
# ---------------------------------------------------------------------------

_TIME_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _extract_per_tag_timestamps(content: str, tag_name: str) -> list[float | None]:
    """Recover ``t=`` floats from each ``<tag_name ...>`` occurrence in order.

    One entry per tag occurrence (in source order); ``None`` when the tag
    had no ``t=`` attribute. Lets callers ``zip()`` against the SDK's parsed
    objects safely.
    """
    tag_with_optional_t = re.compile(
        rf"<{re.escape(tag_name)}\b(?P<attrs>[^>]*)/?>",
        re.DOTALL,
    )
    timestamps: list[float | None] = []
    for m in tag_with_optional_t.finditer(content):
        attrs = m.group("attrs")
        t_match = re.search(r'\bt\s*=\s*"([^"]*)"', attrs)
        if t_match is None:
            timestamps.append(None)
            continue
        nums = _TIME_NUM_RE.findall(t_match.group(1))
        timestamps.append(float(nums[0]) if nums else None)
    return timestamps


def _attach_timestamps(fo_items: list[Any], timestamps: list[float | None]) -> None:
    """Stamp each FiftyOne label with its raw ``t`` (seconds) custom attribute.

    Called after converting SDK objects to FiftyOne labels. Items whose
    corresponding timestamp is ``None`` are skipped. Uses plain ``zip`` so
    any trailing entries (from tags the SDK silently dropped) are ignored —
    the same invariant the callers rely on when building these lists.
    """
    for item, t in zip(fo_items, timestamps):  # noqa: B905 -- trailing entries intentionally dropped
        if t is not None:
            item["t"] = t


# ---------------------------------------------------------------------------
# Clip tag parsing. The API emits ``<clip mention="X" t="START_S END_S" />``
# for events and sometimes falls back to prose like "between 2.0 and 3.0
# seconds". `extract_points` doesn't support clip, so we roll our own.
# ---------------------------------------------------------------------------


_CLIP_TAG_RE = re.compile(
    r'<clip\b[^>]*?mention\s*=\s*"([^"]*)"[^>]*?t\s*=\s*"([^"]*)"[^>]*/?>',
    re.DOTALL,
)
_PROSE_CLIP_RE = re.compile(
    r"between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s+seconds",
    re.IGNORECASE,
)


def _seconds_to_frame(seconds: float, frame_rate: float | None) -> int | None:
    """Convert seconds to a 1-indexed FiftyOne frame number, or ``None``."""
    if frame_rate is None or frame_rate <= 0:
        return None
    # FiftyOne frames are 1-indexed; clamp to 1 so t=0 maps to frame 1.
    return max(1, int(round(seconds * frame_rate)))


def _parse_clip_tags(content: str, *, frame_rate: float | None) -> fo.TemporalDetections:
    """Parse ``<clip ... />`` tags (plus prose fallback) into `TemporalDetections`.

    Each clip carries raw ``t_start_seconds`` / ``t_end_seconds`` attributes
    plus a ``support=[start_frame, end_frame]`` when ``frame_rate`` is known.
    Without a frame rate, ``support`` falls back to the placeholder ``[1, 1]``
    (and we tag the detection ``support_is_placeholder=True``).
    """
    detections: list[fo.TemporalDetection] = []
    saw_tag = False

    for mention, t_str in _CLIP_TAG_RE.findall(content):
        saw_tag = True
        nums = _TIME_NUM_RE.findall(t_str)
        if not nums:
            logger.warning(
                "[perceptron] clip tag with unparseable t=%r; skipping", t_str
            )
            continue
        t_start = float(nums[0])
        t_end = float(nums[1]) if len(nums) >= 2 else t_start
        detections.append(
            _build_temporal_detection(
                label=mention or "event",
                t_start=t_start,
                t_end=t_end,
                frame_rate=frame_rate,
                source="tag",
            )
        )

    if not saw_tag:
        # Prose fallback: model didn't emit a <clip /> tag, but the prose
        # may still describe a time range. We recover one prose interval
        # per response if found -- multi-interval prose is uncommon and we
        # don't try to be clever.
        m = _PROSE_CLIP_RE.search(content)
        if m is not None:
            t_start = float(m.group(1))
            t_end = float(m.group(2))
            logger.info(
                "[perceptron] no <clip /> tag found; recovered prose interval "
                "(%s, %s) seconds",
                t_start,
                t_end,
            )
            detections.append(
                _build_temporal_detection(
                    label="event",
                    t_start=t_start,
                    t_end=t_end,
                    frame_rate=frame_rate,
                    source="prose",
                )
            )

    return fo.TemporalDetections(detections=detections)


# FiftyOne `TemporalDetection` requires a non-null ``support=[first, last]``.
# When ``frame_rate`` is unknown we fall back to this placeholder; the raw
# seconds live on ``t_start_seconds`` / ``t_end_seconds`` attributes so no
# information is lost.
_PLACEHOLDER_SUPPORT: list[int] = [1, 1]


def _build_temporal_detection(
    *,
    label: str,
    t_start: float,
    t_end: float,
    frame_rate: float | None,
    source: str,
) -> fo.TemporalDetection:
    """Build a `TemporalDetection` with raw seconds + frame-index ``support``.

    Falls back to ``_PLACEHOLDER_SUPPORT`` when frame_rate is missing and
    tags the detection ``support_is_placeholder=True``.
    """
    f_start = _seconds_to_frame(t_start, frame_rate)
    f_end = _seconds_to_frame(t_end, frame_rate)

    if f_start is None or f_end is None:
        logger.warning(
            "[perceptron] frame_rate unavailable; TemporalDetection for %r will "
            "use placeholder support=%s. Seconds-precise times are kept on "
            "the t_start_seconds / t_end_seconds attributes.",
            label,
            _PLACEHOLDER_SUPPORT,
        )
        det = fo.TemporalDetection(label=label, support=list(_PLACEHOLDER_SUPPORT))
        det["support_is_placeholder"] = True
    else:
        # Clamp out-of-order ranges -- the model has been observed emitting
        # t="end start" rarely. Support must always be [smaller, bigger].
        if f_end < f_start:
            f_end = f_start
        det = fo.TemporalDetection(label=label, support=[f_start, f_end])

    det["t_start_seconds"] = t_start
    det["t_end_seconds"] = t_end
    det["source"] = source
    return det


# ---------------------------------------------------------------------------
# Classification JSON parsing. Tasks now run with native
# ``response_format=json_schema`` (CLASSIFY_SINGLE / CLASSIFY_MULTI in the
# model class), so the API returns raw JSON. The fenced-block extraction
# fallback machinery has been removed -- `json.loads(content)` is sufficient.
# ---------------------------------------------------------------------------


def _classifications_from_json(parsed: Any, *, multi: bool) -> fo.Classification | fo.Classifications | None:
    """Build FiftyOne classifications from a parsed JSON value.

    Accepts the shapes our system prompts request:
        ``{"classifications": [{"label": ..., "confidence": ...}, ...]}``,
        ``[{"label": ...}, ...]``, or ``{"label": ...}``.

    Returns `Classification` when ``multi=False`` (keeping the first if more
    were returned, with an INFO log), `Classifications` when ``multi=True``,
    or ``None`` if nothing parses.
    """
    items: list[dict[str, Any]] = []
    match parsed:
        case dict() as d if isinstance(d.get("classifications"), list):
            items = [c for c in d["classifications"] if isinstance(c, dict)]
        case dict() as d if "label" in d:
            items = [d]
        case list() as lst:
            items = [c for c in lst if isinstance(c, dict)]
        case _:
            logger.info(
                "[perceptron] classifications JSON had unexpected shape: %s",
                type(parsed).__name__,
            )
            return None

    classifications: list[fo.Classification] = []
    for item in items:
        label = item.get("label")
        if label is None:
            logger.info("[perceptron] classification item without 'label' skipped: %r", item)
            continue
        kwargs: dict[str, Any] = {"label": str(label)}
        confidence = item.get("confidence")
        if isinstance(confidence, (int, float)):
            kwargs["confidence"] = float(confidence)
        classifications.append(fo.Classification(**kwargs))

    if not classifications:
        return None

    if not multi:
        if len(classifications) > 1:
            logger.info(
                "[perceptron] classify_single got %d items; keeping first only",
                len(classifications),
            )
        return classifications[0]
    return fo.Classifications(classifications=classifications)


# ---------------------------------------------------------------------------
# Top-level dispatch.
# ---------------------------------------------------------------------------


def to_fiftyone(
    content: str,
    annotation_format: str,
    *,
    target: str | None = None,
    frame_rate: float | None = None,
) -> FOLabel:
    """Convert the API's raw response text to a FiftyOne label container.

    ``annotation_format`` values and their output types:
        * ``"box"``              -> `fo.Detections`
        * ``"point"``            -> `fo.Keypoints`
        * ``"polygon"``          -> `fo.Polylines`
        * ``"clip"``             -> `fo.TemporalDetections`
        * ``"vqa"`` | ``"caption"`` | ``"ocr_text"`` -> ``str`` (tags stripped)
        * ``"classify_single"``  -> `fo.Classification`
        * ``"classify_multi"``   -> `fo.Classifications`

    Args:
        content: Raw ``message.content`` string from the API.
        annotation_format: One of the values listed above.
        target: Class-label fallback when the API emits a
            ``<track mention="ID">`` wrapper (the wrapper's mention is a
            tracking ID, not a class name).
        frame_rate: Sample's video frame rate. Used for ``support`` mapping
            on clip tasks; optional otherwise.

    Returns:
        A FiftyOne label container, plain string, or ``None`` if the model
        produced content but no recognizable structure (logged INFO).
    """
    preview = content.replace("\n", " ")
    if len(preview) > 200:
        preview = preview[:200] + "..."
    logger.info(
        "[perceptron] to_fiftyone: format=%s len=%d preview=%r",
        annotation_format,
        len(content),
        preview,
    )

    if not content:
        # Empty content is a legitimate "no match" result; return the appropriate
        # empty container so callers don't have to handle None for grounding tasks.
        logger.info("[perceptron] empty content; returning empty container for format=%s", annotation_format)
        return _empty_container_for(annotation_format)

    match annotation_format:
        case "box":
            return _parse_boxes(content, target=target)
        case "point":
            return _parse_points(content, target=target)
        case "polygon":
            return _parse_polygons(content, target=target)
        case "clip":
            return _parse_clip_tags(content, frame_rate=frame_rate)
        case "vqa" | "caption" | "ocr_text":
            cleaned = strip_tags(content).strip()
            logger.info("[perceptron] free-text result: %d chars", len(cleaned))
            return cleaned
        case "classify_single":
            return _classifications_from_json_or_empty(content, multi=False)
        case "classify_multi":
            return _classifications_from_json_or_empty(content, multi=True)
        case _:
            raise ValueError(f"unsupported annotation_format: {annotation_format!r}")


def _empty_container_for(annotation_format: str) -> FOLabel:
    """Return the empty FiftyOne container matching ``annotation_format``."""
    match annotation_format:
        case "box":
            return fo.Detections(detections=[])
        case "point":
            return fo.Keypoints(keypoints=[])
        case "polygon":
            return fo.Polylines(polylines=[])
        case "clip":
            return fo.TemporalDetections(detections=[])
        case "vqa" | "caption" | "ocr_text":
            return ""
        case "classify_single" | "classify_multi":
            return None
        case _:
            raise ValueError(f"unsupported annotation_format: {annotation_format!r}")


def _parse_boxes(content: str, *, target: str | None) -> fo.Detections:
    """Parse ``<point_box>`` tags into `fo.Detections`, handling wrappers / MOT.

    Wrapper variants:
        * ``<collection mention="LABEL">`` -- LABEL propagates as the class
          fallback for unmentioned tags. No tracking plumbing.
        * ``<track mention="ID">`` -- ID is a tracking instance, not a class.
          We create a shared `fo.Instance`, parse a trailing integer as the
          ``index``, store the raw ID as ``tracking_id``, and fall back to
          ``target`` for the class label.
        * No wrapper -- per-tag mentions are authoritative (or ``"object"``).
    """
    boxes = extract_points(content, expected="box")
    timestamps = _extract_per_tag_timestamps(content, "point_box")
    wrapper = _detect_wrapper(content)
    ctx = _resolve_wrapper_context(wrapper, target=target)

    detections: list[fo.Detection] = []
    for box in boxes:
        if not box.mention and ctx.label_fallback:
            box.mention = ctx.label_fallback
        detections.append(
            _box_to_detection(box, instance=ctx.instance, index=ctx.index, tracking_id=ctx.tracking_id)
        )
    # Stamp each detection with its raw t= seconds so write_per_frame_labels
    # can route it to the correct sample.frames[i].
    _attach_timestamps(detections, timestamps)

    logger.info(
        "[perceptron] parsed %d box(es); wrapper=%s tracking_id=%s index=%s",
        len(detections),
        wrapper,
        ctx.tracking_id,
        ctx.index,
    )
    return fo.Detections(detections=detections)


def _parse_points(content: str, *, target: str | None) -> fo.Keypoints:
    """Parse ``<point>`` tags into `fo.Keypoints`, with box-center fallback.

    Same wrapper handling as `_parse_boxes`. When the model returns boxes
    instead of points (ambiguous-prompt edge case), we use box centers.
    """
    points = extract_points(content, expected="point")
    wrapper = _detect_wrapper(content)
    ctx = _resolve_wrapper_context(wrapper, target=target)

    if not points:
        # Known edge case: model returned boxes when we asked for points.
        boxes = extract_points(content, expected="box")
        if boxes:
            logger.info(
                "[perceptron] point format requested but %d <point_box> tag(s) "
                "found; converting to box centers",
                len(boxes),
            )
            point_ts = _extract_per_tag_timestamps(content, "point_box")
            for b in boxes:
                if not b.mention and ctx.label_fallback:
                    b.mention = ctx.label_fallback
            points = [_box_center_point(b) for b in boxes]
        else:
            logger.info("[perceptron] no <point> or <point_box> tags found in content")
            point_ts = []
    else:
        point_ts = _extract_per_tag_timestamps(content, "point")
        for p in points:
            if not p.mention and ctx.label_fallback:
                p.mention = ctx.label_fallback

    keypoints: list[fo.Keypoint] = []
    for p in points:
        keypoints.append(
            _point_to_keypoint(p, instance=ctx.instance, index=ctx.index, tracking_id=ctx.tracking_id)
        )
    _attach_timestamps(keypoints, point_ts)

    logger.info(
        "[perceptron] parsed %d keypoint(s); wrapper=%s tracking_id=%s index=%s",
        len(keypoints),
        wrapper,
        ctx.tracking_id,
        ctx.index,
    )
    return fo.Keypoints(keypoints=keypoints)


def _parse_polygons(content: str, *, target: str | None) -> fo.Polylines:
    """Parse ``<polygon>`` tags into `fo.Polylines`.

    The API emits two variants:

    1. Bare polygon with inline mention::

           <polygon mention="bull"> (325,1000) (330,819) ... </polygon>

    2. Collection wrapper where the mention labels the whole group::

           <collection mention="person">
             <polygon> (702,735) ... </polygon>
             <polygon> (939,738) ... </polygon>
           </collection>

    Coordinates are space-separated ``(x,y)`` pairs in the 0-1000 space.
    Each polygon becomes a closed, filled `fo.Polyline` in [0,1] relative coords.
    """
    wrapper = _detect_wrapper(content)
    ctx = _resolve_wrapper_context(wrapper, target=target)

    polylines: list[fo.Polyline] = []
    for m in _POLYGON_TAG_RE.finditer(content):
        attrs_str = m.group("attrs")
        body = m.group("body")

        # Prefer mention on the tag itself; fall back to wrapper label / target.
        mention_m = _POLYGON_MENTION_RE.search(attrs_str)
        label = (mention_m.group(1) if mention_m else None) or ctx.label_fallback or target or "object"

        # Extract (x,y) integer pairs from the tag body and normalise to [0,1].
        raw_pts = _POLYGON_COORD_RE.findall(body)
        if not raw_pts:
            logger.warning("[perceptron] polygon tag with no coordinate pairs; skipping")
            continue

        points = [(int(x) / COORD_MAX, int(y) / COORD_MAX) for x, y in raw_pts]
        # points is a list-of-lists: one sub-list per disconnected shape.
        polylines.append(fo.Polyline(label=label, points=[points], closed=True, filled=True))

    logger.info("[perceptron] parsed %d polygon(s); wrapper=%s", len(polylines), wrapper)
    return fo.Polylines(polylines=polylines)


@dataclass(slots=True, kw_only=True, frozen=True)
class _WrapperContext:
    """Plumbing derived from a ``<collection>`` or ``<track>`` wrapper.

    Attributes:
        label_fallback: Class label for child tags missing ``mention=``.
            Comes from ``<collection mention=...>``; for ``<track>`` it falls
            back to the caller's ``target`` (since the wrapper mention is an
            instance ID, not a class name).
        instance: Shared `fo.Instance` for ``<track>`` (drives FiftyOne MOT);
            ``None`` otherwise.
        index: Integer track index parsed from a trailing number in a
            ``<track>`` mention (``"vehicle_1"`` -> 1); ``None`` otherwise.
        tracking_id: Raw ``<track>`` mention string, stored on each label
            for debugging.
    """

    label_fallback: str | None
    instance: fo.Instance | None
    index: int | None
    tracking_id: str | None


def _resolve_wrapper_context(
    wrapper: tuple[str, str] | None, *, target: str | None
) -> _WrapperContext:
    """Compute the per-call wrapper plumbing from `_detect_wrapper`'s output."""
    if wrapper is None:
        return _WrapperContext(label_fallback=None, instance=None, index=None, tracking_id=None)

    kind, value = wrapper
    match kind:
        case "collection":
            return _WrapperContext(
                label_fallback=value,
                instance=None,
                index=None,
                tracking_id=None,
            )
        case "track":
            # Parse trailing integer for track numbering: vehicle_1 -> 1.
            m = _TRAILING_INT_RE.search(value)
            track_index = int(m.group(1)) if m is not None else None
            return _WrapperContext(
                label_fallback=target or None,
                instance=fo.Instance(),
                index=track_index,
                tracking_id=value,
            )
        case _:
            # Unreachable given the wrapper regex.
            raise ValueError(f"unknown wrapper kind {kind!r}")


def _classifications_from_json_or_empty(
    content: str, *, multi: bool
) -> fo.Classification | fo.Classifications | None:
    """Parse raw JSON into FiftyOne classifications, with informative logging.

    Assumes ``content`` is raw JSON (the model uses
    ``response_format=json_schema``). Returns ``None`` on a JSON parse error
    -- a legitimate "model returned content but it wasn't valid JSON" state.
    """
    cleaned = content.strip()
    if not cleaned:
        logger.info("[perceptron] empty classification content; returning None")
        return None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.info(
            "[perceptron] classification content was not valid JSON (%s); returning None",
            exc,
        )
        return None
    result = _classifications_from_json(parsed, multi=multi)
    if result is None:
        logger.info("[perceptron] JSON parsed but no usable classifications extracted")
    else:
        kind = "Classifications" if multi else "Classification"
        logger.info("[perceptron] built %s from JSON", kind)
    return result


# ---------------------------------------------------------------------------
# Per-frame writeback for video samples.
# ---------------------------------------------------------------------------


def write_per_frame_labels(
    sample: fo.Sample,
    label_obj: FOLabel,
    field: str,
    *,
    frame_rate: float | None,
) -> dict[str, int]:
    """Route a label container to ``sample.frames[i][field]`` or ``sample[field]``.

    Routing rules:

    * **Video + Detections/Keypoints**: items MUST go on frames -- the App's
      video player renders spatial labels off frames; a sample-level
      ``Detections`` field on a video is a different schema and hidden from
      the timeline. Items need both a ``t`` (seconds) attribute and a usable
      ``frame_rate``; otherwise they're dropped with a warning rather than
      silently saved at sample level where the user can't see them.
    * **Video + non-spatial** (Classification(s), str, TemporalDetections):
      these describe the whole clip, so they go on ``sample[field]``.
    * **Image samples**: always ``sample[field]``.

    Returns a summary dict::

        {
          "per_frame_count": items written across frames,
          "sample_level":    items written at sample level,
          "dropped":         items dropped (no `t=` or missing `frame_rate`),
          "frames_written":  distinct frame indices touched,
        }
    """
    if label_obj is None:
        logger.info("[perceptron] write_per_frame_labels: label_obj is None; nothing to write")
        return _writeback_summary()

    is_video = getattr(sample, "media_type", None) == "video"

    # Non-spatial labels are always sample-level (image or video). FiftyOne
    # supports sample-level Classification / Classifications / str /
    # TemporalDetections on video samples natively.
    if not isinstance(label_obj, (fo.Detections, fo.Keypoints)):
        sample[field] = label_obj
        sample.save()
        n_items = _count_sample_level_items(label_obj)
        logger.info(
            "[perceptron] wrote sample-level %s (%d item(s)) -> sample[%s]",
            type(label_obj).__name__,
            n_items,
            field,
        )
        return _writeback_summary(sample_level=n_items)

    # From here on we're handling Detections or Keypoints.
    items = (
        label_obj.detections
        if isinstance(label_obj, fo.Detections)
        else label_obj.keypoints
    )

    if not items:
        logger.info(
            "[perceptron] %s container is empty; nothing to write for sample[%s]",
            type(label_obj).__name__,
            field,
        )
        return _writeback_summary()

    # Image case: always sample-level.
    if not is_video:
        sample[field] = label_obj
        sample.save()
        logger.info(
            "[perceptron] image sample: wrote sample-level %s (%d item(s)) -> sample[%s]",
            type(label_obj).__name__,
            len(items),
            field,
        )
        return _writeback_summary(sample_level=len(items))

    # Video case: spatial labels MUST live on frames.
    if frame_rate is None or frame_rate <= 0:
        logger.error(
            "[perceptron] video sample %s has frame_rate=%r; %d %s label(s) cannot "
            "be placed on a frame and will be dropped. Run "
            "`sample.compute_metadata()` (or fix the file) and re-run to keep "
            "these labels.",
            sample.filepath,
            frame_rate,
            len(items),
            type(label_obj).__name__,
        )
        return _writeback_summary(dropped=len(items))

    timed = [it for it in items if getattr(it, "t", None) is not None]
    n_untimed = len(items) - len(timed)
    if n_untimed:
        logger.warning(
            "[perceptron] video sample %s: %d/%d %s label(s) had no `t=` timestamp "
            "from the model and will be dropped (only timestamped labels can "
            "be placed on a frame).",
            sample.filepath,
            n_untimed,
            len(items),
            type(label_obj).__name__,
        )

    if not timed:
        # Every item missing `t=` -- nothing left to place. Already logged above.
        return _writeback_summary(dropped=len(items))

    written = _write_items_to_frames(sample, timed, field, frame_rate=frame_rate)
    written["dropped"] += n_untimed
    return written


def _writeback_summary(
    *,
    per_frame_count: int = 0,
    sample_level: int = 0,
    dropped: int = 0,
    frames_written: int = 0,
) -> dict[str, int]:
    """Return-value factory so every code path uses the same shape."""
    return {
        "per_frame_count": per_frame_count,
        "sample_level": sample_level,
        "dropped": dropped,
        "frames_written": frames_written,
    }


def _count_sample_level_items(label_obj: Any) -> int:
    """Best-effort item count for sample-level label objects."""
    match label_obj:
        case fo.Classifications():
            return len(label_obj.classifications)
        case fo.TemporalDetections():
            return len(label_obj.detections)
        case fo.Polylines():
            return len(label_obj.polylines)
        case _:
            # fo.Classification, str, or any other single-item container.
            return 1


def _write_items_to_frames(
    sample: fo.Sample,
    items: list[Any],
    field: str,
    *,
    frame_rate: float,
) -> dict[str, int]:
    """Group timed `items` by frame index and write to `sample.frames[i][field]`.

    Args:
        sample: Video sample. Caller has verified `media_type == "video"`.
        items: Label items, every one with a `t` attribute in seconds.
        field: Target field name on each frame.
        frame_rate: For seconds -> frame index conversion. Caller has already
            verified this is positive.

    Returns:
        Summary dict (see `write_per_frame_labels`).
    """
    by_frame: dict[int, list[Any]] = {}
    for item in items:
        idx = _seconds_to_frame(item.t, frame_rate)
        # We checked frame_rate above; _seconds_to_frame can't return None here,
        # but assert it explicitly so a refactor doesn't silently break.
        assert idx is not None  # noqa: S101 -- internal invariant after frame_rate guard
        by_frame.setdefault(idx, []).append(item)

    # Pick the right container constructor per item type. We know items are
    # uniform because they came from the same Detections / Keypoints container.
    # TRACK is the only dense task that reaches this path and always produces
    # fo.Detections, but keep the TypeError explicit so a future refactor
    # can't silently produce wrong output.
    example = items[0]
    match example:
        case fo.Detection():
            container_cls = fo.Detections
            container_kwarg = "detections"
        case _:
            raise TypeError(
                f"_write_items_to_frames: unexpected item type "
                f"{type(example).__name__}; expected fo.Detection"
            )

    for frame_idx, batch in sorted(by_frame.items()):
        frame_key = max(1, frame_idx)  # FiftyOne uses 1-indexed frames
        frame = sample.frames[frame_key]
        frame[field] = container_cls(**{container_kwarg: batch})

    sample.save()
    logger.info(
        "[perceptron] wrote per-frame %s to sample[%s]: %d label(s) across %d frame(s)",
        type(example).__name__ + "s",
        field,
        len(items),
        len(by_frame),
    )
    return _writeback_summary(
        per_frame_count=len(items),
        frames_written=len(by_frame),
    )
