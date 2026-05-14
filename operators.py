"""FiftyOne operator for the Perceptron plugin.

One operator: `RunPerceptron`. Accessible from the operator browser
(backtick) or the "Perceptron" grid-action button. The form is *conditional*
-- the mode you pick at the top drives which fields below and what
`execute` does.
(https://docs.voxel51.com/plugins/developing_plugins.html#conditional-inputs)

The available modes depend on ``dataset.media_type``:

    Image datasets:
        Semantic Search  -- score each sample yes/no against a free-text query.
        Bootstrap Labels -- single-shot grounding (detect / keypoints / polygon)
                            plus caption, classify, and VQA.

    Video datasets:
        Event Search     -- find targeted moments; writes fo.TemporalDetections
                            and switches the App into a clips view of matches.
        Semantic Search  -- same as above.
        Bootstrap Labels -- dense-frame detection (TRACK), unconstrained clip
                            extraction (KEY_MOMENTS), plus caption, classify, VQA.

Mixed-media datasets are not supported and surface an error in the form.

Failures bubble up untouched.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import re
import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone import ViewField as F
from fiftyone.operators import types

from ._shared import (
    PLUGIN_VERSION,
    get_api_key,
    has_api_key,
    make_run_key,
    notify,
    task_supports_per_frame,
)
from .perceptron_model import DEFAULT_MODEL_NAME, PerceptronModel
from .perceptron_parser import write_per_frame_labels
from .prompts import Task, default_user_prompt

logger = logging.getLogger("perceptron")


# ---------------------------------------------------------------------------
# Mode + task taxonomy. Strings are user-facing AND ctx.params values -- if
# you rename one, update the dispatch tables in execute().
# ---------------------------------------------------------------------------

MODE_EVENT_SEARCH: str = "event_search"
MODE_SEMANTIC_SEARCH: str = "semantic_search"
MODE_BOOTSTRAP: str = "bootstrap"

MODE_LABELS: dict[str, str] = {
    MODE_EVENT_SEARCH: "Event Search",
    MODE_SEMANTIC_SEARCH: "Semantic Search",
    MODE_BOOTSTRAP: "Bootstrap Labels",
}

# One-liner shown next to each mode choice in the dropdown.
MODE_CHOICE_DESCRIPTIONS: dict[str, str] = {
    MODE_EVENT_SEARCH: "Find moments matching a free-text event description. (Video only)",
    MODE_SEMANTIC_SEARCH: "Score each sample yes/no against a free-text query.",
    MODE_BOOTSTRAP: "Bootstrap annotation labels across the target view.",
}

# Longer markdown rendered below the mode selector once a mode is picked.
MODE_LONG_DESCRIPTIONS: dict[str, str] = {
    MODE_EVENT_SEARCH: (
        "**Event Search**  \n"
        "Scans each video for the event you describe. For every match it writes a `fo.TemporalDetection` "
        "(with `support=[start_frame, end_frame]` plus raw `t_start_seconds` / `t_end_seconds` attributes) "
        "onto the configured output field.  \n\n"
        "After the run, the view switches to a clips view with one row per detected event."
    ),
    MODE_SEMANTIC_SEARCH: (
        "**Semantic Search**  \n"
        "Scores each sample as a yes/no answer with a confidence score, written as "
        "`fo.Classification(label, confidence)` on the sample.  \n"
        "After the run, the view filters to samples where `label == \"yes\"` and `confidence >= threshold`."
    ),
    MODE_BOOTSTRAP: (
        "**Bootstrap Labels**  \n"
        "Pick a task below to write detections, keypoints, polygons, captions, classifications, or VQA "
        "answers across the target view. Available tasks and output placement (per-frame vs. sample-level) "
        "depend on the dataset's media type."
    ),
}


# Bootstrap task lists, split by media type. Display order = priority order.
_BOOTSTRAP_VIDEO_TASKS: list[Task] = [
    Task.TRACK,        # dense per-frame detection (with accuracy warning)
    Task.KEY_MOMENTS,  # unconstrained clip extraction
    Task.CLASSIFY_SINGLE,
    Task.CLASSIFY_MULTI,
    Task.VQA,
    Task.CAPTION_CONCISE,
    Task.CAPTION_DETAILED,
]

_BOOTSTRAP_IMAGE_TASKS: list[Task] = [
    Task.DETECT,       # single-shot box grounding
    Task.KEYPOINTS,    # single-shot point grounding
    Task.POLYGON,      # single-shot polygon grounding
    Task.CLASSIFY_SINGLE,
    Task.CLASSIFY_MULTI,
    Task.VQA,
    Task.CAPTION_CONCISE,
    Task.CAPTION_DETAILED,
]


# Short one-liners shown next to each task name in the dropdown.
_BOOTSTRAP_TASK_SHORT: dict[Task, str] = {
    # Image grounding
    Task.DETECT: "Detect and locate objects with bounding boxes. One API call per image.",
    Task.KEYPOINTS: "Point to objects by class. One API call per image.",
    Task.POLYGON: "Outline objects with polygon shapes. One API call per image.",
    # Video tasks
    Task.TRACK: "Per-frame object detection at a chosen stride. N API calls per video.",
    Task.KEY_MOMENTS: "Identify all key events in each video. One API call per video.",
    # Shared
    Task.CLASSIFY_SINGLE: "One-aspect classification with confidence.",
    Task.CLASSIFY_MULTI: "Multi-aspect classification with confidences.",
    Task.VQA: "Free-form Q&A; plain-text answer per sample.",
    Task.CAPTION_CONCISE: "One-sentence caption per sample.",
    Task.CAPTION_DETAILED: "Paragraph caption (picks up signage / fine detail).",
}


_BOOTSTRAP_TASK_LONG: dict[Task, str] = {
    Task.DETECT: (
        "**Detect**  \n"
        "Single-shot box grounding on images. Sends one `image_url` request per sample and writes "
        "`fo.Detections` at the sample level."
    ),
    Task.KEYPOINTS: (
        "**Keypoints**  \n"
        "Single-shot point grounding on images. Sends one `image_url` request per sample and writes "
        "`fo.Keypoints` at the sample level. When the model returns a box instead of a point "
        "(an occasional edge case), the parser substitutes the box center."
    ),
    Task.POLYGON: (
        "**Polygon**  \n"
        "Single-shot polygon grounding on images. Sends one `image_url` request per sample and writes "
        "`fo.Polylines` (closed, filled) at the sample level. Each detected object is outlined with "
        "a true polygon contour."
    ),
    Task.TRACK: (
        "**Track**  \n"
        "Per-frame object detection at the chosen stride. Decomposes each video into image-mode "
        "requests (one API call per sampled frame) and writes `fo.Detections` to "
        "`sample.frames[i][field]`. **No cross-frame instance IDs** -- run a downstream tracker "
        "(ByteTrack / IoU matching) for ID continuity. Cost scales linearly with frame count and "
        "inversely with stride.  \n\n"
        "**Note:** The model is not explicitly trained for video object tracking -- results may vary. "
        "For reliable track continuity, apply a downstream tracker to these per-frame detections."
    ),
    Task.KEY_MOMENTS: (
        "**Key Moments**  \n"
        "Unconstrained clip extraction: the model freely identifies every noteworthy event in each "
        "video and writes `fo.TemporalDetections` at the sample level. No target required.  \n\n"
        "For targeted event search (find *this specific thing*), use the **Event Search** mode instead."
    ),
    Task.CLASSIFY_SINGLE: (
        "**Classify (single label)**  \n"
        "Assign one classification label per sample with a confidence score. Uses the API's strict "
        "`json_schema` mode. Produces `fo.Classification(label, confidence)` at the sample level."
    ),
    Task.CLASSIFY_MULTI: (
        "**Classify (multi-label)**  \n"
        "Assign multiple classification labels per sample, each with a confidence. Strict `json_schema` "
        "mode. Produces `fo.Classifications` at the sample level."
    ),
    Task.VQA: (
        "**Visual Q&A**  \n"
        "Ask a free-text question about each sample. Produces a plain-text answer written to the "
        "output field as a string."
    ),
    Task.CAPTION_CONCISE: (
        "**Caption (concise)**  \n"
        "One-sentence caption per sample. Pairs well with FiftyOne's brain text-similarity methods "
        "after the run for free-text retrieval."
    ),
    Task.CAPTION_DETAILED: (
        "**Caption (detailed)**  \n"
        "Paragraph-length description per sample, picking up visible text (signage, license plates, "
        "etc.) and fine-grained context."
    ),
}


# Per-task spec for the conditional ``target`` input field.
# Tasks absent from this dict (captions, KEY_MOMENTS) have no target slot.
_TARGET_FIELD_SPEC: dict[Task, dict[str, Any]] = {
    Task.DETECT: {
        "label": "Target object",
        "description": "Object class to detect (e.g. 'car', 'person').",
        "required": True,
    },
    Task.KEYPOINTS: {
        "label": "Target object",
        "description": "Object to point at (e.g. 'pedestrian').",
        "required": True,
    },
    Task.POLYGON: {
        "label": "Target object",
        "description": "Object class to outline (e.g. 'vehicle').",
        "required": True,
    },
    Task.TRACK: {
        "label": "Target object",
        "description": "Object class to detect per frame (e.g. 'armored vehicle').",
        "required": True,
    },
    Task.CLASSIFY_SINGLE: {
        "label": "Aspect to classify (optional)",
        "description": "What to classify (e.g. 'scene type', 'weather').",
        "required": False,
    },
    Task.CLASSIFY_MULTI: {
        "label": "Aspects to list (optional)",
        "description": "Aspects to enumerate (e.g. 'scene, vehicles, lighting').",
        "required": False,
    },
    Task.VQA: {
        "label": "Question",
        "description": "Free-text question to ask about each sample.",
        "required": True,
    },
}


_BOOTSTRAP_DEFAULT_FIELDS: dict[Task, str] = {
    Task.DETECT: "perceptron_detections",
    Task.KEYPOINTS: "perceptron_keypoints",
    Task.POLYGON: "perceptron_polygons",
    Task.TRACK: "perceptron_detections",
    Task.KEY_MOMENTS: "perceptron_key_moments",
    Task.CLASSIFY_SINGLE: "perceptron_class",
    Task.CLASSIFY_MULTI: "perceptron_class",
    Task.VQA: "perceptron_answer",
    Task.CAPTION_CONCISE: "caption",
    Task.CAPTION_DETAILED: "caption",
}


# Semantic Search: strict json_schema constraining `label` to {"yes", "no"}.
# Uses native response_format=json_schema rather than prompt-side fenced JSON.
_SEMANTIC_SEARCH_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "SemanticMatch",
        "strict": True,
        "schema": {
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
                            "label": {"type": "string", "enum": ["yes", "no"]},
                            "confidence": {"type": "number"},
                        },
                    },
                },
            },
        },
    },
}

_SEMANTIC_SEARCH_TEMPLATE: str = "Does this video match this description: {query}"
_SEMANTIC_DEFAULT_THRESHOLD: float = 0.7

# Ticker flush interval while a single API call is in flight. 0.5s is fast
# enough to feel "live" without spamming the SSE stream.
_PROGRESS_TICK_INTERVAL_S: float = 0.5

# Sanitize free-text queries into safe field names.
_SANITIZE_FIELD_RE = re.compile(r"[^a-zA-Z0-9_]+")


# Markdown body for the "metadata missing" gate. The `.strip()` is load-
# bearing: MarkdownView renders block constructs (lists) as plain prose if
# the document starts with a blank line.
_METADATA_FIX_INSTRUCTIONS_MD: str = """\
**To fix this:**

- Press `Cancel` to close this operator.
- Press the backtick key (**`**) to open the operator browser.
- Search for and run the built-in **`compute_metadata`** operator on this dataset.
- Re-open this operator.

Metadata is cached on each sample after the first call, so this is a one-time cost per dataset.
""".strip()


# ---------------------------------------------------------------------------
# The operator.
# ---------------------------------------------------------------------------


class RunPerceptron(foo.Operator):
    """Run a Perceptron task on a video dataset, with a mode-driven form."""

    version: str = PLUGIN_VERSION

    @property
    def config(self) -> foo.OperatorConfig:
        return foo.OperatorConfig(
            name="run_perceptron",
            label="Perceptron: run task",
            description=(
                "Run a Perceptron task across a video dataset. Pick a mode "
                "for event search, semantic search, or label bootstrapping; "
                "the form updates based on your selection."
            ),
            dynamic=True,
            # Generator execution lets us yield progress updates the App
            # renders in-modal. Immediate by default; users can opt into
            # background execution via the in-form checkbox.
            execute_as_generator=True,
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            icon="/assets/icon.svg",
        )

    def resolve_placement(self, ctx: Any) -> Any:
        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(
                label="Perceptron",
                icon="/assets/icon.svg",
                prompt=True,
            ),
        )

    # ----------------------------------------------------------------- inputs

    def resolve_input(self, ctx: Any) -> Any:
        inputs = types.Object()

        # 0. API-key gate -- if not configured, the rest of the form is moot.
        if not has_api_key(ctx):
            inputs.message(
                "no_key",
                label=(
                    "PERCEPTRON_API_KEY is not set. Export it before "
                    "launching FiftyOne (see the plugin README)."
                ),
            )
            return types.Property(inputs)

        # 1. Media-type gate: read once here and thread through the form.
        media_type = ctx.dataset.media_type  # "image", "video", or "mixed"
        if media_type == "mixed":
            inputs.view(
                "mixed_media_error",
                types.Error(
                    label="Mixed-media datasets are not supported",
                    description=(
                        "This dataset contains both images and videos. "
                        "Filter the view to a single media type before running Perceptron."
                    ),
                    space=12,
                ),
            )
            return types.Property(inputs, view=types.View(label="Perceptron"))

        # 2. Metadata pre-flight checks.
        mode = ctx.params.get("mode", MODE_EVENT_SEARCH if media_type == "video" else MODE_SEMANTIC_SEARCH)

        # For video + dense/clip tasks: metadata is required (needed to convert
        # t= seconds to frame indices). Block the form with a hard error.
        if _mode_requires_frame_rate(ctx, mode):
            view = ctx.target_view()
            n_missing = _count_missing_metadata(view)
            if n_missing > 0:
                _render_missing_metadata_error(inputs, n_missing, len(view))
                return types.Property(inputs, view=types.View(label="Perceptron"))

        # For image datasets: metadata is not required but enables the
        # resolution advisory. Show a soft notice on form open if it's missing.
        if media_type == "image":
            view = ctx.target_view()
            n_missing = _count_missing_metadata(view)
            if n_missing > 0:
                inputs.view(
                    "_metadata_notice",
                    types.Notice(
                        label=(
                            f"{n_missing} of {len(view)} sample(s) are missing metadata. "
                            f"Run `dataset.compute_metadata()` to enable the resolution advisory."
                        )
                    ),
                )

        # 2b. Soft resolution-floor advisory (non-blocking).
        # Detection accuracy degrades sharply below ~270px short side.
        _render_resolution_advisory(inputs, ctx)

        # 3. Target view selection (full dataset / current view / selected).
        inputs.view_target(ctx)

        # 4. Mode selector: available modes depend on media type.
        # Event Search is video-only (requires temporal clip output).
        available_modes = (
            [MODE_EVENT_SEARCH, MODE_SEMANTIC_SEARCH, MODE_BOOTSTRAP]
            if media_type == "video"
            else [MODE_SEMANTIC_SEARCH, MODE_BOOTSTRAP]
        )
        default_mode = available_modes[0]
        # Keep the user's current selection if it's still valid.
        if mode not in available_modes:
            mode = default_mode

        mode_choices = types.DropdownView()
        for mode_value in available_modes:
            mode_choices.add_choice(
                mode_value,
                label=MODE_LABELS[mode_value],
                description=MODE_CHOICE_DESCRIPTIONS[mode_value],
            )
        inputs.enum(
            "mode",
            mode_choices.values(),
            default=default_mode,
            required=True,
            label="Mode",
            view=mode_choices,
        )
        inputs.str(
            "_mode_description",
            view=types.MarkdownView(read_only=True),
            default=MODE_LONG_DESCRIPTIONS.get(mode, ""),
        )

        # 5. Mode-specific form fields.
        match mode:
            case "event_search":
                _render_event_search_inputs(inputs, ctx)
            case "semantic_search":
                _render_semantic_search_inputs(inputs, ctx)
            case "bootstrap":
                _render_bootstrap_inputs(inputs, ctx, media_type=media_type)
            case _:
                raise ValueError(f"unexpected mode: {mode!r}")

        # 6. Advanced toggles (thinking + focus). Model is always perceptron-mk1.
        inputs.bool(
            "enable_thinking",
            default=False,
            label="Enable thinking (advanced)",
            description=(
                "Slower and more expensive; off by default. May demote "
                "structured output to prose for clip-shaped tasks when "
                "combined with weak prompts. Recommended on for VQA and "
                "reasoning-heavy classification; off for clip extraction "
                "unless the prompt uses canonical phrasing."
            ),
        )
        inputs.bool(
            "focus",
            default=False,
            label="Enable focus",
            description=(
                "When on, the model applies internal focusing tools that zoom "
                "into regions and re-run inference on crops. Useful for "
                "fine-grained grounding; off by default."
            ),
        )

        # 7. Execution-mode checkbox. Authority is `resolve_delegation()`,
        # which replaces the App's built-in execution-mode dropdown.
        _render_execution_mode(inputs, ctx)

        return types.Property(
            inputs,
            view=types.View(label="Perceptron"),
        )

    def resolve_delegation(self, ctx: Any) -> bool:
        # Returning a concrete bool tells FiftyOne not to render its own picker.
        return bool(ctx.params.get("delegate", False))

    # ------------------------------------------------------------- execution

    async def execute(self, ctx: Any) -> AsyncIterator[dict[str, Any]]:
        """Async-generator dispatcher across the three modes.

        Each per-mode executor runs the blocking API call in
        ``asyncio.to_thread`` while this coroutine yields live ticker events.
        """
        mode = ctx.params["mode"]
        # Two log handlers on the "perceptron" logger for the run's duration:
        #   ProgressHandler  -- routes to ctx.set_progress for delegated runs.
        #   _capture_perceptron_logs -- buffers records for the immediate-run
        #     SSE modal (the only path that reaches the progress spinner).
        # Each is a no-op in the other's domain.
        with foo.ProgressHandler(ctx, logger=logger), _capture_perceptron_logs() as logbuf:
            match mode:
                case "event_search":
                    async for msg in _execute_event_search(ctx, self.version, logbuf):
                        yield msg
                case "semantic_search":
                    async for msg in _execute_semantic_search(ctx, self.version, logbuf):
                        yield msg
                case "bootstrap":
                    async for msg in _execute_bootstrap(ctx, self.version, logbuf):
                        yield msg
                case _:
                    raise ValueError(f"unexpected mode in execute: {mode!r}")

    def resolve_output(self, ctx: Any) -> Any:
        outputs = types.Object()
        outputs.str("mode", label="Mode")
        outputs.str("summary", label="Summary", view=types.MarkdownView())
        outputs.str("run_key", label="Custom run key")
        outputs.float("elapsed_seconds", label="Elapsed (s)")
        return types.Property(outputs, view=types.View(label="Perceptron run results"))


# ---------------------------------------------------------------------------
# Live-log -> progress-modal plumbing.
#
# `foo.ProgressHandler` only reaches the App's progress modal for *delegated*
# runs (it routes through `ctx.set_progress`, which the delegated runner
# persists). For *immediate* generator runs, only events explicitly *yielded*
# from `execute()` reach the App via SSE. So we buffer log records in a
# thread-safe deque and drain them between work steps as
# `ctx.ops.set_progress(label=msg)` events. The percentage bar still ticks
# from the iteration-end summary yield; the label flips between log lines.
# ---------------------------------------------------------------------------


class _ProgressLogBuffer(logging.Handler):
    """Bounded logging handler that buffers formatted records for later draining."""

    def __init__(self, capacity: int = 256, level: int = logging.INFO) -> None:
        super().__init__(level=level)
        self.records: collections.deque[str] = collections.deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:  # noqa: BLE001 -- never let logging kill a run
            msg = record.getMessage()
        self.records.append(msg)


@contextlib.contextmanager
def _capture_perceptron_logs() -> Iterator[_ProgressLogBuffer]:
    """Attach a `_ProgressLogBuffer` to the ``"perceptron"`` logger for the scope."""
    buffer = _ProgressLogBuffer()
    perceptron_logger = logging.getLogger("perceptron")
    # Honor the logger's effective level so DEBUG flows when verbosity is up.
    buffer.setLevel(perceptron_logger.getEffectiveLevel())
    perceptron_logger.addHandler(buffer)
    try:
        yield buffer
    finally:
        perceptron_logger.removeHandler(buffer)


def _drain_log_buffer(ctx: Any, buffer: _ProgressLogBuffer) -> Iterator[Any]:
    """Yield one ``set_progress(label=msg)`` per buffered log record.

    Updates only the label, not the progress percentage.
    """
    while buffer.records:
        msg = buffer.records.popleft()
        yield ctx.ops.set_progress(label=msg)


def _label_debug_str(label: Any) -> str:
    """One-line summary of a FiftyOne label for browser console logging."""
    if isinstance(label, fo.Polylines):
        n = len(label.polylines)
        if n and label.polylines[0].points:
            sample_pts = str(label.polylines[0].points[0])[:80]
            return f"{n} polyline(s) | first label={label.polylines[0].label!r} pts[0]={sample_pts}"
        return f"{n} polyline(s) (empty)"
    if isinstance(label, fo.Detections):
        return f"{len(label.detections)} detection(s)"
    if isinstance(label, fo.Keypoints):
        return f"{len(label.keypoints)} keypoint(s)"
    if isinstance(label, fo.TemporalDetections):
        return f"{len(label.detections)} temporal detection(s)"
    if isinstance(label, fo.Classifications):
        return f"{len(label.classifications)} classification(s)"
    if isinstance(label, fo.Classification):
        return f"label={label.label!r} confidence={label.confidence}"
    if isinstance(label, str):
        return repr(label[:80])
    if label is None:
        return "None"
    return repr(label)[:100]


async def _predict_with_progress(
    ctx: Any,
    view: Any,
    *,
    model: PerceptronModel,
    mode_label: str,
    logbuf: _ProgressLogBuffer,
    started_at: float,
    on_sample: Callable[[fo.Sample, Any, int], str],
) -> AsyncIterator[Any]:
    """Drive the per-sample predict loop shared by all three modes.

    For each sample: kick off the blocking API call in a worker thread,
    yield ticker events while it runs, then invoke ``on_sample(sample, label,
    i)`` to perform writeback / stat tracking. The callback returns the
    status string to display in the modal for that iteration's progress
    yield.
    """
    total = len(view)
    for i, sample in enumerate(view):
        predict_task = asyncio.create_task(asyncio.to_thread(_predict, model, sample))
        async for tick in _tick_until_done(
            predict_task,
            ctx,
            mode_label=mode_label,
            done_so_far=i,
            total=total,
            started_at=started_at,
            model=model,
            logbuf=logbuf,
        ):
            yield tick

        label = predict_task.result()

        # Browser console: raw API response + parsed label for each sample.
        raw = getattr(model, "_last_raw_content", "") or ""
        yield ctx.ops.console_log(
            f"[perceptron] [{i+1}/{total}] raw response ({len(raw)} chars): {raw!r}"
        )
        yield ctx.ops.console_log(
            f"[perceptron] [{i+1}/{total}] parsed label: {_label_debug_str(label)}"
        )

        status = on_sample(sample, label, i)

        # Surface every `[perceptron]` log line from this iteration as its own
        # label update before bumping the progress percentage.
        for msg in _drain_log_buffer(ctx, logbuf):
            yield msg
        yield _yield_progress(
            ctx,
            mode_label=mode_label,
            status=status,
            done=i + 1,
            total=total,
            model=model,
            started_at=started_at,
        )


# ---------------------------------------------------------------------------
# Per-mode form renderers.
# ---------------------------------------------------------------------------


def _render_event_search_inputs(inputs: Any, ctx: Any) -> None:
    """Event Search mode: free-text query + prompt preview + auto-derived output field."""
    inputs.str(
        "event_query",
        label="Event description",
        description=(
            "Natural-language description of the event to find "
            "(e.g. 'a pedestrian crosses in front of a moving vehicle')."
        ),
        required=True,
    )

    query = ctx.params.get("event_query") or ""
    # Live prompt preview so users see the exact phrasing sent to the model.
    preview_target = query or "<event description>"
    preview = default_user_prompt(Task.FIND_EVENT, preview_target)
    inputs.str(
        "_prompt_preview",
        view=types.MarkdownView(read_only=True),
        default=f"**Prompt:** `{preview}`",
    )

    default_field = _derive_field_name("event", query)
    inputs.str(
        "event_field",
        label="Output field",
        description="Where to write the TemporalDetections (per video).",
        default=default_field,
        required=True,
    )


def _render_semantic_search_inputs(inputs: Any, ctx: Any) -> None:
    """Semantic Search mode: query + prompt preview + output field + confidence threshold."""
    inputs.str(
        "semantic_query",
        label="Search query",
        description=(
            "Natural-language description of the samples you want "
            "(e.g. 'videos with motorcycles', 'aerial coastline shots')."
        ),
        required=True,
    )

    query = ctx.params.get("semantic_query") or ""
    # Semantic Search uses a fixed question template, not default_user_prompt.
    preview_query = query or "<search query>"
    preview = _SEMANTIC_SEARCH_TEMPLATE.format(query=preview_query)
    inputs.str(
        "_prompt_preview",
        view=types.MarkdownView(read_only=True),
        default=f"**Prompt:** `{preview}`",
    )

    default_field = _derive_field_name("match", query)
    inputs.str(
        "semantic_field",
        label="Output field",
        description="Where to write the per-sample Classification.",
        default=default_field,
        required=True,
    )

    inputs.float(
        "semantic_threshold",
        label="Confidence threshold",
        description=(
            "After scoring, only samples with `label == \"yes\"` AND "
            "`confidence >= threshold` will be kept in the filtered view."
        ),
        default=_SEMANTIC_DEFAULT_THRESHOLD,
    )


def _render_bootstrap_inputs(inputs: Any, ctx: Any, *, media_type: str) -> None:
    """Bootstrap mode: task picker + conditional target + output field.

    The task list shown depends on ``media_type``: image datasets get grounding
    tasks (DETECT, KEYPOINTS, POLYGON) while video datasets get TRACK and
    KEY_MOMENTS. Shared tasks (classify, VQA, caption) appear in both.
    """
    task_list = (
        _BOOTSTRAP_VIDEO_TASKS if media_type == "video" else _BOOTSTRAP_IMAGE_TASKS
    )
    default_task = task_list[0].value

    task_choices = types.DropdownView()
    for task in task_list:
        task_choices.add_choice(
            task.value,
            label=_humanize_task(task),
            description=_BOOTSTRAP_TASK_SHORT[task],
        )

    inputs.enum(
        "bootstrap_task",
        task_choices.values(),
        default=default_task,
        required=True,
        label="Task",
        view=task_choices,
    )

    selected = ctx.params.get("bootstrap_task", default_task)
    # Guard against a stale task value from the wrong media type.
    try:
        task = Task(selected)
    except ValueError:
        task = task_list[0]

    inputs.str(
        "_bootstrap_task_description",
        view=types.MarkdownView(read_only=True),
        default=_BOOTSTRAP_TASK_LONG.get(task, ""),
    )

    # Caption and KEY_MOMENTS tasks have no target input.
    spec = _TARGET_FIELD_SPEC.get(task)
    if spec is not None:
        inputs.str("bootstrap_target", **spec)

    # Live prompt preview. Uses the spec label as the placeholder when the
    # target is empty so users see the grammar even before they start typing.
    target = (ctx.params.get("bootstrap_target") or "").strip()
    placeholder = f"<{spec['label'].lower()}>" if spec else None
    preview_target = target or placeholder
    try:
        preview = default_user_prompt(task, preview_target, media_type=media_type)
        inputs.str(
            "_prompt_preview",
            view=types.MarkdownView(read_only=True),
            default=f"**Prompt:** `{preview}`",
        )
    except ValueError:
        pass  # task requires a target but none is typed yet; skip the preview

    # Dense path (TRACK) needs stride / max_frames controls + cost preview.
    if task_supports_per_frame(task):
        _render_dense_controls(inputs, ctx)

    inputs.str(
        "bootstrap_field",
        label="Output field",
        description="Where to write the labels.",
        default=_BOOTSTRAP_DEFAULT_FIELDS[task],
        required=True,
    )


def _render_dense_controls(inputs: Any, ctx: Any) -> None:
    """Stride / max_frames inputs + a cost-preview Markdown row.

    Only rendered when a dense task (TRACK, KEYPOINTS) is selected. The cost
    preview multiplies the per-sample request count by the target-view size
    to give an honest "this will make X API calls" estimate.
    """
    inputs.int(
        "bootstrap_stride",
        label="Stride (frames between samples)",
        description=(
            "Send every Nth frame as an API call. Smaller = better coverage "
            "but higher cost. Default 3."
        ),
        default=3,
        min=1,
    )
    inputs.int(
        "bootstrap_max_frames",
        label="Max frames per video (optional)",
        description=(
            "Cap the number of API calls per video, regardless of stride. "
            "Leave blank for no cap."
        ),
        required=False,
    )
    _render_dense_cost_preview(inputs, ctx)


def _render_dense_cost_preview(inputs: Any, ctx: Any) -> None:
    """Estimate API calls across the target view and surface as Markdown."""
    stride = max(1, int(ctx.params.get("bootstrap_stride") or 3))
    max_frames = ctx.params.get("bootstrap_max_frames") or None

    view = ctx.target_view()
    n_videos = len(view)

    # Sample the first video's frame count as a representative; metadata
    # gate has already verified this is set.
    avg_frame_count = _avg_frame_count(view)
    if avg_frame_count is None or n_videos == 0:
        preview_text = (
            "**Cost preview**: unable to estimate (no samples in target view "
            "or `metadata.total_frame_count` missing). Each video will use "
            f"`ceil(frame_count / {stride})` API calls."
        )
    else:
        per_video = max(1, -(-avg_frame_count // stride))  # ceil-div
        if max_frames:
            per_video = min(per_video, int(max_frames))
        total = per_video * n_videos
        preview_text = (
            f"**Cost preview**: ~{per_video} API call(s) per video across "
            f"**{n_videos}** sample(s) -> **~{total} total call(s)**.  \n"
            f"_(Estimate based on the target view's average frame count of "
            f"{avg_frame_count}.)_"
        )

    inputs.str(
        "_dense_cost_preview",
        view=types.MarkdownView(read_only=True, space=12),
        default=preview_text,
    )


def _avg_frame_count(view: Any) -> int | None:
    """Average ``metadata.total_frame_count`` across `view`, or `None`.

    Uses FiftyOne's `mean()` aggregation. Returns `None` if the view is
    empty or no samples have the metadata field populated.
    """
    if len(view) == 0:
        return None
    try:
        avg = view.mean("metadata.total_frame_count")
    except Exception:  # noqa: BLE001 -- defensive; aggregation can vary
        return None
    if avg is None:
        return None
    return int(avg)


def _humanize_task(task: Task) -> str:
    """``Task.CLASSIFY_SINGLE`` -> ``"Classify single"`` for dropdown labels."""
    raw = task.value.replace("_", " ")
    return raw.capitalize()


def _render_resolution_advisory(inputs: Any, ctx: Any) -> None:
    """Non-blocking notice when target-view samples fall below the resolution floor.

    Empirical finding #9: detection holds at 50% downscale (~540px short side)
    and collapses at 25% (~270px). 512px is a conservative warning threshold.
    Advisory only; doesn't block submission.

    Uses ``metadata.frame_width/frame_height`` for video and
    ``metadata.width/height`` for images -- FiftyOne uses different field names
    on VideoMetadata vs ImageMetadata.
    """
    view = ctx.target_view()
    if len(view) == 0:
        return
    # Pick the right metadata field names for the media type.
    media_type = ctx.dataset.media_type
    w_field = "metadata.frame_width" if media_type == "video" else "metadata.width"
    h_field = "metadata.frame_height" if media_type == "video" else "metadata.height"
    try:
        min_w = view.min(w_field)
        min_h = view.min(h_field)
    except Exception:  # noqa: BLE001 -- aggregation can fail on partial metadata
        return
    if min_w is None or min_h is None:
        return
    short_side = min(int(min_w), int(min_h))
    if short_side >= 512:
        return
    inputs.view(
        "_resolution_advisory",
        types.Notice(
            label=(
                f"Some samples have a short side of {short_side}px. Detection "
                f"accuracy degrades sharply below ~270px; results on small "
                f"samples may be empty."
            ),
        ),
    )


def _render_missing_metadata_error(inputs: Any, n_missing: int, n_total: int) -> None:
    """Top-of-form error + remediation instructions when metadata is missing.

    The caller must `return` immediately after this so nothing else in the
    form is reachable. ``space=12`` forces full-width rendering in the modal.
    """
    inputs.view(
        "metadata_missing_error",
        types.Error(
            label="Compute video metadata before running Perceptron",
            description=(
                f"{n_missing} of {n_total} samples in the target view are "
                f"missing the `metadata` field, which is needed to map "
                f"per-frame model output back to frames."
            ),
            space=12,
        ),
    )
    inputs.str(
        "_metadata_fix_instructions",
        view=types.MarkdownView(read_only=True, space=12),
        default=_METADATA_FIX_INSTRUCTIONS_MD,
    )


def _render_execution_mode(inputs: Any, ctx: Any) -> None:
    """Immediate-vs-delegated checkbox + advisory notice.

    Authority is `resolve_delegation()`, so FiftyOne hides its built-in picker.
    Pattern from
    https://docs.voxel51.com/plugins/developing_plugins.html#delegated-execution.
    """
    delegate = bool(ctx.params.get("delegate", False))
    description = (
        "Uncheck this box to run immediately in the foreground."
        if delegate
        else "Check this box to delegate this run to a background queue."
    )
    inputs.bool(
        "delegate",
        default=False,
        required=True,
        label="Delegate execution?",
        description=description,
        view=types.CheckboxView(),
    )
    if delegate:
        inputs.view(
            "delegate_notice",
            types.Notice(
                label=(
                    "Delegated execution requires a FiftyOne delegated-"
                    "operation service running in this environment. See "
                    "https://docs.voxel51.com/plugins/index.html#operators "
                    "for setup."
                )
            ),
        )


# ---------------------------------------------------------------------------
# Per-mode execute() bodies. Each is a generator: yields progress and a
# final result dict that becomes ctx.results.
# ---------------------------------------------------------------------------


async def _execute_event_search(
    ctx: Any, version: str, logbuf: _ProgressLogBuffer
) -> AsyncIterator[dict[str, Any]]:
    query = (ctx.params.get("event_query") or "").strip()
    if not query:
        raise ValueError("Event Search requires a non-empty event description.")
    field = (ctx.params.get("event_field") or _derive_field_name("event", query)).strip()
    if not field:
        raise ValueError("Event Search requires a non-empty output field name.")

    view = ctx.target_view()

    # Pre-flight metadata BEFORE the first yield. If we yield first and then
    # raise, the SSE stream is already open and the App's progress modal will
    # hang at "starting on N video(s)" instead of surfacing the exception.
    # Frame rate is required to convert the API's `t=` seconds into
    # `TemporalDetection.support` frame ranges.
    _require_video_metadata(view, reason="Event Search")

    total = len(view)
    notify(ctx, f"Searching {total} video(s) for event '{query}'...")

    # Initial 0% so the spinner appears immediately rather than popping in
    # at the first per-sample update. No model yet -- elapsed / tokens
    # show as placeholders.
    yield _yield_progress(
        ctx,
        mode_label="Event Search",
        status=f"Starting on {total} video(s)...",
        done=0,
        total=total,
    )

    model = _build_model(ctx, task=Task.FIND_EVENT, target=query)
    # Flush the model-init log lines to the modal before the first sample.
    for msg in _drain_log_buffer(ctx, logbuf):
        yield msg
    # Zero token counters so the progress label starts at 0.
    model.reset_usage_totals()

    started_at = time.perf_counter()
    counters = {"clips": 0, "matched": 0}

    def on_sample(sample: fo.Sample, label: Any, _i: int) -> str:
        sample[field] = label
        sample.save()
        if label is not None and getattr(label, "detections", None):
            counters["clips"] += len(label.detections)
            counters["matched"] += 1
        return f"{counters['matched']} match(es), {counters['clips']} clip(s)"

    async for msg in _predict_with_progress(
        ctx,
        view,
        model=model,
        mode_label="Event Search",
        logbuf=logbuf,
        started_at=started_at,
        on_sample=on_sample,
    ):
        yield msg

    n_clips = counters["clips"]
    elapsed_s = time.perf_counter() - started_at

    # The naive `match(field has detections)` view shows full videos with the
    # event timestamps merely highlighted on the timeline. `to_clips(field)`
    # is the FiftyOne idiom for "one row per event": each row's `support`
    # frame range is the clip, and the App video player auto-scrubs to it.
    # https://docs.voxel51.com/user_guide/using_views.html#video-clips
    clips_view = ctx.dataset.to_clips(field)
    n_matched_clips = len(clips_view)
    # We also compute the source-sample count for the audit summary, since
    # ClipsView rows are clips (not samples).
    n_matched_samples = len(
        ctx.dataset.match(F(field) != None).match(  # noqa: E711
            F(f"{field}.detections").length() > 0
        )
    )

    usage = model.usage_totals
    run_key = _register_custom_run(
        ctx,
        version=version,
        operation="event_search",
        summary={
            "mode": MODE_EVENT_SEARCH,
            "query": query,
            "field": field,
            "n_total": total,
            "n_matched_samples": n_matched_samples,
            "n_matched_clips": n_matched_clips,
            "n_temporal_detections": n_clips,
            "elapsed_seconds": round(elapsed_s, 2),
            "api_calls": usage["calls"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
        },
    )

    summary_md = (
        f"**Event Search** completed.  \n"
        f"- Query: `{query}`  \n"
        f"- Matched: **{n_matched_samples}** / {total} videos "
        f"(**{n_matched_clips}** event clip(s))  \n"
        f"- Output field: `{field}`  \n"
        f"- Elapsed: **{_format_elapsed(elapsed_s)}** across "
        f"**{usage['calls']}** API call(s)  \n"
        f"- Tokens: **{usage['prompt_tokens']:,}** in / "
        f"**{usage['completion_tokens']:,}** out  \n"
        f"- View auto-switched to a clips view, one row per detected event."
    )
    notify(
        ctx,
        (
            f"Event Search done: {n_matched_clips} clip(s) across "
            f"{n_matched_samples}/{total} video(s)."
        ),
        variant="success",
    )

    # Switch the App into the clips view + refresh, then settle at 100%.
    yield ctx.ops.set_view(view=clips_view)
    yield ctx.ops.reload_samples()
    yield _yield_progress(
        ctx,
        mode_label="Event Search",
        status=(
            f"Complete -- {n_matched_clips} clip(s) across "
            f"{n_matched_samples}/{total} video(s)."
        ),
        done=total,
        total=total,
        model=model,
        started_at=started_at,
    )

    yield {
        "mode": MODE_EVENT_SEARCH,
        "summary": summary_md,
        "run_key": run_key,
        "elapsed_seconds": round(elapsed_s, 2),
    }


async def _execute_semantic_search(
    ctx: Any, version: str, logbuf: _ProgressLogBuffer
) -> AsyncIterator[dict[str, Any]]:
    query = (ctx.params.get("semantic_query") or "").strip()
    if not query:
        raise ValueError("Semantic Search requires a non-empty query.")
    field = (ctx.params.get("semantic_field") or _derive_field_name("match", query)).strip()
    if not field:
        raise ValueError("Semantic Search requires a non-empty output field name.")
    threshold = float(ctx.params.get("semantic_threshold", _SEMANTIC_DEFAULT_THRESHOLD))

    view = ctx.target_view()
    total = len(view)
    notify(ctx, f"Scoring {total} sample(s) for '{query}'...")

    yield _yield_progress(
        ctx,
        mode_label="Semantic Search",
        status=f"Starting on {total} sample(s)...",
        done=0,
        total=total,
    )

    model = _build_model(
        ctx,
        task=Task.CLASSIFY_SINGLE,
        target=None,
        prompt=_SEMANTIC_SEARCH_TEMPLATE.format(query=query),
        response_format=_SEMANTIC_SEARCH_RESPONSE_FORMAT,
    )
    for msg in _drain_log_buffer(ctx, logbuf):
        yield msg
    model.reset_usage_totals()

    # Semantic Search produces sample-level Classifications -- no per-frame
    # writes, so no frame_rate pre-flight needed.
    started_at = time.perf_counter()
    counters = {"yes": 0}

    def on_sample(sample: fo.Sample, label: Any, _i: int) -> str:
        sample[field] = label
        sample.save()
        if label is not None and getattr(label, "label", None) == "yes":
            counters["yes"] += 1
        return f"{counters['yes']} 'yes' so far"

    async for msg in _predict_with_progress(
        ctx,
        view,
        model=model,
        mode_label="Semantic Search",
        logbuf=logbuf,
        started_at=started_at,
        on_sample=on_sample,
    ):
        yield msg

    n_yes = counters["yes"]
    elapsed_s = time.perf_counter() - started_at

    matched_view = ctx.dataset.match(
        (F(f"{field}.label") == "yes")
        & (F(f"{field}.confidence") >= threshold)
    )
    n_matched = len(matched_view)

    usage = model.usage_totals
    run_key = _register_custom_run(
        ctx,
        version=version,
        operation="semantic_search",
        summary={
            "mode": MODE_SEMANTIC_SEARCH,
            "query": query,
            "field": field,
            "threshold": threshold,
            "n_total": total,
            "n_yes": n_yes,
            "n_matched_samples": n_matched,
            "elapsed_seconds": round(elapsed_s, 2),
            "api_calls": usage["calls"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
        },
    )

    summary_md = (
        f"**Semantic Search** completed.  \n"
        f"- Query: `{query}`  \n"
        f"- Threshold: `{threshold:.2f}`  \n"
        f"- 'yes' answers: **{n_yes}** / {total}  \n"
        f"- Above threshold: **{n_matched}** samples  \n"
        f"- Output field: `{field}`  \n"
        f"- Elapsed: **{_format_elapsed(elapsed_s)}** across "
        f"**{usage['calls']}** API call(s)  \n"
        f"- Tokens: **{usage['prompt_tokens']:,}** in / "
        f"**{usage['completion_tokens']:,}** out  \n"
        f"- View auto-filtered to above-threshold samples."
    )
    notify(
        ctx,
        f"Semantic Search done: {n_matched}/{total} matched at >= {threshold:.2f}.",
        variant="success",
    )

    yield ctx.ops.set_view(view=matched_view)
    yield ctx.ops.reload_samples()
    yield _yield_progress(
        ctx,
        mode_label="Semantic Search",
        status=(
            f"Complete -- {n_matched}/{total} matched at "
            f">= {threshold:.2f}"
        ),
        done=total,
        total=total,
        model=model,
        started_at=started_at,
    )

    yield {
        "mode": MODE_SEMANTIC_SEARCH,
        "summary": summary_md,
        "run_key": run_key,
        "elapsed_seconds": round(elapsed_s, 2),
    }


async def _execute_bootstrap(
    ctx: Any, version: str, logbuf: _ProgressLogBuffer
) -> AsyncIterator[dict[str, Any]]:
    task = Task(ctx.params["bootstrap_task"])
    target = (ctx.params.get("bootstrap_target") or "").strip() or None
    field = (ctx.params.get("bootstrap_field") or _BOOTSTRAP_DEFAULT_FIELDS[task]).strip()
    if not field:
        raise ValueError("Bootstrap Labels requires a non-empty output field name.")

    view = ctx.target_view()

    # Pre-flight metadata BEFORE the first yield (see comment in
    # `_execute_event_search`). Only per-frame tasks need it; sample-level
    # tasks (caption, classify, vqa) skip the check entirely.
    per_frame = task_supports_per_frame(task)
    if per_frame:
        _require_video_metadata(view, reason=f"Bootstrap {task.value}")

    total = len(view)
    notify(ctx, f"Running {task.value} on {total} sample(s)...")

    mode_label = f"Bootstrap {task.value}"
    yield _yield_progress(
        ctx,
        mode_label=mode_label,
        status=f"Starting on {total} sample(s)...",
        done=0,
        total=total,
    )

    model = _build_model(ctx, task=task, target=target)
    for msg in _drain_log_buffer(ctx, logbuf):
        yield msg
    model.reset_usage_totals()

    started_at = time.perf_counter()
    stats = {
        "frame_labels": 0,
        "sample_labels": 0,
        "frames_touched": 0,
        "dropped": 0,
    }

    def on_sample(sample: fo.Sample, label: Any, _i: int) -> str:
        frame_rate = _frame_rate(sample) if per_frame else None
        summary = write_per_frame_labels(sample, label, field, frame_rate=frame_rate)
        stats["frame_labels"] += summary["per_frame_count"]
        stats["sample_labels"] += summary["sample_level"]
        stats["frames_touched"] += summary["frames_written"]
        stats["dropped"] += summary["dropped"]
        n_labels = stats["frame_labels"] + stats["sample_labels"]
        # Surface dropped count so missing `t=` / `frame_rate` issues stand out.
        drop_tail = f"; {stats['dropped']} dropped" if stats["dropped"] else ""
        return f"{n_labels} label(s) written{drop_tail}"

    async for msg in _predict_with_progress(
        ctx,
        view,
        model=model,
        mode_label=mode_label,
        logbuf=logbuf,
        started_at=started_at,
        on_sample=on_sample,
    ):
        yield msg

    n_frame_labels = stats["frame_labels"]
    n_sample_labels = stats["sample_labels"]
    n_frames_touched = stats["frames_touched"]
    n_dropped = stats["dropped"]
    n_labels = n_frame_labels + n_sample_labels
    elapsed_s = time.perf_counter() - started_at

    usage = model.usage_totals
    run_key = _register_custom_run(
        ctx,
        version=version,
        operation=f"bootstrap_{task.value}",
        summary={
            "mode": MODE_BOOTSTRAP,
            "task": task.value,
            "target": target,
            "field": field,
            "n_total": total,
            "n_labels": n_labels,
            "n_frame_labels": n_frame_labels,
            "n_sample_labels": n_sample_labels,
            "n_frames_touched": n_frames_touched,
            "n_dropped": n_dropped,
            "elapsed_seconds": round(elapsed_s, 2),
            "api_calls": usage["calls"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
        },
    )

    summary_md_lines = [
        f"**Bootstrap Labels** ({task.value}) completed.  ",
        f"- Target: `{target or '(none)'}`  ",
        f"- Output field: `{field}`  ",
        f"- Processed: {total} sample(s)  ",
        f"- Frame-level labels: **{n_frame_labels}** across {n_frames_touched} frame(s)  ",
        f"- Sample-level labels: **{n_sample_labels}**  ",
        (
            f"- Elapsed: **{_format_elapsed(elapsed_s)}** across "
            f"**{usage['calls']}** API call(s)  "
        ),
        (
            f"- Tokens: **{usage['prompt_tokens']:,}** in / "
            f"**{usage['completion_tokens']:,}** out  "
        ),
    ]
    if n_dropped:
        summary_md_lines.append(
            f"- Dropped (no `t=` or missing `frame_rate`): **{n_dropped}**  "
        )
    summary_md = "\n".join(summary_md_lines)

    success_msg = (
        f"Bootstrap {task.value} done: {n_labels} label(s) written to '{field}'"
    )
    if n_dropped:
        success_msg += f" ({n_dropped} dropped -- see logs)"
    notify(
        ctx,
        success_msg + ".",
        variant="success" if n_dropped == 0 else "warning",
    )

    # Reload the dataset so new fields appear in the sample-grid sidebar,
    # then settle on a clean 100% so the modal closes cleanly.
    yield ctx.ops.reload_dataset()
    final_drop_tail = f"; {n_dropped} dropped" if n_dropped else ""
    yield _yield_progress(
        ctx,
        mode_label=mode_label,
        status=f"Complete -- {n_labels} label(s) written{final_drop_tail}",
        done=total,
        total=total,
        model=model,
        started_at=started_at,
    )

    yield {
        "mode": MODE_BOOTSTRAP,
        "summary": summary_md,
        "run_key": run_key,
        "elapsed_seconds": round(elapsed_s, 2),
    }


# ---------------------------------------------------------------------------
# Module-private helpers (shared by all three modes).
# ---------------------------------------------------------------------------


def _build_model(
    ctx: Any,
    *,
    task: Task,
    target: str | None,
    prompt: str | None = None,
    response_format: dict[str, Any] | None = None,
) -> PerceptronModel:
    """Construct a `PerceptronModel` from the operator context.

    ``media_type`` is always sourced from ``ctx.dataset.media_type`` so the
    model's predict dispatch and FiftyOne ``Model.media_type`` property are
    always consistent with the dataset being processed.

    ``response_format`` is the optional caller-side override (Semantic Search
    passes a yes/no-constrained json_schema). When absent, the model class
    applies the default CLASSIFY schema for classify tasks.
    """
    stride_param = ctx.params.get("bootstrap_stride")
    max_frames_param = ctx.params.get("bootstrap_max_frames")
    cfg: dict[str, Any] = {
        "model": DEFAULT_MODEL_NAME,
        "task": task,
        "media_type": ctx.dataset.media_type,
        "target": target,
        "prompt": prompt,
        "enable_thinking": bool(ctx.params.get("enable_thinking", False)),
        "focus": bool(ctx.params.get("focus", False)),
        "api_key": get_api_key(ctx),
    }
    if stride_param:
        cfg["stride"] = int(stride_param)
    if max_frames_param:
        cfg["max_frames"] = int(max_frames_param)
    if response_format is not None:
        cfg["response_format"] = response_format
    return PerceptronModel(config=cfg)


def _predict(model: PerceptronModel, sample: fo.Sample) -> Any:
    """`model.predict` with sample context attached to any raised exception."""
    try:
        return model.predict(sample.filepath, sample=sample)
    except Exception as exc:
        exc.add_note(f"While processing sample {sample.id!s} ({sample.filepath})")
        raise


def _format_elapsed(seconds: float) -> str:
    """Compact elapsed format: ``12.3s`` / ``2m 34s`` / ``1h 5m``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h}h {m}m"


def _format_tokens(n: int) -> str:
    """Compact token count: ``1234`` -> ``1.2k``, ``1500000`` -> ``1.5M``."""
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.1f}M"


async def _tick_until_done(
    predict_task: asyncio.Task[Any],
    ctx: Any,
    *,
    mode_label: str,
    done_so_far: int,
    total: int,
    started_at: float,
    model: PerceptronModel,
    logbuf: _ProgressLogBuffer,
) -> AsyncIterator[Any]:
    """Yield progress + log events every ``_PROGRESS_TICK_INTERVAL_S`` while
    ``predict_task`` is still running.

    Each tick drains buffered log records, then yields a `_yield_progress`
    event with the latest elapsed-time / token totals so the modal updates
    live during long calls. Returns when the task completes or raises;
    the caller calls `predict_task.result()` afterwards.

    ``asyncio.shield(predict_task)`` is load-bearing: without it the inner
    `wait_for(..., timeout=...)` would cancel the underlying thread when the
    timeout fires, killing the prediction on every tick.
    """
    pending_status = "Waiting for API response..."
    while not predict_task.done():
        try:
            await asyncio.wait_for(
                asyncio.shield(predict_task),
                timeout=_PROGRESS_TICK_INTERVAL_S,
            )
        except asyncio.TimeoutError:
            for msg in _drain_log_buffer(ctx, logbuf):
                yield msg
            yield _yield_progress(
                ctx,
                mode_label=mode_label,
                status=pending_status,
                done=done_so_far,
                total=total,
                model=model,
                started_at=started_at,
            )


def _yield_progress(
    ctx: Any,
    *,
    mode_label: str,
    status: str,
    done: int,
    total: int,
    model: PerceptronModel | None = None,
    started_at: float | None = None,
) -> Any:
    """Render the in-modal progress UI: circular spinner + stacked stats.

    Uses the ``show_output`` schema pattern (rather than
    `ctx.ops.set_progress`) so we get a circular spinner plus stacked
    `LabelValueView` rows (Status / Elapsed / Tokens In / Tokens Out)
    instead of a single linear bar. When ``model`` / ``started_at`` are
    `None` (the initial "Starting..." yield), elapsed and tokens render
    as ``--`` placeholders.
    """
    pct = int(round(done / total * 100)) if total else 0
    spinner_caption = f"{mode_label}  -  {done}/{total} ({pct}%)"

    schema = types.Object()
    schema.int(
        "progress",
        view=types.ProgressView(
            variant="circular",
            label=spinner_caption,
        ),
    )
    schema.str("status", label="Status", view=types.LabelValueView())
    schema.str("elapsed", label="Elapsed", view=types.LabelValueView())
    schema.str("tokens_in", label="Total Tokens In", view=types.LabelValueView())
    schema.str("tokens_out", label="Total Tokens Out", view=types.LabelValueView())

    if model is not None and started_at is not None:
        elapsed_str = _format_elapsed(time.perf_counter() - started_at)
        usage = model.usage_totals
        tokens_in_str = _format_tokens(usage["prompt_tokens"])
        tokens_out_str = _format_tokens(usage["completion_tokens"])
    else:
        elapsed_str = tokens_in_str = tokens_out_str = "--"

    results: dict[str, Any] = {
        "status": status,
        "elapsed": elapsed_str,
        "tokens_in": tokens_in_str,
        "tokens_out": tokens_out_str,
    }
    if total and done >= total:
        # Passing a value flips MUI CircularProgress from *indeterminate*
        # (spinning) to *determinate* (static). The App multiplies this value
        # by 100 before displaying it, so 1 → 100%.
        results["progress"] = 1

    return ctx.trigger(
        "show_output",
        {
            "outputs": types.Property(schema).to_json(),
            "results": results,
        },
    )


def _missing_metadata_view(view: Any) -> Any:
    """Subview of samples whose `metadata` field is unset.

    `compute_metadata` populates ``metadata`` itself; a rare codec/probe
    failure can leave ``metadata.frame_rate`` unset, but that's handled
    later by `write_per_frame_labels` (drops the affected labels with a
    warning) so we only gate the form on the top-level miss.
    """
    return view.exists("metadata", False)


def _count_missing_metadata(view: Any) -> int:
    return len(_missing_metadata_view(view))


def _require_video_metadata(view: Any, *, reason: str) -> None:
    """Raise if any sample lacks ``metadata`` -- runtime defense for the form gate.

    Must be called BEFORE the first ``yield`` of the executor. Once SSE is
    open, exceptions hang the modal instead of surfacing as an error toast.
    The form gate in `resolve_input` is the primary defense; this catches
    SDK-launched runs that bypass the form.
    """
    missing = _missing_metadata_view(view)
    n_missing = len(missing)
    if n_missing == 0:
        logger.info(
            "[perceptron] metadata pre-flight OK: every sample has metadata.frame_rate"
        )
        return

    examples = [s.filepath for s in missing.limit(5).select_fields("filepath")]
    bullet_list = "\n".join(f"  - {fp}" for fp in examples)
    if n_missing > len(examples):
        bullet_list += f"\n  ...and {n_missing - len(examples)} more."

    raise RuntimeError(
        f"{reason} requires video metadata, but {n_missing} sample(s) in "
        f"the target view are missing it.\n\n"
        f"Run `dataset.compute_metadata()` once and re-launch the operator. "
        f"This is a one-time cost; metadata is cached on each sample after "
        f"the first call.\n\n"
        f"Examples of samples missing metadata:\n{bullet_list}"
    )


def _mode_requires_frame_rate(ctx: Any, mode: str) -> bool:
    """Whether the active mode/task needs ``metadata.frame_rate``.

    Event Search always does (it converts seconds to frame support values).
    Bootstrap does only for TRACK (the one remaining per-frame dense task).
    Semantic Search and all image tasks never do.
    """
    match mode:
        case "event_search":
            return True
        case "bootstrap":
            task_str = ctx.params.get("bootstrap_task")
            if not task_str:
                # No task selected yet; default to requiring metadata so the
                # gate fires early for video datasets, then re-renders when
                # the user picks a task.
                return True
            try:
                return task_supports_per_frame(Task(task_str))
            except ValueError:
                return True
        case _:
            return False


def _frame_rate(sample: fo.Sample) -> float | None:
    """Read ``metadata.frame_rate`` off a sample, or ``None``.

    Fast per-sample read for the work loop; the pre-flight already vetted
    the view.
    """
    metadata = sample.metadata
    if metadata is None:
        return None
    fr = getattr(metadata, "frame_rate", None)
    return float(fr) if fr else None


def _register_custom_run(
    ctx: Any,
    *,
    version: str,
    operation: str,
    summary: dict[str, Any],
) -> str:
    """Register a Custom Run on the dataset and return its key.

    Steps (this order matters -- `init_run_results` calls `get_run_info`,
    which raises if the key isn't already registered):

        1. Build the `RunConfig` with `init_run(...)`.
        2. `register_run(key, config)`.
        3. `init_run_results(key)`.
        4. Populate dynamic attrs on the results object.
        5. `save_run_results(key, results, overwrite=True)`.
    """
    run_key = make_run_key(operation)

    run_config = ctx.dataset.init_run(
        operator="run_perceptron",
        version=version,
        params=dict(ctx.params),
        dataset_name=ctx.dataset.name,
    )
    ctx.dataset.register_run(run_key, run_config)

    run_results = ctx.dataset.init_run_results(run_key)
    # `summary` is a dynamic attribute -- `save_run_results` serializes the
    # object's __dict__, so whatever we set here lands in the audit record.
    run_results.summary = summary
    ctx.dataset.save_run_results(run_key, run_results, overwrite=True)

    logger.info("[perceptron] registered custom run %s with summary keys=%s", run_key, list(summary.keys()))
    return run_key


def _derive_field_name(prefix: str, query: str) -> str:
    """Free-text query -> safe sample-field name, capped at 40 chars.

    ``("event", "a pedestrian crosses the street")`` ->
    ``"event_a_pedestrian_crosses_the_stre"``.
    """
    slug = _SANITIZE_FIELD_RE.sub("_", query.strip().lower()).strip("_")
    if not slug:
        return prefix
    combined = f"{prefix}_{slug}"
    return combined[:40].rstrip("_") or prefix
