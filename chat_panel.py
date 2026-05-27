"""Perceptron Chat Panel — ask questions about the current image or video.

Architecture
------------
A hybrid modal panel. Python lifecycle hooks push the current sample's
filepath and media type to React via ``ctx.panel.set_state()``. The React
component calls three panel methods:

    ``ask``             — validates params, starts an inference daemon thread
                          that streams tokens to a file, returns a run_id.
    ``get_stream_chunk``— reads new bytes from the stream file since the
                          caller's last cursor position; React polls every
                          250 ms to produce a live typing effect.
    ``save_as_label``   — parses the completed stream content using the existing
                          perceptron_parser and saves the resulting FiftyOne
                          label to the sample. Only called when the response
                          contained recognised grounding tags.

Streaming pattern (identical to vlm_prompt_lab)
-----------------------------------------------
The streaming call runs in a daemon thread. It writes token chunks to an
append-only file at ``~/.fiftyone/perceptron_chat/.stream_<run_id>.txt``
and writes lifecycle state to ``~/.fiftyone/perceptron_chat/.status_<run_id>.json``.
React polls ``get_stream_chunk`` with a byte cursor; the method seeks to
that position and returns whatever the thread has written since.

This file-based IPC is required because FiftyOne reimports the plugin
module on every panel-method call, so module-level state is reset between
the ``ask`` and ``get_stream_chunk`` calls.

Multi-turn context
------------------
The image or video is embedded only in the first user message. Subsequent
user messages contain text only. The full conversation history is resent on
every API call so the model always has context.

Grouped datasets
----------------
``_sync_sample`` resolves the active group slice so the panel always sends
the filepath of the sample currently visible in the modal, not the default
slice's filepath.
"""

from __future__ import annotations

import json
import mimetypes
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import bson
from fiftyone import ViewField as F
from openai import OpenAI

import fiftyone.operators as foo
import fiftyone.operators.types as types

from ._shared import get_api_key, has_api_key
from .perceptron_api import to_image_data_uri, to_video_data_uri
from .perceptron_model import DEFAULT_MODEL_NAME
from .perceptron_parser import to_fiftyone

# ---------------------------------------------------------------------------
# Runtime file locations — OUTSIDE the plugin directory to avoid invalidating
# FiftyOne's plugin cache (writing inside the dir changes its mtime).
# ---------------------------------------------------------------------------

_STATUS_DIR = Path.home() / ".fiftyone" / "perceptron_chat"

BASE_URL = "https://api.perceptron.inc/v1"
TIMEOUT_S = 300  # 5 min; video data URIs + reasoning can be slow


def _stream_path(run_id: str) -> Path:
    return _STATUS_DIR / f".stream_{run_id}.txt"


def _status_path(run_id: str) -> Path:
    return _STATUS_DIR / f".status_{run_id}.json"


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def _ensure_dir() -> None:
    _STATUS_DIR.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, data: dict) -> None:
    """Atomic JSON write (write-to-temp + rename)."""
    _ensure_dir()
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _append_stream(run_id: str, text: str) -> None:
    _ensure_dir()
    with open(_stream_path(run_id), "a", encoding="utf-8") as f:
        f.write(text)
        f.flush()


def _clear_run(run_id: str) -> None:
    """Remove stale stream + status files before starting a new inference."""
    for p in (_stream_path(run_id), _status_path(run_id)):
        p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Format detection + label helpers
# ---------------------------------------------------------------------------


def _detect_format(content: str) -> str | None:
    """Detect grounding output format from a completed response.

    Checks for known tag signatures and JSON key patterns. Priority order:
    polygon > box (XML) > point > clip > bbox_2d (JSON). XML tags take
    precedence over JSON so the canonical format is always preferred.
    Returns ``None`` for plain-text responses — the Convert button is hidden.
    """
    if re.search(r"<polygon\b", content, re.IGNORECASE):
        return "polygon"
    if re.search(r"<point_box\b", content, re.IGNORECASE):
        return "box"
    if re.search(r"<point\b", content, re.IGNORECASE):
        return "point"
    if re.search(r"<clip\b", content, re.IGNORECASE):
        return "clip"
    # JSON bbox_2d array — checked last so XML tags always win.
    if re.search(r'"bbox_2d"\s*:', content):
        return "bbox2d"
    return None


def _count_label_items(label: Any) -> int:
    """Best-effort item count for display in the save-success message."""
    if hasattr(label, "detections"):
        return len(label.detections)
    if hasattr(label, "keypoints"):
        return len(label.keypoints)
    if hasattr(label, "polylines"):
        return len(label.polylines)
    if hasattr(label, "classifications"):
        return len(label.classifications)
    return 1


# Default output field names shown in the Convert UI, one per format.
_DEFAULT_FIELDS: dict[str, str] = {
    "box":     "perceptron_detections",
    "bbox2d":  "perceptron_detections",  # same output type as "box"
    "point":   "perceptron_keypoints",
    "polygon": "perceptron_polygons",
    "clip":    "perceptron_events",
}


# ---------------------------------------------------------------------------
# Inference thread
# ---------------------------------------------------------------------------


def _run_stream_thread(
    api_key: str,
    messages: list[dict[str, Any]],
    vision_config: dict[str, Any] | None,
    run_id: str,
) -> None:
    """Stream a Perceptron chat completion and write tokens to the stream file.

    Runs as a daemon thread so it exits automatically if the server process
    exits. Writes ``status="streaming"`` before the first token, then
    ``status="done"`` (with latency) on completion, or ``status="error"``
    (with error message) on failure.
    """
    _write_json(_status_path(run_id), {"status": "streaming", "start_time": time.time()})
    try:
        client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=TIMEOUT_S)

        extra: dict[str, Any] = {}
        if vision_config:
            extra["extra_body"] = {"vision_config": vision_config}

        stream = client.chat.completions.create(
            model=DEFAULT_MODEL_NAME,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **extra,
        )

        t0 = time.time()
        prompt_tokens = completion_tokens = 0

        for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            delta = getattr(choice, "delta", None) if choice else None
            if delta and getattr(delta, "content", None):
                _append_stream(run_id, delta.content)
            # usage arrives on the final chunk when include_usage=True
            if getattr(chunk, "usage", None):
                usage = chunk.usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        # Read back the full streamed content to detect any grounding tags.
        try:
            full_content = _stream_path(run_id).read_text(encoding="utf-8")
        except OSError:
            full_content = ""
        detected_format = _detect_format(full_content)

        _write_json(_status_path(run_id), {
            "status": "done",
            "latency_ms": int((time.time() - t0) * 1000),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            # None when the response is plain text; set to a format string when
            # grounding tags are present so React can show the Convert button.
            "detected_format": detected_format,
            "default_field": _DEFAULT_FIELDS.get(detected_format, "") if detected_format else "",
        })

    except Exception as exc:
        _write_json(_status_path(run_id), {
            "status": "error",
            "error": str(exc),
        })
        print(f"[perceptron_chat] inference error:\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


class PerceptronChatPanel(foo.Panel):
    """Modal panel for asking questions about the current image or video.

    Displayed in the FiftyOne sample modal. Lifecycle hooks keep the active
    sample filepath and media type in panel state; all conversation state
    lives on the React side and survives sample navigation within the session.
    """

    @property
    def config(self):
        return foo.PanelConfig(
            name="perceptron_chat",
            label="Ask Perceptron",
            surfaces="modal",
            help_markdown=(
                "Ask free-form questions about the current image or video. "
                "Responses stream live from Perceptron Mk1. "
                "Turn on **Thinking** for reasoning-heavy questions."
            ),
        )

    # ── Lifecycle hooks ──────────────────────────────────────────────────────

    def on_load(self, ctx):
        ctx.panel.set_state("api_key_missing", not has_api_key(ctx))
        self._sync_sample(ctx)

    def on_change_current_sample(self, ctx):
        self._sync_sample(ctx)

    def on_change_group_slice(self, ctx):
        # Fires when the user clicks a slice tab in the modal.
        # React also detects this via the Recoil atom and calls
        # update_sample directly — both paths converge on the same values.
        self._sync_sample(ctx)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _sync_sample(self, ctx) -> None:
        """Push the active sample's filepath, sample ID, media type, and frame
        rate to React state.

        For grouped datasets, resolves the active group slice so the panel
        always reflects the slice currently visible in the modal.
        """
        if not ctx.current_sample:
            return

        dataset = ctx.dataset
        sample = dataset[ctx.current_sample]
        gf = dataset.group_field

        resolved = sample  # will be replaced by the active slice if needed
        filepath = sample.filepath
        sample_id = ctx.current_sample
        media_type = getattr(sample, "media_type", None) or "image"

        # For grouped datasets, resolve to the slice the user is viewing.
        if gf and ctx.group_slice:
            group_elem = sample[gf]
            if group_elem and group_elem.name != ctx.group_slice:
                try:
                    slice_sample = (
                        dataset
                        .select_group_slices(ctx.group_slice)
                        .match(F(f"{gf}._id") == bson.ObjectId(group_elem.id))
                        .first()
                    )
                    if slice_sample is not None:
                        resolved   = slice_sample
                        filepath   = slice_sample.filepath
                        sample_id  = slice_sample.id
                        media_type = getattr(slice_sample, "media_type", None) or "image"
                except Exception as exc:
                    print(f"[perceptron_chat] slice lookup error: {exc}")

        # frame_rate is needed by save_as_label to convert clip timestamps
        # to frame indices. Read from the already-resolved sample to avoid
        # a redundant database round-trip.
        frame_rate: float | None = None
        meta = resolved.metadata
        if meta is not None:
            raw_fr = getattr(meta, "frame_rate", None)
            if raw_fr:
                frame_rate = float(raw_fr)

        ctx.panel.set_state("filepath",   filepath)
        ctx.panel.set_state("sample_id",  sample_id)
        ctx.panel.set_state("media_type", media_type)
        ctx.panel.set_state("frame_rate", frame_rate)

    # ── Panel methods (called from React via usePanelEvent) ──────────────────

    def ask(self, ctx) -> dict:
        """Start a streaming inference and return a run_id for React to poll.

        Parameters (via ctx.params)
        ---------------------------
        filepath : str
            Absolute path to the current sample's media file.
        media_type : str
            ``"image"`` or ``"video"``.
        question : str
            The user's current question.
        history : list[dict]
            Prior turns as ``[{"role": "user"|"assistant", "content": str}, ...]``.
            The image/video is injected into the first user turn by this method;
            callers must pass plain-text content for all turns.
        enable_thinking : bool
            When true, sets ``vision_config.enable_thinking = True``.

        Returns
        -------
        dict
            ``{"status": "started", "run_id": str}`` on success.
            ``{"error": str}`` on validation failure.
        """
        if not has_api_key(ctx):
            return {"error": "PERCEPTRON_API_KEY is not set."}

        filepath   = ctx.params.get("filepath", "")
        media_type = ctx.params.get("media_type", "image")
        question   = (ctx.params.get("question") or "").strip()
        history: list[dict] = ctx.params.get("history", [])
        enable_thinking = bool(ctx.params.get("enable_thinking", False))

        if not question:
            return {"error": "Question cannot be empty."}
        if not filepath:
            return {"error": "No filepath provided."}

        # Encode the current media as a data URI.
        if media_type == "video":
            media_url = to_video_data_uri(filepath)
            media_part: dict[str, Any] = {"type": "video_url", "video_url": {"url": media_url}}
        else:
            img_bytes = Path(filepath).read_bytes()
            mime = mimetypes.guess_type(filepath)[0] or "image/jpeg"
            media_url = to_image_data_uri(img_bytes, mime=mime)
            media_part = {"type": "image_url", "image_url": {"url": media_url}}

        # Build the messages array. The media is embedded in the first user
        # message only; subsequent turns are text so the context window isn't
        # flooded with repeated media tokens.
        messages: list[dict[str, Any]] = []

        if not history:
            # First turn — include the media.
            messages.append({
                "role": "user",
                "content": [media_part, {"type": "text", "text": question}],
            })
        else:
            # Replay prior turns, injecting media into the first user message.
            first_user_injected = False
            for turn in history:
                if turn["role"] == "user" and not first_user_injected:
                    messages.append({
                        "role": "user",
                        "content": [media_part, {"type": "text", "text": turn["content"]}],
                    })
                    first_user_injected = True
                else:
                    messages.append({"role": turn["role"], "content": turn["content"]})
            # If history was all assistant turns somehow, inject media in the new question.
            if not first_user_injected:
                messages.append({
                    "role": "user",
                    "content": [media_part, {"type": "text", "text": question}],
                })
            else:
                messages.append({"role": "user", "content": question})

        vision_config: dict[str, Any] = {"internal_tools": {"focus": False}}
        if enable_thinking:
            vision_config["enable_thinking"] = True

        # Unique run ID so concurrent panel instances don't share stream files.
        run_id = f"{ctx.current_sample or 'x'}_{int(time.time() * 1000)}"
        _clear_run(run_id)

        thread = threading.Thread(
            target=_run_stream_thread,
            kwargs=dict(
                api_key=get_api_key(ctx),
                messages=messages,
                vision_config=vision_config,
                run_id=run_id,
            ),
            daemon=True,
        )
        thread.start()
        return {"status": "started", "run_id": run_id}

    def get_stream_chunk(self, ctx) -> dict:
        """Return new streamed text since the caller's last cursor position.

        React calls this every 250 ms while inference is running. Uses byte
        offsets so multi-byte UTF-8 characters are handled correctly.

        Parameters (via ctx.params)
        ---------------------------
        run_id : str
            Run identifier returned by ``ask``.
        cursor : int
            Byte offset in the stream file from which to start reading.

        Returns
        -------
        dict
            text         — new decoded text since cursor (may be empty).
            cursor       — updated byte offset.
            done         — True when inference is complete or errored.
            final_status — full status dict when done, else None.
        """
        run_id = ctx.params.get("run_id", "")
        cursor = int(ctx.params.get("cursor", 0))

        try:
            with open(_stream_path(run_id), "rb") as f:
                f.seek(cursor)
                new_bytes = f.read()
            new_cursor = cursor + len(new_bytes)
            new_text = new_bytes.decode("utf-8", errors="replace")
        except (FileNotFoundError, OSError):
            new_text = ""
            new_cursor = cursor

        status = _read_json(_status_path(run_id)) or {}
        done = status.get("status") in ("done", "error")

        return {
            "text": new_text,
            "cursor": new_cursor,
            "done": done,
            "final_status": status if done else None,
        }

    def save_as_label(self, ctx) -> dict:
        """Parse a completed stream and save the result as a FiftyOne label.

        Reads the raw response from the stream file for ``run_id``, detects the
        grounding format, calls ``to_fiftyone``, and writes the label to the
        specified field on the sample.

        Parameters (via ctx.params)
        ---------------------------
        run_id     : str   Run identifier from ``ask``.
        sample_id  : str   FiftyOne sample ID to write the label to.
        field_name : str   Destination field on the sample.
        detected_format : str  One of ``"box"``, ``"point"``, ``"polygon"``, ``"clip"``.
        frame_rate : float | None  Video frame rate for clip → frame-index conversion.

        Returns
        -------
        dict
            On success: ``{saved: True, label_type: str, count: int, field: str}``.
            On failure: ``{error: str}``.
        """
        run_id    = ctx.params.get("run_id", "")
        sample_id = ctx.params.get("sample_id", "")
        field     = (ctx.params.get("field_name") or "").strip()
        fmt       = ctx.params.get("detected_format", "")
        frame_rate = ctx.params.get("frame_rate")

        if not run_id:    return {"error": "No run_id provided."}
        if not sample_id: return {"error": "No sample_id provided."}
        if not field:     return {"error": "Field name is required."}
        if not fmt:       return {"error": "No detected format — nothing to convert."}

        stream_file = _stream_path(run_id)
        if not stream_file.exists():
            return {"error": (
                f"Stream file for run '{run_id}' no longer exists. "
                "Re-ask the question to generate a new stream."
            )}

        try:
            content = stream_file.read_text(encoding="utf-8")
            fr = float(frame_rate) if frame_rate else None
            label = to_fiftyone(content, fmt, frame_rate=fr)

            sample = ctx.dataset[sample_id]
            sample[field] = label
            sample.save()

            # reload_samples() refreshes the grid and sample viewer without
            # closing the modal or resetting the view. reload_dataset() is
            # too aggressive here (triggers a full schema reload that can
            # close the modal).
            ctx.ops.reload_samples()

            label_type = type(label).__name__
            count = _count_label_items(label)
            return {"saved": True, "label_type": label_type, "count": count, "field": field}

        except Exception as exc:
            return {"error": f"Parse / save failed: {exc}"}

    def render(self, ctx):
        return types.Property(
            types.Object(),
            view=types.View(
                component="PerceptronChatPanel",
                composite_view=True,
                ask=self.ask,
                get_stream_chunk=self.get_stream_chunk,
                save_as_label=self.save_as_label,
            ),
        )
