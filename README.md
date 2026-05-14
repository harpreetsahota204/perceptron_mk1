# Perceptron × FiftyOne

A FiftyOne plugin that brings [Perceptron](https://docs.perceptron.inc) into
your data curation workflow. Use it to search, annotate, and understand your
**image and video datasets** directly from the FiftyOne App — no code required.

> **Perceptron Mk1** is Perceptron's frontier vision-language model for images
> and video. This plugin is built specifically for Mk1 — the grounding,
> clipping, thinking, and focus capabilities it exposes are all Mk1-specific.

---

## What you can do

### On image datasets

| Mode | What it does |
|------|-------------|
| **Semantic Search** | Score each image yes/no against a natural-language query, then filter to the matches |
| **Bootstrap Labels → Detect** | Find and locate every instance of an object class with bounding boxes |
| **Bootstrap Labels → Keypoints** | Point to every instance of an object class |
| **Bootstrap Labels → Polygon** | Outline every instance with polygon shapes |
| **Bootstrap Labels → Classify / VQA / Caption** | Add classification labels, answers, or captions to each image |

### On video datasets

| Mode | What it does |
|------|-------------|
| **Event Search** | Find the exact moment a specific event happens across all videos, then browse as clips |
| **Semantic Search** | Score each video yes/no against a natural-language query, then filter to the matches |
| **Bootstrap Labels → Key Moments** | Automatically identify all noteworthy events in each video |
| **Bootstrap Labels → Track** | Write per-frame bounding-box detections at a chosen stride |
| **Bootstrap Labels → Classify / VQA / Caption** | Add classification labels, answers, or captions to each video |

---

## Install

**Requirements:**  Python ≥ 3.11

```bash
pip install fiftyone openai perceptron opencv-python
```

`opencv-python` is only needed for the **Track** task (per-frame video decomposition).

### Install the plugin


```bash
fiftyone plugins download https://github.com/harpreetsahota204/perceptron-mk1
```

### Set your API key

Get a key at [platform.perceptron.inc](https://platform.perceptron.inc), then
export it before launching FiftyOne:

```bash
export PERCEPTRON_API_KEY="ak.<your-key>"
```

The plugin reads the key from FiftyOne's secret system — you never paste it
into the form. The form shows a clear error if the key isn't set before you
run anything.

Then, launch the FiftyOne App:

```bash
fiftyone app launch
```

---

## Opening the plugin

You can launch the operator from two places in the App:

- **Backtick menu** — press `` ` ``, search "Perceptron: run task", press Enter
- **Grid action button** — click **Perceptron** in the action bar above the sample grid

Both open the same conditional form. The modes and tasks available adapt
automatically to your dataset's media type.

---

## One-time setup for videos: compute metadata

Event Search and the Track task convert the API's timestamp output into
FiftyOne frame indices. This requires `metadata.frame_rate` on every video
sample. Run this once per dataset:

```python
dataset.compute_metadata()
```

The result is cached on each sample. The plugin checks this at startup and
shows a clear, actionable error if anything is missing — no API calls are
wasted on a run that would produce unusable output.

Image datasets and video tasks that don't use per-frame output (Key Moments,
Classify, VQA, Caption) skip this check entirely.

---

## Walkthrough: Event Search

*Find the exact moment something specific happens across a video dataset.*

1. Open the operator, pick **Event Search**

2. Describe the event: `a person opens a vehicle door`

3. Click **Execute**

Every video that matched gets a `fo.TemporalDetection` pinpointing the moment
(`support=[start_frame, end_frame]` plus `t_start_seconds` / `t_end_seconds`
for the raw timestamps). The App immediately switches to a clips view — one row
per detected moment, with the video player scrubbed to that clip.

Run the same operator again with a different query. Each query writes to its
own field, so multiple searches stack on the same dataset without overwriting
each other.

---

## Walkthrough: Semantic Search

*Filter a dataset to the images or videos that match a description.*

1. Open the operator, pick **Semantic Search**
2. Enter your query: `aerial footage of a coastline`
3. Set a confidence threshold (default 0.7)
4. Click **Execute**

Each sample receives a `fo.Classification` with `label="yes"` or `"no"` and a
confidence score. The App filters the view to samples above your threshold.
Lower the threshold in the form to broaden the match set; the results are saved
so you can re-filter without re-running.

---

## Walkthrough: Detect objects in images

*Write bounding-box labels across an image dataset.*

1. Open the operator on an image dataset, pick **Bootstrap Labels → Detect**
2. Enter a target: `forklift`
3. Set the output field: `perceptron_detections`
4. Click **Execute**

Each image gets `fo.Detections` written at the sample level. Use FiftyOne's
filtering and tagging tools to review and curate the results.

---

## Walkthrough: Find key moments in every video

*Automatically summarise what happens in each video as temporal clips.*

1. Open the operator on a video dataset, pick **Bootstrap Labels → Key Moments**
2. Set the output field: `perceptron_key_moments`
3. Click **Execute**

No target required — the model decides what's noteworthy. Each video gets a
`fo.TemporalDetections` container with one detection per identified moment and
a label describing what happened.

To search for a *specific* event rather than letting the model choose freely,
use **Event Search** mode instead.

---

## Advanced options: thinking and focus

Every operator run exposes two toggles at the bottom of the form. Both map
directly to fields in the `vision_config` the plugin sends to the API.

### Enable thinking

When on, the model runs an internal reasoning trace before producing its
answer. The trace is stored in the response's `reasoning_content` field;
the final answer (what the plugin parses) is always in `content`.

| Task type | Recommendation |
|-----------|---------------|
| Captioning, VQA, OCR, free-form reasoning | **On** — thinking improves answer quality |
| Video clipping (Event Search, Key Moments) | **On** — reasoning helps the model localize events temporally |
| Spatial grounding (Detect, Keypoints, Polygon, Track) | **Off** — thinking is not recommended for spatial detection tasks and can degrade structured output |

Thinking is off by default because it adds latency and cost, and is actively
counterproductive for detection tasks.

### Enable focus

When on, the model can invoke an internal focus tool that zooms into a region
of the image or video and runs inference again on that crop. This is useful for
fine-grained grounding tasks where small or partially occluded objects might
otherwise be missed.

Off by default. Enable it when you need higher recall on small or detail-rich
objects, accepting the additional latency cost.

---

## All tasks at a glance

### Image tasks

| Task | Output | One API call per |
|------|--------|-----------------|
| `detect` | `fo.Detections` at sample level | image |
| `keypoints` | `fo.Keypoints` at sample level | image |
| `polygon` | `fo.Polylines` at sample level | image |

> **Note on polygons:** Mk1 currently returns bounding-box-shaped polygon
> responses. Detections will appear as rectangular polylines until the model
> natively supports free-form polygon output.

### Video tasks

| Task | Output | One API call per |
|------|--------|-----------------|
| `find_event` | `fo.TemporalDetections` at sample level | video |
| `key_moments` | `fo.TemporalDetections` at sample level | video |
| `track` | `fo.Detections` per-frame | frame (N calls per video) |

### Shared tasks (image and video)

| Task | Output | One API call per |
|------|--------|-----------------|
| `classify_single` | `fo.Classification(label, confidence)` | sample |
| `classify_multi` | `fo.Classifications` | sample |
| `caption_concise` | text | sample |
| `caption_detailed` | text (includes visible signage) | sample |
| `vqa` | text | sample |
