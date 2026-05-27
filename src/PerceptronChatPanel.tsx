import React, { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { usePanelClient } from "./hooks/usePanelClient";
import type { PanelData, PanelSchema, Turn } from "./types";

// ---------------------------------------------------------------------------
// Logging — prefixed for easy filtering in DevTools (type [perceptron_chat])
// ---------------------------------------------------------------------------

function log(...args: unknown[]) {
  console.log("[perceptron_chat]", ...args);
}
function logError(...args: unknown[]) {
  console.error("[perceptron_chat]", ...args);
}

// ---------------------------------------------------------------------------
// Session persistence — per-sample in sessionStorage
// ---------------------------------------------------------------------------

const SESSION_PREFIX = "perceptronChat:";

interface StoredSession {
  turns: Turn[];
  enableThinking: boolean;
  enableFocus: boolean;
  hintFormat: string;
}

function loadSession(sampleId: string): StoredSession | null {
  try {
    const raw = sessionStorage.getItem(SESSION_PREFIX + sampleId);
    return raw ? (JSON.parse(raw) as StoredSession) : null;
  } catch {
    return null;
  }
}

function saveSession(sampleId: string, data: StoredSession): void {
  try {
    sessionStorage.setItem(SESSION_PREFIX + sampleId, JSON.stringify(data));
  } catch {}
}

// ---------------------------------------------------------------------------
// Per-turn save state — tracks the Convert to FiftyOne workflow per turn.
// ---------------------------------------------------------------------------

interface TurnSaveState {
  fieldName: string;
  saving: boolean;
  saved: boolean;
  savedLabelType?: string;
  savedCount?: number;
  error: string | null;
}

// ---------------------------------------------------------------------------
// Global style injection (once per page load)
// ---------------------------------------------------------------------------

let _stylesInjected = false;
function ensureStyles() {
  if (_stylesInjected) return;
  _stylesInjected = true;
  const el = document.createElement("style");
  el.textContent = `
    @keyframes prcSpin  { to { transform: rotate(360deg); } }
    @keyframes prcBlink { 0%,100%{opacity:1} 50%{opacity:0} }
    .prc-md { font-size: 13px; line-height: 1.7; color: var(--fo-palette-text-primary); word-break: break-word; }
    .prc-md p  { margin: 0 0 8px; }
    .prc-md p:last-child { margin-bottom: 0; }
    .prc-md h1,.prc-md h2,.prc-md h3 { margin: 12px 0 5px; font-weight: 600; }
    .prc-md ul,.prc-md ol { margin: 0 0 8px; padding-left: 18px; }
    .prc-md li { margin-bottom: 2px; }
    .prc-md code { font-family: ui-monospace, monospace; font-size: 12px; background: var(--fo-palette-background-level2); color: var(--fo-palette-primary-main); padding: 1px 4px; border-radius: 3px; }
    .prc-md pre { background: var(--fo-palette-background-level2); border: 1px solid var(--fo-palette-divider); border-radius: 4px; padding: 8px 10px; overflow-x: auto; margin: 0 0 8px; }
    .prc-md pre code { background: none; padding: 0; color: var(--fo-palette-text-primary); }
    .prc-md blockquote { margin: 0 0 8px; padding: 3px 10px; border-left: 3px solid var(--fo-palette-primary-main); color: var(--fo-palette-text-secondary); }
    .prc-md strong { font-weight: 600; }
    .prc-md em    { font-style: italic; }
    .prc-md a     { color: var(--fo-palette-primary-main); }
  `;
  document.head.appendChild(el);
}

// ---------------------------------------------------------------------------
// Design tokens
// ---------------------------------------------------------------------------

const V = {
  bg:       "var(--fo-palette-background-body)",
  bg2:      "var(--fo-palette-background-level2)",
  bg3:      "var(--fo-palette-background-level3)",
  divider:  "var(--fo-palette-divider)",
  text:     "var(--fo-palette-text-primary)",
  muted:    "var(--fo-palette-text-secondary)",
  dim:      "var(--fo-palette-text-tertiary)",
  primary:  "var(--fo-palette-primary-main)",
  textInv:  "var(--fo-palette-text-invert)",
  font:     "var(--fo-fontFamily-body)",
  red:      "#e08080",
  redBg:    "#2a0e0e",
  green:    "#4caf81",
  greenBg:  "#0b1a0b",
  greenBorder: "#1a3a1a",
};

// Human-readable label for each grounding format.
const FORMAT_LABELS: Record<string, string> = {
  box:     "fo.Detections (boxes)",
  bbox2d:  "fo.Detections (bbox_2d JSON)",
  point:   "fo.Keypoints",
  polygon: "fo.Polylines",
  clip:    "fo.TemporalDetections",
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type StreamState = "idle" | "running" | "done" | "error";

interface Props {
  data?:   PanelData;
  schema?: PanelSchema;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const PerceptronChatPanel: React.FC<Props> = ({ data, schema }) => {
  const uris = {
    ask:              schema?.view?.ask              ?? "",
    get_stream_chunk: schema?.view?.get_stream_chunk ?? "",
    save_as_label:    schema?.view?.save_as_label    ?? "",
  };
  const { ask, getStreamChunk, saveAsLabel } = usePanelClient(uris);

  // Current sample context pushed by Python.
  const [filepath,  setFilepath]  = useState("");
  const [sampleId,  setSampleId]  = useState("");
  const [mediaType, setMediaType] = useState("image");
  const [frameRate, setFrameRate] = useState<number | null>(null);

  // Conversation turns.
  const [turns,    setTurns]    = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [enableThinking, setEnableThinking] = useState(false);
  const [enableFocus,    setEnableFocus]    = useState(false);
  // "auto" = let the model decide; other values set annotation_format in vision_config.
  const [hintFormat, setHintFormat] = useState("auto");

  // Streaming state for the active (in-progress) turn.
  const [streamState,   setStreamState]   = useState<StreamState>("idle");
  const [streamingText, setStreamingText] = useState("");
  const [streamError,   setStreamError]   = useState<string | null>(null);
  const [latencyMs,     setLatencyMs]     = useState<number | null>(null);
  const [promptTok,     setPromptTok]     = useState<number | null>(null);
  const [completionTok, setCompletionTok] = useState<number | null>(null);

  // Per-turn save state (keyed by turn index).
  const [turnSaveStates, setTurnSaveStates] = useState<Record<number, TurnSaveState>>({});

  const runIdRef      = useRef("");
  const cursorRef     = useRef(0);
  const chunkCountRef = useRef(0);
  const prevSampleId  = useRef("");
  const scrollRef     = useRef<HTMLDivElement>(null);

  useEffect(() => { ensureStyles(); }, []);

  // ── Sync sample from Python ───────────────────────────────────────────────
  useEffect(() => {
    const newPath = data?.filepath   ?? "";
    const newId   = data?.sample_id  ?? "";
    const newMt   = data?.media_type ?? "image";
    const newFr   = data?.frame_rate ?? null;
    if (!newId || !newPath) return;
    if (newId === prevSampleId.current) return;
    prevSampleId.current = newId;

    log("sample synced", { sample_id: newId, filepath: newPath, media_type: newMt, frame_rate: newFr });

    setFilepath(newPath);
    setSampleId(newId);
    setMediaType(newMt);
    setFrameRate(newFr);
    setQuestion("");
    setStreamingText("");
    setStreamState("idle");
    setStreamError(null);
    setLatencyMs(null);
    setPromptTok(null);
    setCompletionTok(null);
    setTurnSaveStates({});
    runIdRef.current  = "";
    cursorRef.current = 0;
    chunkCountRef.current = 0;

    const cached = loadSession(newId);
    if (cached) {
      log("restored session", { sample_id: newId, turns: cached.turns.length });
      setTurns(cached.turns);
      setEnableThinking(cached.enableThinking);
      setEnableFocus(cached.enableFocus ?? false);
      setHintFormat(cached.hintFormat ?? "auto");
    } else {
      setTurns([]);
    }
  }, [data?.filepath, data?.sample_id]); // eslint-disable-line

  // ── Persist turns ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (!sampleId) return;
    saveSession(sampleId, { turns, enableThinking, enableFocus, hintFormat });
  }, [turns, enableThinking, enableFocus, hintFormat, sampleId]);

  // ── Stream polling ────────────────────────────────────────────────────────
  useEffect(() => {
    if (streamState !== "running") return;
    const id = setInterval(() => {
      getStreamChunk(runIdRef.current, cursorRef.current)
        .then((chunk) => {
          if (chunk.text) {
            setStreamingText((prev) => prev + chunk.text);
            cursorRef.current = chunk.cursor;
          }
          chunkCountRef.current++;
          // Log every 10th chunk to avoid flooding the console.
          if (chunkCountRef.current % 10 === 0) {
            log(`stream chunk #${chunkCountRef.current}`, { cursor: chunk.cursor, done: chunk.done });
          }

          if (chunk.done) {
            clearInterval(id);
            const fs = chunk.final_status;
            if (fs?.status === "error") {
              logError("stream error", fs.error);
              setStreamError(fs.error ?? "Inference failed.");
              setStreamState("error");
            } else {
              log("stream done", {
                run_id:           runIdRef.current,
                latency_ms:       fs?.latency_ms,
                prompt_tokens:    fs?.prompt_tokens,
                completion_tokens: fs?.completion_tokens,
                detected_format:  fs?.detected_format ?? null,
              });
              setLatencyMs(fs?.latency_ms ?? null);
              setPromptTok(fs?.prompt_tokens ?? null);
              setCompletionTok(fs?.completion_tokens ?? null);
              setStreamState("done");

              // Commit the completed assistant turn, carrying metadata for the Convert button.
              const detectedFmt  = fs?.detected_format ?? null;
              const defaultField = fs?.default_field ?? "";
              const committedRunId = runIdRef.current;
              setStreamingText((text) => {
                setTurns((prev) => {
                  const last = prev[prev.length - 1];
                  if (!last || last.role !== "assistant") {
                    return [...prev, {
                      role:            "assistant",
                      content:         text,
                      run_id:          committedRunId,
                      detected_format: detectedFmt,
                      default_field:   defaultField,
                    }];
                  }
                  return prev;
                });
                return "";
              });
            }
          }
        })
        .catch((err) => logError("getStreamChunk error", err));
    }, 250);
    return () => clearInterval(id);
  }, [streamState]); // eslint-disable-line

  // ── Auto-scroll ───────────────────────────────────────────────────────────
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [turns.length, streamingText]);

  // ── Send ──────────────────────────────────────────────────────────────────
  const handleSend = useCallback(async () => {
    if (!question.trim() || streamState === "running" || !filepath) return;

    const q = question.trim();
    setQuestion("");
    setStreamingText("");
    setStreamState("running");
    setStreamError(null);
    setLatencyMs(null);
    setPromptTok(null);
    setCompletionTok(null);
    cursorRef.current = 0;
    chunkCountRef.current = 0;

    const historyForApi = [...turns];
    setTurns((prev) => [...prev, { role: "user", content: q }]);

    log("ask →", { filepath, media_type: mediaType, history_len: historyForApi.length,
                    enable_thinking: enableThinking, enable_focus: enableFocus, hint_format: hintFormat });

    try {
      const result = await ask({
        filepath,
        media_type:      mediaType,
        question:        q,
        history:         historyForApi,
        enable_thinking: enableThinking,
        enable_focus:    enableFocus,
        hint_format:     hintFormat,
      });
      log("ask ← run_id:", result.run_id);
      runIdRef.current = result.run_id;
    } catch (e: any) {
      logError("ask failed", e?.message);
      setStreamError(e?.message ?? "Request failed.");
      setStreamState("error");
    }
  }, [question, streamState, filepath, mediaType, turns, enableThinking, enableFocus, hintFormat, ask]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
    },
    [handleSend]
  );

  // ── Convert to FiftyOne ───────────────────────────────────────────────────
  const handleConvert = useCallback(async (turnIdx: number, turn: Turn) => {
    const state = turnSaveStates[turnIdx];
    const fieldName = (state?.fieldName ?? turn.default_field ?? "").trim();
    if (!fieldName || !turn.run_id || !turn.detected_format || !sampleId) return;

    log("convert →", {
      turn_idx:        turnIdx,
      run_id:          turn.run_id,
      detected_format: turn.detected_format,
      field_name:      fieldName,
      sample_id:       sampleId,
      frame_rate:      frameRate,
    });

    setTurnSaveStates((prev) => ({
      ...prev,
      [turnIdx]: { ...prev[turnIdx], fieldName, saving: true, saved: false, error: null },
    }));

    try {
      const result = await saveAsLabel({
        run_id:           turn.run_id,
        sample_id:        sampleId,
        field_name:       fieldName,
        detected_format:  turn.detected_format,
        frame_rate:       frameRate,
      });
      log("convert ←", { label_type: result.label_type, count: result.count, field: result.field });
      setTurnSaveStates((prev) => ({
        ...prev,
        [turnIdx]: {
          ...prev[turnIdx],
          saving:          false,
          saved:           true,
          savedLabelType:  result.label_type,
          savedCount:      result.count,
          error:           null,
        },
      }));
    } catch (e: any) {
      logError("convert failed", e?.message);
      setTurnSaveStates((prev) => ({
        ...prev,
        [turnIdx]: { ...prev[turnIdx], saving: false, saved: false, error: e?.message ?? "Save failed." },
      }));
    }
  }, [turnSaveStates, sampleId, frameRate, saveAsLabel]);

  // ── API key warning ───────────────────────────────────────────────────────
  if (data?.api_key_missing) {
    return (
      <div style={{ padding: "20px 24px", fontFamily: V.font, color: V.red,
                    background: V.redBg, borderRadius: 6, margin: 16,
                    border: "1px solid #7a3030", lineHeight: 1.6 }}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>⚠ PERCEPTRON_API_KEY is not set</div>
        <div style={{ fontSize: 12, color: "#f5c6c6" }}>
          Export the key before launching FiftyOne, then restart the server.
        </div>
        <code style={{ display: "block", marginTop: 10, background: "#1a1a1a",
                       color: "#90cdf4", fontFamily: "monospace", fontSize: 11,
                       padding: "6px 10px", borderRadius: 4 }}>
          export PERCEPTRON_API_KEY="ak...."
        </code>
      </div>
    );
  }

  if (!filepath) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center",
                    height: "100%", color: V.muted, fontSize: 13, fontFamily: V.font,
                    textAlign: "center", padding: 24 }}>
        Open a sample to start chatting.
      </div>
    );
  }

  const canSend    = !!question.trim() && streamState !== "running";
  const mediaLabel = mediaType === "video" ? "video" : "image";

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%",
                  overflow: "hidden", fontFamily: V.font, fontSize: 13,
                  color: V.text, background: V.bg, boxSizing: "border-box" }}>

      {/* ── Sample info bar ── */}
      <div style={{ flexShrink: 0, padding: "6px 12px",
                    borderBottom: `1px solid ${V.divider}`,
                    display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ fontSize: 10, padding: "2px 7px", borderRadius: 10,
                       background: V.bg3, color: V.muted, textTransform: "uppercase",
                       letterSpacing: "0.05em", flexShrink: 0 }}>
          {mediaLabel}
        </span>
        <span style={{ fontSize: 11, color: V.dim, overflow: "hidden",
                       textOverflow: "ellipsis", whiteSpace: "nowrap" }}
              title={filepath}>
          {filepath.split("/").pop()}
        </span>
      </div>

      {/* ── Chat history ── */}
      <div ref={scrollRef} style={{ flex: 1, overflowY: "auto",
                                     padding: "12px 14px", minHeight: 0 }}>

        {turns.length === 0 && streamState === "idle" && (
          <div style={{ color: V.muted, fontSize: 12, textAlign: "center",
                        padding: "24px 12px" }}>
            Ask anything about this {mediaLabel}…
          </div>
        )}

        {turns.map((turn, i) => (
          <div key={i} style={{ marginBottom: 10 }}>
            {/* Bubble */}
            <div style={{
              display:        "flex",
              justifyContent: turn.role === "user" ? "flex-end" : "flex-start",
            }}>
              <div style={{
                maxWidth:     "88%",
                background:   turn.role === "user" ? V.primary : V.bg2,
                color:        turn.role === "user" ? V.textInv  : V.text,
                borderRadius: turn.role === "user" ? "12px 12px 4px 12px" : "12px 12px 12px 4px",
                padding:      "8px 12px",
                fontSize:     13,
                lineHeight:   1.5,
              }}>
                {turn.role === "assistant" ? (
                  <div className="prc-md">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{turn.content}</ReactMarkdown>
                  </div>
                ) : (
                  turn.content
                )}
              </div>
            </div>

            {/* Convert to FiftyOne — only for completed assistant turns with grounding output */}
            {turn.role === "assistant" && turn.detected_format && (
              <ConvertBar
                turnIdx={i}
                turn={turn}
                saveState={turnSaveStates[i]}
                onConvert={handleConvert}
                onFieldChange={(idx, name) =>
                  setTurnSaveStates((prev) => ({
                    ...prev,
                    [idx]: { ...prev[idx], fieldName: name, saved: false, error: null,
                             saving: false, savedLabelType: undefined, savedCount: undefined },
                  }))
                }
              />
            )}
          </div>
        ))}

        {/* Active streaming turn */}
        {streamState === "running" && (
          <div style={{ display: "flex", justifyContent: "flex-start", marginBottom: 10 }}>
            <div style={{
              maxWidth: "88%", background: V.bg2, color: V.text,
              borderRadius: "12px 12px 12px 4px", padding: "8px 12px",
              fontSize: 13, lineHeight: 1.5,
            }}>
              {streamingText ? (
                <div className="prc-md">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{streamingText}</ReactMarkdown>
                  <span style={{
                    display: "inline-block", width: 2, height: "1em",
                    background: V.primary, marginLeft: 1, verticalAlign: "text-bottom",
                    animation: "prcBlink 1s step-end infinite",
                  }} />
                </div>
              ) : (
                <div style={{ display: "flex", alignItems: "center", gap: 6, color: V.muted }}>
                  <span style={{
                    width: 12, height: 12, borderRadius: "50%",
                    border: `2px solid ${V.divider}`, borderTop: `2px solid ${V.primary}`,
                    animation: "prcSpin 0.7s linear infinite",
                    display: "inline-block", flexShrink: 0,
                  }} />
                  Waiting for response…
                </div>
              )}
            </div>
          </div>
        )}

        {streamError && (
          <div style={{ padding: "8px 10px", background: V.redBg, color: V.red,
                        borderRadius: 6, fontSize: 12, marginBottom: 8 }}>
            {streamError}
          </div>
        )}

        {streamState === "done" && (latencyMs != null || completionTok != null) && (
          <div style={{ fontSize: 11, color: V.dim, textAlign: "center", padding: "2px 0 8px" }}>
            {[
              completionTok != null && `${completionTok} tokens`,
              latencyMs    != null && `${(latencyMs / 1000).toFixed(1)}s`,
              promptTok    != null && `${promptTok} in`,
            ].filter(Boolean).join(" · ")}
          </div>
        )}
      </div>

      {/* ── Bottom bar ── */}
      <div style={{ flexShrink: 0, padding: "8px 12px",
                    borderTop: `1px solid ${V.divider}`,
                    background: V.bg2, display: "flex",
                    flexDirection: "column", gap: 7 }}>

        {/* Format hint + thinking toggle — same row */}
        <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" as const }}>

          {/* Output format selector */}
          <div style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: V.muted }}>
            <span style={{ flexShrink: 0 }}>Output:</span>
            <select
              value={hintFormat}
              onChange={(e) => setHintFormat(e.target.value)}
              style={{
                background: V.bg2, color: V.text,
                border: `1px solid ${V.divider}`, borderRadius: 4,
                padding: "2px 5px", fontSize: 11, fontFamily: V.font,
                cursor: "pointer", outline: "none",
              }}
            >
              <option value="auto">Auto (detect)</option>
              <option value="box">Boxes</option>
              <option value="point">Keypoints</option>
              <option value="polygon">Polygons</option>
              <option value="clip">Clips</option>
            </select>
          </div>

          {/* Thinking toggle */}
          <label style={{ display: "flex", alignItems: "center", gap: 6,
                           cursor: "pointer", userSelect: "none",
                           fontSize: 11, color: V.muted }}>
            <input
              type="checkbox" checked={enableThinking}
              onChange={(e) => setEnableThinking(e.target.checked)}
              style={{ width: 13, height: 13, cursor: "pointer", accentColor: V.primary, flexShrink: 0 }}
            />
            Thinking
          </label>

          {/* Focus toggle */}
          <label style={{ display: "flex", alignItems: "center", gap: 6,
                           cursor: "pointer", userSelect: "none",
                           fontSize: 11, color: V.muted }}>
            <input
              type="checkbox" checked={enableFocus}
              onChange={(e) => setEnableFocus(e.target.checked)}
              style={{ width: 13, height: 13, cursor: "pointer", accentColor: V.primary, flexShrink: 0 }}
            />
            Focus
          </label>
        </div>

        <div style={{ display: "flex", gap: 6, alignItems: "flex-end" }}>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={streamState === "running"}
            placeholder={
              streamState === "running"
                ? "Waiting for response…"
                : `Ask about this ${mediaLabel}… (Enter ↵ to send)`
            }
            rows={1}
            style={{
              flex: 1, background: V.bg, color: streamState === "running" ? V.dim : V.text,
              border: `1px solid ${V.divider}`, borderRadius: 6,
              padding: "7px 10px", fontSize: 13, fontFamily: V.font,
              lineHeight: 1.5, outline: "none", resize: "none" as const,
              minHeight: 38, overflow: "auto", boxSizing: "border-box" as const,
              cursor: streamState === "running" ? "not-allowed" : "text",
            }}
          />
          <button
            onClick={handleSend} disabled={!canSend}
            title="Send (Enter)"
            style={{
              background: "none", border: "none", padding: "0 2px",
              cursor:  canSend ? "pointer" : "default",
              color:   streamState === "running" ? V.muted : canSend ? V.primary : V.dim,
              opacity: canSend || streamState === "running" ? 1 : 0.35,
              lineHeight: 1, display: "flex", alignItems: "center",
              flexShrink: 0, marginBottom: 6,
            }}
          >
            {streamState === "running" ? (
              <span style={{
                width: 16, height: 16, borderRadius: "50%",
                border: `2px solid ${V.divider}`, borderTop: `2px solid ${V.primary}`,
                animation: "prcSpin 0.7s linear infinite", display: "inline-block",
              }} />
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
              </svg>
            )}
          </button>
        </div>

        {turns.length > 0 && streamState !== "running" && (
          <div style={{ display: "flex", justifyContent: "flex-end" }}>
            <button
              onClick={() => {
                setTurns([]);
                setStreamingText("");
                setStreamState("idle");
                setStreamError(null);
                setTurnSaveStates({});
                log("conversation cleared for sample", sampleId);
              }}
              style={{
                background: "none", border: "none", cursor: "pointer",
                color: V.dim, fontSize: 11, padding: "0 2px",
                textDecoration: "underline", fontFamily: V.font,
              }}
            >
              Clear conversation
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// ConvertBar — rendered below each completed assistant turn that has grounding
// output. Contains the field-name input and the Convert button.
// ---------------------------------------------------------------------------

interface ConvertBarProps {
  turnIdx:       number;
  turn:          Turn;
  saveState:     TurnSaveState | undefined;
  onConvert:     (idx: number, turn: Turn) => void;
  onFieldChange: (idx: number, name: string) => void;
}

const ConvertBar: React.FC<ConvertBarProps> = ({
  turnIdx, turn, saveState, onConvert, onFieldChange,
}) => {
  const fieldName  = saveState?.fieldName ?? turn.default_field ?? "";
  const saving     = saveState?.saving    ?? false;
  const saved      = saveState?.saved     ?? false;
  const saveError  = saveState?.error     ?? null;
  const fmtLabel   = FORMAT_LABELS[turn.detected_format ?? ""] ?? turn.detected_format;
  const canConvert = !!fieldName.trim() && !saving && !saved;

  return (
    <div style={{
      marginTop:    6,
      marginLeft:   0,
      padding:      "7px 10px",
      background:   V.bg3,
      borderRadius: "0 0 8px 8px",
      border:       `1px solid ${V.divider}`,
      borderTop:    "none",
      display:      "flex",
      flexDirection: "column" as const,
      gap:          5,
    }}>
      {/* Format badge */}
      <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: V.muted }}>
        <span style={{
          fontSize: 10, padding: "1px 6px", borderRadius: 8,
          background: V.bg2, color: V.primary,
          border: `1px solid ${V.divider}`,
        }}>
          {fmtLabel}
        </span>
        <span>detected — save to dataset?</span>
      </div>

      {/* Field name input + Convert button */}
      <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
        <input
          type="text"
          value={fieldName}
          onChange={(e) => onFieldChange(turnIdx, e.target.value)}
          disabled={saving || saved}
          placeholder="field name"
          style={{
            flex:         1,
            background:   (saving || saved) ? V.bg : V.bg2,
            color:        (saving || saved) ? V.muted : V.text,
            border:       `1px solid ${V.divider}`,
            borderRadius: 4,
            padding:      "4px 8px",
            fontSize:     11,
            fontFamily:   "ui-monospace, monospace",
            outline:      "none",
            cursor:       (saving || saved) ? "not-allowed" : "text",
          }}
        />
        <button
          onClick={() => onConvert(turnIdx, { ...turn, default_field: fieldName })}
          disabled={!canConvert}
          title={saved ? "Already saved" : `Save as ${fmtLabel} to field "${fieldName}"`}
          style={{
            background:   saved  ? V.greenBg  : canConvert ? V.bg2 : V.bg,
            color:        saved  ? V.green     : canConvert ? V.text : V.dim,
            border:       `1px solid ${saved ? V.greenBorder : V.divider}`,
            borderRadius: 4,
            padding:      "4px 10px",
            fontSize:     11,
            fontFamily:   V.font,
            cursor:       canConvert ? "pointer" : "not-allowed",
            flexShrink:   0,
            whiteSpace:   "nowrap" as const,
            display:      "flex",
            alignItems:   "center",
            gap:          4,
          }}
        >
          {saving ? (
            <>
              <span style={{
                width: 10, height: 10, borderRadius: "50%",
                border: `2px solid ${V.divider}`, borderTop: `2px solid ${V.primary}`,
                animation: "prcSpin 0.7s linear infinite", display: "inline-block",
              }} />
              Saving…
            </>
          ) : saved ? (
            `✓ ${saveState?.savedCount ?? ""} ${saveState?.savedLabelType ?? ""}`
          ) : (
            "Convert to FiftyOne ▶"
          )}
        </button>
      </div>

      {saveError && (
        <div style={{ fontSize: 11, color: V.red, padding: "2px 0" }}>
          {saveError}
        </div>
      )}
    </div>
  );
};

export default PerceptronChatPanel;
