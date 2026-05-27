import React, { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { usePanelClient } from "./hooks/usePanelClient";
import type { PanelData, PanelSchema, Turn } from "./types";

// ---------------------------------------------------------------------------
// Session persistence — survives modal close/reopen within the browser tab.
// Keyed by sample_id so navigating back to a sample restores its conversation.
// ---------------------------------------------------------------------------

const SESSION_PREFIX = "perceptronChat:";

interface StoredSession {
  turns: Turn[];
  enableThinking: boolean;
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
// Inject global styles (spin keyframe + markdown scoping) — once per page load.
// ---------------------------------------------------------------------------

let _stylesInjected = false;
function ensureStyles() {
  if (_stylesInjected) return;
  _stylesInjected = true;
  const el = document.createElement("style");
  el.textContent = `
    @keyframes prcSpin  { to { transform: rotate(360deg); } }
    @keyframes prcBlink { 0%,100%{opacity:1} 50%{opacity:0} }

    /* Scoped markdown styles — all rules under .prc-md */
    .prc-md { font-size: 13px; line-height: 1.7; color: var(--fo-palette-text-primary); word-break: break-word; }
    .prc-md p  { margin: 0 0 8px; }
    .prc-md p:last-child { margin-bottom: 0; }
    .prc-md h1,.prc-md h2,.prc-md h3 { margin: 12px 0 5px; font-weight: 600; }
    .prc-md ul,.prc-md ol { margin: 0 0 8px; padding-left: 18px; }
    .prc-md li { margin-bottom: 2px; }
    .prc-md code {
      font-family: ui-monospace, monospace; font-size: 12px;
      background: var(--fo-palette-background-level2);
      color: var(--fo-palette-primary-main);
      padding: 1px 4px; border-radius: 3px;
    }
    .prc-md pre {
      background: var(--fo-palette-background-level2);
      border: 1px solid var(--fo-palette-divider);
      border-radius: 4px; padding: 8px 10px; overflow-x: auto; margin: 0 0 8px;
    }
    .prc-md pre code { background: none; padding: 0; color: var(--fo-palette-text-primary); }
    .prc-md blockquote {
      margin: 0 0 8px; padding: 3px 10px;
      border-left: 3px solid var(--fo-palette-primary-main);
      color: var(--fo-palette-text-secondary);
    }
    .prc-md strong { font-weight: 600; }
    .prc-md em    { font-style: italic; }
    .prc-md a     { color: var(--fo-palette-primary-main); }
  `;
  document.head.appendChild(el);
}

// ---------------------------------------------------------------------------
// Design tokens — reference FiftyOne CSS variables for automatic theming.
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
};

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

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
  };
  const { ask, getStreamChunk } = usePanelClient(uris);

  // Current sample context pushed by Python lifecycle hooks.
  const [filepath,  setFilepath]  = useState("");
  const [sampleId,  setSampleId]  = useState("");
  const [mediaType, setMediaType] = useState("image");

  // Conversation turns — stored per sample in sessionStorage.
  const [turns,     setTurns]     = useState<Turn[]>([]);

  // Input state.
  const [question,       setQuestion]       = useState("");
  const [enableThinking, setEnableThinking] = useState(false);

  // Streaming state.
  type StreamState = "idle" | "running" | "done" | "error";
  const [streamState,    setStreamState]    = useState<StreamState>("idle");
  const [streamingText,  setStreamingText]  = useState("");   // current assistant turn in progress
  const [streamError,    setStreamError]    = useState<string | null>(null);
  const [latencyMs,      setLatencyMs]      = useState<number | null>(null);
  const [promptTokens,   setPromptTokens]   = useState<number | null>(null);
  const [completionTok,  setCompletionTok]  = useState<number | null>(null);

  const runIdRef     = useRef("");
  const cursorRef    = useRef(0);
  const prevSampleId = useRef("");
  const scrollRef    = useRef<HTMLDivElement>(null);
  const inputRef     = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { ensureStyles(); }, []);

  // ── Sync sample from Python lifecycle props ──────────────────────────────
  useEffect(() => {
    const newPath = data?.filepath   ?? "";
    const newId   = data?.sample_id  ?? "";
    const newMt   = data?.media_type ?? "image";
    if (!newId || !newPath) return;
    if (newId === prevSampleId.current) return;
    prevSampleId.current = newId;

    setFilepath(newPath);
    setSampleId(newId);
    setMediaType(newMt);
    setQuestion("");
    setStreamingText("");
    setStreamState("idle");
    setStreamError(null);
    setLatencyMs(null);
    setPromptTokens(null);
    setCompletionTok(null);
    runIdRef.current  = "";
    cursorRef.current = 0;

    // Restore conversation for this sample or start fresh.
    const cached = loadSession(newId);
    if (cached) {
      setTurns(cached.turns);
      setEnableThinking(cached.enableThinking);
    } else {
      setTurns([]);
    }
  }, [data?.filepath, data?.sample_id]); // eslint-disable-line

  // ── Persist turns to sessionStorage ──────────────────────────────────────
  useEffect(() => {
    if (!sampleId) return;
    saveSession(sampleId, { turns, enableThinking });
  }, [turns, enableThinking, sampleId]);

  // ── Stream polling — every 250 ms while inference is running ─────────────
  useEffect(() => {
    if (streamState !== "running") return;
    const id = setInterval(() => {
      getStreamChunk(runIdRef.current, cursorRef.current)
        .then((chunk) => {
          if (chunk.text) {
            setStreamingText((prev) => prev + chunk.text);
            cursorRef.current = chunk.cursor;
          }
          if (chunk.done) {
            clearInterval(id);
            const fs = chunk.final_status;
            if (fs?.status === "error") {
              setStreamError(fs.error ?? "Inference failed.");
              setStreamState("error");
            } else {
              setLatencyMs(fs?.latency_ms ?? null);
              setPromptTokens(fs?.prompt_tokens ?? null);
              setCompletionTok(fs?.completion_tokens ?? null);
              setStreamState("done");
              // Commit the completed assistant turn to history.
              setStreamingText((text) => {
                setTurns((prev) => {
                  const last = prev[prev.length - 1];
                  // The user turn was appended optimistically on send;
                  // now append the completed assistant turn.
                  if (!last || last.role !== "assistant") {
                    return [...prev, { role: "assistant", content: text }];
                  }
                  return prev;
                });
                return ""; // clear streaming buffer
              });
            }
          }
        })
        .catch(() => {});
    }, 250);
    return () => clearInterval(id);
  }, [streamState]); // eslint-disable-line

  // ── Auto-scroll to bottom on new content ─────────────────────────────────
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
    setPromptTokens(null);
    setCompletionTok(null);
    cursorRef.current = 0;

    // Optimistically append the user turn to the visible history.
    const newUserTurn: Turn = { role: "user", content: q };
    const historyForApi = [...turns]; // history before this question
    setTurns((prev) => [...prev, newUserTurn]);

    try {
      const result = await ask({
        filepath,
        media_type: mediaType,
        question: q,
        history: historyForApi,
        enable_thinking: enableThinking,
      });
      runIdRef.current = result.run_id;
    } catch (e: any) {
      setStreamError(e?.message ?? "Request failed.");
      setStreamState("error");
    }
  }, [question, streamState, filepath, mediaType, turns, enableThinking, ask]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

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

  const canSend = !!question.trim() && streamState !== "running";
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
          <div key={i} style={{
            display: "flex",
            justifyContent: turn.role === "user" ? "flex-end" : "flex-start",
            marginBottom: 10,
          }}>
            <div style={{
              maxWidth: "88%",
              background: turn.role === "user" ? V.primary : V.bg2,
              color:      turn.role === "user" ? V.textInv  : V.text,
              borderRadius: turn.role === "user" ? "12px 12px 4px 12px" : "12px 12px 12px 4px",
              padding:    "8px 12px",
              fontSize:   13,
              lineHeight: 1.5,
            }}>
              {turn.role === "assistant" ? (
                <div className="prc-md">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {turn.content}
                  </ReactMarkdown>
                </div>
              ) : (
                turn.content
              )}
            </div>
          </div>
        ))}

        {/* Streaming assistant turn in progress */}
        {streamState === "running" && (
          <div style={{ display: "flex", justifyContent: "flex-start", marginBottom: 10 }}>
            <div style={{
              maxWidth: "88%", background: V.bg2, color: V.text,
              borderRadius: "12px 12px 12px 4px", padding: "8px 12px",
              fontSize: 13, lineHeight: 1.5,
            }}>
              {streamingText ? (
                <div className="prc-md">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {streamingText}
                  </ReactMarkdown>
                  <span style={{
                    display: "inline-block", width: 2, height: "1em",
                    background: V.primary, marginLeft: 1, verticalAlign: "text-bottom",
                    animation: "prcBlink 1s step-end infinite",
                  }} />
                </div>
              ) : (
                /* Waiting for first token */
                <div style={{ display: "flex", alignItems: "center", gap: 6, color: V.muted }}>
                  <span style={{
                    width: 12, height: 12, borderRadius: "50%",
                    border: `2px solid ${V.divider}`,
                    borderTop: `2px solid ${V.primary}`,
                    animation: "prcSpin 0.7s linear infinite",
                    display: "inline-block", flexShrink: 0,
                  }} />
                  Thinking…
                </div>
              )}
            </div>
          </div>
        )}

        {/* Error display */}
        {streamError && (
          <div style={{ padding: "8px 10px", background: V.redBg, color: V.red,
                        borderRadius: 6, fontSize: 12, marginBottom: 8 }}>
            {streamError}
          </div>
        )}

        {/* Token / latency footer after completion */}
        {streamState === "done" && (latencyMs != null || completionTok != null) && (
          <div style={{ fontSize: 11, color: V.dim, textAlign: "center",
                        padding: "2px 0 8px" }}>
            {[
              completionTok != null && `${completionTok} tokens`,
              latencyMs    != null && `${(latencyMs / 1000).toFixed(1)}s`,
              promptTokens != null && `${promptTokens} in`,
            ].filter(Boolean).join(" · ")}
          </div>
        )}
      </div>

      {/* ── Bottom bar ── */}
      <div style={{ flexShrink: 0, padding: "8px 12px",
                    borderTop: `1px solid ${V.divider}`,
                    background: V.bg2, display: "flex",
                    flexDirection: "column", gap: 7 }}>

        {/* Thinking toggle */}
        <label style={{ display: "flex", alignItems: "center", gap: 6,
                         cursor: "pointer", userSelect: "none",
                         fontSize: 11, color: V.muted }}>
          <input
            type="checkbox"
            checked={enableThinking}
            onChange={(e) => setEnableThinking(e.target.checked)}
            style={{ width: 13, height: 13, cursor: "pointer",
                     accentColor: V.primary, flexShrink: 0 }}
          />
          Enable thinking (slower, better for reasoning)
        </label>

        {/* Textarea + send button */}
        <div style={{ display: "flex", gap: 6, alignItems: "flex-end" }}>
          <textarea
            ref={inputRef}
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
              flex:         1,
              background:   streamState === "running" ? V.bg : V.bg,
              color:        streamState === "running" ? V.dim : V.text,
              border:       `1px solid ${V.divider}`,
              borderRadius: 6,
              padding:      "7px 10px",
              fontSize:     13,
              fontFamily:   V.font,
              lineHeight:   1.5,
              outline:      "none",
              resize:       "none" as const,
              minHeight:    38,
              overflow:     "auto",
              boxSizing:    "border-box" as const,
              cursor:       streamState === "running" ? "not-allowed" : "text",
            }}
          />
          <button
            onClick={handleSend}
            disabled={!canSend}
            title="Send (Enter)"
            style={{
              background:   "none",
              border:       "none",
              padding:      "0 2px",
              cursor:       canSend ? "pointer" : "default",
              color:        streamState === "running"
                              ? V.muted
                              : canSend ? V.primary : V.dim,
              opacity:      canSend || streamState === "running" ? 1 : 0.35,
              lineHeight:   1,
              display:      "flex",
              alignItems:   "center",
              flexShrink:   0,
              marginBottom: 6,
            }}
          >
            {streamState === "running" ? (
              <span style={{
                width: 16, height: 16, borderRadius: "50%",
                border: `2px solid ${V.divider}`,
                borderTop: `2px solid ${V.primary}`,
                animation: "prcSpin 0.7s linear infinite",
                display: "inline-block",
              }} />
            ) : (
              /* Paper plane icon */
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
              </svg>
            )}
          </button>
        </div>

        {/* Clear conversation */}
        {turns.length > 0 && streamState !== "running" && (
          <div style={{ display: "flex", justifyContent: "flex-end" }}>
            <button
              onClick={() => {
                setTurns([]);
                setStreamingText("");
                setStreamState("idle");
                setStreamError(null);
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

export default PerceptronChatPanel;
