export interface StreamChunk {
  text: string;
  cursor: number;
  done: boolean;
  final_status?: {
    status: "done" | "error";
    latency_ms?: number;
    prompt_tokens?: number;
    completion_tokens?: number;
    /** Non-null when the response contained grounding tags. */
    detected_format?: string | null;
    /** Pre-computed default field name for the Convert UI. */
    default_field?: string;
    error?: string;
  };
}

export interface AskResult {
  status: "started";
  run_id: string;
}

export interface SaveLabelResult {
  saved: boolean;
  label_type: string;
  count: number;
  field: string;
}

/** One turn in the conversation history. */
export interface Turn {
  role: "user" | "assistant";
  /** Plain text content (media is injected by Python on the first user turn). */
  content: string;
  /** Present on completed assistant turns — used by the Convert button. */
  run_id?: string;
  /** Non-null when the response contained parseable grounding tags. */
  detected_format?: string | null;
  /** Pre-computed default field name for the Convert UI. */
  default_field?: string;
}

export interface PanelData {
  filepath?: string;
  sample_id?: string;
  media_type?: string;
  frame_rate?: number | null;
  api_key_missing?: boolean;
}

export interface PanelSchema {
  view?: {
    ask?: string;
    get_stream_chunk?: string;
    save_as_label?: string;
  };
}
