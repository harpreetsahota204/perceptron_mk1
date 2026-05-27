export interface StreamChunk {
  text: string;
  cursor: number;
  done: boolean;
  final_status?: {
    status: "done" | "error";
    latency_ms?: number;
    prompt_tokens?: number;
    completion_tokens?: number;
    error?: string;
  };
}

export interface AskResult {
  status: "started";
  run_id: string;
}

/** One turn in the conversation history. */
export interface Turn {
  role: "user" | "assistant";
  /** Plain text content (media is injected by Python on the first user turn). */
  content: string;
}

export interface PanelData {
  filepath?: string;
  sample_id?: string;
  media_type?: string;
  api_key_missing?: boolean;
}

export interface PanelSchema {
  view?: {
    ask?: string;
    get_stream_chunk?: string;
  };
}
