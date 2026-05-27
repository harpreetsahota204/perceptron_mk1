import { useCallback } from "react";
import { usePanelEvent } from "@fiftyone/operators";
import type { AskResult, SaveLabelResult, StreamChunk, Turn } from "../types";

interface PanelUris {
  ask:              string;
  get_stream_chunk: string;
  save_as_label:    string;
}

/**
 * Bridge to the Python PerceptronChatPanel methods.
 *
 * Each method wraps ``usePanelEvent`` in a Promise so callers can use
 * async/await. Python-side ``{error}`` responses surface as rejected Promises.
 */
export function usePanelClient(uris: PanelUris) {
  const handleEvent = usePanelEvent();

  const call = useCallback(
    <T>(methodName: string, uri: string, params: Record<string, unknown>): Promise<T> =>
      new Promise((resolve, reject) => {
        handleEvent(methodName, {
          operator: uri,
          params,
          callback: (result: any) => {
            const r = result?.result as (T & { error?: string }) | undefined;
            if (r?.error) reject(new Error(r.error));
            else resolve(r as T);
          },
        });
      }),
    [handleEvent]
  );

  const ask = useCallback(
    (params: {
      filepath: string;
      media_type: string;
      question: string;
      history: Turn[];
      enable_thinking: boolean;
    }) => call<AskResult>("ask", uris.ask, params),
    [call, uris.ask]
  );

  const getStreamChunk = useCallback(
    (run_id: string, cursor: number) =>
      call<StreamChunk>("get_stream_chunk", uris.get_stream_chunk, { run_id, cursor }),
    [call, uris.get_stream_chunk]
  );

  const saveAsLabel = useCallback(
    (params: {
      run_id: string;
      sample_id: string;
      field_name: string;
      detected_format: string;
      frame_rate: number | null;
    }) => call<SaveLabelResult>("save_as_label", uris.save_as_label, params),
    [call, uris.save_as_label]
  );

  return { ask, getStreamChunk, saveAsLabel };
}
