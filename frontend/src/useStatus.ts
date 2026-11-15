import { useEffect, useState } from "react";

export interface Status {
  mode: string;
  raw: string | null;
  stable: string | null;
  confidence: number;
  fps: number;
  latency_ms: number;
}

export function useStatus(url = "ws://127.0.0.1:8000/ws") {
  const [status, setStatus] = useState<Status | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const ws = new WebSocket(url);
    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (e) => setStatus(JSON.parse(e.data));
    return () => ws.close();
  }, [url]);

  return { status, connected };
}