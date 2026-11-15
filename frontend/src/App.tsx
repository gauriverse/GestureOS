import { useEffect, useState, useRef } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { useStatus } from "./useStatus";

const MAX_POINTS = 120;

export default function App() {
  const { status, connected } = useStatus();
  const [history, setHistory] = useState<{ t: number; fps: number; latency: number }[]>([]);
  const [usage, setUsage] = useState<Record<string, number>>({});
  const prevStable = useRef<string | null>(null);


  useEffect(() => {
    if (!status) return;
    const cur = status.stable;
    const newGesture = cur && cur !== prevStable.current && cur !== "open_palm" ? cur : null;
    prevStable.current = cur;

    queueMicrotask(() => {
      setHistory((h) => [...h, { t: Date.now(), fps: status.fps, latency: status.latency_ms }].slice(-MAX_POINTS));
      if (newGesture) {
        setUsage((u) => ({ ...u, [newGesture]: (u[newGesture] ?? 0) + 1 }));
      }
    });
  }, [status]);

  return (
    <div style={{ fontFamily: "system-ui", padding: 24, maxWidth: 900, margin: "0 auto" }}>
      <h1>GestureOS</h1>
      <p>{connected ? "Connected" : "Disconnected: is the backend running on port 8000?"}</p>

      {status && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 24 }}>
          <Card label="Mode" value={status.mode} />
          <Card label="Gesture" value={status.stable ?? "-"} />
          <Card label="FPS" value={status.fps.toFixed(1)} />
          <Card label="Latency (ms)" value={status.latency_ms.toFixed(1)} />
        </div>
      )}

      <h3>Latency (ms)</h3>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={history}>
          <XAxis dataKey="t" hide />
          <YAxis />
          <Tooltip />
          <Line type="monotone" dataKey="latency" stroke="#e11d48" dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>

      <h3>Gesture events this session</h3>
      <ul>
        {Object.entries(usage).sort((a, b) => b[1] - a[1]).map(([name, n]) => (
          <li key={name}>{name}: {n}</li>
        ))}
      </ul>
    </div>
  );
}

function Card({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ border: "1px solid #ddd", borderRadius: 8, padding: 12 }}>
      <div style={{ fontSize: 12, color: "#666" }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 600 }}>{value}</div>
    </div>
  );
}