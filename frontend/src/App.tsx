import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { useStatus } from "./useStatus";

const MAX_POINTS = 120; // about 12 seconds at 10 updates per second

export default function App() {
  const { status, connected } = useStatus();
  const [history, setHistory] = useState<{ t: number; fps: number; latency: number }[]>([]);
  const [usage, setUsage] = useState<Record<string, number>>({});

  // Build the time series and count gesture usage each time a new status arrives.
  // Schedule the state updates on the next tick so React does not trigger a
  // synchronous render cascade directly from this effect.
  useEffect(() => {
    if (!status) return;

    const timeoutId = window.setTimeout(() => {
      setHistory((h) => [...h, { t: Date.now(), fps: status.fps, latency: status.latency_ms }].slice(-MAX_POINTS));

      if (status.stable && status.stable !== "open_palm") {
        setUsage((u) => ({ ...u, [status.stable as string]: (u[status.stable as string] ?? 0) + 1 }));
      }
    }, 0);

    return () => window.clearTimeout(timeoutId);
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

      <h3>Gesture usage this session</h3>
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