import { useEffect, useState, useRef } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { useStatus } from "./useStatus";

const MAX_POINTS = 120;

type PerClassMetric = {
  precision: number;
  recall: number;
  "f1-score": number;
  support: number;
};

type ClassifierResults = {
  accuracy: number;
  per_class: Record<string, PerClassMetric>;
};

type MetricsData = {
  model: string;
  classifier: string;
  test_session: string;
  results: Record<string, ClassifierResults>;
};

export default function App() {
  const { status, connected } = useStatus();
  const [history, setHistory] = useState<{ t: number; fps: number; latency: number }[]>([]);
  const [usage, setUsage] = useState<Record<string, number>>({});
  const prevStable = useRef<string | null>(null);

  const [metrics, setMetrics] = useState<MetricsData | null>(null);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/metrics")
      .then((r) => r.json())
      .then(setMetrics)
      .catch(() => setMetrics(null));
  }, []);


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

            {metrics && (
        <>
          <h3>Model evaluation: {metrics.model} ({metrics.classifier})</h3>
          <p style={{ color: "#b45309" }}>
            Tested on session {metrics.test_session}. Features were chosen after seeing this session's
            errors, so treat these numbers as optimistic until a fresh session is recorded.
          </p>
          <p>Overall accuracy: {(metrics.results[metrics.classifier].accuracy * 100).toFixed(1)}%</p>
          <table style={{ borderCollapse: "collapse", width: "100%" }}>
            <thead>
              <tr>
                <th align="left">Gesture</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(metrics.results[metrics.classifier].per_class).map(([name, m]: [string, PerClassMetric]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td align="center">{m.precision.toFixed(3)}</td>
                  <td align="center">{m.recall.toFixed(3)}</td>
                  <td align="center">{m["f1-score"].toFixed(3)}</td>
                  <td align="center">{m.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
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