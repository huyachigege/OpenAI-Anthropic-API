import { useState, useEffect, useCallback } from "react";

const MODELS = [
  { id: "gpt-5.2", provider: "openai" },
  { id: "gpt-5-mini", provider: "openai" },
  { id: "gpt-5-nano", provider: "openai" },
  { id: "o4-mini", provider: "openai" },
  { id: "o3", provider: "openai" },
  { id: "claude-opus-4-6", provider: "anthropic" },
  { id: "claude-sonnet-4-6", provider: "anthropic" },
  { id: "claude-haiku-4-5", provider: "anthropic" },
];

const ENDPOINTS = [
  {
    method: "GET",
    path: "/v1/models",
    type: "both",
    label: "List Models",
    desc: "Returns all available OpenAI and Anthropic models.",
  },
  {
    method: "POST",
    path: "/v1/chat/completions",
    type: "openai",
    label: "Chat Completions",
    desc: "OpenAI-compatible chat completions. Works with any OpenAI SDK. Supports streaming and tool calls.",
  },
  {
    method: "POST",
    path: "/v1/messages",
    type: "anthropic",
    label: "Messages",
    desc: "Anthropic Messages API. Drop-in replacement for the official Anthropic SDK. Supports streaming and tool use.",
  },
];

const STEPS = [
  {
    num: 1,
    title: "Open Settings",
    desc: "In CherryStudio, go to Settings → Model Provider.",
  },
  {
    num: 2,
    title: "Add a Custom Provider",
    desc: 'Click "Add Provider" and choose either OpenAI or Anthropic as the provider type. Both formats are supported.',
  },
  {
    num: 3,
    title: "Enter Connection Details",
    desc: "Paste your Base URL and API key (your PROXY_API_KEY) into the corresponding fields.",
  },
  {
    num: 4,
    title: "Select a Model & Chat",
    desc: "Save and pick any model from the list. Start chatting — requests are proxied through Replit AI Integrations.",
  },
];

function CopyButton({ text, small }: { text: string; small?: boolean }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      if (navigator.clipboard) {
        await navigator.clipboard.writeText(text);
      } else {
        const el = document.createElement("textarea");
        el.value = text;
        document.body.appendChild(el);
        el.select();
        document.execCommand("copy");
        document.body.removeChild(el);
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // ignore
    }
  }, [text]);

  return (
    <button
      onClick={handleCopy}
      style={{
        background: copied ? "rgba(34,197,94,0.15)" : "rgba(255,255,255,0.07)",
        border: `1px solid ${copied ? "rgba(34,197,94,0.4)" : "rgba(255,255,255,0.12)"}`,
        color: copied ? "#86efac" : "#a1afc9",
        borderRadius: 6,
        padding: small ? "3px 10px" : "5px 14px",
        fontSize: small ? 11 : 12,
        fontFamily: "inherit",
        cursor: "pointer",
        transition: "all 0.15s",
        whiteSpace: "nowrap",
        flexShrink: 0,
      }}
    >
      {copied ? "Copied!" : "Copy"}
    </button>
  );
}

export default function App() {
  const [online, setOnline] = useState<boolean | null>(null);
  const origin = window.location.origin;

  useEffect(() => {
    const check = () => {
      fetch("/api/healthz")
        .then((r) => setOnline(r.ok))
        .catch(() => setOnline(false));
    };
    check();
    const id = setInterval(check, 15000);
    return () => clearInterval(id);
  }, []);

  const curlExample = `curl ${origin}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_PROXY_API_KEY" \\
  -d '{
    "model": "claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'`;

  const s: Record<string, React.CSSProperties> = {
    page: {
      minHeight: "100vh",
      background: "hsl(222,47%,11%)",
      color: "#e2e8f0",
      fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
      fontSize: 14,
    },
    header: {
      borderBottom: "1px solid rgba(255,255,255,0.07)",
      padding: "18px 32px",
      display: "flex",
      alignItems: "center",
      gap: 14,
      position: "sticky" as const,
      top: 0,
      background: "hsl(222,47%,11%)",
      zIndex: 10,
    },
    icon: {
      width: 36,
      height: 36,
      borderRadius: 8,
      background: "linear-gradient(135deg,#6366f1,#a855f7)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontSize: 18,
      flexShrink: 0,
    },
    statusDot: {
      width: 8,
      height: 8,
      borderRadius: "50%",
      background: online === null ? "#fbbf24" : online ? "#22c55e" : "#ef4444",
      boxShadow: `0 0 8px 2px ${online === null ? "rgba(251,191,36,0.5)" : online ? "rgba(34,197,94,0.5)" : "rgba(239,68,68,0.5)"}`,
      flexShrink: 0,
    },
    main: { maxWidth: 820, margin: "0 auto", padding: "40px 24px 80px" },
    sectionTitle: {
      fontSize: 13,
      fontWeight: 600,
      letterSpacing: "0.08em",
      textTransform: "uppercase" as const,
      color: "#64748b",
      marginBottom: 16,
    },
    card: {
      background: "hsl(222,47%,14%)",
      border: "1px solid rgba(255,255,255,0.07)",
      borderRadius: 12,
      padding: "20px 24px",
      marginBottom: 24,
    },
    row: { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 },
    label: { fontSize: 12, color: "#64748b", marginBottom: 6 },
    mono: {
      fontFamily: "ui-monospace, 'Cascadia Code', 'Fira Code', monospace",
      fontSize: 13,
      color: "#c4b5fd",
    },
  };

  return (
    <div style={s.page}>
      {/* Header */}
      <header style={s.header}>
        <div style={s.icon}>⚡</div>
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 700, fontSize: 16, lineHeight: 1.2 }}>AI Proxy</div>
          <div style={{ fontSize: 12, color: "#64748b", marginTop: 2 }}>
            OpenAI + Anthropic dual-compatible reverse proxy
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={s.statusDot} />
          <span style={{ fontSize: 12, color: "#64748b" }}>
            {online === null ? "Checking…" : online ? "Online" : "Offline"}
          </span>
        </div>
      </header>

      <main style={s.main}>
        {/* Connection Details */}
        <div style={{ marginBottom: 32 }}>
          <div style={s.sectionTitle}>Connection Details</div>
          <div style={s.card}>
            <div style={{ marginBottom: 18 }}>
              <div style={s.label}>Base URL</div>
              <div style={{ ...s.row }}>
                <code style={s.mono}>{origin}</code>
                <CopyButton text={origin} small />
              </div>
            </div>
            <div
              style={{
                height: 1,
                background: "rgba(255,255,255,0.06)",
                margin: "16px 0",
              }}
            />
            <div>
              <div style={s.label}>Auth Header</div>
              <div style={s.row}>
                <code style={s.mono}>Authorization: Bearer YOUR_PROXY_API_KEY</code>
                <CopyButton text="Authorization: Bearer YOUR_PROXY_API_KEY" small />
              </div>
              <div style={{ fontSize: 11, color: "#475569", marginTop: 6 }}>
                Replace YOUR_PROXY_API_KEY with the value you set in Replit Secrets as PROXY_API_KEY.
              </div>
            </div>
          </div>
        </div>

        {/* Endpoints */}
        <div style={{ marginBottom: 32 }}>
          <div style={s.sectionTitle}>API Endpoints</div>
          <div style={{ display: "flex", flexDirection: "column" as const, gap: 12 }}>
            {ENDPOINTS.map((ep) => (
              <div key={ep.path} style={s.card}>
                <div style={{ ...s.row, marginBottom: 10 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" as const }}>
                    <span
                      style={{
                        background: ep.method === "GET" ? "rgba(34,197,94,0.15)" : "rgba(168,85,247,0.15)",
                        color: ep.method === "GET" ? "#86efac" : "#c4b5fd",
                        border: `1px solid ${ep.method === "GET" ? "rgba(34,197,94,0.3)" : "rgba(168,85,247,0.3)"}`,
                        borderRadius: 5,
                        padding: "2px 8px",
                        fontSize: 11,
                        fontWeight: 700,
                        fontFamily: "ui-monospace, monospace",
                        letterSpacing: "0.04em",
                      }}
                    >
                      {ep.method}
                    </span>
                    <code style={{ ...s.mono, fontSize: 14 }}>
                      {origin}{ep.path}
                    </code>
                    <span
                      style={{
                        background:
                          ep.type === "openai"
                            ? "rgba(59,130,246,0.12)"
                            : ep.type === "anthropic"
                            ? "rgba(245,158,11,0.12)"
                            : "rgba(100,116,139,0.15)",
                        color:
                          ep.type === "openai"
                            ? "#93c5fd"
                            : ep.type === "anthropic"
                            ? "#fcd34d"
                            : "#94a3b8",
                        border: `1px solid ${
                          ep.type === "openai"
                            ? "rgba(59,130,246,0.25)"
                            : ep.type === "anthropic"
                            ? "rgba(245,158,11,0.25)"
                            : "rgba(100,116,139,0.25)"
                        }`,
                        borderRadius: 5,
                        padding: "2px 8px",
                        fontSize: 11,
                        fontWeight: 500,
                      }}
                    >
                      {ep.type === "openai" ? "OpenAI" : ep.type === "anthropic" ? "Anthropic" : "Both"}
                    </span>
                  </div>
                  <CopyButton text={`${origin}${ep.path}`} small />
                </div>
                <div style={{ fontSize: 13, color: "#94a3b8", lineHeight: 1.6 }}>{ep.desc}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Available Models */}
        <div style={{ marginBottom: 32 }}>
          <div style={s.sectionTitle}>Available Models</div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
              gap: 10,
            }}
          >
            {MODELS.map((m) => (
              <div
                key={m.id}
                style={{
                  background: "hsl(222,47%,14%)",
                  border: "1px solid rgba(255,255,255,0.07)",
                  borderRadius: 10,
                  padding: "12px 16px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: 8,
                }}
              >
                <code style={{ ...s.mono, fontSize: 12, color: "#e2e8f0" }}>{m.id}</code>
                <span
                  style={{
                    background:
                      m.provider === "openai" ? "rgba(59,130,246,0.12)" : "rgba(245,158,11,0.12)",
                    color: m.provider === "openai" ? "#93c5fd" : "#fcd34d",
                    border: `1px solid ${m.provider === "openai" ? "rgba(59,130,246,0.25)" : "rgba(245,158,11,0.25)"}`,
                    borderRadius: 4,
                    padding: "1px 7px",
                    fontSize: 10,
                    fontWeight: 600,
                    letterSpacing: "0.04em",
                    whiteSpace: "nowrap" as const,
                  }}
                >
                  {m.provider === "openai" ? "OpenAI" : "Anthropic"}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* CherryStudio Setup */}
        <div style={{ marginBottom: 32 }}>
          <div style={s.sectionTitle}>CherryStudio Setup</div>
          <div style={s.card}>
            <div style={{ display: "flex", flexDirection: "column" as const, gap: 20 }}>
              {STEPS.map((step, i) => (
                <div key={step.num} style={{ display: "flex", gap: 16 }}>
                  <div
                    style={{
                      width: 32,
                      height: 32,
                      borderRadius: "50%",
                      background: `linear-gradient(135deg,${
                        ["#6366f1","#8b5cf6","#a855f7","#c026d3"][i]
                      },${
                        ["#8b5cf6","#a855f7","#c026d3","#e879f9"][i]
                      })`,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontWeight: 700,
                      fontSize: 13,
                      flexShrink: 0,
                    }}
                  >
                    {step.num}
                  </div>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 4 }}>{step.title}</div>
                    <div style={{ fontSize: 13, color: "#94a3b8", lineHeight: 1.6 }}>{step.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Quick Test */}
        <div style={{ marginBottom: 32 }}>
          <div style={s.sectionTitle}>Quick Test (curl)</div>
          <div
            style={{
              ...s.card,
              padding: 0,
              overflow: "hidden",
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "12px 20px",
                borderBottom: "1px solid rgba(255,255,255,0.06)",
              }}
            >
              <span style={{ fontSize: 12, color: "#64748b" }}>Shell</span>
              <CopyButton text={curlExample} small />
            </div>
            <pre
              style={{
                margin: 0,
                padding: "20px",
                fontFamily: "ui-monospace, 'Cascadia Code', monospace",
                fontSize: 12.5,
                lineHeight: 1.7,
                overflowX: "auto" as const,
                color: "#e2e8f0",
              }}
            >
              <span style={{ color: "#7dd3fc" }}>curl</span>
              {" "}
              <span style={{ color: "#c4b5fd" }}>{origin}/v1/chat/completions</span>
              {" \\\n  "}
              <span style={{ color: "#86efac" }}>-H</span>
              {" "}
              <span style={{ color: "#fcd34d" }}>"Content-Type: application/json"</span>
              {" \\\n  "}
              <span style={{ color: "#86efac" }}>-H</span>
              {" "}
              <span style={{ color: "#fcd34d" }}>"Authorization: Bearer YOUR_PROXY_API_KEY"</span>
              {" \\\n  "}
              <span style={{ color: "#86efac" }}>-d</span>
              {" "}
              <span style={{ color: "#fcd34d" }}>{`'{\n    "model": "claude-sonnet-4-6",\n    "messages": [{"role": "user", "content": "Hello!"}]\n  }'`}</span>
            </pre>
          </div>
        </div>

        {/* Footer */}
        <div
          style={{
            textAlign: "center" as const,
            fontSize: 12,
            color: "#334155",
            borderTop: "1px solid rgba(255,255,255,0.05)",
            paddingTop: 24,
          }}
        >
          Built on Replit · Powered by Replit AI Integrations (OpenAI + Anthropic) · No API keys required
        </div>
      </main>
    </div>
  );
}
