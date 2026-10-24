import React, { useEffect, useState } from 'react';

export default function ApiGatewayApp({ activeModel }) {
  const [prompt, setPrompt] = useState('Say hello in one sentence.');
  const [response, setResponse] = useState(null);
  const [busy, setBusy] = useState(false);
  const [keys, setKeys] = useState([]);
  const [issued, setIssued] = useState(null);
  const [label, setLabel] = useState('my-app');
  const [copied, setCopied] = useState(false);
  const baseUrl = 'http://127.0.0.1:8000/v1';

  const refreshKeys = async () => {
    const res = await fetch('/api/gateway/status');
    if (res.ok) setKeys((await res.json()).keys || []);
  };

  useEffect(() => { refreshKeys(); }, []);

  const snippet = `from openai import OpenAI

client = OpenAI(base_url="${baseUrl}", api_key="YOUR_KEY")
print(client.chat.completions.create(
    model="${activeModel || 'your-model'}",
    messages=[{"role": "user", "content": "Hello"}],
).choices[0].message.content)`;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16, maxWidth: 860 }}>
      <div className="card-panel">
        <h3>Give other apps this address</h3>
        <p className="hint" style={{ margin: '8px 0 12px' }}>
          Any tool that speaks the OpenAI API can use your hub. They should call this address and the model you are using now: <strong>{activeModel || 'none yet'}</strong>.
        </p>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <input className="form-input" readOnly value={baseUrl} />
          <button className="btn btn-primary" onClick={() => { navigator.clipboard.writeText(baseUrl); setCopied(true); setTimeout(() => setCopied(false), 1500); }}>
            {copied ? 'Copied' : 'Copy address'}
          </button>
        </div>
      </div>

      <div className="card-panel">
        <h3>Try it from here</h3>
        <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
          <input className="form-input" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
          <button className="btn btn-primary" disabled={busy} onClick={async () => {
            setBusy(true);
            try {
              const res = await fetch('/v1/chat/completions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: activeModel, messages: [{ role: 'user', content: prompt }], max_tokens: 64 }),
              });
              setResponse(await res.json());
            } catch (e) {
              setResponse({ error: e.message });
            } finally {
              setBusy(false);
            }
          }}>
            Send
          </button>
        </div>
        {response && (
          <p style={{ marginTop: 12, lineHeight: 1.5 }}>
            {response.choices?.[0]?.message?.content || response.error || JSON.stringify(response)}
          </p>
        )}
      </div>

      <div className="card-panel">
        <h3>Access keys</h3>
        <p className="hint" style={{ margin: '8px 0 12px' }}>
          This screen does not need a key. If another app sends a key, it must match one you create here.
        </p>
        <div style={{ display: 'flex', gap: 8 }}>
          <input className="form-input" value={label} onChange={(e) => setLabel(e.target.value)} placeholder="Name, e.g. website" />
          <button className="btn btn-primary" onClick={async () => {
            const res = await fetch('/api/gateway/keys', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ label }),
            });
            const data = await res.json();
            setIssued(data.key);
            refreshKeys();
          }}>Create key</button>
        </div>
        {issued && <div className="notice success" style={{ marginTop: 12 }}>Copy this key now. It will not be shown again: {issued.secret}</div>}
        {keys.map((k) => (
          <div key={k.id} style={{ display: 'flex', justifyContent: 'space-between', marginTop: 10 }}>
            <span>{k.label} · {k.secret_preview}</span>
            <button className="btn btn-secondary" onClick={async () => { await fetch(`/api/gateway/keys/${k.id}`, { method: 'DELETE' }); refreshKeys(); }}>Revoke</button>
          </div>
        ))}
      </div>

      <div className="card-panel">
        <h3>Example for a Python app</h3>
        <pre className="hint" style={{ whiteSpace: 'pre-wrap', marginTop: 10 }}>{snippet}</pre>
      </div>
    </div>
  );
}
