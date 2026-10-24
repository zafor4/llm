import React, { useEffect, useState } from 'react';
import { Copy, Check, Globe, Key } from 'lucide-react';

export default function ApiGatewayApp({ activeModel }) {
  const [copied, setCopied] = useState(null);
  const [prompt, setPrompt] = useState('Write a Python function to reverse a linked list.');
  const [response, setResponse] = useState(null);
  const [busy, setBusy] = useState(false);
  const [lang, setLang] = useState('python');
  const [keys, setKeys] = useState([]);
  const [issued, setIssued] = useState(null);
  const [label, setLabel] = useState('product-app');
  const baseUrl = 'http://127.0.0.1:8000/v1';

  const refreshKeys = async () => {
    const res = await fetch('/api/gateway/status');
    if (res.ok) {
      const data = await res.json();
      setKeys(data.keys || []);
    }
  };

  useEffect(() => { refreshKeys(); }, []);

  const snippets = {
    python: `from openai import OpenAI\n\nclient = OpenAI(base_url="${baseUrl}", api_key="YOUR_HUB_KEY")\nprint(client.chat.completions.create(\n    model="${activeModel || 'GPT-Mini-256'}",\n    messages=[{"role": "user", "content": "Write quicksort."}],\n).choices[0].message.content)`,
    javascript: `import OpenAI from "openai";\nconst openai = new OpenAI({ baseURL: "${baseUrl}", apiKey: "YOUR_HUB_KEY" });\nconst out = await openai.chat.completions.create({\n  model: "${activeModel || 'GPT-Mini-256'}",\n  messages: [{ role: "user", content: "Hello" }],\n});`,
    curl: `curl -X POST "${baseUrl}/chat/completions" -H "Content-Type: application/json" -H "Authorization: Bearer YOUR_HUB_KEY" -d "{\\"model\\":\\"${activeModel || 'GPT-Mini-256'}\\",\\"messages\\":[{\\"role\\":\\"user\\",\\"content\\":\\"Hello\\"}]}"`,
  };

  const copy = (text, key) => {
    navigator.clipboard.writeText(text);
    setCopied(key);
    setTimeout(() => setCopied(null), 2000);
  };

  const test = async () => {
    setBusy(true);
    try {
      const res = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: activeModel,
          messages: [{ role: 'user', content: prompt }],
          max_tokens: 64,
        }),
      });
      setResponse(await res.json());
    } catch (e) {
      setResponse({ error: e.message });
    } finally {
      setBusy(false);
    }
  };

  const issue = async () => {
    const res = await fetch('/api/gateway/keys', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label }),
    });
    const data = await res.json();
    setIssued(data.key);
    refreshKeys();
  };

  const revoke = async (id) => {
    await fetch(`/api/gateway/keys/${id}`, { method: 'DELETE' });
    refreshKeys();
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(260px,1fr))', gap: 12 }}>
        <div className="card-panel">
          <Globe size={16} color="var(--accent-blue)" />
          <div style={{ fontFamily: 'var(--font-mono)', marginTop: 8 }}>{baseUrl}</div>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Drop-in for api.openai.com/v1</p>
        </div>
        <div className="card-panel">
          <Key size={16} color="var(--accent-emerald)" />
          <div style={{ marginTop: 8 }}>{activeModel || 'none loaded'}</div>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Currently routed engine</p>
        </div>
      </div>

      <div className="card-panel">
        <h3>Shareable keys</h3>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', margin: '8px 0' }}>
          Local playground calls need no key. If a Bearer token is sent, it must match an issued key.
        </p>
        <div style={{ display: 'flex', gap: 8 }}>
          <input className="form-input" value={label} onChange={(e) => setLabel(e.target.value)} />
          <button className="btn btn-primary" onClick={issue}>Issue key</button>
        </div>
        {issued && (
          <div className="card-panel" style={{ marginTop: 10, background: 'var(--bg-input)' }}>
            Copy this secret now: <code>{issued.secret}</code>
          </div>
        )}
        {keys.map((k) => (
          <div key={k.id} style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8, fontSize: '0.85rem' }}>
            <span>{k.label} · {k.secret_preview}</span>
            <button className="btn btn-secondary" onClick={() => revoke(k.id)}>Revoke</button>
          </div>
        ))}
      </div>

      <div className="card-panel">
        <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
          {Object.keys(snippets).map((k) => (
            <button key={k} className={`btn ${lang === k ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setLang(k)}>{k}</button>
          ))}
          <button className="btn btn-secondary" onClick={() => copy(snippets[lang], lang)}>
            {copied === lang ? <Check size={14} /> : <Copy size={14} />} Copy
          </button>
        </div>
        <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.82rem' }}>{snippets[lang]}</pre>
      </div>

      <div className="card-panel">
        <h3>Live /v1 tester</h3>
        <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
          <input className="form-input" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
          <button className="btn btn-primary" disabled={busy} onClick={test}>Send</button>
        </div>
        {response && <pre style={{ marginTop: 12, whiteSpace: 'pre-wrap', fontSize: '0.8rem' }}>{JSON.stringify(response, null, 2)}</pre>}
      </div>
    </div>
  );
}
