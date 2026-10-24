import React, { useState } from 'react';
import { Terminal, Copy, Check, Globe, Send, Play, Sparkles, Key, ExternalLink, Code2 } from 'lucide-react';

export default function ApiGatewayApp({ activeModel }) {
  const [copiedSnippet, setCopiedSnippet] = useState(null);
  const [testPrompt, setTestPrompt] = useState('Write a Python function to reverse a linked list.');
  const [testResponse, setTestResponse] = useState(null);
  const [isCalling, setIsCalling] = useState(false);
  const [selectedLang, setSelectedLang] = useState('python');

  const baseUrl = 'http://127.0.0.1:8000/v1';

  const snippets = {
    python: `from openai import OpenAI

client = OpenAI(
    base_url="${baseUrl}",
    api_key="local-intelligence-token"  # Any string works for local mode
)

response = client.chat.completions.create(
    model="${activeModel}",
    messages=[
        {"role": "user", "content": "Write a Python quicksort function."}
    ],
    max_tokens=128,
    temperature=0.7
)

print(response.choices[0].message.content)`,
    javascript: `import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "${baseUrl}",
  apiKey: "local-intelligence-token",
  dangerouslyAllowBrowser: true,
});

const completion = await openai.chat.completions.create({
  model: "${activeModel}",
  messages: [{ role: "user", content: "Explain how async/await works." }],
});

console.log(completion.choices[0].message.content);`,
    curl: `curl -X POST "${baseUrl}/chat/completions" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${activeModel}",
    "messages": [{"role": "user", "content": "Hello Intelligence Hub!"}],
    "max_tokens": 64
  }'`,
    langchain: `from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="${baseUrl}",
    api_key="local-intelligence-token",
    model="${activeModel}",
    temperature=0.7
)

response = llm.invoke("Summarize the key principles of clean code.")
print(response.content)`
  };

  const copyToClipboard = (text, key) => {
    navigator.clipboard.writeText(text);
    setCopiedSnippet(key);
    setTimeout(() => setCopiedSnippet(null), 2500);
  };

  const handleTestApi = async () => {
    setIsCalling(true);
    setTestResponse(null);
    try {
      const res = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: activeModel,
          messages: [{ role: 'user', content: testPrompt }],
          max_tokens: 64,
          temperature: 0.7
        })
      });
      const data = await res.json();
      setTestResponse(data);
    } catch (err) {
      setTestResponse({ error: err.message });
    } finally {
      setIsCalling(false);
    }
  };

  return (
    <div className="gateway-app">
      <div className="app-header">
        <div>
          <h2>External API Gateway (House of Intelligence)</h2>
          <p>Expose your local and custom intelligence models to external apps via standard OpenAI-compatible endpoints</p>
        </div>
      </div>

      {/* Gateway Status Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>API Base URL</span>
            <Globe size={16} color="var(--accent-blue)" />
          </div>
          <div style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--accent-blue)', fontFamily: 'monospace' }}>
            {baseUrl}
          </div>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>
            Drop-in replacement for <code>https://api.openai.com/v1</code>
          </p>
        </div>

        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Currently Routing</span>
            <Key size={16} color="var(--accent-emerald)" />
          </div>
          <div style={{ fontSize: '1.05rem', fontWeight: 600, color: '#34d399' }}>
            {activeModel}
          </div>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>
            All requests to <code>/v1/chat/completions</code> resolve to this engine
          </p>
        </div>

        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>API Compatibility</span>
            <Code2 size={16} color="var(--accent-purple)" />
          </div>
          <div style={{ fontSize: '1.05rem', fontWeight: 600, color: '#c084fc' }}>
            OpenAI v1 Spec
          </div>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>
            Compatible with Cursor, Continue.dev, LangChain, LlamaIndex
          </p>
        </div>
      </div>

      {/* Code Snippets Integration Section */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem', marginBottom: '1rem' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>External Integration Snippets</h3>
          
          <div style={{ display: 'flex', gap: '0.4rem' }}>
            {['python', 'javascript', 'curl', 'langchain'].map((lang) => (
              <button
                key={lang}
                className={`btn ${selectedLang === lang ? 'btn-primary' : 'btn-secondary'}`}
                style={{ fontSize: '0.75rem', textTransform: 'capitalize', padding: '4px 10px' }}
                onClick={() => setSelectedLang(lang)}
              >
                {lang}
              </button>
            ))}
          </div>
        </div>

        <div style={{ position: 'relative', background: '#0a0d14', borderRadius: '8px', padding: '1.25rem', border: '1px solid var(--border-color)' }}>
          <button
            onClick={() => copyToClipboard(snippets[selectedLang], selectedLang)}
            className="btn btn-secondary"
            style={{ position: 'absolute', top: '12px', right: '12px', fontSize: '0.75rem', padding: '4px 8px' }}
          >
            {copiedSnippet === selectedLang ? <Check size={14} color="#34d399" /> : <Copy size={14} />}
            {copiedSnippet === selectedLang ? 'Copied!' : 'Copy Code'}
          </button>
          <pre style={{ margin: 0, fontFamily: 'monospace', fontSize: '0.85rem', color: '#e2e8f0', overflowX: 'auto', whiteSpace: 'pre-wrap' }}>
            {snippets[selectedLang]}
          </pre>
        </div>
      </div>

      {/* Live Interactive API Tester */}
      <div className="card">
        <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.5rem' }}>Live Gateway Request Tester</h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
          Test sending a real payload through <code>POST /v1/chat/completions</code> right now.
        </p>

        <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1rem' }}>
          <input
            type="text"
            className="input-field"
            value={testPrompt}
            onChange={(e) => setTestPrompt(e.target.value)}
            placeholder="Type a test message for the external gateway..."
            style={{ flex: 1 }}
          />
          <button
            className="btn btn-primary"
            onClick={handleTestApi}
            disabled={isCalling || !testPrompt.trim()}
          >
            {isCalling ? 'Querying Gateway...' : 'Send API Call'}
          </button>
        </div>

        {testResponse && (
          <div style={{ background: '#0a0d14', borderRadius: '8px', padding: '1rem', border: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.5rem', display: 'flex', justifyContent: 'space-between' }}>
              <span>HTTP 200 OK — Response Payload</span>
              <span>Model: {testResponse.model || activeModel}</span>
            </div>
            <pre style={{ margin: 0, fontFamily: 'monospace', fontSize: '0.85rem', color: '#38bdf8', whiteSpace: 'pre-wrap', maxHeight: '250px', overflowY: 'auto' }}>
              {JSON.stringify(testResponse, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}
