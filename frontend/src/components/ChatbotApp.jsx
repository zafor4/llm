import React, { useEffect, useRef, useState } from 'react';
import { Send, Trash2, Sliders, Sparkles, Copy, Check, Bot } from 'lucide-react';

export default function ChatbotApp({ activeModel, modelsList, setActiveModel, onRefresh }) {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Playground chat. History is sent to the active hub engine. Product apps should use /v1 instead.' },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(64);
  const [systemPrompt, setSystemPrompt] = useState('You are a concise coding assistant.');
  const [showSettings, setShowSettings] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState(null);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const switchModel = async (id) => {
    setActiveModel?.(id);
    await fetch('/api/models/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_name: id, activate: true }),
    });
    onRefresh?.();
  };

  const handleSend = async (textToSend = null) => {
    const query = textToSend || input;
    if (!query.trim() || loading) return;
    const userMsg = { role: 'user', content: query.trim() };
    const next = [...messages, userMsg];
    setMessages(next);
    setInput('');
    setLoading(true);
    try {
      const history = next
        .filter((m) => m.role === 'user' || m.role === 'assistant')
        .slice(0, -1)
        .map((m) => ({ role: m.role, content: m.content }));
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: query.trim(),
          history,
          temperature,
          max_tokens: maxTokens,
          system_prompt: systemPrompt,
          model: activeModel,
        }),
      });
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: data.content || data.detail || 'No response', latency: data.latency_sec, model: data.model_used },
      ]);
    } catch (e) {
      setMessages((prev) => [...prev, { role: 'assistant', content: `Error: ${e.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 120px)', maxWidth: 960, margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
        <select className="form-select" style={{ width: 'auto' }} value={activeModel || ''} onChange={(e) => switchModel(e.target.value)}>
          {(modelsList || []).map((m) => (
            <option key={m.id || m} value={m.id || m}>{m.name || m}</option>
          ))}
        </select>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-secondary" onClick={() => setShowSettings(!showSettings)}><Sliders size={14} /> Parameters</button>
          <button className="btn btn-secondary" onClick={() => setMessages([])}><Trash2 size={14} /> Clear</button>
        </div>
      </div>

      {showSettings && (
        <div className="card-panel" style={{ marginBottom: 12 }}>
          <label>System prompt</label>
          <input className="form-input" value={systemPrompt} onChange={(e) => setSystemPrompt(e.target.value)} />
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 10 }}>
            <div>
              <span>Temperature {temperature.toFixed(2)}</span>
              <input type="range" min="0.1" max="1.5" step="0.05" value={temperature} onChange={(e) => setTemperature(+e.target.value)} style={{ width: '100%' }} />
            </div>
            <div>
              <span>Max tokens {maxTokens}</span>
              <input type="range" min="16" max="256" step="8" value={maxTokens} onChange={(e) => setMaxTokens(+e.target.value)} style={{ width: '100%' }} />
            </div>
          </div>
        </div>
      )}

      <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 14 }}>
        {messages.map((m, idx) => (
          <div key={idx} style={{ alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start', maxWidth: '85%' }}>
            <div className="card-panel" style={{ padding: 14, background: m.role === 'user' ? 'linear-gradient(135deg,#1e3a8a,#2563eb)' : undefined }}>
              <div style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>
              {m.role !== 'user' && (
                <div style={{ marginTop: 8, fontSize: '0.7rem', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between' }}>
                  <span>{m.model || activeModel} {m.latency ? `· ${m.latency}s` : ''}</span>
                  <button style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer' }} onClick={() => { navigator.clipboard.writeText(m.content); setCopiedIndex(idx); }}>
                    {copiedIndex === idx ? <Check size={12} /> : <Copy size={12} />}
                  </button>
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && <div className="card-panel" style={{ width: 'fit-content' }}><Bot size={14} /> generating…</div>}
        <div ref={endRef} />
      </div>

      <div style={{ display: 'flex', gap: 8, margin: '10px 0', overflowX: 'auto' }}>
        {['def quick_sort(arr):', 'def binary_search(arr, target):', 'Explain residual connections'].map((p) => (
          <button key={p} className="btn btn-secondary" style={{ fontSize: '0.75rem' }} onClick={() => handleSend(p)}>
            <Sparkles size={11} /> {p.slice(0, 22)}
          </button>
        ))}
      </div>

      <div style={{ display: 'flex', gap: 10, background: 'var(--bg-input)', padding: 8, borderRadius: 12, border: '1px solid var(--border-subtle)' }}>
        <input className="form-input" style={{ border: 'none', background: 'transparent' }} value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && handleSend()} placeholder="Message the active engine…" />
        <button className="btn btn-primary" disabled={loading || !input.trim()} onClick={() => handleSend()}><Send size={16} /></button>
      </div>
    </div>
  );
}
