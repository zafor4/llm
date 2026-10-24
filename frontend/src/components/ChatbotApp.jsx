import React, { useEffect, useRef, useState } from 'react';
import { Send } from 'lucide-react';

export default function ChatbotApp({ activeModel, modelsList, setActiveModel, onRefresh }) {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'This is a test chat. Other apps should use Connect, not this screen.' },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(64);
  const [showSettings, setShowSettings] = useState(false);
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
          model: activeModel,
        }),
      });
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: data.content || data.detail || 'No reply', latency: data.latency_sec, model: data.model_used },
      ]);
    } catch (e) {
      setMessages((prev) => [...prev, { role: 'assistant', content: `Could not reach the hub: ${e.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 130px)', maxWidth: 820, margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, marginBottom: 12, flexWrap: 'wrap' }}>
        <label className="field-label" style={{ margin: 0, display: 'flex', alignItems: 'center', gap: 8 }}>
          Talking to
          <select className="form-select" style={{ width: 'auto' }} value={activeModel || ''} onChange={(e) => switchModel(e.target.value)}>
            {(modelsList || []).map((m) => (
              <option key={m.id || m} value={m.id || m}>{m.name || m}</option>
            ))}
          </select>
        </label>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-secondary" onClick={() => setShowSettings(!showSettings)}>Options</button>
          <button className="btn btn-secondary" onClick={() => setMessages([])}>Clear</button>
        </div>
      </div>

      {showSettings && (
        <div className="card-panel" style={{ marginBottom: 12 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div>
              <div className="field-label">Creativity {temperature.toFixed(1)}</div>
              <input type="range" min="0.1" max="1.5" step="0.1" value={temperature} onChange={(e) => setTemperature(+e.target.value)} style={{ width: '100%' }} />
            </div>
            <div>
              <div className="field-label">Reply length {maxTokens}</div>
              <input type="range" min="16" max="256" step="8" value={maxTokens} onChange={(e) => setMaxTokens(+e.target.value)} style={{ width: '100%' }} />
            </div>
          </div>
        </div>
      )}

      <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 12 }}>
        {messages.map((m, idx) => (
          <div key={idx} style={{ alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start', maxWidth: '88%' }}>
            <div className="card-panel" style={{ padding: 14, background: m.role === 'user' ? 'rgba(37, 99, 235, 0.35)' : undefined }}>
              <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.5 }}>{m.content}</div>
              {m.role !== 'user' && m.model && <div className="hint" style={{ marginTop: 8 }}>{m.model}{m.latency ? ` · ${m.latency}s` : ''}</div>}
            </div>
          </div>
        ))}
        {loading && <div className="hint">Thinking…</div>}
        <div ref={endRef} />
      </div>

      <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
        {['Write a short hello function', 'What can you do?'].map((p) => (
          <button key={p} className="btn btn-secondary" onClick={() => handleSend(p)}>{p}</button>
        ))}
      </div>

      <div className="card-panel" style={{ marginTop: 10, padding: 10, display: 'flex', gap: 8 }}>
        <input
          className="form-input"
          style={{ border: 'none', background: 'transparent' }}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Type a message"
        />
        <button className="btn btn-primary" disabled={loading || !input.trim()} onClick={() => handleSend()}>
          <Send size={16} />
        </button>
      </div>
    </div>
  );
}
