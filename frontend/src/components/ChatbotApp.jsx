import React, { useState, useRef, useEffect } from 'react';
import { Send, Trash2, Sliders, Sparkles, Copy, Check, Bot } from 'lucide-react';

export default function ChatbotApp({ activeModel, modelsList, setActiveModel }) {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: `Hello! I am your conversational coding assistant powered by **${activeModel}**. Ask me to write algorithms, explain code, or generate Python routines.`,
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(64);
  const [showSettings, setShowSettings] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState(null);

  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleSend = async (textToSend = null) => {
    const query = textToSend || input;
    if (!query.trim() || loading) return;

    const userMsg = { role: 'user', content: query.trim() };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: query.trim(),
          temperature,
          max_tokens: maxTokens,
        }),
      });

      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.content,
          latency: data.latency_sec,
          model: data.model_used,
        },
      ]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Error: ${e.message}. Ensure backend server is running.` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text, idx) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(idx);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 120px)', maxWidth: '960px', margin: '0 auto' }}>
      {/* Chat Top Controls */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', paddingBottom: '12px', borderBottom: '1px solid var(--border-subtle)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>Engine:</span>
          <select 
            className="form-select" 
            style={{ width: 'auto', padding: '6px 12px', fontSize: '0.8125rem' }}
            value={activeModel}
            onChange={(e) => setActiveModel(e.target.value)}
          >
            {modelsList.map((m) => (
              <option key={m.id} value={m.name}>{m.name}</option>
            ))}
          </select>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button 
            className="btn btn-secondary" 
            style={{ padding: '6px 12px', fontSize: '0.8125rem' }}
            onClick={() => setShowSettings(!showSettings)}
          >
            <Sliders size={14} />
            Parameters
          </button>
          <button 
            className="btn btn-secondary" 
            style={{ padding: '6px 12px', fontSize: '0.8125rem', color: 'var(--accent-rose)' }}
            onClick={() => setMessages([])}
          >
            <Trash2 size={14} />
            Clear
          </button>
        </div>
      </div>

      {/* Collapsible Parameters Drawer */}
      {showSettings && (
        <div className="card-panel" style={{ padding: '16px', marginBottom: '14px', background: 'var(--bg-input)' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '6px', color: 'var(--text-secondary)' }}>
                <span>Temperature (Creativity):</span>
                <span>{temperature.toFixed(2)}</span>
              </div>
              <input 
                type="range" 
                min="0.1" 
                max="1.5" 
                step="0.05" 
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--accent-blue)' }}
              />
            </div>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '6px', color: 'var(--text-secondary)' }}>
                <span>Max New Tokens:</span>
                <span>{maxTokens}</span>
              </div>
              <input 
                type="range" 
                min="16" 
                max="128" 
                step="8" 
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--accent-blue)' }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Messages Scroll Area */}
      <div style={{ flex: 1, overflowY: 'auto', paddingRight: '6px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {messages.map((m, idx) => {
          const isUser = m.role === 'user';
          return (
            <div 
              key={idx} 
              style={{ 
                display: 'flex', 
                gap: '12px', 
                alignSelf: isUser ? 'flex-end' : 'flex-start',
                maxWidth: '85%'
              }}
            >
              {!isUser && (
                <div style={{ width: '32px', height: '32px', borderRadius: '50%', background: 'linear-gradient(135deg, var(--accent-blue), var(--accent-purple))', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, color: 'white' }}>
                  <Bot size={16} />
                </div>
              )}

              <div 
                style={{ 
                  background: isUser ? 'linear-gradient(135deg, #1e3a8a, #2563eb)' : 'var(--bg-card)', 
                  border: isUser ? 'none' : '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-md)',
                  padding: '12px 16px',
                  color: '#ffffff',
                  fontSize: '0.9375rem',
                  lineHeight: 1.55,
                  position: 'relative',
                }}
              >
                <div style={{ whiteSpace: 'pre-wrap', fontFamily: m.content.includes('def ') ? 'var(--font-mono)' : 'var(--font-sans)', fontSize: m.content.includes('def ') ? '0.85rem' : '0.9375rem' }}>
                  {m.content}
                </div>

                {!isUser && (
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '8px', paddingTop: '6px', borderTop: '1px solid rgba(255,255,255,0.06)', fontSize: '0.6875rem', color: 'var(--text-muted)' }}>
                    <span>{m.model || activeModel} {m.latency ? `• ${m.latency}s` : ''}</span>
                    <button 
                      style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px' }}
                      onClick={() => copyToClipboard(m.content, idx)}
                    >
                      {copiedIndex === idx ? <Check size={12} color="var(--accent-emerald)" /> : <Copy size={12} />}
                      <span>{copiedIndex === idx ? 'Copied' : 'Copy'}</span>
                    </button>
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {loading && (
          <div style={{ display: 'flex', gap: '12px', alignSelf: 'flex-start' }}>
            <div style={{ width: '32px', height: '32px', borderRadius: '50%', background: 'linear-gradient(135deg, var(--accent-blue), var(--accent-purple))', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white' }}>
              <Bot size={16} />
            </div>
            <div className="card-panel" style={{ padding: '10px 16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span className="pulse-dot" style={{ background: 'var(--accent-blue)' }}></span>
              <span style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>{activeModel} generating tokens...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Preset Chips */}
      <div style={{ display: 'flex', gap: '8px', margin: '10px 0', overflowX: 'auto', paddingBottom: '4px' }}>
        {[
          'def quick_sort(arr: list[int]) -> list[int]:',
          'def binary_search(arr: list[int], target: int) -> int:',
          'def fibonacci(n: int) -> int:',
          'def is_palindrome(s: str) -> bool:',
        ].map((preset, i) => (
          <button 
            key={i} 
            className="btn btn-secondary" 
            style={{ fontSize: '0.75rem', padding: '4px 10px', whiteSpace: 'nowrap', fontFamily: 'var(--font-mono)' }}
            onClick={() => handleSend(preset)}
          >
            <Sparkles size={11} color="var(--accent-blue)" />
            {preset.split('(')[0].replace('def ', '')}()
          </button>
        ))}
      </div>

      {/* Bottom Input */}
      <div style={{ display: 'flex', gap: '10px', alignItems: 'center', background: 'var(--bg-input)', padding: '8px 14px', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border-subtle)' }}>
        <input 
          type="text" 
          className="form-input" 
          placeholder="Ask SmallGPT or enter a Python prompt..." 
          style={{ border: 'none', background: 'transparent', padding: '6px 0' }}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
        />
        <button 
          className="btn btn-primary" 
          style={{ width: '38px', height: '38px', padding: 0, borderRadius: '50%', flexShrink: 0 }}
          disabled={loading || !input.trim()}
          onClick={() => handleSend()}
        >
          <Send size={16} />
        </button>
      </div>
    </div>
  );
}
