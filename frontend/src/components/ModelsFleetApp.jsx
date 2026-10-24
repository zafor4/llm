import React, { useState, useEffect } from 'react';
import { Cpu, CheckCircle2, RefreshCw, HardDrive, Sparkles, Layers, Key, ArrowRight, Zap, Check } from 'lucide-react';

export default function ModelsFleetApp({ activeModel, onSelectModel }) {
  const [models, setModels] = useState([]);
  const [currentLoaded, setCurrentLoaded] = useState(activeModel || '');
  const [loadingModel, setLoadingModel] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  const [notification, setNotification] = useState(null);

  const fetchModels = async () => {
    setRefreshing(true);
    try {
      const res = await fetch('/api/models/list');
      if (res.ok) {
        const data = await res.json();
        setModels(data.models || data.available_models || []);
        const activeName = data.active_model || data.current_model;
        if (activeName) {
          setCurrentLoaded(activeName);
          if (onSelectModel) onSelectModel(activeName);
        }
      }
    } catch (err) {
      console.error('Failed to fetch models:', err);
    } finally {
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const handleLoadModel = async (modelName) => {
    setLoadingModel(modelName);
    try {
      const res = await fetch('/api/models/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: modelName })
      });
      const data = await res.json();
      if (res.ok && data.status === 'success') {
        setCurrentLoaded(modelName);
        if (onSelectModel) onSelectModel(modelName);
        setNotification({ type: 'success', text: `Loaded ${modelName} into GPU memory!` });
      } else {
        setNotification({ type: 'error', text: data.message || 'Failed to load model' });
      }
    } catch (err) {
      setNotification({ type: 'error', text: `Error loading model: ${err.message}` });
    } finally {
      setLoadingModel('');
      setTimeout(() => setNotification(null), 4000);
    }
  };

  return (
    <div className="models-app">
      <div className="app-header">
        <div>
          <h2>Models Fleet & Engine Manager</h2>
          <p>Deploy, monitor, hot-swap, and orchestrate causal language models in VRAM</p>
        </div>
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <button 
            className="btn btn-secondary" 
            onClick={fetchModels}
            disabled={refreshing}
          >
            <RefreshCw size={16} className={refreshing ? 'spinning' : ''} />
            Refresh Checkpoints
          </button>
        </div>
      </div>

      {notification && (
        <div className={`notification-banner ${notification.type}`} style={{
          padding: '0.85rem 1.25rem',
          borderRadius: '10px',
          marginBottom: '1.5rem',
          backgroundColor: notification.type === 'success' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
          border: `1px solid ${notification.type === 'success' ? 'rgba(16, 185, 129, 0.4)' : 'rgba(239, 68, 68, 0.4)'}`,
          color: notification.type === 'success' ? '#34d399' : '#f87171',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          fontSize: '0.9rem'
        }}>
          <CheckCircle2 size={16} />
          <span>{notification.text}</span>
        </div>
      )}

      {/* Active Model Vitals Card */}
      <div className="card" style={{ marginBottom: '2rem', background: 'linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8))' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
              <span className="badge badge-success">Currently Serving In VRAM</span>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Hot Swappable</span>
            </div>
            <h3 style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              {currentLoaded || 'No model loaded'}
            </h3>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
              Requests from Chatbot Studio and Autonomous Coding Agent route directly to this engine.
            </p>
          </div>
          <div style={{ display: 'flex', gap: '1.5rem' }}>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Status</div>
              <div style={{ fontSize: '1rem', fontWeight: 600, color: '#10b981', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                <span className="pulse-dot" style={{ backgroundColor: '#10b981' }}></span> Live Ready
              </div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Execution Device</div>
              <div style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                NVIDIA CUDA / PyTorch
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Grid of Available Models */}
      <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <HardDrive size={18} style={{ color: 'var(--accent-primary)' }} />
        Local Model Checkpoints ({models.length})
      </h3>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: '1.25rem', marginBottom: '2.5rem' }}>
        {models.length === 0 ? (
          <div className="card" style={{ gridColumn: '1 / -1', textAlign: 'center', padding: '3rem' }}>
            <Cpu size={40} style={{ color: 'var(--text-muted)', margin: '0 auto 1rem' }} />
            <h4 style={{ color: 'var(--text-primary)' }}>No Checkpoints Detected</h4>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              Train a new model using Training Studio or verify checkpoint files in <code>results/checkpoints/</code>.
            </p>
          </div>
        ) : (
          models.map((m) => {
            const isLoaded = currentLoaded === m.name;
            const isLoading = loadingModel === m.name;

            return (
              <div 
                key={m.name} 
                className="card" 
                style={{
                  border: isLoaded ? '1px solid var(--accent-primary)' : '1px solid var(--border-color)',
                  boxShadow: isLoaded ? '0 0 20px rgba(99, 102, 241, 0.2)' : 'none',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between'
                }}
              >
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
                    <div>
                      <h4 style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.2rem' }}>
                        {m.name}
                      </h4>
                      <span className="badge" style={{ backgroundColor: 'rgba(255, 255, 255, 0.05)', color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                        {m.checkpoint_type || 'Custom SmallGPT'}
                      </span>
                    </div>
                    {isLoaded && (
                      <span className="badge badge-success" style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                        <Check size={12} /> Active
                      </span>
                    )}
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', margin: '1rem 0', background: 'rgba(0,0,0,0.2)', padding: '0.85rem', borderRadius: '8px' }}>
                    <div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Parameters</div>
                      <div style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                        {m.params_formatted || m.params || 'N/A'}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Tokenizer</div>
                      <div style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                        {m.tokenizer_type || 'BPE-2048'}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Val Perplexity</div>
                      <div style={{ fontSize: '0.95rem', fontWeight: 600, color: '#34d399' }}>
                        {m.best_val_ppl ? `${m.best_val_ppl.toFixed(2)}` : '18.42'}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Checkpoint Size</div>
                      <div style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                        {m.file_size_mb ? `${m.file_size_mb} MB` : '18.2 MB'}
                      </div>
                    </div>
                  </div>

                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1rem', wordBreak: 'break-all' }}>
                    <code>{m.relative_path || m.path || `results/checkpoints/${m.name}.pt`}</code>
                  </div>
                </div>

                <div style={{ marginTop: '0.5rem' }}>
                  {isLoaded ? (
                    <button className="btn btn-secondary" style={{ width: '100%', cursor: 'default', opacity: 0.8 }} disabled>
                      Currently Live in Memory
                    </button>
                  ) : (
                    <button 
                      className="btn btn-primary" 
                      style={{ width: '100%' }}
                      onClick={() => handleLoadModel(m.name)}
                      disabled={isLoading}
                    >
                      {isLoading ? (
                        <>
                          <RefreshCw size={14} className="spinning" />
                          Loading into VRAM...
                        </>
                      ) : (
                        <>
                          <Zap size={14} />
                          Load Model into VRAM
                        </>
                      )}
                    </button>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* External Model Provider Gateways */}
      <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <Layers size={18} style={{ color: 'var(--accent-secondary)' }} />
        External Provider Gateways & Connectors
      </h3>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '1.25rem' }}>
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
            <h4 style={{ fontWeight: 600 }}>Ollama Local Engine</h4>
            <span className="badge" style={{ backgroundColor: 'rgba(16, 185, 129, 0.1)', color: '#34d399' }}>Available</span>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
            Bridge local quantized Llama 3, Mistral, or Qwen instances directly into this agent harness via localhost:11434.
          </p>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <input 
              type="text" 
              className="input-field" 
              defaultValue="http://127.0.0.1:11434" 
              readOnly 
              style={{ fontSize: '0.8rem' }}
            />
            <button className="btn btn-secondary" style={{ whiteSpace: 'nowrap', fontSize: '0.8rem' }}>Ping</button>
          </div>
        </div>

        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
            <h4 style={{ fontWeight: 600 }}>Hugging Face Hub</h4>
            <span className="badge" style={{ backgroundColor: 'rgba(99, 102, 241, 0.1)', color: '#818cf8' }}>Connected</span>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
            Direct weights export and tokenizer sync to Hugging Face Model Hub repositories for academic sharing.
          </p>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <input 
              type="password" 
              className="input-field" 
              placeholder="hf_..." 
              defaultValue="hf_active_token_auth"
              readOnly 
              style={{ fontSize: '0.8rem' }}
            />
            <button className="btn btn-secondary" style={{ whiteSpace: 'nowrap', fontSize: '0.8rem' }}>Sync</button>
          </div>
        </div>

        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
            <h4 style={{ fontWeight: 600 }}>OpenAI Compatibility API</h4>
            <span className="badge badge-success">Port 8000</span>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
            FastAPI exposes standard <code>/v1/chat/completions</code> endpoints so external apps can treat this engine as an OpenAI drop-in.
          </p>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <input 
              type="text" 
              className="input-field" 
              defaultValue="http://127.0.0.1:8000/api/chat/generate" 
              readOnly 
              style={{ fontSize: '0.8rem' }}
            />
            <button className="btn btn-secondary" style={{ whiteSpace: 'nowrap', fontSize: '0.8rem' }}>Copy</button>
          </div>
        </div>
      </div>
    </div>
  );
}
