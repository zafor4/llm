import React, { useEffect, useState } from 'react';
import { Cpu, RefreshCw, Zap, Trash2, Flame, Check } from 'lucide-react';

export default function ModelsFleetApp({ activeModel, onSelectModel, onContinueTrain, onRefresh }) {
  const [models, setModels] = useState([]);
  const [loaded, setLoaded] = useState([]);
  const [busy, setBusy] = useState('');
  const [notice, setNotice] = useState(null);
  const [hfId, setHfId] = useState('Qwen/Qwen2.5-0.5B-Instruct');
  const [ollamaName, setOllamaName] = useState('qwen2.5:0.5b');
  const [ollamaHost, setOllamaHost] = useState('http://127.0.0.1:11434');
  const [ollamaProbe, setOllamaProbe] = useState(null);

  const loadCatalog = async () => {
    const res = await fetch('/api/models/list');
    if (!res.ok) return;
    const data = await res.json();
    setModels(data.models || []);
    setLoaded(data.loaded_models || []);
    if (data.active_model && onSelectModel) onSelectModel(data.active_model);
  };

  useEffect(() => {
    loadCatalog();
  }, []);

  const flash = (type, text) => {
    setNotice({ type, text });
    setTimeout(() => setNotice(null), 4000);
  };

  const post = async (url, body) => {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || data.message || 'Request failed');
    return data;
  };

  const handleLoad = async (id) => {
    setBusy(id);
    try {
      await post('/api/models/load', { model_name: id, activate: true });
      flash('success', `Loaded ${id}`);
      await loadCatalog();
      onRefresh?.();
    } catch (e) {
      flash('error', e.message);
    } finally {
      setBusy('');
    }
  };

  const handleUnload = async (id) => {
    try {
      await post('/api/models/unload', { model_name: id });
      await loadCatalog();
      onRefresh?.();
    } catch (e) {
      flash('error', e.message);
    }
  };

  const handleDelete = async (id) => {
    if (!confirm(`Remove ${id} from the catalog?`)) return;
    const res = await fetch(`/api/models/${encodeURIComponent(id)}`, { method: 'DELETE' });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      flash('error', data.detail || 'Delete failed');
      return;
    }
    await loadCatalog();
    onRefresh?.();
  };

  const importHF = async () => {
    setBusy('hf');
    try {
      const data = await post('/api/models/import/huggingface', { model_id: hfId, load_now: false });
      flash('success', `Registered ${data.model.id}. Load it when you want it in VRAM.`);
      await loadCatalog();
    } catch (e) {
      flash('error', e.message);
    } finally {
      setBusy('');
    }
  };

  const importOllama = async () => {
    setBusy('ollama');
    try {
      const data = await post('/api/models/import/ollama', {
        model_name: ollamaName,
        host: ollamaHost,
        load_now: true,
      });
      flash('success', `Ollama model ${data.model.name} is active.`);
      await loadCatalog();
      onRefresh?.();
    } catch (e) {
      flash('error', e.message);
    } finally {
      setBusy('');
    }
  };

  const pingOllama = async () => {
    const res = await fetch(`/api/models/providers/ollama?host=${encodeURIComponent(ollamaHost)}`);
    setOllamaProbe(await res.json());
  };

  return (
    <div>
      {notice && (
        <div className="card-panel" style={{ marginBottom: 16, borderColor: notice.type === 'success' ? 'rgba(16,185,129,0.4)' : 'rgba(244,63,94,0.4)' }}>
          {notice.text}
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <p style={{ color: 'var(--text-secondary)' }}>
          Local checkpoints, Hugging Face imports, and Ollama services. Product apps consume whatever is Active.
        </p>
        <button className="btn btn-secondary" onClick={loadCatalog}><RefreshCw size={14} /> Refresh</button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginBottom: 20 }}>
        <div className="card-panel">
          <h3 style={{ marginBottom: 10 }}>Import Hugging Face</h3>
          <input className="form-input" value={hfId} onChange={(e) => setHfId(e.target.value)} placeholder="org/model" />
          <button className="btn btn-primary" style={{ marginTop: 10 }} disabled={busy === 'hf'} onClick={importHF}>
            Register model
          </button>
        </div>
        <div className="card-panel">
          <h3 style={{ marginBottom: 10 }}>Connect Ollama</h3>
          <input className="form-input" style={{ marginBottom: 8 }} value={ollamaHost} onChange={(e) => setOllamaHost(e.target.value)} />
          <input className="form-input" value={ollamaName} onChange={(e) => setOllamaName(e.target.value)} placeholder="model:tag" />
          <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
            <button className="btn btn-secondary" onClick={pingOllama}>Ping</button>
            <button className="btn btn-primary" disabled={busy === 'ollama'} onClick={importOllama}>Import & activate</button>
          </div>
          {ollamaProbe && (
            <div style={{ fontSize: '0.8rem', marginTop: 8, color: ollamaProbe.ok ? '#34d399' : '#f87171' }}>
              {ollamaProbe.ok ? `Online · ${ (ollamaProbe.models || []).join(', ') || 'no tags' }` : ollamaProbe.error}
            </div>
          )}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 14 }}>
        {models.map((m) => {
          const isActive = m.id === activeModel || m.is_active;
          const isLoaded = m.is_loaded || loaded.some((x) => x.id === m.id);
          return (
            <div key={m.id} className="card-panel" style={{ borderColor: isActive ? 'var(--accent-blue)' : undefined }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <div>
                  <h4>{m.name}</h4>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{m.type || m.provider}</div>
                </div>
                {isActive && <span className="active-model-badge"><Check size={12} /> Active</span>}
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, margin: '12px 0', fontSize: '0.8rem' }}>
                <div>Provider<br /><strong>{m.provider}</strong></div>
                <div>PPL<br /><strong>{m.val_perplexity ?? '—'}</strong></div>
                <div>Size<br /><strong>{m.file_size_mb ? `${m.file_size_mb} MB` : '—'}</strong></div>
                <div>State<br /><strong>{isLoaded ? 'In memory' : 'Registered'}</strong></div>
              </div>
              {m.notes && <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{m.notes}</p>}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 12 }}>
                {!isLoaded ? (
                  <button className="btn btn-primary" disabled={busy === m.id} onClick={() => handleLoad(m.id)}>
                    <Zap size={14} /> Load
                  </button>
                ) : (
                  <button className="btn btn-secondary" onClick={() => handleUnload(m.id)}>Unload</button>
                )}
                {m.provider === 'local' && (
                  <button className="btn btn-secondary" onClick={() => onContinueTrain?.(m.id)}>
                    <Flame size={14} /> Continue train
                  </button>
                )}
                <button className="btn btn-secondary" onClick={() => handleDelete(m.id)}>
                  <Trash2 size={14} /> Remove
                </button>
              </div>
            </div>
          );
        })}
        {models.length === 0 && (
          <div className="card-panel" style={{ gridColumn: '1 / -1', textAlign: 'center' }}>
            <Cpu size={36} style={{ opacity: 0.4 }} />
            <p>No models in the catalog. Train one or import Hugging Face / Ollama.</p>
          </div>
        )}
      </div>
    </div>
  );
}
