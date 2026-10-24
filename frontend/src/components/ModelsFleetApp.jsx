import React, { useEffect, useState } from 'react';
import { RefreshCw } from 'lucide-react';

export default function ModelsFleetApp({
  activeModel,
  loadedModels = [],
  onSelectModel,
  onContinueTrain,
  onRefresh,
}) {
  const [models, setModels] = useState([]);
  const [loaded, setLoaded] = useState(loadedModels);
  const [tab, setTab] = useState('yours');
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
    if (data.active_model) onSelectModel?.(data.active_model);
  };

  useEffect(() => {
    loadCatalog();
  }, []);

  const flash = (type, text) => {
    setNotice({ type, text });
    setTimeout(() => setNotice(null), 4500);
  };

  const post = async (url, body) => {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Something went wrong');
    return data;
  };

  const handleLoad = async (id) => {
    setBusy(id);
    try {
      await post('/api/models/load', { model_name: id, activate: true });
      flash('success', `${id} is now the model you are using.`);
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
    if (!window.confirm(`Remove ${id} from your library?`)) return;
    const res = await fetch(`/api/models/${encodeURIComponent(id)}`, { method: 'DELETE' });
    if (!res.ok) {
      flash('error', 'Could not remove that model.');
      return;
    }
    await loadCatalog();
    onRefresh?.();
  };

  const kindLabel = (m) => {
    if (m.provider === 'local') return 'Trained here';
    if (m.provider === 'huggingface') return 'Hugging Face';
    if (m.provider === 'ollama') return 'Ollama';
    return m.provider;
  };

  return (
    <div>
      {notice && <div className={`notice ${notice.type}`}>{notice.text}</div>}

      <div className="page-header">
        <p className="page-lead" style={{ margin: 0 }}>
          Use a model to make it Active. Add ones from Hugging Face or Ollama when you want something stronger than a local checkpoint.
        </p>
        <button className="btn btn-secondary" onClick={loadCatalog}><RefreshCw size={14} /> Refresh</button>
      </div>

      <div className="soft-tabs">
        {[
          ['yours', 'Your models'],
          ['add', 'Add a model'],
          ['memory', 'In memory'],
        ].map(([id, label]) => (
          <button key={id} className={`btn ${tab === id ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setTab(id)}>
            {label}
          </button>
        ))}
      </div>

      {tab === 'yours' && (
        <div className="model-grid">
          {models.map((m) => {
            const isActive = m.id === activeModel || m.is_active;
            const isLoaded = m.is_loaded || loaded.some((x) => x.id === m.id);
            return (
              <div key={m.id} className="card-panel">
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                  <div>
                    <h3>{m.name}</h3>
                    <div className="hint">{kindLabel(m)}</div>
                  </div>
                  {isActive && <span className="chip on">In use</span>}
                </div>
                <div className="hint" style={{ margin: '12px 0' }}>
                  {isLoaded ? 'Loaded and ready.' : 'Saved, but not in memory yet.'}
                  {m.val_perplexity ? ` Quality score: ${Number(m.val_perplexity).toFixed(2)}` : ''}
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                  {!isActive && (
                    <button className="btn btn-primary" disabled={busy === m.id} onClick={() => handleLoad(m.id)}>
                      Use this
                    </button>
                  )}
                  {isLoaded && (
                    <button className="btn btn-secondary" onClick={() => handleUnload(m.id)}>Remove from memory</button>
                  )}
                  {m.provider === 'local' && (
                    <button className="btn btn-secondary" onClick={() => onContinueTrain?.(m.id)}>Improve</button>
                  )}
                  <button className="btn btn-secondary" onClick={() => handleDelete(m.id)}>Delete</button>
                </div>
              </div>
            );
          })}
          {models.length === 0 && (
            <div className="card-panel empty-state" style={{ gridColumn: '1 / -1' }}>
              <p>No models yet. Train one, or open Add a model.</p>
            </div>
          )}
        </div>
      )}

      {tab === 'add' && (
        <div className="action-grid">
          <div className="card-panel">
            <h3>From Hugging Face</h3>
            <p className="hint" style={{ margin: '8px 0 12px' }}>
              Paste a model name from the Hub. It is added to your library. Load it when you want it in memory.
            </p>
            <label className="field-label">Model name</label>
            <input className="form-input" value={hfId} onChange={(e) => setHfId(e.target.value)} />
            <button className="btn btn-primary" style={{ marginTop: 12 }} disabled={busy === 'hf'} onClick={async () => {
              setBusy('hf');
              try {
                const data = await post('/api/models/import/huggingface', { model_id: hfId, load_now: false });
                flash('success', `${data.model.id} was added. Open Your models and click Use this.`);
                await loadCatalog();
                setTab('yours');
              } catch (e) {
                flash('error', e.message);
              } finally {
                setBusy('');
              }
            }}>
              Add to library
            </button>
          </div>
          <div className="card-panel">
            <h3>From Ollama on this computer</h3>
            <p className="hint" style={{ margin: '8px 0 12px' }}>
              Ollama must already be running. Check the connection, then add a model you have pulled.
            </p>
            <label className="field-label">Ollama address</label>
            <input className="form-input" value={ollamaHost} onChange={(e) => setOllamaHost(e.target.value)} />
            <label className="field-label">Model tag</label>
            <input className="form-input" value={ollamaName} onChange={(e) => setOllamaName(e.target.value)} />
            <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
              <button className="btn btn-secondary" onClick={async () => {
                const res = await fetch(`/api/models/providers/ollama?host=${encodeURIComponent(ollamaHost)}`);
                setOllamaProbe(await res.json());
              }}>Check</button>
              <button className="btn btn-primary" disabled={busy === 'ollama'} onClick={async () => {
                setBusy('ollama');
                try {
                  const data = await post('/api/models/import/ollama', { model_name: ollamaName, host: ollamaHost, load_now: true });
                  flash('success', `${data.model.name} is ready to use.`);
                  await loadCatalog();
                  onRefresh?.();
                  setTab('yours');
                } catch (e) {
                  flash('error', e.message);
                } finally {
                  setBusy('');
                }
              }}>Add and use</button>
            </div>
            {ollamaProbe && (
              <p className="hint" style={{ marginTop: 10 }}>
                {ollamaProbe.ok
                  ? `Connected. Available: ${(ollamaProbe.models || []).join(', ') || 'none listed'}`
                  : `Could not reach Ollama. ${ollamaProbe.error || ''}`}
              </p>
            )}
          </div>
        </div>
      )}

      {tab === 'memory' && (
        <div className="card-panel">
          <h3 style={{ marginBottom: 8 }}>What is loaded right now</h3>
          <p className="hint" style={{ marginBottom: 14 }}>
            Models in memory use your computer. Remove one if you need space.
          </p>
          {loaded.length === 0 && <p className="hint">Nothing is loaded. Use a model from Your models.</p>}
          {loaded.map((m) => (
            <div key={m.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 0', borderBottom: '1px solid var(--border-subtle)' }}>
              <div>
                <strong>{m.id}</strong>
                <div className="hint">{m.provider}{m.id === activeModel ? ' · in use' : ''}</div>
              </div>
              <button className="btn btn-secondary" onClick={() => handleUnload(m.id)}>Remove from memory</button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
