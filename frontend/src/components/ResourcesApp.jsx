import React, { useCallback, useEffect, useState } from 'react';
import { RefreshCw } from 'lucide-react';

function Bar({ used, total }) {
  const pct = total ? Math.min(100, (Number(used) / Number(total)) * 100) : 0;
  return (
    <div className="progress-track" style={{ marginTop: 8 }}>
      <div className="progress-fill" style={{ width: `${pct}%` }} />
    </div>
  );
}

export default function ResourcesApp({ onRefresh, onSelectModel }) {
  const [board, setBoard] = useState(null);
  const [busy, setBusy] = useState('');
  const [notice, setNotice] = useState(null);

  const load = useCallback(async () => {
    const res = await fetch('/api/resources');
    if (!res.ok) return;
    const data = await res.json();
    setBoard(data);
    if (data.active_model) onSelectModel?.(data.active_model);
  }, [onSelectModel]);

  useEffect(() => {
    load();
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, [load]);

  const flash = (type, text) => {
    setNotice({ type, text });
    setTimeout(() => setNotice(null), 4000);
  };

  const post = async (url, body, after) => {
    setBusy(url);
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: body ? JSON.stringify(body) : '{}',
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Request failed');
      flash('success', after);
      await load();
      onRefresh?.();
    } catch (e) {
      flash('error', e.message);
    } finally {
      setBusy('');
    }
  };

  const host = board?.host || {};
  const assigned = board?.assigned || [];
  const available = board?.available || [];
  const training = board?.training || {};

  return (
    <div>
      {notice && <div className={`notice ${notice.type}`}>{notice.text}</div>}

      <div className="page-header">
        <p className="page-lead" style={{ margin: 0 }}>
          Watch this computer, put models on it, and take them off when you need space.
          Chat and other apps only use the model marked In use.
        </p>
        <button className="btn btn-secondary" onClick={load}><RefreshCw size={14} /> Refresh</button>
      </div>

      <h3 style={{ fontFamily: 'var(--font-display)', marginBottom: 10 }}>This computer</h3>
      <div className="stat-grid" style={{ marginBottom: 22 }}>
        <div className="card-panel">
          <div className="hint">Graphics memory (GPU)</div>
          <strong>{host.cuda_available ? `${host.vram_used_mb || 0} / ${host.vram_total_mb || 0} MB` : 'No GPU — using CPU'}</strong>
          {host.cuda_available && <Bar used={host.vram_used_mb} total={host.vram_total_mb} />}
          <p className="hint" style={{ marginTop: 8 }}>{host.device_name || '—'}</p>
        </div>
        <div className="card-panel">
          <div className="hint">System memory (RAM)</div>
          <strong>{host.ram_used_gb || 0} / {host.ram_total_gb || 0} GB</strong>
          <Bar used={host.ram_used_gb} total={host.ram_total_gb} />
        </div>
        <div className="card-panel">
          <div className="hint">Processor (CPU)</div>
          <strong>{host.cpu_usage_percent || 0}% of {host.cpu_count || 1} cores</strong>
          <Bar used={host.cpu_usage_percent} total={100} />
        </div>
        <div className="card-panel">
          <div className="hint">Disk space</div>
          <strong>{host.disk_free_gb || 0} GB free of {host.disk_total_gb || 0} GB</strong>
          <Bar used={(host.disk_total_gb || 0) - (host.disk_free_gb || 0)} total={host.disk_total_gb} />
        </div>
      </div>

      <div className="card-panel" style={{ marginBottom: 22 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
          <div>
            <h3>Training</h3>
            <p className="hint" style={{ marginTop: 6 }}>
              {training.is_training
                ? `${training.model_name} is training (step ${training.current_step}/${training.total_steps}). This uses GPU and RAM until it finishes.`
                : 'No training job is using resources right now.'}
            </p>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn btn-secondary" disabled={!!busy} onClick={() => post('/api/resources/free-cache', {}, 'Unused GPU memory was released.')}>
              Free unused GPU
            </button>
            <button className="btn btn-secondary" disabled={!!busy || assigned.length === 0} onClick={() => post('/api/resources/release-all', {}, 'All models were taken out of memory.')}>
              Release everything
            </button>
          </div>
        </div>
      </div>

      <h3 style={{ fontFamily: 'var(--font-display)', marginBottom: 8 }}>On this computer now</h3>
      <p className="hint" style={{ marginBottom: 12 }}>These models are using memory. You can make one In use, or release it.</p>
      <div className="model-grid" style={{ marginBottom: 22 }}>
        {assigned.length === 0 && (
          <div className="card-panel empty-state" style={{ gridColumn: '1 / -1' }}>
            <p>Nothing is loaded. Include a model from the list below.</p>
          </div>
        )}
        {assigned.map((m) => {
          const inUse = m.id === board?.active_model;
          return (
            <div key={m.id} className="card-panel">
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <div>
                  <h3>{m.id}</h3>
                  <div className="hint">{m.provider} · {m.device || m.kind}</div>
                </div>
                {inUse && <span className="chip on">In use</span>}
              </div>
              <div style={{ display: 'flex', gap: 8, marginTop: 12, flexWrap: 'wrap' }}>
                {!inUse && (
                  <button className="btn btn-primary" disabled={!!busy} onClick={() => post('/api/resources/activate', { model_name: m.id }, `${m.id} is now in use.`)}>
                    Use this
                  </button>
                )}
                <button className="btn btn-secondary" disabled={!!busy} onClick={() => post('/api/resources/release', { model_name: m.id }, `${m.id} was released.`)}>
                  Release
                </button>
              </div>
            </div>
          );
        })}
      </div>

      <h3 style={{ fontFamily: 'var(--font-display)', marginBottom: 8 }}>Include a model</h3>
      <p className="hint" style={{ marginBottom: 12 }}>
        These are in your library but not using this computer yet. Include one to put it in memory.
      </p>
      <div className="model-grid">
        {available.length === 0 && (
          <div className="card-panel empty-state" style={{ gridColumn: '1 / -1' }}>
            <p>Every model in your library is already included, or the library is empty.</p>
          </div>
        )}
        {available.map((m) => (
          <div key={m.id} className="card-panel">
            <h3>{m.name}</h3>
            <p className="hint">{m.provider === 'local' ? 'Trained here' : m.provider} · not in memory</p>
            <div style={{ display: 'flex', gap: 8, marginTop: 12, flexWrap: 'wrap' }}>
              <button
                className="btn btn-primary"
                disabled={!!busy}
                onClick={() => post('/api/resources/include', { model_name: m.id, activate: true }, `${m.name} is included and in use.`)}
              >
                Include and use
              </button>
              <button
                className="btn btn-secondary"
                disabled={!!busy}
                onClick={() => post('/api/resources/include', { model_name: m.id, activate: false }, `${m.name} is included in memory.`)}
              >
                Include only
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
