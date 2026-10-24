import React from 'react';
import { Gauge, Trash2, RefreshCw } from 'lucide-react';

function Bar({ used, total, color = 'var(--accent-blue)' }) {
  const pct = total ? Math.min(100, (used / total) * 100) : 0;
  return (
    <div style={{ height: 8, background: 'rgba(255,255,255,0.06)', borderRadius: 99, overflow: 'hidden' }}>
      <div style={{ width: `${pct}%`, height: '100%', background: color }} />
    </div>
  );
}

export default function ResourcesApp({ vitals, loadedModels, activeModel, onRefresh }) {
  const unload = async (id) => {
    await fetch('/api/models/unload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_name: id }),
    });
    onRefresh?.();
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h3 style={{ fontFamily: 'var(--font-display)' }}>Host & accelerator</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
            Evict engines to free VRAM. Product traffic uses whatever is Active.
          </p>
        </div>
        <button className="btn btn-secondary" onClick={onRefresh}><RefreshCw size={14} /> Refresh</button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 14 }}>
        <div className="card-panel">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
            <span>VRAM</span>
            <Gauge size={16} color="var(--accent-emerald)" />
          </div>
          <div style={{ marginBottom: 8 }}>{vitals.vram_used_mb || 0} / {vitals.vram_total_mb || 0} MB</div>
          <Bar used={vitals.vram_used_mb} total={vitals.vram_total_mb} color="var(--accent-emerald)" />
        </div>
        <div className="card-panel">
          <div style={{ marginBottom: 8 }}>System RAM</div>
          <div style={{ marginBottom: 8 }}>{vitals.ram_used_gb || 0} / {vitals.ram_total_gb || 0} GB</div>
          <Bar used={vitals.ram_used_gb} total={vitals.ram_total_gb} color="var(--accent-purple)" />
        </div>
        <div className="card-panel">
          <div style={{ marginBottom: 8 }}>CPU</div>
          <div style={{ marginBottom: 8 }}>{vitals.cpu_usage_percent || 0}%</div>
          <Bar used={vitals.cpu_usage_percent} total={100} color="var(--accent-amber)" />
        </div>
        <div className="card-panel">
          <div style={{ marginBottom: 8 }}>Disk free</div>
          <div>{vitals.disk_free_gb || 0} GB / {vitals.disk_total_gb || 0} GB</div>
        </div>
      </div>

      <div className="card-panel">
        <h3 style={{ marginBottom: 12 }}>Engines in memory</h3>
        {(loadedModels || []).length === 0 && (
          <p style={{ color: 'var(--text-muted)' }}>Nothing loaded. Open the catalog to load a model.</p>
        )}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {(loadedModels || []).map((m) => (
            <div key={m.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: 10, background: 'var(--bg-input)', borderRadius: 8 }}>
              <div>
                <strong>{m.id}</strong>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  {m.provider} · {m.device} {m.id === activeModel ? '· ACTIVE' : ''}
                </div>
              </div>
              <button className="btn btn-secondary" onClick={() => unload(m.id)}>
                <Trash2 size={14} /> Unload
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
