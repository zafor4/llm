import React, { useEffect, useState } from 'react';
import { Bot, MessageSquare, Flame, Layers, Gauge, Globe, ArrowRight } from 'lucide-react';

export default function DashboardApp({ vitals, setActiveApp, activeModel, modelsList, loadedModels }) {
  const [overview, setOverview] = useState(null);

  useEffect(() => {
    fetch('/api/dashboard/overview')
      .then((r) => (r.ok ? r.json() : null))
      .then(setOverview)
      .catch(() => {});
  }, [activeModel, vitals.vram_used_mb]);

  const cards = [
    { title: 'Model Catalog', desc: 'Register local, Hugging Face, and Ollama models. Load, unload, annotate.', appId: 'models', icon: Layers, tag: 'Control' },
    { title: 'Training Jobs', desc: 'Train from scratch or continue a checkpoint with live telemetry.', appId: 'training', icon: Flame, tag: 'Control' },
    { title: 'Resources', desc: 'Watch VRAM, RAM, CPU and evict engines from memory.', appId: 'resources', icon: Gauge, tag: 'Control' },
    { title: 'API Sharing', desc: 'OpenAI-compatible /v1 endpoint and shareable local keys.', appId: 'gateway', icon: Globe, tag: 'Product' },
    { title: 'Chat Playground', desc: 'Smoke-test any loaded engine before wiring a product app.', appId: 'chat', icon: MessageSquare, tag: 'Playground' },
    { title: 'Agent Playground', desc: 'Oracle benches or live planning with an instruct model.', appId: 'agent', icon: Bot, tag: 'Playground' },
  ];

  const activity = overview?.activity || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      <div className="card-panel" style={{ background: 'linear-gradient(135deg, rgba(30,58,138,0.3), rgba(88,28,135,0.25))' }}>
        <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#38bdf8', textTransform: 'uppercase', marginBottom: 6 }}>
          House of Intelligence
        </div>
        <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.65rem', fontWeight: 800, marginBottom: 6 }}>
          Control plane for every model you train or import
        </h2>
        <p style={{ color: 'var(--text-secondary)', maxWidth: 720 }}>
          This hub owns catalog, training, memory, and the public API. Chat and the coding agent here are playgrounds.
          Product apps should call <code>/v1/chat/completions</code> or the hub REST routes.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 14 }}>
        {[
          ['Device', vitals.device_name || '—'],
          ['VRAM', `${vitals.vram_used_mb || 0} / ${vitals.vram_total_mb || 0} MB`],
          ['Host RAM', `${vitals.ram_used_gb || 0} / ${vitals.ram_total_gb || 0} GB`],
          ['Loaded / Catalog', `${(loadedModels || []).length} / ${(modelsList || []).length}`],
          ['Training', vitals.training_status || 'IDLE'],
          ['Active', activeModel || 'none'],
        ].map(([label, value]) => (
          <div key={label} className="card-panel" style={{ padding: 16 }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>{label}</div>
            <div style={{ fontWeight: 700, marginTop: 6, fontSize: '0.95rem' }}>{value}</div>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16 }}>
        <div>
          <h3 style={{ fontFamily: 'var(--font-display)', marginBottom: 12 }}>Surfaces</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 12 }}>
            {cards.map((c) => {
              const Icon = c.icon;
              return (
                <div key={c.appId} className="card-panel" style={{ cursor: 'pointer' }} onClick={() => setActiveApp(c.appId)}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
                    <Icon size={18} color="var(--accent-blue)" />
                    <span style={{ fontSize: '0.68rem', color: 'var(--accent-cyan)' }}>{c.tag}</span>
                  </div>
                  <h4 style={{ marginBottom: 6 }}>{c.title}</h4>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 10 }}>{c.desc}</p>
                  <span style={{ fontSize: '0.8rem', color: 'var(--accent-blue)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                    Open <ArrowRight size={12} />
                  </span>
                </div>
              );
            })}
          </div>
        </div>
        <div className="card-panel">
          <h3 style={{ marginBottom: 12 }}>Recent activity</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, maxHeight: 420, overflowY: 'auto' }}>
            {activity.length === 0 && <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No events yet. Load a model or start a job.</p>}
            {activity.map((ev) => (
              <div key={ev.id} style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: 8 }}>
                <div style={{ fontSize: '0.7rem', color: 'var(--accent-cyan)' }}>{ev.kind} · {ev.source}</div>
                <div style={{ fontSize: '0.82rem' }}>{ev.message}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
