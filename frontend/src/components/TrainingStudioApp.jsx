import React, { useEffect, useState } from 'react';
import { Flame, Square } from 'lucide-react';

export default function TrainingStudioApp({ onModelTrained, modelsList, continueTarget }) {
  const localModels = (modelsList || []).filter((m) => m.provider === 'local');
  const [mode, setMode] = useState(continueTarget ? 'continue' : 'scratch');
  const [modelName, setModelName] = useState(continueTarget ? `${continueTarget}-ft` : 'Custom-GPT-Micro');
  const [baseModel, setBaseModel] = useState(continueTarget || localModels[0]?.id || '');
  const [layers, setLayers] = useState(4);
  const [heads, setHeads] = useState(4);
  const [dim, setDim] = useState(192);
  const [blockSize, setBlockSize] = useState(128);
  const [lr, setLr] = useState(0.0005);
  const [batchSize, setBatchSize] = useState(16);
  const [maxSteps, setMaxSteps] = useState(100);
  const [corpusText, setCorpusText] = useState('');
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [jobs, setJobs] = useState([]);

  useEffect(() => {
    if (continueTarget) {
      setMode('continue');
      setBaseModel(continueTarget);
      setModelName(`${continueTarget}-ft`);
    }
  }, [continueTarget]);

  useEffect(() => {
    fetch('/api/trainer/jobs').then((r) => r.json()).then((d) => setJobs(d.jobs || [])).catch(() => {});
  }, [isTraining, trainingStatus?.status]);

  useEffect(() => {
    if (!isTraining) return undefined;
    const interval = setInterval(async () => {
      const res = await fetch('/api/trainer/status');
      const data = await res.json();
      setTrainingStatus(data);
      if (data.status === 'COMPLETED' || data.status === 'ERROR' || data.status === 'CANCELLED') {
        setIsTraining(false);
        if (data.status === 'COMPLETED') onModelTrained?.(data.model_name);
      }
    }, 500);
    return () => clearInterval(interval);
  }, [isTraining, onModelTrained]);

  const handleStartTraining = async () => {
    setIsTraining(true);
    setTrainingStatus({ status: 'TRAINING', current_step: 0, total_steps: maxSteps });
    try {
      const res = await fetch('/api/trainer/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: modelName,
          mode,
          base_model: mode === 'continue' ? baseModel : null,
          n_layer: layers,
          n_head: heads,
          n_embd: dim,
          block_size: blockSize,
          learning_rate: lr,
          batch_size: batchSize,
          max_steps: maxSteps,
          corpus_text: corpusText,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        alert(data.detail || 'Failed to start');
        setIsTraining(false);
      }
    } catch (e) {
      alert(e.message);
      setIsTraining(false);
    }
  };

  const cancel = async () => {
    await fetch('/api/trainer/cancel', { method: 'POST' });
  };

  const pct = Math.min(100, ((trainingStatus?.current_step || 0) / Math.max(1, trainingStatus?.total_steps || maxSteps)) * 100);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '440px 1fr', gap: 24 }}>
      <div className="card-panel">
        <h3 style={{ marginBottom: 12 }}>Job configuration</h3>
        <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
          {['scratch', 'continue'].map((m) => (
            <button key={m} className={`btn ${mode === m ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setMode(m)}>
              {m === 'scratch' ? 'From scratch' : 'Continue / finetune'}
            </button>
          ))}
        </div>
        <label>Output checkpoint name</label>
        <input className="form-input" value={modelName} onChange={(e) => setModelName(e.target.value)} />
        {mode === 'continue' && (
          <div style={{ marginTop: 10 }}>
            <label>Base local model</label>
            <select className="form-select" value={baseModel} onChange={(e) => setBaseModel(e.target.value)}>
              {localModels.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select>
          </div>
        )}
        {mode === 'scratch' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginTop: 10 }}>
            <div><label>Layers</label><input className="form-input" type="number" value={layers} onChange={(e) => setLayers(+e.target.value)} /></div>
            <div><label>Heads</label><input className="form-input" type="number" value={heads} onChange={(e) => setHeads(+e.target.value)} /></div>
            <div><label>Embed dim</label><input className="form-input" type="number" value={dim} onChange={(e) => setDim(+e.target.value)} /></div>
            <div><label>Context</label><input className="form-input" type="number" value={blockSize} onChange={(e) => setBlockSize(+e.target.value)} /></div>
          </div>
        )}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginTop: 10 }}>
          <div><label>Steps</label><input className="form-input" type="number" value={maxSteps} onChange={(e) => setMaxSteps(+e.target.value)} /></div>
          <div><label>Learning rate</label><input className="form-input" type="number" step="0.0001" value={lr} onChange={(e) => setLr(+e.target.value)} /></div>
        </div>
        <textarea
          className="form-textarea code-font"
          rows={4}
          style={{ marginTop: 10 }}
          placeholder="Optional corpus. Blank uses the built-in algorithm set."
          value={corpusText}
          onChange={(e) => setCorpusText(e.target.value)}
        />
        <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
          <button className="btn btn-primary" disabled={isTraining} onClick={handleStartTraining}>
            <Flame size={16} /> {isTraining ? 'Running…' : 'Start job'}
          </button>
          {isTraining && (
            <button className="btn btn-secondary" onClick={cancel}><Square size={14} /> Cancel</button>
          )}
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        <div className="card-panel">
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{trainingStatus?.status || 'IDLE'} · {trainingStatus?.mode || mode}</div>
              <h3>{trainingStatus?.model_name || modelName}</h3>
            </div>
            <div>Step {trainingStatus?.current_step || 0} / {trainingStatus?.total_steps || maxSteps}</div>
          </div>
          <div style={{ height: 8, background: 'rgba(255,255,255,0.06)', borderRadius: 99, overflow: 'hidden', margin: '12px 0' }}>
            <div style={{ width: `${pct}%`, height: '100%', background: 'linear-gradient(90deg, var(--accent-amber), var(--accent-rose))' }} />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
            {[
              ['Loss', trainingStatus?.current_loss || 0],
              ['Val PPL', trainingStatus?.val_perplexity || 0],
              ['tok/s', trainingStatus?.tokens_per_sec || 0],
              ['VRAM', `${trainingStatus?.peak_vram_mb || 0} MB`],
            ].map(([k, v]) => (
              <div key={k} style={{ background: 'var(--bg-input)', padding: 10, borderRadius: 8 }}>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{k}</div>
                <div style={{ fontFamily: 'var(--font-mono)' }}>{v}</div>
              </div>
            ))}
          </div>
          {trainingStatus?.error_message && (
            <p style={{ color: '#f87171', marginTop: 10 }}>{trainingStatus.error_message}</p>
          )}
        </div>

        <div className="card-panel">
          <h3 style={{ marginBottom: 10 }}>Job history</h3>
          {jobs.length === 0 && <p style={{ color: 'var(--text-muted)' }}>No jobs recorded yet.</p>}
          {jobs.map((j) => (
            <div key={j.job_id} style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--border-subtle)', fontSize: '0.85rem' }}>
              <span>{j.model_name} · {j.mode || 'scratch'}</span>
              <span style={{ color: j.status === 'COMPLETED' ? '#34d399' : '#94a3b8' }}>{j.status}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
