import React, { useEffect, useState } from 'react';

const PRESETS = [
  { id: 'quick', label: 'Quick try', steps: 50, hint: 'A few minutes. Good for a test.' },
  { id: 'standard', label: 'Standard', steps: 100, hint: 'Balanced default.' },
  { id: 'longer', label: 'Longer', steps: 250, hint: 'More time, usually better fit.' },
];

export default function TrainingStudioApp({ onModelTrained, modelsList, continueTarget }) {
  const localModels = (modelsList || []).filter((m) => m.provider === 'local');
  const [mode, setMode] = useState(continueTarget ? 'continue' : 'scratch');
  const [modelName, setModelName] = useState(continueTarget ? `${continueTarget}-v2` : 'My-First-Model');
  const [baseModel, setBaseModel] = useState(continueTarget || localModels[0]?.id || '');
  const [preset, setPreset] = useState('standard');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [layers, setLayers] = useState(4);
  const [heads, setHeads] = useState(4);
  const [dim, setDim] = useState(192);
  const [blockSize, setBlockSize] = useState(128);
  const [lr, setLr] = useState(0.0005);
  const [maxSteps, setMaxSteps] = useState(100);
  const [corpusText, setCorpusText] = useState('');
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [jobs, setJobs] = useState([]);

  useEffect(() => {
    if (continueTarget) {
      setMode('continue');
      setBaseModel(continueTarget);
      setModelName(`${continueTarget}-v2`);
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
      if (['COMPLETED', 'ERROR', 'CANCELLED'].includes(data.status)) {
        setIsTraining(false);
        if (data.status === 'COMPLETED') onModelTrained?.(data.model_name);
      }
    }, 500);
    return () => clearInterval(interval);
  }, [isTraining, onModelTrained]);

  const applyPreset = (id) => {
    const p = PRESETS.find((x) => x.id === id);
    setPreset(id);
    if (p) setMaxSteps(p.steps);
  };

  const start = async () => {
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
          batch_size: 16,
          max_steps: maxSteps,
          corpus_text: corpusText,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        alert(typeof data.detail === 'string' ? data.detail : 'Could not start training.');
        setIsTraining(false);
      }
    } catch (e) {
      alert(e.message);
      setIsTraining(false);
    }
  };

  const pct = Math.min(100, ((trainingStatus?.current_step || 0) / Math.max(1, trainingStatus?.total_steps || maxSteps)) * 100);

  return (
    <div className="split-2">
      <div className="card-panel">
        <h3>New training job</h3>
        <label className="field-label">What kind of job?</label>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className={`btn ${mode === 'scratch' ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setMode('scratch')}>New model</button>
          <button className={`btn ${mode === 'continue' ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setMode('continue')}>Improve one I have</button>
        </div>

        <label className="field-label">Name for the result</label>
        <input className="form-input" value={modelName} onChange={(e) => setModelName(e.target.value)} />

        {mode === 'continue' && (
          <>
            <label className="field-label">Start from</label>
            <select className="form-select" value={baseModel} onChange={(e) => setBaseModel(e.target.value)}>
              {localModels.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select>
          </>
        )}

        <label className="field-label">How long?</label>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {PRESETS.map((p) => (
            <button
              key={p.id}
              className={`btn ${preset === p.id ? 'btn-primary' : 'btn-secondary'}`}
              style={{ justifyContent: 'space-between' }}
              onClick={() => applyPreset(p.id)}
            >
              <span>{p.label}</span>
              <span className="hint">{p.hint}</span>
            </button>
          ))}
        </div>

        <label className="field-label">Your own text (optional)</label>
        <textarea
          className="form-textarea"
          rows={4}
          placeholder="Leave blank to use the built-in coding examples."
          value={corpusText}
          onChange={(e) => setCorpusText(e.target.value)}
        />

        <button className="btn btn-secondary" style={{ marginTop: 12 }} onClick={() => setShowAdvanced(!showAdvanced)}>
          {showAdvanced ? 'Hide extra settings' : 'Show extra settings'}
        </button>
        {showAdvanced && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginTop: 10 }}>
            <div><label className="field-label">Layers</label><input className="form-input" type="number" value={layers} onChange={(e) => setLayers(+e.target.value)} /></div>
            <div><label className="field-label">Heads</label><input className="form-input" type="number" value={heads} onChange={(e) => setHeads(+e.target.value)} /></div>
            <div><label className="field-label">Size</label><input className="form-input" type="number" value={dim} onChange={(e) => setDim(+e.target.value)} /></div>
            <div><label className="field-label">Context</label><input className="form-input" type="number" value={blockSize} onChange={(e) => setBlockSize(+e.target.value)} /></div>
            <div><label className="field-label">Steps</label><input className="form-input" type="number" value={maxSteps} onChange={(e) => { setMaxSteps(+e.target.value); setPreset('custom'); }} /></div>
            <div><label className="field-label">Learning rate</label><input className="form-input" type="number" step="0.0001" value={lr} onChange={(e) => setLr(+e.target.value)} /></div>
          </div>
        )}

        <div style={{ display: 'flex', gap: 8, marginTop: 16 }}>
          <button className="btn btn-primary" disabled={isTraining} onClick={start}>
            {isTraining ? 'Training…' : 'Start training'}
          </button>
          {isTraining && (
            <button className="btn btn-secondary" onClick={() => fetch('/api/trainer/cancel', { method: 'POST' })}>Stop</button>
          )}
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        <div className="card-panel">
          <div className="hint">{trainingStatus?.status || 'Waiting'}</div>
          <h3 style={{ margin: '4px 0 12px' }}>{trainingStatus?.model_name || modelName}</h3>
          <div className="progress-track" style={{ marginBottom: 14 }}>
            <div className="progress-fill" style={{ width: `${pct}%` }} />
          </div>
          <div className="stat-grid">
            <div><div className="hint">Step</div><strong>{trainingStatus?.current_step || 0} / {trainingStatus?.total_steps || maxSteps}</strong></div>
            <div><div className="hint">How well it fits</div><strong>{trainingStatus?.val_perplexity || '—'}</strong></div>
            <div><div className="hint">Speed</div><strong>{trainingStatus?.tokens_per_sec || 0}</strong></div>
          </div>
          {trainingStatus?.error_message && <p className="notice error" style={{ marginTop: 12 }}>{trainingStatus.error_message}</p>}
        </div>
        <div className="card-panel">
          <h3 style={{ marginBottom: 10 }}>Recent jobs</h3>
          {jobs.length === 0 && <p className="hint">Nothing trained yet.</p>}
          {jobs.map((j) => (
            <div key={j.job_id} style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--border-subtle)' }}>
              <span>{j.model_name}</span>
              <span className="hint">{j.status === 'COMPLETED' ? 'Done' : j.status}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
