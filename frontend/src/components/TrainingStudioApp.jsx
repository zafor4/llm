import React, { useState, useEffect } from 'react';
import { Flame, Play, CheckCircle2, AlertTriangle, Cpu, Zap, Activity } from 'lucide-react';

export default function TrainingStudioApp({ onModelTrained }) {
  const [modelName, setModelName] = useState('Custom-GPT-Micro');
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

  useEffect(() => {
    let interval = null;
    if (isTraining) {
      interval = setInterval(async () => {
        try {
          const res = await fetch('/api/trainer/status');
          const data = await res.json();
          setTrainingStatus(data);

          if (data.status === 'COMPLETED' || data.status === 'ERROR') {
            setIsTraining(false);
            if (data.status === 'COMPLETED' && onModelTrained) {
              onModelTrained(data.model_name);
            }
          }
        } catch (e) {
          console.error('Error polling training', e);
        }
      }, 500);
    }
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
      if (!data.success) {
        alert(`Failed to start training: ${data.detail || data.error}`);
        setIsTraining(false);
      }
    } catch (e) {
      alert(`Training request failed: ${e.message}`);
      setIsTraining(false);
    }
  };

  const pct = Math.min(
    100,
    ((trainingStatus?.current_step || 0) / Math.max(1, trainingStatus?.total_steps || maxSteps)) * 100
  );

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '440px 1fr', gap: '24px', height: 'calc(100vh - 120px)' }}>
      {/* Left Column: Hyperparameters & Dataset */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', overflowY: 'auto' }}>
        <div className="card-panel">
          <div className="card-title-row">
            <h3>Model Architecture Config</h3>
            <span style={{ fontSize: '0.6875rem', color: 'var(--accent-amber)', background: 'rgba(245,158,11,0.1)', padding: '2px 8px', borderRadius: '9999px', fontWeight: 600 }}>
              PyTorch Causal Transformer
            </span>
          </div>

          <div style={{ marginBottom: '12px' }}>
            <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '4px' }}>
              Model Checkpoint Identifier:
            </label>
            <input 
              type="text" 
              className="form-input" 
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
            />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '12px' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '4px' }}>
                Layers (n_layer):
              </label>
              <input 
                type="number" 
                className="form-input" 
                value={layers}
                min="2"
                max="12"
                onChange={(e) => setLayers(parseInt(e.target.value))}
              />
            </div>
            <div>
              <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '4px' }}>
                Heads (n_head):
              </label>
              <input 
                type="number" 
                className="form-input" 
                value={heads}
                min="2"
                max="12"
                onChange={(e) => setHeads(parseInt(e.target.value))}
              />
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '12px' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '4px' }}>
                Embedding Dim:
              </label>
              <input 
                type="number" 
                className="form-input" 
                value={dim}
                step="32"
                min="64"
                max="512"
                onChange={(e) => setDim(parseInt(e.target.value))}
              />
            </div>
            <div>
              <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '4px' }}>
                Context Window (L):
              </label>
              <input 
                type="number" 
                className="form-input" 
                value={blockSize}
                step="64"
                min="64"
                max="512"
                onChange={(e) => setBlockSize(parseInt(e.target.value))}
              />
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '14px' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '4px' }}>
                Max Steps:
              </label>
              <input 
                type="number" 
                className="form-input" 
                value={maxSteps}
                min="25"
                max="1000"
                onChange={(e) => setMaxSteps(parseInt(e.target.value))}
              />
            </div>
            <div>
              <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '4px' }}>
                Learning Rate:
              </label>
              <input 
                type="number" 
                className="form-input" 
                value={lr}
                step="0.0001"
                onChange={(e) => setLr(parseFloat(e.target.value))}
              />
            </div>
          </div>

          <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '4px' }}>
              Training Corpus (Leave empty to use curated algorithmic dataset):
            </label>
            <textarea 
              className="form-textarea code-font" 
              rows={4}
              placeholder="Paste custom training code or text snippets..."
              value={corpusText}
              onChange={(e) => setCorpusText(e.target.value)}
            />
          </div>

          <button 
            className="btn btn-primary" 
            style={{ width: '100%', background: 'linear-gradient(135deg, var(--accent-amber), var(--accent-rose))' }}
            disabled={isTraining}
            onClick={handleStartTraining}
          >
            <Flame size={16} />
            {isTraining ? 'Training in Progress...' : 'Launch GPU Training Job'}
          </button>
        </div>
      </div>

      {/* Right Column: Live Training Monitor & Loss Convergence */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', overflowY: 'auto' }}>
        {/* Progress Bar Panel */}
        <div className="card-panel">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <div>
              <span style={{ fontSize: '0.7rem', fontWeight: 700, textTransform: 'uppercase', color: isTraining ? '#38bdf8' : trainingStatus?.status === 'COMPLETED' ? '#10b981' : '#94a3b8' }}>
                Status: {trainingStatus?.status || 'IDLE'}
              </span>
              <h3 style={{ fontSize: '1.15rem', color: '#ffffff', marginTop: '2px' }}>
                {trainingStatus?.model_name || modelName}
              </h3>
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9rem', color: '#ffffff', fontWeight: 700 }}>
              Step {trainingStatus?.current_step || 0} / {trainingStatus?.total_steps || maxSteps}
            </div>
          </div>

          {/* Progress Bar */}
          <div style={{ height: '8px', background: 'rgba(255,255,255,0.06)', borderRadius: '9999px', overflow: 'hidden', marginBottom: '16px' }}>
            <div 
              style={{ 
                height: '100%', 
                width: `${pct}%`, 
                background: 'linear-gradient(90deg, var(--accent-amber), var(--accent-rose))', 
                transition: 'width 0.3s ease' 
              }} 
            />
          </div>

          {/* Real-time telemetry cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
            <div style={{ background: 'var(--bg-input)', padding: '12px', borderRadius: 'var(--radius-sm)', textAlign: 'center' }}>
              <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Cross-Entropy Loss</span>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.15rem', fontWeight: 700, color: '#f43f5e' }}>
                {trainingStatus?.current_loss ? trainingStatus.current_loss.toFixed(4) : '0.0000'}
              </div>
            </div>

            <div style={{ background: 'var(--bg-input)', padding: '12px', borderRadius: 'var(--radius-sm)', textAlign: 'center' }}>
              <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Val Perplexity</span>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.15rem', fontWeight: 700, color: '#10b981' }}>
                {trainingStatus?.val_perplexity ? trainingStatus.val_perplexity.toFixed(2) : '0.00'}
              </div>
            </div>

            <div style={{ background: 'var(--bg-input)', padding: '12px', borderRadius: 'var(--radius-sm)', textAlign: 'center' }}>
              <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Throughput</span>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.15rem', fontWeight: 700, color: '#38bdf8' }}>
                {trainingStatus?.tokens_per_sec ? `${trainingStatus.tokens_per_sec}` : '0'} tok/s
              </div>
            </div>

            <div style={{ background: 'var(--bg-input)', padding: '12px', borderRadius: 'var(--radius-sm)', textAlign: 'center' }}>
              <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Peak VRAM</span>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.15rem', fontWeight: 700, color: '#fbbf24' }}>
                {trainingStatus?.peak_vram_mb ? `${trainingStatus.peak_vram_mb} MB` : '0 MB'}
              </div>
            </div>
          </div>
        </div>

        {/* Training Log Table */}
        <div className="card-panel" style={{ flex: 1 }}>
          <div className="card-title-row">
            <h3>Step-by-Step Training Telemetry</h3>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Logged every 25 optimization steps
            </span>
          </div>

          <div style={{ overflowX: 'auto', maxHeight: '280px' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8125rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)', textAlign: 'left' }}>
                  <th style={{ padding: '8px' }}>Step</th>
                  <th style={{ padding: '8px' }}>Train Loss</th>
                  <th style={{ padding: '8px' }}>Val Loss</th>
                  <th style={{ padding: '8px' }}>Val PPL</th>
                  <th style={{ padding: '8px' }}>Throughput</th>
                  <th style={{ padding: '8px' }}>VRAM</th>
                </tr>
              </thead>
              <tbody>
                {trainingStatus?.history?.length > 0 ? (
                  trainingStatus.history.map((h, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                      <td style={{ padding: '8px', fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>#{h.step}</td>
                      <td style={{ padding: '8px', fontFamily: 'var(--font-mono)' }}>{h.train_loss.toFixed(4)}</td>
                      <td style={{ padding: '8px', fontFamily: 'var(--font-mono)' }}>{h.val_loss.toFixed(4)}</td>
                      <td style={{ padding: '8px', fontFamily: 'var(--font-mono)', color: '#10b981' }}>{h.val_perplexity.toFixed(2)}</td>
                      <td style={{ padding: '8px', fontFamily: 'var(--font-mono)' }}>{h.tokens_per_sec.toFixed(0)} tok/s</td>
                      <td style={{ padding: '8px', fontFamily: 'var(--font-mono)' }}>{h.peak_gpu_mb.toFixed(1)} MB</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="6" style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)' }}>
                      No training logs yet. Start a training job to see real-time metrics.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
