import React, { useState, useEffect } from 'react';
import { 
  LineChart, 
  Play, 
  CheckCircle2, 
  XCircle, 
  FileText, 
  Download, 
  Layers, 
  Award, 
  ExternalLink,
  RefreshCw,
  Terminal,
  Activity,
  Cpu,
  BookOpen
} from 'lucide-react';

export default function BenchmarksApp() {
  const [testResults, setTestResults] = useState(null);
  const [isRunningTests, setIsRunningTests] = useState(false);
  const [activeTab, setActiveTab] = useState('models');
  const [lmRows, setLmRows] = useState([]);
  const [agentRows, setAgentRows] = useState([]);

  useEffect(() => {
    fetch('/api/benchmarks/scaling-experiments')
      .then((r) => (r.ok ? r.json() : { lm_experiments: [] }))
      .then((d) => setLmRows(d.lm_experiments || []))
      .catch(() => {});
    fetch('/api/benchmarks/results')
      .then((r) => (r.ok ? r.json() : []))
      .then((rows) => setAgentRows(Array.isArray(rows) ? rows : []))
      .catch(() => {});
  }, []);

  const runPytestSuite = async () => {
    setIsRunningTests(true);
    setTestResults({ status: 'running', log: 'Starting Pytest test runner via FastAPI...' });
    try {
      const res = await fetch('/api/tests/run', { method: 'POST' });
      const data = await res.json();
      setTestResults(data);
    } catch (err) {
      setTestResults({ status: 'error', log: `Failed to execute tests: ${err.message}` });
    } finally {
      setIsRunningTests(false);
    }
  };

  return (
    <div className="benchmarks-app">
      <div className="app-header">
        <div>
          <h2>System Benchmarks & Verification</h2>
          <p>Real-time engine throughput, autonomous task benchmarks, and automated test diagnostics</p>
        </div>
        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button 
            className="btn btn-primary"
            onClick={runPytestSuite}
            disabled={isRunningTests}
          >
            {isRunningTests ? <RefreshCw size={16} className="spinning" /> : <Play size={16} />}
            Execute Unit Tests Suite
          </button>
        </div>
      </div>

      {/* Navigation tabs */}
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>
        <button 
          className={`btn ${activeTab === 'models' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('models')}
          style={{ fontSize: '0.875rem' }}
        >
          Model Performance & Throughput
        </button>
        <button 
          className={`btn ${activeTab === 'agent' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('agent')}
          style={{ fontSize: '0.875rem' }}
        >
          Coding Agent Benchmarks
        </button>
        <button 
          className={`btn ${activeTab === 'tests' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('tests')}
          style={{ fontSize: '0.875rem' }}
        >
          Automated Test Suite
        </button>
        <button 
          className={`btn ${activeTab === 'docs' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('docs')}
          style={{ fontSize: '0.875rem' }}
        >
          Production Documentation
        </button>
      </div>

      {/* Tab: Model Performance & Throughput */}
      {activeTab === 'models' && (
        <div>
          <div className="card" style={{ marginBottom: '1.5rem' }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.5rem' }}>
              Model Architecture & Throughput Metrics
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.25rem' }}>
              Evaluated on NVIDIA GeForce RTX 3050 (6GB VRAM) using FP16 mixed precision and FlashAttention SDPA.
            </p>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.875rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-muted)' }}>
                    <th style={{ padding: '0.75rem' }}>Model</th>
                    <th style={{ padding: '0.75rem' }}>Parameters</th>
                    <th style={{ padding: '0.75rem' }}>Config (L / H / D)</th>
                    <th style={{ padding: '0.75rem' }}>Tokenizer</th>
                    <th style={{ padding: '0.75rem' }}>Train Loss</th>
                    <th style={{ padding: '0.75rem' }}>Val Loss</th>
                    <th style={{ padding: '0.75rem' }}>Val Perplexity</th>
                    <th style={{ padding: '0.75rem' }}>Throughput</th>
                  </tr>
                </thead>
                <tbody>
                  {lmRows.map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                      <td style={{ padding: '0.75rem', fontWeight: 600, color: 'var(--accent-primary)' }}>{row.model_name}</td>
                      <td style={{ padding: '0.75rem' }}>{row.total_params?.toLocaleString?.() || row.total_params}</td>
                      <td style={{ padding: '0.75rem' }}>{row.n_layer}L / {row.n_head}H / {row.n_embd}D</td>
                      <td style={{ padding: '0.75rem' }}>
                        <span className="badge" style={{ backgroundColor: 'rgba(255,255,255,0.05)', fontSize: '0.75rem' }}>BPE-500</span>
                      </td>
                      <td style={{ padding: '0.75rem' }}>—</td>
                      <td style={{ padding: '0.75rem' }}>{row.val_loss}</td>
                      <td style={{ padding: '0.75rem', fontWeight: 600, color: '#34d399' }}>{row.val_perplexity}</td>
                      <td style={{ padding: '0.75rem', color: 'var(--text-muted)' }}>{Math.round(row.tokens_per_sec || 0)} tok/s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', gap: '1.5rem' }}>
            <div className="card">
              <h4 style={{ fontWeight: 600, marginBottom: '0.5rem' }}>Model Convergence & Scaling</h4>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
                Shows cross-entropy loss reduction across parameter configurations from 1.2M to 19.2M parameters.
              </p>
              <div style={{ background: 'rgba(0,0,0,0.3)', borderRadius: '8px', overflow: 'hidden', padding: '0.5rem', textAlign: 'center' }}>
                <img 
                  src="/results/scaling_laws_ppl.png" 
                  alt="Scaling Laws Curve" 
                  style={{ width: '100%', maxHeight: '300px', objectFit: 'contain' }}
                  onError={(e) => { e.target.style.display = 'none'; }}
                />
              </div>
            </div>

            <div className="card">
              <h4 style={{ fontWeight: 600, marginBottom: '0.5rem' }}>Tokenizer Compression Ratio</h4>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
                BPE-2048 delivers a 3.42x token compression over raw characters, multiplying effective context length.
              </p>
              <div style={{ background: 'rgba(0,0,0,0.3)', borderRadius: '8px', overflow: 'hidden', padding: '0.5rem', textAlign: 'center' }}>
                <img 
                  src="/results/tokenizer_comparison.png" 
                  alt="Tokenizer Comparison" 
                  style={{ width: '100%', maxHeight: '300px', objectFit: 'contain' }}
                  onError={(e) => { e.target.style.display = 'none'; }}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tab: Coding Agent Benchmarks */}
      {activeTab === 'agent' && (
        <div>
          <div className="card" style={{ marginBottom: '1.5rem' }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.5rem' }}>
              Coding Agent Task Performance (SWE-Style Evaluation)
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.25rem' }}>
              Comparing Zero-Shot direct code generation versus the Antigravity Autonomous Agent Control Loop (Reason-Act-Verify).
            </p>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.875rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-muted)' }}>
                    <th style={{ padding: '0.75rem' }}>Task · Mode</th>
                    <th style={{ padding: '0.75rem' }}>Success</th>
                    <th style={{ padding: '0.75rem' }}>Steps</th>
                    <th style={{ padding: '0.75rem' }}>Latency</th>
                    <th style={{ padding: '0.75rem' }}>Category</th>
                  </tr>
                </thead>
                <tbody>
                  {agentRows.map((b, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                      <td style={{ padding: '0.75rem', fontWeight: 600 }}>{b.task_id} · {b.agent_mode}</td>
                      <td style={{ padding: '0.75rem', color: b.success ? '#34d399' : '#f87171' }}>{String(b.success)}</td>
                      <td style={{ padding: '0.75rem', color: '#34d399', fontWeight: 700 }}>{b.steps_taken}</td>
                      <td style={{ padding: '0.75rem', color: 'var(--text-muted)' }}>{b.latency_sec}s</td>
                      <td style={{ padding: '0.75rem' }}>
                        <span className="badge badge-success">{b.category}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card">
            <h4 style={{ fontWeight: 600, marginBottom: '0.5rem' }}>Autonomous Task Resolution Graph</h4>
            <div style={{ background: 'rgba(0,0,0,0.3)', borderRadius: '8px', overflow: 'hidden', padding: '0.5rem', textAlign: 'center' }}>
              <img 
                src="/results/agent_benchmark_comparison.png" 
                alt="Agent Benchmark Comparison" 
                style={{ width: '100%', maxHeight: '340px', objectFit: 'contain' }}
                onError={(e) => { e.target.style.display = 'none'; }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Tab: Automated Test Suite */}
      {activeTab === 'tests' && (
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Automated Pytest Suite Runner</h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                Validates the entire codebase: Neural Architectures, Tokenizers, Safe Workspace Sandbox, and Agent Tool Execution.
              </p>
            </div>
            <button 
              className="btn btn-primary"
              onClick={runPytestSuite}
              disabled={isRunningTests}
            >
              {isRunningTests ? <RefreshCw size={16} className="spinning" /> : <Play size={16} />}
              {isRunningTests ? 'Running Pytest...' : 'Execute Full Test Suite'}
            </button>
          </div>

          <div style={{ background: '#0a0d14', borderRadius: '8px', padding: '1.25rem', fontFamily: 'monospace', fontSize: '0.85rem', border: '1px solid var(--border-color)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', marginBottom: '0.75rem', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '0.5rem' }}>
              <Terminal size={14} />
              <span>pytest tests/ -v</span>
            </div>
            
            {testResults ? (
              <div>
                <div style={{ marginBottom: '0.75rem', color: testResults.exit_code === 0 ? '#34d399' : '#f87171', fontWeight: 600 }}>
                  {testResults.exit_code === 0 ? 'STATUS: ALL TESTS PASSED' : 'STATUS: FAILED'}
                </div>
                <pre style={{ whiteSpace: 'pre-wrap', color: 'var(--text-primary)', maxHeight: '380px', overflowY: 'auto' }}>
                  {testResults.output || testResults.stdout || testResults.log || 'No output recorded.'}
                </pre>
              </div>
            ) : (
              <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '2rem' }}>
                Click "Execute Full Test Suite" to run all 16 tests in real time.
              </div>
            )}
          </div>
        </div>
      )}

      {/* Tab: Production Documentation */}
      {activeTab === 'docs' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.25rem' }}>
          <div className="card">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
              <BookOpen size={24} style={{ color: 'var(--accent-primary)' }} />
              <div>
                <h4 style={{ fontWeight: 600 }}>System Architecture</h4>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Component Design & Data Flow</span>
              </div>
            </div>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
              Detailed specification of the FastAPI backend, SmallGPT model architecture, and the secure workspace sandbox.
            </p>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', background: 'rgba(0,0,0,0.2)', padding: '0.5rem', borderRadius: '6px' }}>
              <code>docs/ARCHITECTURE.md</code>
            </div>
          </div>

          <div className="card">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
              <Terminal size={24} style={{ color: '#34d399' }} />
              <div>
                <h4 style={{ fontWeight: 600 }}>REST API Reference</h4>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>FastAPI Endpoints & Schemas</span>
              </div>
            </div>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
              Complete schemas for chat completions, agent execution, checkpoint hot-swapping, and telemetry.
            </p>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', background: 'rgba(0,0,0,0.2)', padding: '0.5rem', borderRadius: '6px' }}>
              <code>docs/API_REFERENCE.md</code>
            </div>
          </div>

          <div className="card">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
              <Cpu size={24} style={{ color: '#f59e0b' }} />
              <div>
                <h4 style={{ fontWeight: 600 }}>Deployment & Operations</h4>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Production Deployment</span>
              </div>
            </div>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
              Production instructions, environment configurations, and GPU optimization setups.
            </p>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', background: 'rgba(0,0,0,0.2)', padding: '0.5rem', borderRadius: '6px' }}>
              <code>docs/DEPLOYMENT.md</code>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
