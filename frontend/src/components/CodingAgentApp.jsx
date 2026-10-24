import React, { useEffect, useState } from 'react';
import { Bot, Play, CheckCircle2, XCircle, RotateCw, ShieldCheck, ChevronDown, ChevronRight } from 'lucide-react';

export default function CodingAgentApp({ activeModel, modelsList }) {
  const [tasks, setTasks] = useState([]);
  const [selectedTask, setSelectedTask] = useState('');
  const [customGoal, setCustomGoal] = useState('');
  const [mode, setMode] = useState('verified');
  const [planner, setPlanner] = useState('auto');
  const [model, setModel] = useState(activeModel || '');
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [expanded, setExpanded] = useState({});

  useEffect(() => {
    fetch('/api/agent/tasks').then((r) => r.json()).then((data) => {
      setTasks(data);
      if (data[0]) setSelectedTask(data[0].task_id);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (activeModel) setModel(activeModel);
  }, [activeModel]);

  const run = async () => {
    setIsRunning(true);
    setResult(null);
    try {
      const res = await fetch('/api/agent/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task_id: selectedTask || null,
          goal: customGoal,
          mode,
          planner,
          model,
          max_steps: 8,
        }),
      });
      const data = await res.json();
      setResult(data);
      const open = {};
      (data.steps || []).forEach((_, i) => { open[i] = true; });
      setExpanded(open);
    } catch (e) {
      alert(e.message);
    } finally {
      setIsRunning(false);
    }
  };

  const current = tasks.find((t) => t.task_id === selectedTask);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '380px 1fr', gap: 24, height: 'calc(100vh - 120px)' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16, overflowY: 'auto' }}>
        <div className="card-panel">
          <h3>Playground run</h3>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', margin: '8px 0 14px' }}>
            Auto uses the oracle on benchmark tasks. Choose Active / hub planner to drive tools with an instruct model (Ollama or HF).
          </p>
          <label>Benchmark task</label>
          <select className="form-select" value={selectedTask} onChange={(e) => setSelectedTask(e.target.value)}>
            <option value="">Custom goal only</option>
            {tasks.map((t) => <option key={t.task_id} value={t.task_id}>{t.task_id}: {t.name}</option>)}
          </select>
          {current && <p style={{ fontSize: '0.8rem', marginTop: 8, color: 'var(--text-secondary)' }}>{current.goal}</p>}
          <label style={{ marginTop: 12, display: 'block' }}>Custom goal</label>
          <textarea className="form-textarea" rows={3} value={customGoal} onChange={(e) => setCustomGoal(e.target.value)} placeholder="Used when no task is selected, or as extra context." />
          <label style={{ marginTop: 12, display: 'block' }}>Planner</label>
          <select className="form-select" value={planner} onChange={(e) => setPlanner(e.target.value)}>
            <option value="auto">Auto (oracle on benches, hub if capable)</option>
            <option value="oracle">Oracle (reproducible benches)</option>
            <option value="active">Active hub model</option>
          </select>
          <label style={{ marginTop: 12, display: 'block' }}>Engine</label>
          <select className="form-select" value={model} onChange={(e) => setModel(e.target.value)}>
            {(modelsList || []).map((m) => <option key={m.id || m} value={m.id || m}>{m.name || m}</option>)}
          </select>
          <div style={{ marginTop: 12 }}>
            {['verified', 'react', 'single_turn'].map((id) => (
              <button key={id} className={`btn ${mode === id ? 'btn-primary' : 'btn-secondary'}`} style={{ marginRight: 6, marginBottom: 6 }} onClick={() => setMode(id)}>{id}</button>
            ))}
          </div>
          <button className="btn btn-primary" style={{ width: '100%', marginTop: 8 }} disabled={isRunning} onClick={run}>
            {isRunning ? <><RotateCw size={16} /> Running…</> : <><Play size={16} /> Execute</>}
          </button>
        </div>
        <div className="card-panel">
          <div style={{ display: 'flex', gap: 8, color: 'var(--accent-emerald)', fontWeight: 700, marginBottom: 8 }}>
            <ShieldCheck size={16} /> Sandbox
          </div>
          <ul style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', paddingLeft: 18 }}>
            <li>Path jail inside a temp workspace</li>
            <li>Python / pytest timeouts</li>
            <li>Observations tagged untrusted</li>
          </ul>
        </div>
      </div>

      <div style={{ overflowY: 'auto' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 10, marginBottom: 14 }}>
          {[
            ['Status', result?.status || (isRunning ? 'RUNNING' : 'IDLE')],
            ['Planner', result?.planner || planner],
            ['Steps', result?.current_step || 0],
            ['Tools', result?.total_tool_calls || 0],
            ['Latency', result?.latency_sec ? `${result.latency_sec}s` : '0s'],
          ].map(([k, v]) => (
            <div key={k} className="card-panel" style={{ padding: 12 }}>
              <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>{k}</div>
              <div style={{ fontWeight: 700 }}>{v}</div>
            </div>
          ))}
        </div>
        <div className="card-panel">
          {(result?.steps || []).map((step, idx) => (
            <div key={idx} style={{ borderLeft: `4px solid ${step.observation_success ? '#10b981' : '#f43f5e'}`, marginBottom: 10, background: 'var(--bg-input)', borderRadius: 8 }}>
              <div style={{ padding: 10, cursor: 'pointer', display: 'flex', justifyContent: 'space-between' }} onClick={() => setExpanded((p) => ({ ...p, [idx]: !p[idx] }))}>
                <span>{expanded[idx] ? <ChevronDown size={14} /> : <ChevronRight size={14} />} Step {step.step_index} · {step.action_tool}</span>
                {step.observation_success ? <CheckCircle2 size={14} color="#10b981" /> : <XCircle size={14} color="#f43f5e" />}
              </div>
              {expanded[idx] && (
                <div style={{ padding: 12, fontSize: '0.8rem' }}>
                  <div><strong>Thought:</strong> {step.thought}</div>
                  <pre style={{ whiteSpace: 'pre-wrap', marginTop: 8 }}>{step.observation}</pre>
                </div>
              )}
            </div>
          ))}
          {!result && !isRunning && (
            <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: 40 }}>
              <Bot size={36} style={{ opacity: 0.3 }} />
              <p>Run a benchmark or a custom goal.</p>
            </div>
          )}
          {result?.final_report && <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.75rem', marginTop: 12 }}>{result.final_report}</pre>}
        </div>
      </div>
    </div>
  );
}
