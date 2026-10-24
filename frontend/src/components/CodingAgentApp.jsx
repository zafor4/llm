import React, { useEffect, useState } from 'react';

export default function CodingAgentApp({ activeModel, modelsList }) {
  const [tasks, setTasks] = useState([]);
  const [selectedTask, setSelectedTask] = useState('');
  const [customGoal, setCustomGoal] = useState('');
  const [showMore, setShowMore] = useState(false);
  const [mode, setMode] = useState('verified');
  const [planner, setPlanner] = useState('auto');
  const [model, setModel] = useState(activeModel || '');
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState(null);

  useEffect(() => {
    fetch('/api/agent/tasks')
      .then((r) => r.json())
      .then((data) => {
        setTasks(data);
        if (data[0]) setSelectedTask(data[0].task_id);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (activeModel) setModel(activeModel);
  }, [activeModel]);

  const current = tasks.find((t) => t.task_id === selectedTask);

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
      setResult(await res.json());
    } catch (e) {
      alert(e.message);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="split-2">
      <div className="card-panel">
        <h3>Pick a practice job</h3>
        <p className="hint" style={{ margin: '8px 0 14px' }}>
          These are safe practice tasks. The agent works in a temporary folder and cannot touch the rest of your computer.
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {tasks.map((t) => (
            <button
              key={t.task_id}
              className={`btn ${selectedTask === t.task_id ? 'btn-primary' : 'btn-secondary'}`}
              style={{ justifyContent: 'flex-start', textAlign: 'left', height: 'auto', padding: '10px 12px' }}
              onClick={() => setSelectedTask(t.task_id)}
            >
              <span>
                <strong>{t.name}</strong>
                <span className="nav-sub">{t.category}</span>
              </span>
            </button>
          ))}
        </div>
        {current && <p className="hint" style={{ marginTop: 12 }}>{current.goal}</p>}

        <button className="btn btn-secondary" style={{ marginTop: 14 }} onClick={() => setShowMore(!showMore)}>
          {showMore ? 'Hide extra options' : 'More options'}
        </button>
        {showMore && (
          <div>
            <label className="field-label">Your own goal (optional)</label>
            <textarea className="form-textarea" rows={3} value={customGoal} onChange={(e) => setCustomGoal(e.target.value)} />
            <label className="field-label">How it should work</label>
            <select className="form-select" value={planner} onChange={(e) => setPlanner(e.target.value)}>
              <option value="auto">Recommended (built-in practice plan)</option>
              <option value="oracle">Always use the practice plan</option>
              <option value="active">Let the current model decide</option>
            </select>
            <label className="field-label">Model</label>
            <select className="form-select" value={model} onChange={(e) => setModel(e.target.value)}>
              {(modelsList || []).map((m) => <option key={m.id || m} value={m.id || m}>{m.name || m}</option>)}
            </select>
            <label className="field-label">Check the result?</label>
            <select className="form-select" value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="verified">Yes, confirm it worked</option>
              <option value="react">Just run the steps</option>
              <option value="single_turn">One answer, no tools</option>
            </select>
          </div>
        )}

        <button className="btn btn-primary" style={{ width: '100%', marginTop: 16 }} disabled={isRunning} onClick={run}>
          {isRunning ? 'Working…' : 'Start'}
        </button>
      </div>

      <div className="card-panel">
        <div className="stat-grid" style={{ marginBottom: 16 }}>
          <div><div className="hint">Status</div><strong>{isRunning ? 'Working' : (result?.status || 'Ready')}</strong></div>
          <div><div className="hint">Steps</div><strong>{result?.current_step || 0}</strong></div>
          <div><div className="hint">Time</div><strong>{result?.latency_sec ? `${result.latency_sec}s` : '—'}</strong></div>
        </div>
        {!result && !isRunning && (
          <div className="empty-state">
            <p>Choose a job on the left and press Start. You will see each step here.</p>
          </div>
        )}
        {(result?.steps || []).map((step) => (
          <div key={step.step_index} style={{ padding: '10px 0', borderBottom: '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <strong>Step {step.step_index}</strong>
              <span className={`chip ${step.observation_success ? 'on' : ''}`}>
                {step.observation_success ? 'Worked' : 'Needs a fix'}
              </span>
            </div>
            <p style={{ margin: '6px 0' }}>{step.thought}</p>
            <p className="hint">Action: {step.action_tool}</p>
          </div>
        ))}
        {result?.final_report && (
          <pre className="hint" style={{ whiteSpace: 'pre-wrap', marginTop: 14 }}>{result.final_report}</pre>
        )}
      </div>
    </div>
  );
}
