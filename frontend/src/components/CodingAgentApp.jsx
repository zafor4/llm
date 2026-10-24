import React, { useState, useEffect } from 'react';
import { 
  Bot, 
  Play, 
  CheckCircle2, 
  XCircle, 
  RotateCw, 
  Terminal, 
  FileCode, 
  ShieldCheck, 
  Clock, 
  Wrench,
  ChevronDown,
  ChevronRight
} from 'lucide-react';

export default function CodingAgentApp() {
  const [tasks, setTasks] = useState([]);
  const [selectedTask, setSelectedTask] = useState('TASK_01');
  const [customGoal, setCustomGoal] = useState('');
  const [mode, setMode] = useState('verified'); // 'verified', 'react', 'single_turn'
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [expandedSteps, setExpandedSteps] = useState({});

  useEffect(() => {
    fetch('/api/agent/tasks')
      .then((res) => res.json())
      .then((data) => {
        setTasks(data);
        if (data.length > 0) setSelectedTask(data[0].task_id);
      })
      .catch((e) => console.error('Failed to load tasks', e));
  }, []);

  const handleRunAgent = async () => {
    setIsRunning(true);
    setResult(null);
    setExpandedSteps({});

    try {
      const res = await fetch('/api/agent/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task_id: selectedTask,
          goal: customGoal,
          mode: mode,
          max_steps: 8,
        }),
      });

      const data = await res.json();
      setResult(data);
      // Expand all steps by default
      const initialExpanded = {};
      data.steps?.forEach((_, idx) => {
        initialExpanded[idx] = true;
      });
      setExpandedSteps(initialExpanded);
    } catch (e) {
      alert(`Agent execution failed: ${e.message}`);
    } finally {
      setIsRunning(false);
    }
  };

  const toggleStep = (idx) => {
    setExpandedSteps((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  const currentTaskObj = tasks.find((t) => t.task_id === selectedTask);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '380px 1fr', gap: '24px', height: 'calc(100vh - 120px)' }}>
      {/* Left: Task & Mode Configuration */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '18px', overflowY: 'auto' }}>
        <div className="card-panel">
          <div className="card-title-row">
            <h3>Autonomous Task Config</h3>
            <span style={{ fontSize: '0.6875rem', color: 'var(--accent-blue)', background: 'rgba(59,130,246,0.1)', padding: '2px 8px', borderRadius: '9999px', fontWeight: 600 }}>
              Isolated Sandbox
            </span>
          </div>

          <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px' }}>
              Select Benchmark Task:
            </label>
            <select 
              className="form-select"
              value={selectedTask}
              onChange={(e) => setSelectedTask(e.target.value)}
            >
              {tasks.map((t) => (
                <option key={t.task_id} value={t.task_id}>
                  {t.task_id}: {t.name}
                </option>
              ))}
            </select>
          </div>

          {currentTaskObj && (
            <div style={{ background: 'var(--bg-input)', padding: '12px', borderRadius: 'var(--radius-sm)', border: '1px dashed var(--border-subtle)', marginBottom: '16px' }}>
              <div style={{ fontSize: '0.6875rem', fontWeight: 700, color: '#38bdf8', textTransform: 'uppercase', marginBottom: '4px' }}>
                Category: {currentTaskObj.category}
              </div>
              <div style={{ fontSize: '0.8125rem', color: 'var(--text-main)', lineHeight: 1.45 }}>
                {currentTaskObj.goal}
              </div>
            </div>
          )}

          <div style={{ marginBottom: '18px' }}>
            <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px' }}>
              Control Architecture:
            </label>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {[
                { id: 'verified', label: 'Mode C: Verified Agent', sub: 'Observe-Act-Reflect with pytest verification' },
                { id: 'react', label: 'Mode B: ReAct Agent', sub: 'Iterative Observe-Act without verification' },
                { id: 'single_turn', label: 'Mode A: Single-Response LLM', sub: 'Zero-shot prompt without tool loop' },
              ].map((m) => (
                <div 
                  key={m.id}
                  onClick={() => setMode(m.id)}
                  style={{
                    padding: '8px 12px',
                    borderRadius: 'var(--radius-sm)',
                    background: mode === m.id ? 'rgba(59,130,246,0.12)' : 'var(--bg-input)',
                    border: mode === m.id ? '1px solid var(--accent-blue)' : '1px solid var(--border-subtle)',
                    cursor: 'pointer',
                    transition: 'var(--transition)',
                  }}
                >
                  <div style={{ fontSize: '0.8125rem', fontWeight: 600, color: mode === m.id ? '#ffffff' : 'var(--text-secondary)' }}>
                    {m.label}
                  </div>
                  <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)' }}>
                    {m.sub}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <button 
            className="btn btn-primary" 
            style={{ width: '100%' }}
            disabled={isRunning}
            onClick={handleRunAgent}
          >
            {isRunning ? (
              <>
                <RotateCw size={16} className="animate-spin" />
                Executing Control Loop...
              </>
            ) : (
              <>
                <Play size={16} />
                Execute Autonomous Agent
              </>
            )}
          </button>
        </div>

        {/* Sandbox Boundaries Box */}
        <div className="card-panel" style={{ padding: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-emerald)', fontSize: '0.8125rem', fontWeight: 700, marginBottom: '8px' }}>
            <ShieldCheck size={16} />
            Active Security Guarantees
          </div>
          <ul style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', paddingLeft: '18px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <li>Strict path resolution within workspace root</li>
            <li>Subprocess code runner with 10s timeout</li>
            <li>Tool outputs tagged as untrusted observations</li>
            <li>Approval gate on destructive file modifications</li>
          </ul>
        </div>
      </div>

      {/* Right: Live Telemetry & Step Trace */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', overflowY: 'auto' }}>
        {/* Telemetry Counters */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '12px' }}>
          <div className="card-panel" style={{ padding: '12px', textAlign: 'center' }}>
            <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Status</span>
            <div style={{ fontSize: '1rem', fontWeight: 700, color: result?.status === 'PASSED' ? '#10b981' : result?.status === 'FAILED' ? '#f43f5e' : '#94a3b8' }}>
              {isRunning ? 'RUNNING' : result?.status || 'IDLE'}
            </div>
          </div>
          <div className="card-panel" style={{ padding: '12px', textAlign: 'center' }}>
            <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Steps</span>
            <div style={{ fontSize: '1rem', fontWeight: 700, color: '#ffffff' }}>
              {result?.current_step || 0}
            </div>
          </div>
          <div className="card-panel" style={{ padding: '12px', textAlign: 'center' }}>
            <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Tool Calls</span>
            <div style={{ fontSize: '1rem', fontWeight: 700, color: '#ffffff' }}>
              {result?.total_tool_calls || 0}
            </div>
          </div>
          <div className="card-panel" style={{ padding: '12px', textAlign: 'center' }}>
            <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Self-Corrections</span>
            <div style={{ fontSize: '1rem', fontWeight: 700, color: '#f59e0b' }}>
              {result?.error_recovery_attempts || 0}
            </div>
          </div>
          <div className="card-panel" style={{ padding: '12px', textAlign: 'center' }}>
            <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Latency</span>
            <div style={{ fontSize: '1rem', fontWeight: 700, color: '#ffffff' }}>
              {result?.latency_sec ? `${result.latency_sec}s` : '0.00s'}
            </div>
          </div>
        </div>

        {/* Timeline of Steps */}
        <div className="card-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div className="card-title-row">
            <h3>Execution Trace & Observability</h3>
            {result?.verification_message && (
              <span style={{ fontSize: '0.75rem', color: result.status === 'PASSED' ? '#34d399' : '#f87171' }}>
                {result.verification_message}
              </span>
            )}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', overflowY: 'auto' }}>
            {(!result?.steps || result.steps.length === 0) && !isRunning && (
              <div style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-muted)' }}>
                <Bot size={40} style={{ margin: '0 auto 12px', opacity: 0.3 }} />
                <p>Click "Execute Autonomous Agent" to run the control loop.</p>
              </div>
            )}

            {isRunning && (
              <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
                <RotateCw size={24} className="animate-spin" style={{ margin: '0 auto 12px', color: 'var(--accent-blue)' }} />
                <p>Agent is inspecting workspace, executing tools, and verifying code assertions...</p>
              </div>
            )}

            {result?.steps?.map((step, idx) => {
              const isExpanded = expandedSteps[idx];
              const isPass = step.observation_success;
              return (
                <div 
                  key={idx}
                  style={{
                    background: 'var(--bg-input)',
                    border: '1px solid var(--border-subtle)',
                    borderLeft: `4px solid ${isPass ? 'var(--accent-emerald)' : 'var(--accent-rose)'}`,
                    borderRadius: 'var(--radius-sm)',
                    overflow: 'hidden',
                  }}
                >
                  <div 
                    onClick={() => toggleStep(idx)}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      padding: '10px 14px',
                      cursor: 'pointer',
                      background: 'rgba(255,255,255,0.02)',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', fontWeight: 700, color: '#38bdf8' }}>
                        STEP {step.step_index}
                      </span>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', background: 'rgba(30,41,59,0.8)', padding: '2px 8px', borderRadius: '4px', color: '#cbd5e1' }}>
                        {step.action_tool}({JSON.stringify(step.action_params)})
                      </span>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      {isPass ? (
                        <span style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#10b981', fontSize: '0.6875rem', fontWeight: 700 }}>
                          <CheckCircle2 size={12} /> PASS
                        </span>
                      ) : (
                        <span style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#f43f5e', fontSize: '0.6875rem', fontWeight: 700 }}>
                          <XCircle size={12} /> FAIL
                        </span>
                      )}
                    </div>
                  </div>

                  {isExpanded && (
                    <div style={{ padding: '12px 14px', borderTop: '1px solid var(--border-subtle)', fontSize: '0.8125rem' }}>
                      <div style={{ color: '#e2e8f0', marginBottom: '8px' }}>
                        <strong style={{ color: '#94a3b8' }}>Thought:</strong> {step.thought}
                      </div>

                      <div style={{ background: '#050810', padding: '8px 12px', borderRadius: '4px', fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: '#94a3b8', maxHeight: '110px', overflowY: 'auto', whiteSpace: 'pre-wrap' }}>
                        {step.observation}
                      </div>

                      {step.reflection && (
                        <div style={{ marginTop: '8px', padding: '6px 10px', background: 'rgba(245,158,11,0.1)', border: '1px solid rgba(245,158,11,0.25)', borderRadius: '4px', color: '#fbbf24', fontSize: '0.75rem' }}>
                          ⚡ <strong>Self-Correction Diagnostic:</strong> {step.reflection}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {result?.final_report && (
            <div style={{ marginTop: '16px', padding: '14px', background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.3)', borderRadius: 'var(--radius-sm)' }}>
              <h4 style={{ fontSize: '0.875rem', fontWeight: 700, color: '#34d399', marginBottom: '6px' }}>
                Verified Final Execution Report
              </h4>
              <pre style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: '#e2e8f0', whiteSpace: 'pre-wrap' }}>
                {result.final_report}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
