/**
 * SmallGPT & Autonomous Agent Dashboard — Frontend Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements - Navigation
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  // DOM Elements - Agent Studio
  const taskSelect = document.getElementById('task-select');
  const taskGoalBox = document.getElementById('task-goal-box');
  const agentModeSelect = document.getElementById('agent-mode-select');
  const btnRunAgent = document.getElementById('btn-run-agent');
  const statStatus = document.getElementById('stat-status');
  const statSteps = document.getElementById('stat-steps');
  const statTools = document.getElementById('stat-tools');
  const statRecoveries = document.getElementById('stat-recoveries');
  const statLatency = document.getElementById('stat-latency');
  const stepsContainer = document.getElementById('steps-timeline-container');
  const finalReportContainer = document.getElementById('final-report-container');
  const finalReportText = document.getElementById('final-report-text');

  // DOM Elements - SmallGPT Playground
  const promptInput = document.getElementById('prompt-input');
  const tempSlider = document.getElementById('temperature-slider');
  const tempVal = document.getElementById('temp-val');
  const tokensSlider = document.getElementById('tokens-slider');
  const tokensVal = document.getElementById('tokens-val');
  const toppSlider = document.getElementById('topp-slider');
  const toppVal = document.getElementById('topp-val');
  const btnGenerate = document.getElementById('btn-generate');
  const gptOutput = document.getElementById('gpt-output-text');
  const gptStatTokens = document.getElementById('gpt-stat-tokens');
  const gptStatLatency = document.getElementById('gpt-stat-latency');
  const gptStatSpeed = document.getElementById('gpt-stat-speed');

  // DOM Elements - Tests
  const btnRunTests = document.getElementById('btn-run-tests');
  const testsTerminal = document.getElementById('tests-terminal-output');

  // DOM Elements - System
  const gpuInfoText = document.getElementById('gpu-info-text');

  let benchmarkTasksCache = [];

  // ---------------------------------------------------------
  // 1. Tab Switching
  // ---------------------------------------------------------
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const targetId = btn.getAttribute('data-tab');
      document.getElementById(targetId).classList.add('active');
    });
  });

  // ---------------------------------------------------------
  // 2. Fetch System Status & Tasks
  // ---------------------------------------------------------
  async function fetchStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      if (data.cuda_available) {
        gpuInfoText.textContent = `${data.device} (${data.vram_total_mb} MB VRAM) | PyTorch ${data.pytorch_version}`;
      } else {
        gpuInfoText.textContent = `CPU Engine | PyTorch ${data.pytorch_version}`;
      }
    } catch (e) {
      gpuInfoText.textContent = 'Server Online';
    }
  }

  async function fetchTasks() {
    try {
      const res = await fetch('/api/tasks');
      benchmarkTasksCache = await res.json();
      updateTaskGoalDisplay();
    } catch (e) {
      console.error('Failed to load tasks', e);
    }
  }

  function updateTaskGoalDisplay() {
    const selectedId = taskSelect.value;
    const task = benchmarkTasksCache.find(t => t.task_id === selectedId);
    if (task) {
      taskGoalBox.innerHTML = `<strong>Goal:</strong> ${task.goal}<br><span style="color:#94a3b8; font-size:0.8rem; margin-top:4px; display:inline-block;">Category: ${task.category}</span>`;
    }
  }

  taskSelect.addEventListener('change', updateTaskGoalDisplay);

  // ---------------------------------------------------------
  // 3. Agent Execution
  // ---------------------------------------------------------
  btnRunAgent.addEventListener('click', async () => {
    const taskId = taskSelect.value;
    const mode = agentModeSelect.value;

    btnRunAgent.disabled = true;
    btnRunAgent.innerHTML = `<span class="pulse-indicator"></span> Running Agent...`;
    statStatus.textContent = 'RUNNING';
    statStatus.style.color = '#38bdf8';
    stepsContainer.innerHTML = '<div class="empty-state"><p>Observing workspace, planning subtasks, and executing tools...</p></div>';
    finalReportContainer.style.display = 'none';

    try {
      const startTime = performance.now();
      const res = await fetch('/api/agent/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: taskId, mode: mode })
      });
      const data = await res.json();
      const clientElapsed = ((performance.now() - startTime) / 1000).toFixed(2);

      // Update telemetry
      const isSuccess = data.success;
      statStatus.textContent = isSuccess ? 'PASSED' : 'FAILED';
      statStatus.style.color = isSuccess ? '#10b981' : '#f43f5e';
      statSteps.textContent = data.current_step;
      statTools.textContent = data.total_tool_calls;
      statRecoveries.textContent = data.error_recovery_attempts;
      statLatency.textContent = `${data.latency_sec}s`;

      // Render step timeline
      renderStepTimeline(data.history);

      // Render final report
      if (data.final_report) {
        finalReportContainer.style.display = 'block';
        finalReportText.textContent = data.final_report;
      }
    } catch (e) {
      statStatus.textContent = 'ERROR';
      statStatus.style.color = '#f43f5e';
      stepsContainer.innerHTML = `<div class="step-card fail"><p>Execution error: ${e.message}</p></div>`;
    } finally {
      btnRunAgent.disabled = false;
      btnRunAgent.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg> Execute Autonomous Agent`;
    }
  });

  function renderStepTimeline(history) {
    if (!history || history.length === 0) {
      stepsContainer.innerHTML = '<div class="empty-state"><p>No actions recorded.</p></div>';
      return;
    }

    stepsContainer.innerHTML = '';
    history.forEach(step => {
      const card = document.createElement('div');
      const passClass = step.observation_success ? 'pass' : 'fail';
      card.className = `step-card ${passClass}`;

      let reflectionHtml = '';
      if (step.reflection) {
        reflectionHtml = `<div class="step-reflection"><strong>Self-Correction / Reflection:</strong> ${escapeHtml(step.reflection)}</div>`;
      }

      card.innerHTML = `
        <div class="step-header">
          <span class="step-num">STEP ${step.step_index}</span>
          <span class="step-action-tag">${escapeHtml(step.action_tool)}(${escapeHtml(JSON.stringify(step.action_params))})</span>
        </div>
        <div class="step-thought"><strong>Thought:</strong> ${escapeHtml(step.thought)}</div>
        <div class="step-obs-preview">${escapeHtml(step.observation)}</div>
        ${reflectionHtml}
      `;
      stepsContainer.appendChild(card);
    });
  }

  // ---------------------------------------------------------
  // 4. SmallGPT Playground Controls
  // ---------------------------------------------------------
  tempSlider.addEventListener('input', (e) => tempVal.textContent = parseFloat(e.target.value).toFixed(2));
  tokensSlider.addEventListener('input', (e) => tokensVal.textContent = e.target.value);
  toppSlider.addEventListener('input', (e) => toppVal.textContent = parseFloat(e.target.value).toFixed(2));

  document.querySelectorAll('.btn-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      promptInput.value = chip.getAttribute('data-prompt');
    });
  });

  btnGenerate.addEventListener('click', async () => {
    const prompt = promptInput.value;
    const temp = parseFloat(tempSlider.value);
    const tokens = parseInt(tokensSlider.value);
    const topp = parseFloat(toppSlider.value);

    btnGenerate.disabled = true;
    btnGenerate.innerHTML = `<span class="pulse-indicator"></span> Generating...`;
    gptOutput.textContent = 'Generating tokens with SmallGPT...';

    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt,
          temperature: temp,
          max_new_tokens: tokens,
          top_p: topp
        })
      });
      const data = await res.json();
      if (data.error) {
        gptOutput.textContent = `Error: ${data.error}`;
      } else {
        gptOutput.textContent = data.full_text;
        gptStatTokens.textContent = data.tokens_generated;
        gptStatLatency.textContent = `${data.latency_sec}s`;
        gptStatSpeed.textContent = `${data.tokens_per_sec} tok/s`;
      }
    } catch (e) {
      gptOutput.textContent = `Generation failed: ${e.message}`;
    } finally {
      btnGenerate.disabled = false;
      btnGenerate.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg> Generate Autoregressive Completion`;
    }
  });

  // ---------------------------------------------------------
  // 5. Unit Tests Runner
  // ---------------------------------------------------------
  btnRunTests.addEventListener('click', async () => {
    btnRunTests.disabled = true;
    btnRunTests.innerHTML = `<span class="pulse-indicator"></span> Running Pytest...`;
    testsTerminal.textContent = 'Running pytest test suite (16 tests)...';

    try {
      const res = await fetch('/api/tests/run', { method: 'POST' });
      const data = await res.json();
      testsTerminal.textContent = data.output;
      testsTerminal.style.color = data.success ? '#10b981' : '#f43f5e';
    } catch (e) {
      testsTerminal.textContent = `Test execution failed: ${e.message}`;
    } finally {
      btnRunTests.disabled = false;
      btnRunTests.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg> Run Pytest (16 Tests)`;
    }
  });

  // ---------------------------------------------------------
  // 6. Fetch Benchmarks Table
  // ---------------------------------------------------------
  async function fetchBenchmarkTable() {
    const tableBody = document.querySelector('#benchmark-summary-table tbody');
    try {
      const res = await fetch('/api/benchmarks');
      const records = await res.json();
      if (!records || records.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="7">No records found. Run `python src/run_experiments.py` first.</td></tr>';
        return;
      }
      tableBody.innerHTML = '';
      records.forEach(r => {
        const tr = document.createElement('tr');
        const successBadge = r.success 
          ? '<span style="color:#10b981; font-weight:600;">PASS</span>' 
          : '<span style="color:#f43f5e; font-weight:600;">FAIL</span>';
        tr.innerHTML = `
          <td><code>${escapeHtml(r.task_id)}</code></td>
          <td>${escapeHtml(r.category)}</td>
          <td><span class="badge ${r.agent_mode === 'verified' ? 'badge-green' : r.agent_mode === 'react' ? 'badge-yellow' : 'badge-purple'}">${escapeHtml(r.agent_mode)}</span></td>
          <td>${successBadge}</td>
          <td>${r.steps_taken}</td>
          <td>${r.tool_calls}</td>
          <td>${parseFloat(r.latency_sec).toFixed(4)}s</td>
        `;
        tableBody.appendChild(tr);
      });
    } catch (e) {
      tableBody.innerHTML = `<tr><td colspan="7">Failed to load benchmark table: ${e.message}</td></tr>`;
    }
  }

  function escapeHtml(str) {
    if (typeof str !== 'string') return String(str);
    return str.replace(/&/g, '&amp;')
              .replace(/</g, '&lt;')
              .replace(/>/g, '&gt;')
              .replace(/"/g, '&quot;')
              .replace(/'/g, '&#039;');
  }

  // Initial loads
  fetchStatus();
  fetchTasks();
  fetchBenchmarkTable();
});
