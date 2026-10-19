/**
 * Antigravity AI Studio — Frontend Application Logic
 * Supports: Conversational Chat, Autonomous Tool-using Agent, Live Model Training, and Checkpoint Switching.
 */

document.addEventListener('DOMContentLoaded', () => {
  // State
  let currentMode = 'agent'; // 'agent' or 'chat'
  let activeModel = 'GPT-Mini-256';
  let isGenerating = false;
  let trainPollInterval = null;

  // DOM Elements - Header
  const headerModelSelect = document.getElementById('header-model-select');
  const modeAgentBtn = document.getElementById('mode-agent-btn');
  const modeChatBtn = document.getElementById('mode-chat-btn');
  const gpuStatusText = document.getElementById('gpu-status-text');
  const btnOpenDrawer = document.getElementById('btn-open-drawer');

  // DOM Elements - Chat Workspace
  const chatMessagesContainer = document.getElementById('chat-messages-container');
  const chatWelcomeHero = document.getElementById('chat-welcome-hero');
  const chatInput = document.getElementById('chat-input');
  const btnSend = document.getElementById('btn-send');
  const btnClearChat = document.getElementById('btn-clear-chat');
  const benchmarkQuickSelect = document.getElementById('benchmark-quick-select');
  const activeModeCaption = document.getElementById('active-mode-caption');

  // DOM Elements - Drawer
  const controlDrawer = document.getElementById('control-drawer');
  const drawerOverlay = document.getElementById('drawer-overlay');
  const btnCloseDrawer = document.getElementById('btn-close-drawer');
  const drawerTabBtns = document.querySelectorAll('.drawer-tab-btn');
  const drawerPanes = document.querySelectorAll('.drawer-pane');

  // Drawer - Training Studio
  const trainModelName = document.getElementById('train-model-name');
  const trainSteps = document.getElementById('train-steps');
  const trainLayers = document.getElementById('train-layers');
  const trainHeads = document.getElementById('train-heads');
  const trainDim = document.getElementById('train-dim');
  const trainBlockSize = document.getElementById('train-block-size');
  const trainLr = document.getElementById('train-lr');
  const trainBatchSize = document.getElementById('train-batch-size');
  const trainCorpusText = document.getElementById('train-corpus-text');
  const btnStartTraining = document.getElementById('btn-start-training');
  const trainingMonitorBox = document.getElementById('training-monitor-box');
  const trainStatusBadge = document.getElementById('train-status-badge');
  const trainStepIndicator = document.getElementById('train-step-indicator');
  const trainProgressFill = document.getElementById('train-progress-fill');
  const trainStatLoss = document.getElementById('train-stat-loss');
  const trainStatPpl = document.getElementById('train-stat-ppl');
  const trainStatSpeed = document.getElementById('train-stat-speed');
  const trainStatVram = document.getElementById('train-stat-vram');

  // Drawer - Models List
  const modelCardsContainer = document.getElementById('model-cards-container');

  // Drawer - Behavior Settings
  const settingsSystemPrompt = document.getElementById('settings-system-prompt');
  const settingsTemp = document.getElementById('settings-temp');
  const settingsTempVal = document.getElementById('settings-temp-val');
  const settingsTopp = document.getElementById('settings-topp');
  const settingsToppVal = document.getElementById('settings-topp-val');
  const btnSaveSettings = document.getElementById('btn-save-settings');

  // Drawer - Monitor & Tests
  const vGpuName = document.getElementById('v-gpu-name');
  const vVramUsage = document.getElementById('v-vram-usage');
  const vActiveModel = document.getElementById('v-active-model');
  const btnRunAllTests = document.getElementById('btn-run-all-tests');
  const drawerTerminalOutput = document.getElementById('drawer-terminal-output');

  // ---------------------------------------------------------
  // 1. Drawer Management & Tabs
  // ---------------------------------------------------------
  function openDrawer(tabName = null) {
    controlDrawer.classList.add('active');
    drawerOverlay.classList.add('active');
    if (tabName) {
      switchDrawerTab(tabName);
    }
  }

  function closeDrawer() {
    controlDrawer.classList.remove('active');
    drawerOverlay.classList.remove('active');
  }

  btnOpenDrawer.addEventListener('click', () => openDrawer());
  btnCloseDrawer.addEventListener('click', closeDrawer);
  drawerOverlay.addEventListener('click', closeDrawer);

  function switchDrawerTab(targetDtab) {
    drawerTabBtns.forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-dtab') === targetDtab);
    });
    drawerPanes.forEach(pane => {
      pane.classList.toggle('active', pane.id === targetDtab);
    });
  }

  drawerTabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      switchDrawerTab(btn.getAttribute('data-dtab'));
    });
  });

  // ---------------------------------------------------------
  // 2. Mode Switching (Agent vs Chatbot)
  // ---------------------------------------------------------
  modeAgentBtn.addEventListener('click', () => setMode('agent'));
  modeChatBtn.addEventListener('click', () => setMode('chat'));

  function setMode(mode) {
    currentMode = mode;
    modeAgentBtn.classList.toggle('active', mode === 'agent');
    modeChatBtn.classList.toggle('active', mode === 'chat');

    if (mode === 'agent') {
      activeModeCaption.textContent = 'Agent Mode Active';
      chatInput.placeholder = 'Give the Autonomous Agent a goal or select a task preset...';
    } else {
      activeModeCaption.textContent = 'Chatbot Mode Active';
      chatInput.placeholder = `Chat with ${activeModel} (direct autoregressive generation)...`;
    }
  }

  // ---------------------------------------------------------
  // 3. Quick Starter Cards & Presets
  // ---------------------------------------------------------
  document.querySelectorAll('.starter-card').forEach(card => {
    card.addEventListener('click', () => {
      const taskId = card.getAttribute('data-task');
      benchmarkQuickSelect.value = taskId;
      const taskText = card.querySelector('.starter-text').textContent;
      chatInput.value = taskText;
      sendMessage();
    });
  });

  benchmarkQuickSelect.addEventListener('change', () => {
    const val = benchmarkQuickSelect.value;
    if (val) {
      setMode('agent');
      const selectedOption = benchmarkQuickSelect.options[benchmarkQuickSelect.selectedIndex].text;
      chatInput.value = `Execute ${selectedOption}`;
    }
  });

  // Auto-resize chat textarea
  chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
  });

  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  btnSend.addEventListener('click', sendMessage);

  btnClearChat.addEventListener('click', () => {
    chatMessagesContainer.innerHTML = '';
    chatWelcomeHero.style.display = 'block';
    chatMessagesContainer.appendChild(chatWelcomeHero);
  });

  // ---------------------------------------------------------
  // 4. Send Message & Render Flow
  // ---------------------------------------------------------
  async function sendMessage() {
    const text = chatInput.value.trim();
    const taskId = benchmarkQuickSelect.value;

    if (!text && !taskId) return;
    if (isGenerating) return;

    isGenerating = true;
    btnSend.disabled = true;

    // Hide welcome hero on first message
    if (chatWelcomeHero) {
      chatWelcomeHero.style.display = 'none';
    }

    // Append User Message
    appendUserMessage(text || `Execute task preset: ${taskId}`);
    chatInput.value = '';
    chatInput.style.height = 'auto';

    // Append Assistant Loading Indicator
    const loadingElem = appendLoadingMessage();
    scrollChatToBottom();

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          mode: currentMode,
          task_id: taskId || null
        })
      });
      const data = await res.json();
      loadingElem.remove();

      if (data.mode === 'agent') {
        renderAgentResponse(data);
      } else {
        renderChatbotResponse(data);
      }
    } catch (e) {
      loadingElem.remove();
      appendAssistantMessage(`Error communicating with backend server: ${e.message}`);
    } finally {
      isGenerating = false;
      btnSend.disabled = false;
      benchmarkQuickSelect.value = '';
      scrollChatToBottom();
    }
  }

  function appendUserMessage(text) {
    const row = document.createElement('div');
    row.className = 'message-row user';
    row.innerHTML = `
      <div class="message-body">
        <div class="message-bubble">${escapeHtml(text)}</div>
      </div>
      <div class="message-avatar">You</div>
    `;
    chatMessagesContainer.appendChild(row);
  }

  function appendLoadingMessage() {
    const row = document.createElement('div');
    row.className = 'message-row assistant';
    row.innerHTML = `
      <div class="message-avatar">AG</div>
      <div class="message-body">
        <div class="message-bubble" style="display: flex; align-items: center; gap: 8px;">
          <span class="pulse-indicator"></span>
          <span style="color: var(--text-muted); font-size: 0.8125rem;">
            ${currentMode === 'agent' ? 'Autonomous agent observing workspace & planning tools...' : 'SmallGPT generating response...'}
          </span>
        </div>
      </div>
    `;
    chatMessagesContainer.appendChild(row);
    return row;
  }

  function appendAssistantMessage(text) {
    const row = document.createElement('div');
    row.className = 'message-row assistant';
    row.innerHTML = `
      <div class="message-avatar">AG</div>
      <div class="message-body">
        <div class="message-bubble">${formatMarkdown(text)}</div>
      </div>
    `;
    chatMessagesContainer.appendChild(row);
  }

  function renderChatbotResponse(data) {
    const row = document.createElement('div');
    row.className = 'message-row assistant';
    row.innerHTML = `
      <div class="message-avatar">AG</div>
      <div class="message-body">
        <div class="message-bubble">${formatMarkdown(data.response)}</div>
        <div style="font-size: 0.6875rem; color: var(--text-muted); padding-left: 4px;">
          Generated with <strong>${escapeHtml(data.model_used || activeModel)}</strong> in ${data.latency_sec}s
        </div>
      </div>
    `;
    chatMessagesContainer.appendChild(row);
  }

  function renderAgentResponse(data) {
    const row = document.createElement('div');
    row.className = 'message-row assistant';

    const steps = data.steps || [];
    let stepsHtml = '';

    steps.forEach((s) => {
      const passClass = s.observation_success ? 'pass' : 'fail';
      const badge = s.observation_success ? '<span class="tool-badge-pass">PASS</span>' : '<span class="tool-badge-fail">FAIL</span>';
      let reflectionHtml = '';
      if (s.reflection) {
        reflectionHtml = `<div style="margin-top:6px; color:#fbbf24; font-size:0.75rem;">⚡ <strong>Self-Correction:</strong> ${escapeHtml(s.reflection)}</div>`;
      }

      stepsHtml += `
        <div class="tool-step-card ${passClass}" style="margin-bottom: 8px;">
          <div class="tool-step-top">
            <span class="tool-cmd">${escapeHtml(s.action_tool)}(${escapeHtml(JSON.stringify(s.action_params))})</span>
            ${badge}
          </div>
          <div style="color: #cbd5e1; font-size: 0.8125rem; margin-bottom: 4px;"><strong>Thought:</strong> ${escapeHtml(s.thought)}</div>
          <div class="tool-obs-box">${escapeHtml(s.observation.slice(0, 300))}${s.observation.length > 300 ? '...' : ''}</div>
          ${reflectionHtml}
        </div>
      `;
    });

    const thoughtAccordion = `
      <div class="thought-accordion open">
        <div class="thought-header">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
          Agent Execution Trace (${steps.length} Tool Steps) — Status: [${data.status}]
        </div>
        <div class="thought-content">
          ${stepsHtml}
        </div>
      </div>
    `;

    row.innerHTML = `
      <div class="message-avatar">AG</div>
      <div class="message-body">
        ${thoughtAccordion}
        <div class="message-bubble">${formatMarkdown(data.response)}</div>
        <div style="font-size: 0.6875rem; color: var(--text-muted); padding-left: 4px;">
          Completed in ${data.latency_sec}s • ${data.total_tool_calls} tool calls • ${data.failed_tool_calls || 0} error recoveries
        </div>
      </div>
    `;

    chatMessagesContainer.appendChild(row);

    // Bind accordion toggles
    row.querySelector('.thought-header').addEventListener('click', function() {
      this.parentElement.classList.toggle('open');
    });
  }

  function scrollChatToBottom() {
    chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
  }

  // ---------------------------------------------------------
  // 5. Model Manager & Checkpoint Loading
  // ---------------------------------------------------------
  async function fetchModels() {
    try {
      const res = await fetch('/api/models');
      const models = await res.json();
      renderModelCards(models);
      updateHeaderDropdown(models);
    } catch (e) {
      modelCardsContainer.innerHTML = `<div style="color:red">Failed to load models: ${e.message}</div>`;
    }
  }

  function updateHeaderDropdown(models) {
    headerModelSelect.innerHTML = '';
    models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.name;
      opt.textContent = `${m.name} ${m.val_perplexity !== 'N/A' ? `(Val PPL: ${m.val_perplexity})` : ''}`;
      if (m.is_active) {
        opt.selected = true;
        activeModel = m.name;
        vActiveModel.textContent = m.name;
      }
      headerModelSelect.appendChild(opt);
    });
  }

  headerModelSelect.addEventListener('change', async () => {
    const chosen = headerModelSelect.value;
    await loadModel(chosen);
  });

  async function loadModel(name) {
    try {
      const res = await fetch('/api/models/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: name })
      });
      const data = await res.json();
      if (data.success) {
        activeModel = data.active_model;
        vActiveModel.textContent = activeModel;
        await fetchModels();
      } else {
        alert(`Failed to load model: ${data.error}`);
      }
    } catch (e) {
      alert(`Model load request failed: ${e.message}`);
    }
  }

  function renderModelCards(models) {
    modelCardsContainer.innerHTML = '';
    models.forEach(m => {
      const card = document.createElement('div');
      card.className = `model-card-item ${m.is_active ? 'active' : ''}`;
      card.innerHTML = `
        <div class="model-card-info">
          <h4>${escapeHtml(m.name)} ${m.is_active ? '<span style="color:#10b981; font-size:0.7rem;">[ACTIVE]</span>' : ''}</h4>
          <p>${escapeHtml(m.type)} • Val Perplexity: ${m.val_perplexity}</p>
        </div>
        <div>
          ${!m.is_active ? `<button class="btn-load-model" data-name="${escapeHtml(m.name)}">Load</button>` : ''}
        </div>
      `;
      modelCardsContainer.appendChild(card);
    });

    modelCardsContainer.querySelectorAll('.btn-load-model').forEach(btn => {
      btn.addEventListener('click', () => {
        loadModel(btn.getAttribute('data-name'));
      });
    });
  }

  // ---------------------------------------------------------
  // 6. Live Model Training Studio
  // ---------------------------------------------------------
  btnStartTraining.addEventListener('click', async () => {
    const payload = {
      model_name: trainModelName.value.trim() || `Custom-GPT-${Date.now()}`,
      max_steps: parseInt(trainSteps.value),
      n_layer: parseInt(trainLayers.value),
      n_head: parseInt(trainHeads.value),
      n_embd: parseInt(trainDim.value),
      block_size: parseInt(trainBlockSize.value),
      learning_rate: parseFloat(trainLr.value),
      batch_size: parseInt(trainBatchSize.value),
      corpus_text: trainCorpusText.value.trim()
    };

    btnStartTraining.disabled = true;
    trainingMonitorBox.style.display = 'block';
    trainStatusBadge.textContent = 'INITIALIZING TRAINING RUN...';
    trainStatusBadge.style.color = '#38bdf8';

    try {
      const res = await fetch('/api/train/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.success) {
        startTrainingPolling();
      } else {
        alert(`Training start failed: ${data.error}`);
        btnStartTraining.disabled = false;
      }
    } catch (e) {
      alert(`Error starting training: ${e.message}`);
      btnStartTraining.disabled = false;
    }
  });

  function startTrainingPolling() {
    if (trainPollInterval) clearInterval(trainPollInterval);

    trainPollInterval = setInterval(async () => {
      try {
        const res = await fetch('/api/train/status');
        const state = await res.json();

        if (state.status === 'TRAINING') {
          trainStatusBadge.textContent = `TRAINING: ${state.model_name}`;
          trainStatusBadge.style.color = '#38bdf8';
          const pct = Math.min(100, (state.current_step / Math.max(1, state.total_steps)) * 100);
          trainProgressFill.style.width = `${pct}%`;
          trainStepIndicator.textContent = `Step ${state.current_step} / ${state.total_steps}`;
          trainStatLoss.textContent = state.current_loss.toFixed(4);
          trainStatPpl.textContent = state.val_perplexity.toFixed(2);
          trainStatSpeed.textContent = `${state.tokens_per_sec} tok/s`;
          trainStatVram.textContent = `${state.peak_vram_mb} MB`;
        } else if (state.status === 'COMPLETED') {
          clearInterval(trainPollInterval);
          trainStatusBadge.textContent = `TRAINING COMPLETED & LOADED!`;
          trainStatusBadge.style.color = '#10b981';
          trainProgressFill.style.width = '100%';
          btnStartTraining.disabled = false;
          await fetchModels();
          await fetchStatus();
        } else if (state.status === 'ERROR') {
          clearInterval(trainPollInterval);
          trainStatusBadge.textContent = `ERROR: ${state.error_message}`;
          trainStatusBadge.style.color = '#f43f5e';
          btnStartTraining.disabled = false;
        }
      } catch (e) {
        console.error('Error polling training', e);
      }
    }, 500);
  }

  // ---------------------------------------------------------
  // 7. Settings & Behavior
  // ---------------------------------------------------------
  settingsTemp.addEventListener('input', (e) => settingsTempVal.textContent = parseFloat(e.target.value).toFixed(2));
  settingsTopp.addEventListener('input', (e) => settingsToppVal.textContent = parseFloat(e.target.value).toFixed(2));

  btnSaveSettings.addEventListener('click', async () => {
    const enabledTools = Array.from(document.querySelectorAll('.tool-check input:checked')).map(i => i.value);
    const payload = {
      system_prompt: settingsSystemPrompt.value.trim(),
      temperature: parseFloat(settingsTemp.value),
      top_p: parseFloat(settingsTopp.value),
      enabled_tools: enabledTools
    };

    try {
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.success) {
        btnSaveSettings.textContent = 'Saved Successfully!';
        btnSaveSettings.style.background = '#10b981';
        setTimeout(() => {
          btnSaveSettings.textContent = 'Save Behavior Settings';
          btnSaveSettings.style.background = '';
        }, 1500);
      }
    } catch (e) {
      alert(`Failed to save settings: ${e.message}`);
    }
  });

  // ---------------------------------------------------------
  // 8. Test Runner
  // ---------------------------------------------------------
  btnRunAllTests.addEventListener('click', async () => {
    btnRunAllTests.disabled = true;
    drawerTerminalOutput.textContent = 'Executing pytest suite across 16 tests in sandbox...';

    try {
      const res = await fetch('/api/tests/run', { method: 'POST' });
      const data = await res.json();
      drawerTerminalOutput.textContent = data.output;
      drawerTerminalOutput.style.color = data.success ? '#34d399' : '#f43f5e';
    } catch (e) {
      drawerTerminalOutput.textContent = `Test run failed: ${e.message}`;
    } finally {
      btnRunAllTests.disabled = false;
    }
  });

  // ---------------------------------------------------------
  // 9. System Status Polling
  // ---------------------------------------------------------
  async function fetchStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();

      if (data.cuda_available) {
        gpuStatusText.textContent = `${data.device} • ${data.vram_used_mb}/${data.vram_total_mb} MB VRAM`;
        vGpuName.textContent = data.device;
        vVramUsage.textContent = `${data.vram_used_mb} MB / ${data.vram_total_mb} MB`;
      } else {
        gpuStatusText.textContent = 'CPU Mode Active';
        vGpuName.textContent = 'CPU Host';
      }

      if (data.active_model) {
        activeModel = data.active_model;
        vActiveModel.textContent = activeModel;
      }
    } catch (e) {
      gpuStatusText.textContent = 'Connecting...';
    }
  }

  // Formatting helpers
  function escapeHtml(str) {
    if (typeof str !== 'string') return String(str);
    return str.replace(/&/g, '&amp;')
              .replace(/</g, '&lt;')
              .replace(/>/g, '&gt;')
              .replace(/"/g, '&quot;')
              .replace(/'/g, '&#039;');
  }

  function formatMarkdown(text) {
    if (!text) return '';
    let escaped = escapeHtml(text);
    // Simple code blocks
    escaped = escaped.replace(/```([\s\S]*?)```/g, '<pre class="code-font" style="background:#050810; padding:10px; border-radius:6px; margin:8px 0; overflow-x:auto;"><code>$1</code></pre>');
    // Inline code
    escaped = escaped.replace(/`([^`]+)`/g, '<code class="code-font" style="background:rgba(255,255,255,0.08); padding:2px 6px; border-radius:4px; color:#38bdf8;">$1</code>');
    // Bold
    escaped = escaped.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    // Line breaks
    escaped = escaped.replace(/\n/g, '<br>');
    return escaped;
  }

  // Initial Boot
  fetchStatus();
  fetchModels();
  setInterval(fetchStatus, 4000);
});
