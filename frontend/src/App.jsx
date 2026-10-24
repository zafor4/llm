import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import DashboardApp from './components/DashboardApp';
import ChatbotApp from './components/ChatbotApp';
import CodingAgentApp from './components/CodingAgentApp';
import TrainingStudioApp from './components/TrainingStudioApp';
import ModelsFleetApp from './components/ModelsFleetApp';
import BenchmarksApp from './components/BenchmarksApp';
import ApiGatewayApp from './components/ApiGatewayApp';
import { Sparkles, Terminal, Activity, Layers } from 'lucide-react';
import './App.css';

export default function App() {
  const [activeApp, setActiveApp] = useState('dashboard');
  const [vitals, setVitals] = useState({
    cuda_available: true,
    device_name: 'NVIDIA GeForce RTX 3050 Laptop GPU',
    vram_used_mb: 1840,
    vram_total_mb: 6144,
    ram_used_gb: 9.8,
    ram_total_gb: 15.8,
    cpu_usage_percent: 18,
    pytorch_version: '2.6.0',
    checkpoints_count: 4,
    active_checkpoint: 'GPT-Mini-256'
  });
  const [activeModel, setActiveModel] = useState('GPT-Mini-256');
  const [modelsList, setModelsList] = useState(['GPT-Micro-128', 'GPT-Mini-256', 'GPT-Small-256', 'GPT-Mini-512']);

  const fetchVitals = async () => {
    try {
      const res = await fetch('/api/dashboard/vitals');
      if (res.ok) {
        const data = await res.json();
        setVitals(data);
        if (data.active_checkpoint) {
          setActiveModel(data.active_checkpoint);
        }
      }
    } catch (err) {
      // Backend polling fallback
    }
  };

  const fetchModels = async () => {
    try {
      const res = await fetch('/api/models/list');
      if (res.ok) {
        const data = await res.json();
        const list = data.models || data.available_models || [];
        if (list.length > 0) {
          setModelsList(list.map(m => m.name || m.id));
        }
        const act = data.active_model || data.current_model;
        if (act) {
          setActiveModel(act);
        }
      }
    } catch (err) {
      // Backend polling fallback
    }
  };

  useEffect(() => {
    fetchVitals();
    fetchModels();
    const interval = setInterval(fetchVitals, 5000);
    return () => clearInterval(interval);
  }, []);

  const getAppTitle = () => {
    switch (activeApp) {
      case 'dashboard': return 'Platform Control Hub';
      case 'chat': return 'Conversational Chatbot Studio';
      case 'agent': return 'Autonomous Coding Agent';
      case 'training': return 'Model Training Studio';
      case 'models': return 'Models & Providers Fleet';
      case 'gateway': return 'External API Gateway (House of Intelligence)';
      case 'benchmarks': return 'Benchmarks & System Diagnostics';
      default: return 'Antigravity AI Platform';
    }
  };

  return (
    <div className="app-shell">
      <Sidebar 
        activeApp={activeApp} 
        setActiveApp={setActiveApp} 
        activeModel={activeModel} 
        vitals={vitals} 
      />

      <main className="main-viewport">
        <header className="top-app-bar">
          <div className="top-bar-title">
            <h2>{getAppTitle()}</h2>
            <div className="active-model-badge" title="Active Checkpoint in VRAM">
              Serving: {activeModel}
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
              <Activity size={14} color="var(--accent-emerald)" />
              <span>GPU: {vitals.vram_used_mb}MB / {vitals.vram_total_mb}MB</span>
            </div>
            <button 
              className="btn btn-secondary" 
              style={{ padding: '6px 12px', fontSize: '0.8rem' }}
              onClick={() => setActiveApp(activeApp === 'models' ? 'dashboard' : 'models')}
            >
              <Layers size={14} />
              Switch Model
            </button>
          </div>
        </header>

        <section className="app-content-view">
          {activeApp === 'dashboard' && (
            <DashboardApp 
              vitals={vitals} 
              setActiveApp={setActiveApp} 
              activeModel={activeModel} 
            />
          )}

          {activeApp === 'chat' && (
            <ChatbotApp 
              activeModel={activeModel} 
              modelsList={modelsList} 
              setActiveModel={setActiveModel} 
            />
          )}

          {activeApp === 'agent' && (
            <CodingAgentApp />
          )}

          {activeApp === 'training' && (
            <TrainingStudioApp 
              onModelTrained={() => {
                fetchVitals();
                fetchModels();
              }} 
            />
          )}

          {activeApp === 'models' && (
            <ModelsFleetApp 
              activeModel={activeModel} 
              onSelectModel={(m) => {
                setActiveModel(m);
                fetchVitals();
              }} 
            />
          )}

          {activeApp === 'gateway' && (
            <ApiGatewayApp activeModel={activeModel} />
          )}

          {activeApp === 'benchmarks' && (
            <BenchmarksApp />
          )}
        </section>
      </main>
    </div>
  );
}
