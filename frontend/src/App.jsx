import React, { useState, useEffect, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import DashboardApp from './components/DashboardApp';
import ChatbotApp from './components/ChatbotApp';
import CodingAgentApp from './components/CodingAgentApp';
import TrainingStudioApp from './components/TrainingStudioApp';
import ModelsFleetApp from './components/ModelsFleetApp';
import BenchmarksApp from './components/BenchmarksApp';
import ApiGatewayApp from './components/ApiGatewayApp';
import ResourcesApp from './components/ResourcesApp';
import { Activity, Layers } from 'lucide-react';
import './App.css';

export default function App() {
  const [activeApp, setActiveApp] = useState('dashboard');
  const [vitals, setVitals] = useState({});
  const [activeModel, setActiveModel] = useState(null);
  const [modelsList, setModelsList] = useState([]);
  const [loadedModels, setLoadedModels] = useState([]);
  const [continueTarget, setContinueTarget] = useState(null);

  const fetchVitals = useCallback(async () => {
    try {
      const res = await fetch('/api/dashboard/vitals');
      if (!res.ok) return;
      const data = await res.json();
      setVitals(data);
      if (data.active_model || data.active_checkpoint) {
        setActiveModel(data.active_model || data.active_checkpoint);
      }
    } catch {
      /* backend offline */
    }
  }, []);

  const fetchModels = useCallback(async () => {
    try {
      const res = await fetch('/api/models/list');
      if (!res.ok) return;
      const data = await res.json();
      const list = data.models || data.available_models || [];
      setModelsList(list);
      setLoadedModels(data.loaded_models || []);
      if (data.active_model) setActiveModel(data.active_model);
    } catch {
      /* backend offline */
    }
  }, []);

  const refreshHub = useCallback(() => {
    fetchVitals();
    fetchModels();
  }, [fetchVitals, fetchModels]);

  useEffect(() => {
    refreshHub();
    const interval = setInterval(fetchVitals, 4000);
    return () => clearInterval(interval);
  }, [refreshHub, fetchVitals]);

  const titles = {
    dashboard: 'Control Center',
    chat: 'Chat Playground',
    agent: 'Agent Playground',
    training: 'Training Jobs',
    models: 'Model Catalog',
    resources: 'Resource Manager',
    gateway: 'API Sharing',
    benchmarks: 'Diagnostics',
  };

  const openContinueTraining = (modelId) => {
    setContinueTarget(modelId);
    setActiveApp('training');
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
            <h2>{titles[activeApp] || 'Intelligence Hub'}</h2>
            <div className="active-model-badge">Serving: {activeModel || 'none'}</div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
              <Activity size={14} color="var(--accent-emerald)" />
              <span>VRAM {vitals.vram_used_mb || 0} / {vitals.vram_total_mb || 0} MB</span>
            </div>
            <button
              className="btn btn-secondary"
              style={{ padding: '6px 12px', fontSize: '0.8rem' }}
              onClick={() => setActiveApp('models')}
            >
              <Layers size={14} />
              Manage Models
            </button>
          </div>
        </header>

        <section className="app-content-view">
          {activeApp === 'dashboard' && (
            <DashboardApp
              vitals={vitals}
              setActiveApp={setActiveApp}
              activeModel={activeModel}
              modelsList={modelsList}
              loadedModels={loadedModels}
            />
          )}
          {activeApp === 'chat' && (
            <ChatbotApp
              activeModel={activeModel}
              modelsList={modelsList}
              setActiveModel={setActiveModel}
              onRefresh={refreshHub}
            />
          )}
          {activeApp === 'agent' && (
            <CodingAgentApp activeModel={activeModel} modelsList={modelsList} />
          )}
          {activeApp === 'training' && (
            <TrainingStudioApp
              modelsList={modelsList}
              continueTarget={continueTarget}
              onModelTrained={refreshHub}
            />
          )}
          {activeApp === 'models' && (
            <ModelsFleetApp
              activeModel={activeModel}
              onSelectModel={(m) => {
                setActiveModel(m);
                refreshHub();
              }}
              onContinueTrain={openContinueTraining}
              onRefresh={refreshHub}
            />
          )}
          {activeApp === 'resources' && (
            <ResourcesApp
              vitals={vitals}
              loadedModels={loadedModels}
              activeModel={activeModel}
              onRefresh={refreshHub}
            />
          )}
          {activeApp === 'gateway' && <ApiGatewayApp activeModel={activeModel} />}
          {activeApp === 'benchmarks' && <BenchmarksApp />}
        </section>
      </main>
    </div>
  );
}
