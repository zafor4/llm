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
import DatasetsApp from './components/DatasetsApp';
import './App.css';

export default function App() {
  const [activeApp, setActiveApp] = useState('dashboard');
  const [vitals, setVitals] = useState({});
  const [activeModel, setActiveModel] = useState(null);
  const [modelsList, setModelsList] = useState([]);
  const [loadedModels, setLoadedModels] = useState([]);
  const [continueTarget, setContinueTarget] = useState(null);
  const [trainDatasetId, setTrainDatasetId] = useState(null);

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
      /* offline */
    }
  }, []);

  const fetchModels = useCallback(async () => {
    try {
      const res = await fetch('/api/models/list');
      if (!res.ok) return;
      const data = await res.json();
      setModelsList(data.models || []);
      setLoadedModels(data.loaded_models || []);
      if (data.active_model) setActiveModel(data.active_model);
    } catch {
      /* offline */
    }
  }, []);

  const refreshHub = useCallback(() => {
    fetchVitals();
    fetchModels();
  }, [fetchVitals, fetchModels]);

  useEffect(() => {
    refreshHub();
    const interval = setInterval(fetchVitals, 5000);
    return () => clearInterval(interval);
  }, [refreshHub, fetchVitals]);

  const titles = {
    dashboard: ['Home', 'Start here. Train, add, or try a model.'],
    models: ['Models', 'Everything you own, plus what is in memory.'],
    resources: ['Resources', 'See this computer, include models, and free memory.'],
    datasets: ['Datasets', 'Include training data, then attach it to any local model.'],
    training: ['Train', 'Create a new model or improve one you already have.'],
    chat: ['Chat', 'A quick way to talk to the model you are using.'],
    agent: ['Tasks', 'Watch an agent work through a practice job.'],
    gateway: ['Connect', 'Let other apps use your models.'],
    benchmarks: ['Health', 'Checks and recorded results.'],
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
          <div>
            <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.15rem' }}>
              {titles[activeApp]?.[0] || 'Hub'}
            </h2>
            <p className="hint">{titles[activeApp]?.[1]}</p>
          </div>
          <button className="btn btn-secondary" onClick={() => setActiveApp('models')}>
            Using {activeModel || 'none'} · change
          </button>
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
              datasetTarget={trainDatasetId}
              onNeedDataset={() => setActiveApp('datasets')}
              onModelTrained={refreshHub}
            />
          )}
          {activeApp === 'datasets' && (
            <DatasetsApp
              onUseForTraining={(id) => {
                setTrainDatasetId(id);
                setActiveApp('training');
              }}
            />
          )}
          {activeApp === 'models' && (
            <ModelsFleetApp
              activeModel={activeModel}
              loadedModels={loadedModels}
              onSelectModel={(m) => {
                setActiveModel(m);
                refreshHub();
              }}
              onContinueTrain={(id) => {
                setContinueTarget(id);
                setActiveApp('training');
              }}
              onRefresh={refreshHub}
            />
          )}
          {activeApp === 'resources' && (
            <ResourcesApp
              onRefresh={refreshHub}
              onSelectModel={(m) => {
                setActiveModel(m);
                refreshHub();
              }}
            />
          )}
          {activeApp === 'gateway' && <ApiGatewayApp activeModel={activeModel} />}
          {activeApp === 'benchmarks' && <BenchmarksApp />}
        </section>
      </main>
    </div>
  );
}
