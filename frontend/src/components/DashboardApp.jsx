import React from 'react';
import { MessageSquare, PlusCircle, ListChecks, Boxes, Gauge, Library } from 'lucide-react';

export default function DashboardApp({ vitals, setActiveApp, activeModel, modelsList, loadedModels }) {
  const actions = [
    { id: 'chat', title: 'Chat', desc: 'Ask the model you are using a question.', icon: MessageSquare },
    { id: 'datasets', title: 'Include a dataset', desc: 'Add your own files or text, then train a model on them.', icon: Library },
    { id: 'training', title: 'Train a model', desc: 'Create a new one, or improve one you already have.', icon: PlusCircle },
    { id: 'agent', title: 'Run a practice task', desc: 'See an agent fix a bug or complete a small job.', icon: ListChecks },
    { id: 'models', title: 'Add or switch models', desc: 'Bring in Hugging Face, Ollama, or a local checkpoint.', icon: Boxes },
    { id: 'resources', title: 'Manage resources', desc: 'See GPU and RAM, include a model, or free space.', icon: Gauge },
  ];

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Welcome back</h2>
          <p className="page-lead">
            This hub stores your models, trains new ones, and lets other apps call them.
            Chat and Tasks are just for trying things out.
          </p>
        </div>
      </div>

      <div className="card-panel" style={{ marginBottom: 20 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16, flexWrap: 'wrap', alignItems: 'center' }}>
          <div>
            <div className="hint">You are using</div>
            <div style={{ fontSize: '1.25rem', fontWeight: 700, marginTop: 4 }}>{activeModel || 'No model loaded yet'}</div>
            <p className="hint" style={{ marginTop: 6 }}>
              {activeModel
                ? 'Chat, Tasks, and Connect all go through this model until you change it.'
                : 'Open Models and load one, or train a new model first.'}
            </p>
          </div>
          <button className="btn btn-primary" onClick={() => setActiveApp('models')}>
            {activeModel ? 'Change model' : 'Choose a model'}
          </button>
        </div>
      </div>

      <h3 style={{ fontFamily: 'var(--font-display)', marginBottom: 12 }}>What do you want to do?</h3>
      <div className="action-grid" style={{ marginBottom: 24 }}>
        {actions.map((a) => {
          const Icon = a.icon;
          return (
            <button key={a.id} className="card-panel action-card" onClick={() => setActiveApp(a.id)}>
              <Icon size={20} color="var(--accent-blue)" />
              <h3>{a.title}</h3>
              <p className="hint">{a.desc}</p>
            </button>
          );
        })}
      </div>

      <div className="stat-grid">
        <div className="card-panel">
          <div className="hint">Models you have</div>
          <div style={{ fontSize: '1.4rem', fontWeight: 700 }}>{(modelsList || []).length}</div>
        </div>
        <div className="card-panel">
          <div className="hint">Ready in memory</div>
          <div style={{ fontSize: '1.4rem', fontWeight: 700 }}>{(loadedModels || []).length}</div>
        </div>
        <div className="card-panel">
          <div className="hint">Datasets you have</div>
          <div style={{ fontSize: '1.4rem', fontWeight: 700 }}>{vitals.datasets_count ?? 0}</div>
        </div>
        <div className="card-panel">
          <div className="hint">Computer</div>
          <div style={{ fontSize: '1.05rem', fontWeight: 700, marginTop: 6 }}>
            {vitals.cuda_available ? 'GPU is available' : 'Using CPU'}
          </div>
        </div>
        <div className="card-panel">
          <div className="hint">Training</div>
          <div style={{ fontSize: '1.05rem', fontWeight: 700, marginTop: 6 }}>
            {vitals.training_status === 'TRAINING' ? 'A job is running' : 'Idle'}
          </div>
        </div>
      </div>
    </div>
  );
}
