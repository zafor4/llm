import React from 'react';
import {
  Home,
  MessageSquare,
  ListChecks,
  Boxes,
  PlusCircle,
  Gauge,
  Share2,
  HeartPulse,
  Sparkles,
} from 'lucide-react';

export default function Sidebar({ activeApp, setActiveApp, activeModel, vitals }) {
  const main = [
    { id: 'dashboard', label: 'Home', sub: 'What to do next', icon: Home },
    { id: 'models', label: 'Models', sub: 'Your library', icon: Boxes },
    { id: 'resources', label: 'Resources', sub: 'Memory and GPU', icon: Gauge },
    { id: 'training', label: 'Train', sub: 'Create or improve', icon: PlusCircle },
  ];
  const tryItems = [
    { id: 'chat', label: 'Chat', sub: 'Talk to a model', icon: MessageSquare },
    { id: 'agent', label: 'Tasks', sub: 'Try an agent', icon: ListChecks },
  ];

  const row = (item) => {
    const Icon = item.icon;
    return (
      <button
        key={item.id}
        className={`nav-item ${activeApp === item.id ? 'active' : ''}`}
        onClick={() => setActiveApp(item.id)}
      >
        <Icon size={18} />
        <span>
          {item.label}
          <span className="nav-sub">{item.sub}</span>
        </span>
      </button>
    );
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="brand-icon">
          <Sparkles size={20} />
        </div>
        <div className="brand-info">
          <h1>Intelligence Hub</h1>
          <span>Your model workshop</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        <div className="nav-section-title">Workshop</div>
        {main.map(row)}
        <div className="nav-section-title">Try</div>
        {tryItems.map(row)}
        <div className="nav-section-title">Share</div>
        {row({ id: 'gateway', label: 'Connect', sub: 'Use in other apps', icon: Share2 })}
      </nav>

      <div className="sidebar-footer">
        <button
          className={`nav-item ${activeApp === 'benchmarks' ? 'active' : ''}`}
          onClick={() => setActiveApp('benchmarks')}
          style={{ marginBottom: 10 }}
        >
          <HeartPulse size={16} />
          <span>
            Health
            <span className="nav-sub">Tests and reports</span>
          </span>
        </button>
        <div className="system-pill">
          <span className="pulse-dot" />
          <span>{vitals?.cuda_available ? 'GPU ready' : 'Running on CPU'}</span>
        </div>
        <div className="hint" style={{ marginTop: 6 }}>
          Using {activeModel || 'no model yet'}
        </div>
      </div>
    </aside>
  );
}
