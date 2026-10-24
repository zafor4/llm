import React from 'react';
import {
  LayoutDashboard,
  MessageSquare,
  Bot,
  Cpu,
  Flame,
  LineChart,
  Sparkles,
  Globe,
  Gauge,
} from 'lucide-react';

export default function Sidebar({ activeApp, setActiveApp, activeModel, vitals }) {
  const controlPlane = [
    { id: 'dashboard', label: 'Control Center', icon: LayoutDashboard },
    { id: 'models', label: 'Model Catalog', icon: Cpu },
    { id: 'training', label: 'Training Jobs', icon: Flame },
    { id: 'resources', label: 'Resources', icon: Gauge },
    { id: 'gateway', label: 'API Sharing', icon: Globe },
    { id: 'benchmarks', label: 'Diagnostics', icon: LineChart },
  ];

  const playgrounds = [
    { id: 'chat', label: 'Chat Playground', icon: MessageSquare },
    { id: 'agent', label: 'Agent Playground', icon: Bot },
  ];

  const renderGroup = (items) =>
    items.map((item) => {
      const Icon = item.icon;
      const isActive = activeApp === item.id;
      return (
        <button
          key={item.id}
          className={`nav-item ${isActive ? 'active' : ''}`}
          onClick={() => setActiveApp(item.id)}
        >
          <Icon size={18} />
          <span>{item.label}</span>
        </button>
      );
    });

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="brand-icon">
          <Sparkles size={20} />
        </div>
        <div className="brand-info">
          <h1>Intelligence Hub</h1>
          <span>Control plane</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        <div className="nav-section-title">Control Plane</div>
        {renderGroup(controlPlane)}
        <div className="nav-section-title" style={{ marginTop: '1rem' }}>Playgrounds</div>
        {renderGroup(playgrounds)}
      </nav>

      <div className="sidebar-footer">
        <div className="system-pill">
          <span className="pulse-dot"></span>
          <span>
            {vitals?.cuda_available
              ? `${(vitals.device_name || 'GPU').split(' ').slice(-2).join(' ')} • ${vitals.vram_used_mb || 0}MB`
              : 'CPU Host Active'}
          </span>
        </div>
        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 6 }}>
          Active: {activeModel || 'none'}
        </div>
      </div>
    </aside>
  );
}
