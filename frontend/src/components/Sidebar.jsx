import React from 'react';
import { 
  LayoutDashboard, 
  MessageSquare, 
  Bot, 
  Cpu, 
  Flame, 
  LineChart, 
  Sparkles,
  Globe
} from 'lucide-react';

export default function Sidebar({ activeApp, setActiveApp, activeModel, vitals }) {
  const intelligenceCore = [
    { id: 'dashboard', label: 'Platform Hub', icon: LayoutDashboard },
    { id: 'training', label: 'Model Training Studio', icon: Flame },
    { id: 'models', label: 'Models & Providers', icon: Cpu },
    { id: 'gateway', label: 'External API Gateway', icon: Globe },
    { id: 'benchmarks', label: 'System Diagnostics', icon: LineChart },
  ];

  const productPlayground = [
    { id: 'chat', label: 'ChatGPT Studio', icon: MessageSquare },
    { id: 'agent', label: 'Autonomous Coding Agent', icon: Bot },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="brand-icon">
          <Sparkles size={20} />
        </div>
        <div className="brand-info">
          <h1>Antigravity AI</h1>
          <span>House of Intelligence</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        <div className="nav-section-title">Intelligence Core</div>
        {intelligenceCore.map((item) => {
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
        })}

        <div className="nav-section-title" style={{ marginTop: '1rem' }}>Product Playground</div>
        {productPlayground.map((item) => {
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
        })}
      </nav>

      <div className="sidebar-footer">
        <div className="system-pill">
          <span className="pulse-dot"></span>
          <span>
            {vitals?.cuda_available 
              ? `${vitals.device_name?.split(' ')[1] || 'GPU'} • ${vitals.vram_used_mb || 0}MB`
              : 'CPU Host Active'}
          </span>
        </div>
      </div>
    </aside>
  );
}
