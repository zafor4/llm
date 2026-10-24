import React from 'react';
import { 
  LayoutDashboard, 
  MessageSquare, 
  Bot, 
  Cpu, 
  Flame, 
  LineChart, 
  Sparkles 
} from 'lucide-react';

export default function Sidebar({ activeApp, setActiveApp, activeModel, vitals }) {
  const navItems = [
    { id: 'dashboard', label: 'Platform Hub', icon: LayoutDashboard },
    { id: 'chat', label: 'Chatbot Studio', icon: MessageSquare },
    { id: 'agent', label: 'Coding Agent', icon: Bot },
    { id: 'training', label: 'Training Studio', icon: Flame },
    { id: 'models', label: 'Models Fleet', icon: Cpu },
    { id: 'benchmarks', label: 'Benchmarks & Diagnostics', icon: LineChart },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="brand-icon">
          <Sparkles size={20} />
        </div>
        <div className="brand-info">
          <h1>Antigravity AI</h1>
          <span>Autonomous Systems OS</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        <div className="nav-section-title">Applications</div>
        {navItems.map((item) => {
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
