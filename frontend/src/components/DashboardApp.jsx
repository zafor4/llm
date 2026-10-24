import React from 'react';
import { 
  Cpu, 
  HardDrive, 
  Zap, 
  ShieldCheck, 
  ArrowRight, 
  Bot, 
  MessageSquare, 
  Flame, 
  Layers 
} from 'lucide-react';

export default function DashboardApp({ vitals, setActiveApp, activeModel }) {
  const cards = [
    {
      title: 'Autonomous Coding Agent',
      desc: 'Execute real software goals in a secure sandbox. Self-corrects on failed assertions using runtime feedback.',
      icon: Bot,
      appId: 'agent',
      tag: 'SWE-bench Style',
      accent: 'var(--accent-blue)',
    },
    {
      title: 'Chatbot Studio',
      desc: 'Interactive conversational chat with local SmallGPT checkpoints and customizable system prompts.',
      icon: MessageSquare,
      appId: 'chat',
      tag: 'ChatGPT Style',
      accent: 'var(--accent-purple)',
    },
    {
      title: 'Model Training Studio',
      desc: 'Train new causal GPT models on your GPU from scratch with live loss curves and progress telemetry.',
      icon: Flame,
      appId: 'training',
      tag: 'PyTorch Engine',
      accent: 'var(--accent-amber)',
    },
    {
      title: 'Models Fleet Manager',
      desc: 'Manage and switch between checkpoints (Micro 1.9M, Mini 6.2M, Small 14.5M) and local endpoints.',
      icon: Layers,
      appId: 'models',
      tag: 'Multi-Model',
      accent: 'var(--accent-emerald)',
    },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Hero Welcome Banner */}
      <div 
        className="card-panel" 
        style={{ 
          background: 'linear-gradient(135deg, rgba(30, 58, 138, 0.3) 0%, rgba(88, 28, 135, 0.25) 100%)',
          border: '1px solid rgba(59, 130, 246, 0.3)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '16px'
        }}
      >
        <div>
          <div style={{ display: 'inline-block', fontSize: '0.75rem', fontWeight: 700, color: '#38bdf8', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '6px' }}>
            Production AI Platform
          </div>
          <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.65rem', fontWeight: 800, color: '#ffffff', marginBottom: '6px' }}>
            Welcome to Antigravity AI Studio
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9375rem', maxWidth: '650px' }}>
            A unified operating platform combining custom compact causal Transformers with goal-directed autonomous coding agents and live GPU model training.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '12px' }}>
          <button className="btn btn-primary" onClick={() => setActiveApp('agent')}>
            <Bot size={16} />
            Launch Agent
          </button>
          <button className="btn btn-secondary" onClick={() => setActiveApp('training')}>
            <Flame size={16} />
            Train Model
          </button>
        </div>
      </div>

      {/* Real-time Hardware Vitals Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px' }}>
        <div className="card-panel" style={{ padding: '18px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase' }}>Compute Device</span>
            <Cpu size={16} color="var(--accent-blue)" />
          </div>
          <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#ffffff' }}>
            {vitals?.device_name || 'Loading...'}
          </div>
          <div style={{ fontSize: '0.75rem', color: '#10b981', marginTop: '4px' }}>
            PyTorch {vitals?.pytorch_version || '2.6.0'} • CUDA Active
          </div>
        </div>

        <div className="card-panel" style={{ padding: '18px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase' }}>GPU VRAM Usage</span>
            <Zap size={16} color="var(--accent-emerald)" />
          </div>
          <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#ffffff' }}>
            {vitals?.vram_used_mb || 0} / {vitals?.vram_total_mb || 0} MB
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Dedicated Video Memory
          </div>
        </div>

        <div className="card-panel" style={{ padding: '18px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase' }}>Host Memory</span>
            <HardDrive size={16} color="var(--accent-purple)" />
          </div>
          <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#ffffff' }}>
            {vitals?.ram_used_gb || 0} / {vitals?.ram_total_gb || 0} GB
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
            CPU Usage: {vitals?.cpu_usage_percent || 0}%
          </div>
        </div>

        <div className="card-panel" style={{ padding: '18px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase' }}>Active Checkpoints</span>
            <ShieldCheck size={16} color="var(--accent-amber)" />
          </div>
          <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#ffffff' }}>
            {vitals?.checkpoints_count || 4} Trained Models
          </div>
          <div style={{ fontSize: '0.75rem', color: '#60a5fa', marginTop: '4px' }}>
            Active: {activeModel}
          </div>
        </div>
      </div>

      {/* App Launchers Grid */}
      <div>
        <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1.15rem', fontWeight: 700, marginBottom: '14px', color: '#ffffff' }}>
          Integrated Applications & Subsystems
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
          {cards.map((c) => {
            const Icon = c.icon;
            return (
              <div 
                key={c.appId} 
                className="card-panel" 
                style={{ 
                  cursor: 'pointer', 
                  transition: 'var(--transition)',
                  border: '1px solid var(--border-subtle)',
                }}
                onClick={() => setActiveApp(c.appId)}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = c.accent;
                  e.currentTarget.style.transform = 'translateY(-3px)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = 'var(--border-subtle)';
                  e.currentTarget.style.transform = 'none';
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                  <div style={{ width: '40px', height: '40px', borderRadius: 'var(--radius-sm)', background: `${c.accent}20`, display: 'flex', alignItems: 'center', justifyContent: 'center', color: c.accent }}>
                    <Icon size={20} />
                  </div>
                  <span style={{ fontSize: '0.6875rem', fontWeight: 600, color: c.accent, background: `${c.accent}15`, padding: '3px 8px', borderRadius: '9999px' }}>
                    {c.tag}
                  </span>
                </div>

                <h4 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#ffffff', marginBottom: '6px' }}>
                  {c.title}
                </h4>
                <p style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '16px' }}>
                  {c.desc}
                </p>

                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8125rem', fontWeight: 600, color: c.accent }}>
                  <span>Launch Application</span>
                  <ArrowRight size={14} />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
