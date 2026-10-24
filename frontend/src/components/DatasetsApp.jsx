import React, { useEffect, useState } from 'react';
import { RefreshCw } from 'lucide-react';

const SECTORS = [
  { id: 'general', label: 'General' },
  { id: 'language', label: 'Language' },
  { id: 'code', label: 'Code' },
  { id: 'chat', label: 'Conversation' },
];

export default function DatasetsApp({ onUseForTraining }) {
  const [datasets, setDatasets] = useState([]);
  const [tab, setTab] = useState('yours');
  const [includeMode, setIncludeMode] = useState('paste');
  const [busy, setBusy] = useState('');
  const [notice, setNotice] = useState(null);
  const [name, setName] = useState('');
  const [sector, setSector] = useState('general');
  const [notes, setNotes] = useState('');
  const [split, setSplit] = useState('auto');
  const [text, setText] = useState('');
  const [path, setPath] = useState('');
  const [file, setFile] = useState(null);
  const [previewId, setPreviewId] = useState(null);
  const [preview, setPreview] = useState([]);

  const loadList = async () => {
    const res = await fetch('/api/datasets');
    if (!res.ok) return;
    const data = await res.json();
    setDatasets(data.datasets || []);
  };

  useEffect(() => {
    loadList();
  }, []);

  const flash = (type, message) => {
    setNotice({ type, text: message });
    setTimeout(() => setNotice(null), 4500);
  };

  const resetForm = () => {
    setName('');
    setNotes('');
    setText('');
    setPath('');
    setFile(null);
    setSplit('auto');
  };

  const afterInclude = async (record) => {
    flash('success', `${record.name} is in your library (${record.sample_count} samples).`);
    resetForm();
    await loadList();
    setTab('yours');
  };

  const includePaste = async () => {
    if (!name.trim() || !text.trim()) {
      flash('error', 'Give it a name and paste some text.');
      return;
    }
    setBusy('paste');
    try {
      const res = await fetch('/api/datasets/include', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: name.trim(),
          text,
          sector,
          notes,
          split,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Could not include that text.');
      await afterInclude(data.dataset);
    } catch (err) {
      flash('error', err.message);
    } finally {
      setBusy('');
    }
  };

  const includePath = async () => {
    if (!name.trim() || !path.trim()) {
      flash('error', 'Give it a name and a file or folder path on this computer.');
      return;
    }
    setBusy('path');
    try {
      const res = await fetch('/api/datasets/include-path', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: name.trim(),
          path: path.trim(),
          sector,
          notes,
          split,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Could not include that path.');
      await afterInclude(data.dataset);
    } catch (err) {
      flash('error', err.message);
    } finally {
      setBusy('');
    }
  };

  const includeUpload = async () => {
    if (!file) {
      flash('error', 'Choose a file first.');
      return;
    }
    setBusy('upload');
    try {
      const body = new FormData();
      body.append('file', file);
      body.append('name', name.trim() || file.name);
      body.append('sector', sector);
      body.append('notes', notes);
      body.append('split', split);
      const res = await fetch('/api/datasets/upload', { method: 'POST', body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Could not include that file.');
      await afterInclude(data.dataset);
    } catch (err) {
      flash('error', err.message);
    } finally {
      setBusy('');
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm(`Remove ${id} from the library?`)) return;
    const res = await fetch(`/api/datasets/${encodeURIComponent(id)}`, { method: 'DELETE' });
    if (!res.ok) {
      flash('error', 'Could not remove that dataset.');
      return;
    }
    if (previewId === id) {
      setPreviewId(null);
      setPreview([]);
    }
    await loadList();
  };

  const showPreview = async (id) => {
    if (previewId === id) {
      setPreviewId(null);
      setPreview([]);
      return;
    }
    const res = await fetch(`/api/datasets/${encodeURIComponent(id)}/samples?limit=5`);
    const data = await res.json().catch(() => ({}));
    setPreviewId(id);
    setPreview(data.samples || []);
  };

  const sectorLabel = (id) => SECTORS.find((s) => s.id === id)?.label || id;

  return (
    <div>
      {notice && <div className={`notice ${notice.type}`}>{notice.text}</div>}

      <div className="page-header">
        <p className="page-lead" style={{ margin: 0 }}>
          Include any text dataset you have — a file, a folder, or pasted examples. Then attach it when you train a local model.
        </p>
        <button className="btn btn-secondary" onClick={loadList}><RefreshCw size={14} /> Refresh</button>
      </div>

      <div className="soft-tabs">
        {[
          ['yours', 'Your datasets'],
          ['include', 'Include a dataset'],
        ].map(([id, label]) => (
          <button key={id} className={`btn ${tab === id ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setTab(id)}>
            {label}
          </button>
        ))}
      </div>

      {tab === 'yours' && (
        <div className="model-grid">
          {datasets.map((d) => (
            <div key={d.id} className="card-panel">
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                <div>
                  <h3>{d.name}</h3>
                  <div className="hint">{sectorLabel(d.sector)} · {d.source}</div>
                </div>
                {d.source === 'builtin' && <span className="chip">Built-in</span>}
              </div>
              <div className="hint" style={{ margin: '12px 0' }}>
                {d.sample_count} samples
                {d.used_by?.length ? ` · used by ${d.used_by.slice(-2).join(', ')}` : ''}
              </div>
              {d.preview?.[0] && (
                <p className="hint" style={{ marginBottom: 12 }}>{d.preview[0]}</p>
              )}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                <button className="btn btn-primary" onClick={() => onUseForTraining?.(d.id)}>
                  Train with this
                </button>
                <button className="btn btn-secondary" onClick={() => showPreview(d.id)}>
                  {previewId === d.id ? 'Hide samples' : 'Preview'}
                </button>
                {d.source !== 'builtin' && (
                  <button className="btn btn-secondary" onClick={() => handleDelete(d.id)}>Delete</button>
                )}
              </div>
              {previewId === d.id && (
                <div style={{ marginTop: 12 }}>
                  {preview.length === 0 && <p className="hint">No samples to show.</p>}
                  {preview.map((sample, i) => (
                    <pre key={i} className="hint" style={{ whiteSpace: 'pre-wrap', margin: '8px 0' }}>
                      {sample.slice(0, 400)}{sample.length > 400 ? '…' : ''}
                    </pre>
                  ))}
                </div>
              )}
            </div>
          ))}
          {datasets.length === 0 && (
            <div className="card-panel empty-state" style={{ gridColumn: '1 / -1' }}>
              <p>No datasets yet. Open Include a dataset.</p>
            </div>
          )}
        </div>
      )}

      {tab === 'include' && (
        <div className="card-panel">
          <h3 style={{ marginBottom: 8 }}>Add to the library</h3>
          <p className="hint" style={{ marginBottom: 14 }}>
            Accepted files: .txt, .md, .json, .jsonl, .csv, .tsv. Chat JSON with messages, prompt/completion, or instruction/output is understood.
          </p>

          <div className="soft-tabs" style={{ marginBottom: 16 }}>
            {[
              ['paste', 'Paste text'],
              ['upload', 'Upload a file'],
              ['path', 'Path on this computer'],
            ].map(([id, label]) => (
              <button key={id} className={`btn ${includeMode === id ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setIncludeMode(id)}>
                {label}
              </button>
            ))}
          </div>

          <label className="field-label">Name</label>
          <input className="form-input" value={name} onChange={(e) => setName(e.target.value)} placeholder="Customer support chats" />

          <label className="field-label">What kind of data is this?</label>
          <select className="form-select" value={sector} onChange={(e) => setSector(e.target.value)}>
            {SECTORS.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
          </select>

          <label className="field-label">How to split plain text</label>
          <select className="form-select" value={split} onChange={(e) => setSplit(e.target.value)}>
            <option value="auto">Decide automatically</option>
            <option value="blank_line">Blank line between examples</option>
            <option value="line">One example per line</option>
            <option value="whole">Treat as one long document</option>
          </select>

          <label className="field-label">Notes (optional)</label>
          <input className="form-input" value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="Where this came from, or what it is for" />

          {includeMode === 'paste' && (
            <>
              <label className="field-label">Examples</label>
              <textarea
                className="form-textarea"
                rows={8}
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder={"User: hi\nAssistant:\nHello. How can I help?\n\nUser: thanks\nAssistant:\nYou are welcome."}
              />
              <button className="btn btn-primary" style={{ marginTop: 12 }} disabled={busy === 'paste'} onClick={includePaste}>
                {busy === 'paste' ? 'Including…' : 'Include in library'}
              </button>
            </>
          )}

          {includeMode === 'upload' && (
            <>
              <label className="field-label">File</label>
              <input className="form-input" type="file" accept=".txt,.md,.json,.jsonl,.csv,.tsv" onChange={(e) => setFile(e.target.files?.[0] || null)} />
              <button className="btn btn-primary" style={{ marginTop: 12 }} disabled={busy === 'upload'} onClick={includeUpload}>
                {busy === 'upload' ? 'Including…' : 'Include file'}
              </button>
            </>
          )}

          {includeMode === 'path' && (
            <>
              <label className="field-label">File or folder path</label>
              <input className="form-input" value={path} onChange={(e) => setPath(e.target.value)} placeholder="C:\\data\\support.jsonl" />
              <p className="hint" style={{ marginTop: 6 }}>A folder pulls in every supported file inside it.</p>
              <button className="btn btn-primary" style={{ marginTop: 12 }} disabled={busy === 'path'} onClick={includePath}>
                {busy === 'path' ? 'Including…' : 'Include from path'}
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}
