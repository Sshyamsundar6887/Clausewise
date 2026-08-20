const { useState } = React;

const tabs = [
  { id: "overview", label: "Overview" },
  { id: "simplification", label: "Plain English" },
  { id: "entities", label: "Entities" },
  { id: "clauses", label: "Clauses" },
  { id: "classification", label: "Classification" },
];

function Metric({ label, value, tone = "neutral" }) {
  return <div className={`metric metric-${tone}`}><span>{label}</span><strong>{value || "None found"}</strong></div>;
}

function EmptyState() {
  return <div className="empty-state"><div className="empty-mark">CW</div><h2>Your analysis will appear here</h2><p>Upload a contract to surface its structure, obligations, entities, and risk signals.</p></div>;
}

function Overview({ data }) {
  const summary = data.summary || {};
  const info = data.document_info || {};
  return <div className="view-stack">
    <div className="result-heading"><div><p className="eyebrow">Document overview</p><h2>{summary.documentType || "Legal agreement"}</h2></div><span className={`risk-pill risk-${(summary.riskLevel || "low").toLowerCase().replace("-", "")}`}>{summary.riskLevel || "Low"} risk</span></div>
    <div className="metrics-grid">
      <Metric label="Key parties" value={summary.parties?.join(", ")} />
      <Metric label="Critical dates" value={summary.criticalDates?.join(", ")} />
      <Metric label="Key terms" value={summary.keyTerms?.join(", ")} />
      <Metric label="Obligations" value={summary.obligations?.join(", ")} />
    </div>
    <section className="content-section"><div className="section-label">File details</div><div className="file-detail"><strong>{info.filename}</strong><span>{info.size} | {info.upload_date?.slice(0, 10)}</span></div></section>
    <section className="content-section"><div className="section-label">Text preview</div><p className="preview">{data.text_content}</p></section>
  </div>;
}

function Simplification({ data }) {
  const sim = data.simplification || {};
  return <div className="view-stack"><div className="result-heading"><div><p className="eyebrow">Translation layer</p><h2>Plain English comparison</h2></div></div><div className="comparison"><article><div className="panel-kicker">Original clause</div><p>{sim.original || "No clause available."}</p></article><article className="plain-panel"><div className="panel-kicker">Plain English</div><p>{sim.simplified || "No simplified clause available."}</p></article></div></div>;
}

function Entities({ data }) {
  const entities = data.entities || [];
  return <div className="view-stack"><div className="result-heading"><div><p className="eyebrow">Named entity recognition</p><h2>{entities.length} entities identified</h2></div></div>{entities.length ? <div className="entity-list">{entities.slice(0, 40).map((entity, index) => <div className="entity-row" key={`${entity.text}-${index}`}><strong>{entity.text}</strong><span>{entity.label}</span><em>{Math.round((entity.confidence || 0) * 100)}%</em></div>)}</div> : <p className="muted">No named entities were found in this document.</p>}</div>;
}

function Clauses({ data }) {
  const clauses = data.clauses || [];
  return <div className="view-stack"><div className="result-heading"><div><p className="eyebrow">Clause intelligence</p><h2>{clauses.length} clauses analyzed</h2></div></div>{clauses.length ? <div className="clause-list">{clauses.map((clause) => <article className={`clause-card ${clause.importance?.toLowerCase()}`} key={clause.id}><div className="clause-top"><strong>{clause.title}</strong><span>{clause.importance}</span></div><p>{clause.text}</p><small>{clause.category}</small></article>)}</div> : <p className="muted">No clauses were long enough to analyze.</p>}</div>;
}

function Classification({ data }) {
  const classification = data.classification || {};
  return <div className="view-stack"><div className="result-heading"><div><p className="eyebrow">Document classifier</p><h2>Likely document types</h2></div></div><div className="prediction-list">{(classification.predictions || []).map((prediction, index) => <div className={`prediction ${index === 0 ? "top-prediction" : ""}`} key={prediction.type}><div><span className="rank">0{index + 1}</span><strong>{prediction.type}</strong></div><div className="confidence"><span>{prediction.confidence.toFixed(1)}%</span><div><i style={{ width: `${prediction.confidence}%` }} /></div></div></div>)}</div><div className="indicator-wrap"><div className="section-label">Signals detected</div><div className="indicator-list">{(classification.keyIndicators || []).map((indicator) => <span key={indicator}>{indicator}</span>)}</div></div></div>;
}

function Results({ data, tab }) {
  if (tab === "overview") return <Overview data={data} />;
  if (tab === "simplification") return <Simplification data={data} />;
  if (tab === "entities") return <Entities data={data} />;
  if (tab === "clauses") return <Clauses data={data} />;
  return <Classification data={data} />;
}

function App() {
  const [darkMode, setDarkMode] = useState(() => localStorage.getItem("clausewise-theme") === "dark");
  const [file, setFile] = useState(null);
  const [data, setData] = useState(null);
  const [tab, setTab] = useState("overview");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  document.body.dataset.theme = darkMode ? "dark" : "light";

  function toggleTheme() {
    const nextTheme = darkMode ? "light" : "dark";
    localStorage.setItem("clausewise-theme", nextTheme);
    setDarkMode(!darkMode);
  }

  async function analyze() {
    if (!file) return;
    setLoading(true); setError("");
    const formData = new FormData(); formData.append("file", file);
    try {
      const response = await fetch("/api/analyze", { method: "POST", body: formData });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Unable to analyze the document.");
      setData(payload); setTab("overview");
    } catch (requestError) { setError(requestError.message); }
    finally { setLoading(false); }
  }

  return <main>
    <header className="topbar"><a className="brand" href="/"><span className="brand-mark">C</span><span>ClauseWise</span></a><div className="topbar-actions"><span className="status"><i /> Local analysis workspace</span><button className="theme-toggle" type="button" onClick={toggleTheme} aria-label={`Switch to ${darkMode ? "light" : "dark"} mode`} title={`Switch to ${darkMode ? "light" : "dark"} mode`}>{darkMode ? "☼" : "☾"}</button></div></header>
    <section className="hero"><div className="hero-copy"><p className="eyebrow">Legal document intelligence</p><h1>Read the fine print<br /><em>with a clearer lens.</em></h1><p>Turn dense agreements into a concise map of what matters, who is involved, and where the pressure points are.</p></div><div className="hero-note"><span>01</span><p>Upload once.<br />Understand more.</p></div></section>
    <section className="workspace">
      <aside className="upload-panel"><div className="section-label">Start an analysis</div><label className={`dropzone ${file ? "has-file" : ""}`}><input type="file" accept=".pdf,.docx,.txt" onChange={(event) => { setFile(event.target.files[0]); setError(""); }} /><span className="upload-icon">{file ? "OK" : "+"}</span><strong>{file ? file.name : "Choose a document"}</strong><small>{file ? `${(file.size / 1024).toFixed(1)} KB ready` : "PDF, DOCX, or TXT up to 10 MB"}</small></label><button className="analyze-button" disabled={!file || loading} onClick={analyze}>{loading ? "Analyzing document..." : "Analyze document"}<span>{"->"}</span></button>{error && <p className="error-message">{error}</p>}<div className="privacy-note"><span>i</span><p>Your file is processed locally and removed after analysis.</p></div></aside>
      <section className="results-panel"><div className="results-top"><div><div className="section-label">Analysis workspace</div><h2>{data ? data.document_info?.filename : "No document loaded"}</h2></div>{data && <span className="ready-label"><i /> Analysis ready</span>}</div>{data && <nav className="tabs">{tabs.map((item) => <button className={tab === item.id ? "active" : ""} key={item.id} onClick={() => setTab(item.id)}>{item.label}</button>)}</nav>}{data ? <Results data={data} tab={tab} /> : <EmptyState />}</section>
    </section>
    <footer><span>ClauseWise</span><span>AI-assisted understanding for everyday agreements</span></footer>
  </main>;
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
