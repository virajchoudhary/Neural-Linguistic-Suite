import { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import "./App.css";

const API = "http://127.0.0.1:8000";

const ALL_LANGUAGES = [
  { id: "en", label: "English", flag: "En" },
  { id: "hi", label: "Hindi", flag: "हि" },
  { id: "es", label: "Spanish", flag: "Es" },
];

const TRANSLATION_PAIRS = {
  "en→hi": "/translate/multi",
  "en→es": "/translate/multi",
  "hi→en": "/translate/hindi-to-english",
  "es→en": "/translate/spanish-to-english",
};

function getEndpoint(srcId, trgId) {
  return TRANSLATION_PAIRS[`${srcId}→${trgId}`] || null;
}

function getTargetsForSource(srcId) {
  return ALL_LANGUAGES.filter(
    (l) => l.id !== srcId && TRANSLATION_PAIRS[`${srcId}→${l.id}`],
  );
}

/* ── Section ── */

function Section({ title, subtitle, children, id }) {
  return (
    <div className="card" id={id}>
      <div className="card-header">
        <div>
          <h2>{title}</h2>
          {subtitle && <p className="card-subtitle">{subtitle}</p>}
        </div>
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

/* ── Tooltip ── */

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="custom-tooltip">
      <div className="label">Epoch {label}</div>
      {payload.map((p, i) => (
        <div key={i} className="value" style={{ color: p.stroke }}>
          {p.name}: {p.value?.toFixed(4)}
        </div>
      ))}
    </div>
  );
}

/* ── Theory ── */

function TheorySection() {
  return (
    <>
      <Section
        title="Transformer Architecture"
        subtitle="Core concepts of parallel sequence modeling"
      >
        <div className="theory-grid">
          <div className="theory-item">
            <h4>What is a Transformer?</h4>
            <p>
              A neural network architecture that relies entirely on{" "}
              <strong>Self-Attention</strong> mechanisms to model global
              dependencies between tokens. Unlike recurrent models, Transformers
              process entire sequences in parallel — dramatically increasing
              throughput and contextual depth.
            </p>
          </div>

          <div className="theory-item">
            <h4>Self-Attention</h4>
            <p>
              The mechanism that allows every token in a sequence to attend to
              every other token simultaneously. Attention scores are computed as
              scaled dot-products of Query, Key, and Value projections,
              capturing rich linguistic relationships regardless of distance in
              the sequence.
            </p>
          </div>

          <div className="theory-item">
            <h4>Multi-Head Attention</h4>
            <p>
              Rather than computing a single attention function, multiple heads
              run in parallel across independent representation subspaces. Each
              head can specialise in different aspects of language — syntax,
              semantics, coreference — and their outputs are concatenated and
              projected.
            </p>
          </div>

          <div className="theory-item">
            <h4>Positional Encoding</h4>
            <p>
              Since self-attention is permutation-invariant, positional
              encodings inject sequence-order information into the embeddings.
              Sinusoidal functions of varying frequency allow the model to
              generalise to sequence lengths unseen during training.
            </p>
          </div>

          <div className="theory-item">
            <h4>Neural Linguistic Suite</h4>
            <p>
              This suite routes all inference through the HuggingFace Inference
              API, combining Helsinki-NLP MarianMT models for multilingual
              translation with DistilBART for abstractive summarization —
              delivering production-quality results with zero local GPU
              overhead.
            </p>
          </div>

          <div className="theory-item">
            <h4>Subword Tokenization</h4>
            <p>
              MarianMT uses SentencePiece to segment text into subword units,
              enabling the model to handle rare and out-of-vocabulary words
              gracefully. DistilBART uses a BPE vocabulary of 50,265 tokens.
              Both strategies balance vocabulary size against coverage.
            </p>
          </div>
        </div>
      </Section>

      <Section
        title="Core Architecture Flow"
        subtitle="Encoder–Decoder processing pipeline"
      >
        <div className="diagram-container">
          <img
            src="/encoder_decoder_diagram.png"
            alt="Transformer encoder-decoder architecture diagram"
            className="arch-diagram-img"
          />
        </div>

        <div className="algorithm" style={{ marginTop: 16 }}>
          <p>
            <strong>Autoregressive Decoding:</strong>
          </p>
          <p className="indent1">
            <strong>for</strong> step = 1, ..., max_len:
          </p>
          <p className="indent2"> memory = Encoder(source_tokens)</p>
          <p className="indent2"> logits = Decoder(generated_so_far, memory)</p>
          <p className="indent2">
            {" "}
            candidates = TopK(logits, k=beam_size){" "}
            <span style={{ color: "#6b7280" }}>// Beam Search</span>
          </p>
          <p className="indent2"> next_token = argmax(score + log_prob)</p>
          <p className="indent2">
            {" "}
            <strong>if</strong> next_token == EOS: <strong>break</strong>
          </p>
        </div>
      </Section>
    </>
  );
}

/* ── Translation ── */

function TranslationSection() {
  const [srcLang, setSrcLang] = useState(ALL_LANGUAGES[0]);
  const [trgLang, setTrgLang] = useState(ALL_LANGUAGES[1]);
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [srcMenuOpen, setSrcMenuOpen] = useState(false);
  const [trgMenuOpen, setTrgMenuOpen] = useState(false);

  const availableTargets = getTargetsForSource(srcLang.id);

  const translate = async () => {
    if (!text.trim()) return;
    const endpoint = getEndpoint(srcLang.id, trgLang.id);
    if (!endpoint) {
      setResult({
        translation: `Translation for ${srcLang.label} → ${trgLang.label} is not available.`,
      });
      return;
    }
    setLoading(true);
    setResult(null);
    try {
      const response = await fetch(
        `${API}/translate/multi?target_lang=${trgLang.id}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: text.trim() }),
        },
      );

      if (response.status === 503) {
        const err = await response.json();
        setResult({
          translation:
            err.detail || "Model is warming up. Please retry in a moment.",
        });
        return;
      }
      if (!response.ok) {
        setResult({ translation: "Translation failed. Please try again." });
        return;
      }

      const data = await response.json();
      setResult(data);
    } catch {
      setResult({
        translation:
          "Could not reach the server. Please check your connection.",
      });
    } finally {
      setLoading(false);
    }
  };

  const selectSource = (lang) => {
    setSrcLang(lang);
    setSrcMenuOpen(false);
    setResult(null);
    if (trgLang.id === lang.id) {
      const targets = getTargetsForSource(lang.id);
      if (targets.length > 0) setTrgLang(targets[0]);
    }
    const targets = getTargetsForSource(lang.id);
    if (!targets.find((t) => t.id === trgLang.id) && targets.length > 0) {
      setTrgLang(targets[0]);
    }
  };

  const selectTarget = (lang) => {
    setTrgLang(lang);
    setTrgMenuOpen(false);
    setResult(null);
  };

  const swapLanguages = () => {
    const reverseEndpoint = getEndpoint(trgLang.id, srcLang.id);
    if (reverseEndpoint) {
      const oldSrc = srcLang;
      const oldTrg = trgLang;
      setSrcLang(oldTrg);
      setTrgLang(oldSrc);
      if (result?.translation && !result.translation.startsWith("[")) {
        setText(result.translation);
      }
      setResult(null);
    }
  };

  const canSwap = !!getEndpoint(trgLang.id, srcLang.id);

  const availableSources = ALL_LANGUAGES.filter((l) =>
    ALL_LANGUAGES.some(
      (t) => t.id !== l.id && TRANSLATION_PAIRS[`${l.id}→${t.id}`],
    ),
  );

  return (
    <Section
      title="Translation"
      subtitle="Translate text instantly between English, Hindi, and Spanish"
      id="translation"
    >
      <div className="lang-bar">
        <div className="lang-target-wrapper">
          <button
            className="lang-selector"
            onClick={() => {
              setSrcMenuOpen(!srcMenuOpen);
              setTrgMenuOpen(false);
            }}
          >
            <span className="lang-badge active">{srcLang.flag}</span>
            <span className="lang-name">{srcLang.label}</span>
            <span className="lang-chevron">{srcMenuOpen ? "▲" : "▼"}</span>
          </button>
          {srcMenuOpen && (
            <div className="lang-dropdown">
              {availableSources.map((lang) => (
                <button
                  key={lang.id}
                  className={`lang-dropdown-item ${lang.id === srcLang.id ? "selected" : ""}`}
                  onClick={() => selectSource(lang)}
                >
                  <span className="lang-badge-sm">{lang.flag}</span>
                  {lang.label}
                </button>
              ))}
            </div>
          )}
        </div>

        <button
          className={`swap-btn ${canSwap ? "" : "disabled"}`}
          onClick={swapLanguages}
          disabled={!canSwap}
          title={canSwap ? "Swap languages" : "Reverse pair not available"}
        >
          ⇄
        </button>

        <div className="lang-target-wrapper">
          <button
            className="lang-selector"
            onClick={() => {
              setTrgMenuOpen(!trgMenuOpen);
              setSrcMenuOpen(false);
            }}
          >
            <span className="lang-badge">{trgLang.flag}</span>
            <span className="lang-name">{trgLang.label}</span>
            <span className="lang-chevron">{trgMenuOpen ? "▲" : "▼"}</span>
          </button>
          {trgMenuOpen && (
            <div className="lang-dropdown">
              {availableTargets.map((lang) => (
                <button
                  key={lang.id}
                  className={`lang-dropdown-item ${lang.id === trgLang.id ? "selected" : ""}`}
                  onClick={() => selectTarget(lang)}
                >
                  <span className="lang-badge-sm">{lang.flag}</span>
                  {lang.label}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="translate-panels">
        <div className="translate-panel">
          <div className="panel-label">{srcLang.label}</div>
          <textarea
            className="translate-textarea"
            placeholder={`Enter ${srcLang.label} text...`}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                translate();
              }
            }}
            rows={4}
          />
        </div>

        <div className="translate-panel">
          <div className="panel-label">{trgLang.label}</div>
          <div className="translate-output">
            {loading ? (
              <div className="translate-loading">
                <span className="spinner spinner-light"></span>
                Translating...
              </div>
            ) : result ? (
              <span>{result.translation}</span>
            ) : (
              <span className="placeholder-text">
                Translation will appear here
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="translate-actions">
        <button
          className="btn-primary"
          onClick={translate}
          disabled={loading || !text.trim()}
        >
          {loading ? <span className="spinner"></span> : null}
          {loading ? "Translating..." : "Translate"}
        </button>
      </div>

      {result?.tokens?.length > 0 && (
        <div className="output-box" style={{ marginTop: 20 }}>
          <div className="token-breakdown">
            <div className="token-label">Input Tokenization</div>
            <div className="token-list">
              {result.tokens.map((tok, i) => (
                <span key={i} className="token-chip">
                  {tok}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </Section>
  );
}

/* ── Summarization ── */

function SummarizationSection() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const summarize = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch(`${API}/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim() }),
      });

      if (res.status === 503) {
        const err = await res.json();
        setResult({
          summary:
            err.detail || "Model is warming up. Please retry in a moment.",
          compression_ratio: "—",
        });
        return;
      }
      if (!res.ok) {
        setResult({
          summary: "Summarization failed. Please try again.",
          compression_ratio: "—",
        });
        return;
      }

      const data = await res.json();
      setResult(data);
    } catch {
      setResult({
        summary: "Could not reach the server. Please check your connection.",
        compression_ratio: "—",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Section
      title="Text Summarization"
      subtitle="Generate concise summaries from any text passage"
      id="summarization"
    >
      <textarea
        className="text-input"
        placeholder="Enter a longer text passage to summarize..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={5}
      />
      <div style={{ marginTop: 16 }}>
        <button
          className="btn-primary"
          onClick={summarize}
          disabled={loading || !text.trim()}
        >
          {loading ? <span className="spinner"></span> : null}
          {loading ? "Summarizing..." : "Summarize"}
        </button>
      </div>

      {result && (
        <div className="output-box">
          <div className="output-label">Summary</div>
          <div className="output-text">{result.summary}</div>
          <div className="compression-chip">
            Compression Ratio: {result.compression_ratio}
          </div>
        </div>
      )}
    </Section>
  );
}

/* ── Architecture ── */

function ArchitectureSection() {
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API}/model-info`)
      .then((res) => res.json())
      .then((data) => {
        setInfo(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  if (loading)
    return (
      <Section title="Model Architecture">
        <div className="status-message">Loading model info...</div>
      </Section>
    );
  if (!info)
    return (
      <Section title="Model Architecture">
        <div className="status-message">
          Could not load model info. Is the backend running?
        </div>
      </Section>
    );

  return (
    <>
      <Section
        title="Model Architecture"
        subtitle="Pre-trained Transformer architecture and model specifications"
      >
        <div className="arch-grid">
          <div className="arch-component">
            <h4>{info.encoder?.type || "Encoder"}</h4>
            <table className="theory-table">
              <thead>
                <tr>
                  <th>Layer</th>
                  <th>Type</th>
                  <th>Parameters</th>
                </tr>
              </thead>
              <tbody>
                {info.encoder?.layers?.map((layer, i) => (
                  <tr key={i}>
                    <td>{layer.name}</td>
                    <td>
                      <span className="type-badge">{layer.type}</span>
                    </td>
                    <td
                      style={{
                        fontFamily: "'JetBrains Mono', monospace",
                        fontSize: "0.85rem",
                      }}
                    >
                      {layer.params}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="arch-component">
            <h4>{info.decoder?.type || "Decoder"}</h4>
            <table className="theory-table">
              <thead>
                <tr>
                  <th>Layer</th>
                  <th>Type</th>
                  <th>Parameters</th>
                </tr>
              </thead>
              <tbody>
                {info.decoder?.layers?.map((layer, i) => (
                  <tr key={i}>
                    <td>{layer.name}</td>
                    <td>
                      <span className="type-badge">{layer.type}</span>
                    </td>
                    <td
                      style={{
                        fontFamily: "'JetBrains Mono', monospace",
                        fontSize: "0.85rem",
                      }}
                    >
                      {layer.params}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </Section>

      <Section
        title="Vocabulary Sizes"
        subtitle="Token counts across all translation and summarization models"
      >
        <div className="vocab-grid">
          {info.vocab_sizes &&
            Object.entries(info.vocab_sizes).map(([key, val]) => (
              <div key={key} className="vocab-item">
                <div className="vocab-label">{key.replace(/_/g, " ")}</div>
                <div className="vocab-value">{val.toLocaleString()}</div>
              </div>
            ))}
        </div>
      </Section>

      <Section
        title="Inference Configuration"
        subtitle="Decoding parameters and API specifications"
      >
        <div className="config-row">
          {info.training_config &&
            Object.entries(info.training_config).map(([key, val]) => (
              <span key={key} className="config-chip">
                {key.replace(/_/g, " ")}: {val}
              </span>
            ))}
        </div>
      </Section>
    </>
  );
}

/* ── Metrics ── */

function MetricsSection() {
  const [translationLogs, setTranslationLogs] = useState(null);
  const [translationEsLogs, setTranslationEsLogs] = useState(null);
  const [summarizationLogs, setSummarizationLogs] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/logs/translation`).then((r) => r.json()),
      fetch(`${API}/logs/translation-es`).then((r) => r.json()),
      fetch(`${API}/logs/summarization`).then((r) => r.json()),
    ])
      .then(([tl, tles, sl]) => {
        setTranslationLogs(tl);
        setTranslationEsLogs(tles);
        setSummarizationLogs(sl);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  if (loading)
    return (
      <Section title="Model Development Metrics">
        <div className="status-message">Loading metrics...</div>
      </Section>
    );

  const formatTranslationData = (logs) => {
    if (!logs?.train_loss) return [];
    return logs.train_loss.map((_, i) => ({
      epoch: i + 1,
      train_loss: logs.train_loss[i],
      val_loss: logs.val_loss?.[i],
      bleu: logs.bleu_scores?.[i],
    }));
  };

  const formatSummarizationData = (logs) => {
    if (!logs?.train_loss) return [];
    return logs.train_loss.map((_, i) => ({
      epoch: i + 1,
      train_loss: logs.train_loss[i],
      val_loss: logs.val_loss?.[i],
    }));
  };

  const hiData = formatTranslationData(translationLogs);
  const esData = formatTranslationData(translationEsLogs);
  const sumData = formatSummarizationData(summarizationLogs);

  return (
    <>
      <Section
        title="Model Development Metrics"
        subtitle="Baseline loss curves and BLEU scores from custom model training"
      >
        <div className="charts-row">
          <div className="chart-container">
            <div className="chart-title">English → Hindi</div>
            {hiData.length > 0 ? (
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={hiData}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="rgba(255,255,255,0.05)"
                  />
                  <XAxis
                    dataKey="epoch"
                    stroke="#64748b"
                    minTickGap={25}
                    tickMargin={10}
                  />
                  <YAxis stroke="#64748b" tickMargin={10} width={50} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="train_loss"
                    name="Train Loss"
                    stroke="#ffffff"
                    strokeWidth={2.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="val_loss"
                    name="Val Loss"
                    stroke="#6b7280"
                    strokeWidth={2.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="bleu"
                    name="BLEU Score"
                    stroke="#6b7280"
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                    strokeDasharray="5 5"
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="status-message">No data available</div>
            )}
          </div>

          <div className="chart-container">
            <div className="chart-title">Summarization</div>
            {sumData.length > 0 ? (
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={sumData}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="rgba(255,255,255,0.05)"
                  />
                  <XAxis
                    dataKey="epoch"
                    stroke="#64748b"
                    minTickGap={25}
                    tickMargin={10}
                  />
                  <YAxis stroke="#64748b" tickMargin={10} width={50} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="train_loss"
                    name="Train Loss"
                    stroke="#ffffff"
                    strokeWidth={2.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="val_loss"
                    name="Val Loss"
                    stroke="#6b7280"
                    strokeWidth={2.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="status-message">No data available</div>
            )}
          </div>
        </div>
      </Section>

      <Section
        title="English → Spanish"
        subtitle="Loss curves and BLEU score from Spanish translation baseline"
      >
        <div className="chart-container" style={{ maxWidth: 600 }}>
          <div className="chart-title">English → Spanish</div>
          {esData.length > 0 ? (
            <ResponsiveContainer width="100%" height={350}>
              <LineChart data={esData}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="rgba(255,255,255,0.05)"
                />
                <XAxis
                  dataKey="epoch"
                  stroke="#64748b"
                  minTickGap={25}
                  tickMargin={10}
                />
                <YAxis stroke="#64748b" tickMargin={10} width={50} />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="train_loss"
                  name="Train Loss"
                  stroke="#ffffff"
                  strokeWidth={2.5}
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="val_loss"
                  name="Val Loss"
                  stroke="#6b7280"
                  strokeWidth={2.5}
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="bleu"
                  name="BLEU Score"
                  stroke="#6b7280"
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                  strokeDasharray="5 5"
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="status-message">No data available</div>
          )}
        </div>
      </Section>
    </>
  );
}

/* ── Footer ── */

function Footer() {
  return (
    <footer className="app-footer">
      <span className="footer-copy">© Viraj Choudhary</span>

      <div className="footer-links">
        <a
          href="https://github.com/virajchoudhary"
          target="_blank"
          rel="noopener noreferrer"
        >
          GitHub
        </a>

        <a
          href="https://www.linkedin.com/in/virajchoudhary"
          target="_blank"
          rel="noopener noreferrer"
        >
          LinkedIn
        </a>

        <a
          href="https://x.com/virajchoudhary_"
          target="_blank"
          rel="noopener noreferrer"
        >
          Twitter
        </a>

        <a href="mailto:virajc188@gmail.com">Email</a>
      </div>
    </footer>
  );
}
/* ── App ── */

export default function App() {
  const [activeTab, setActiveTab] = useState("theory");

  const tabs = [
    { id: "theory", label: "Theory", component: <TheorySection /> },
    {
      id: "translation",
      label: "Translation",
      component: <TranslationSection />,
    },
    {
      id: "summarization",
      label: "Summarization",
      component: <SummarizationSection />,
    },
    {
      id: "architecture",
      label: "Architecture",
      component: <ArchitectureSection />,
    },
    { id: "metrics", label: "Metrics", component: <MetricsSection /> },
  ];

  return (
    <div className="app-layout">
      <nav className="sidebar">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={`sidebar-item ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>
      <main className="main-content">
        <header className="app-header">
          <h1>Neural Linguistic Suite</h1>
          <p>
            Multilingual Translation &amp; Abstractive Summarization via
            Transformer Models
          </p>
        </header>
        {tabs.find((t) => t.id === activeTab).component}
        <Footer />
      </main>
    </div>
  );
}
