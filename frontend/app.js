/* AI Debate Arena — frontend (Preact + htm via one vendored module, no
   build step). The backend state arrives as a WebSocket snapshot and is
   kept current by incremental events; keyed declarative rendering means
   DOM nodes persist across updates, so entrance animations play exactly
   once and updates never flash. */

import {
  html, render, useEffect, useReducer, useRef, useState,
} from "./vendor/preact-htm.module.js";

const STEP_ORDER = ["model", "research", "prep", "debate", "judging", "done"];
const STEP_LABELS = { model: "Model", research: "Research", prep: "Prep",
                      debate: "Debate", judging: "Judging", done: "Verdict" };

const EXAMPLES = [
  { label: "driverless cars",
    topic: "Driverless cars are safe enough for public roads." },
  { label: "NDIS overhaul",
    topic: "The Australian government's current overhaul of the NDIS is " +
           "moving too far and too fast at the expense of vulnerable citizens." },
  { label: "social media ban",
    topic: "Social media should be banned for children under 16." },
];

/* ---------------- Backend state mirror ---------------- */

/* Mirror of backend _apply(): mutates the draft, reducer returns a fresh
   top-level object so Preact re-renders. */
function applyEvent(s, ev) {
  switch (ev.type) {
    case "phase": s.phase = ev.phase; s.log.push(ev.message); break;
    case "status": s.log.push(ev.message); break;
    case "download_progress": s.download = ev; break;
    case "research_source": s.sources.push(ev); break;
    case "research_done":
      s.num_chunks = ev.num_chunks;
      s.semantic = ev.semantic || false;
      break;
    case "prep_positions":
      s.prep = { positions: { pro: ev.pro, con: ev.con }, stage: "positions",
                 window: 0, total_windows: 0, source: "", quotes: [],
                 sort_done: 0, briefs: { pro: null, con: null } };
      break;
    case "prep_start":
      if (s.prep) { s.prep.stage = "mining"; s.prep.total_windows = ev.total; }
      break;
    case "prep_window":
      if (s.prep) { s.prep.window = ev.index; s.prep.source = ev.source; }
      break;
    case "prep_quote":
      if (s.prep) s.prep.quotes.push({ quote: ev.quote, source: ev.source, side: null });
      break;
    case "prep_sort_start":
      if (s.prep) s.prep.stage = "sorting";
      break;
    case "prep_sort":
      if (s.prep && s.prep.quotes[ev.index]) {
        s.prep.quotes[ev.index].side = ev.side;
        s.prep.sort_done++;
      }
      break;
    case "prep_brief":
      if (s.prep) s.prep.briefs[ev.side] = ev.quotes;
      break;
    case "prep_done":
      if (s.prep) s.prep.stage = "done";
      break;
    case "turn_prep":
      s.current_prep = { speaker: ev.speaker, label: ev.label, queries: ev.queries };
      break;
    case "turn_start":
      s.current = { speaker: ev.speaker, phase: ev.phase, round: ev.round,
                    label: ev.label, text: "" };
      s.current_prep = null;
      break;
    case "token":
      if (s.current) s.current.text += ev.text;
      break;
    case "turn_end":
      s.transcript.push({ speaker: ev.speaker, phase: ev.phase,
                          round: ev.round, label: ev.label, text: ev.text });
      s.current = null;
      break;
    case "judge_start":
      s.judges.forEach((j) => { if (j.id === ev.judge_id) j.status = "deliberating"; });
      break;
    case "judge_criterion":
      s.judges.forEach((j) => {
        if (j.id === ev.judge_id) j.partial[ev.criterion] = ev.result;
      });
      break;
    case "judge_result":
      s.judges.forEach((j) => {
        if (j.id === ev.judge_id) { j.status = "done"; j.ballot = ev.ballot; }
      });
      break;
    case "verdict": s.verdict = ev; break;
    case "error": s.phase = "error"; s.error = ev.message; break;
  }
}

function reducer(state, ev) {
  if (ev.type === "snapshot") return ev.state;
  if (!state) return state;
  applyEvent(state, ev);
  return { ...state };
}

function useBackendState() {
  const [state, dispatch] = useReducer(reducer, null);
  useEffect(() => {
    let ws;
    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.onmessage = (msg) => dispatch(JSON.parse(msg.data));
      ws.onclose = () => setTimeout(connect, 1500);
    };
    connect();
    const ping = setInterval(() => {
      if (ws && ws.readyState === 1) ws.send("ping");
    }, 25000);
    return () => clearInterval(ping);
  }, []);
  return state;
}

/* ---------------- Setup view ---------------- */

/* XHR instead of fetch: only XHR exposes upload byte progress. */
function uploadMaterial(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/materials");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(Math.round((100 * e.loaded) / e.total));
    };
    // All bytes sent — anything from here on is server-side parsing.
    xhr.upload.onload = () => onProgress(100);
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) return resolve();
      let detail = xhr.statusText;
      try { detail = JSON.parse(xhr.responseText).detail || detail; } catch {}
      reject(new Error(typeof detail === "string" ? detail : JSON.stringify(detail)));
    };
    xhr.onerror = () => reject(new Error(`${file.name}: upload failed`));
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}

function Setup() {
  const [models, setModels] = useState([]);
  const [fakeLlm, setFakeLlm] = useState(false);
  const [ram, setRam] = useState(null);
  const [modelId, setModelId] = useState(null);
  const [topic, setTopic] = useState("");
  const [proPersonality, setProPersonality] = useState("");
  const [conPersonality, setConPersonality] = useState("");
  const [rounds, setRounds] = useState(2);
  const [materials, setMaterials] = useState([]);
  const [pending, setPending] = useState([]);
  const [matsOnly, setMatsOnly] = useState(false);
  const [matError, setMatError] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const fileRef = useRef();

  const refreshMaterials = async () => {
    const res = await fetch("/api/materials").then((r) => r.json());
    setMaterials(res.materials);
    // Materials-only with zero materials would leave the debaters with no
    // research at all — untick it when the last document is removed.
    if (res.materials.length === 0) setMatsOnly(false);
  };

  useEffect(() => {
    fetch("/api/models").then((r) => r.json()).then((d) => {
      setModels(d.models);
      setFakeLlm(d.fake_llm);
      const def = d.models.find((m) => m.default) || d.models[0];
      if (def) setModelId(def.id);
    });
    fetch("/api/system").then((r) => r.json()).then((d) => setRam(d.available_ram_gb));
    refreshMaterials();
  }, []);

  const onFiles = async (ev) => {
    const files = [...ev.target.files];
    ev.target.value = "";
    setMatError("");
    const failures = [];
    let added = 0;
    for (const file of files) {
      const pid = `${Date.now()}-${Math.random()}`;
      setPending((p) => [...p, { id: pid, name: file.name, pct: 0 }]);
      try {
        await uploadMaterial(file, (pct) =>
          setPending((p) => p.map((x) => (x.id === pid ? { ...x, pct } : x))));
        added++;
      } catch (e) {
        failures.push(e.message);
      }
      setPending((p) => p.filter((x) => x.id !== pid));
      await refreshMaterials(); // each finished file appears right away
    }
    // Uploading your own sources usually means you want the debate grounded
    // in them; switch to materials-only (the user can still untick it).
    if (added) setMatsOnly(true);
    if (failures.length) setMatError(failures.join(" — "));
  };

  const removeMaterial = async (id) => {
    await fetch(`/api/materials/${id}`, { method: "DELETE" });
    refreshMaterials();
  };

  const begin = async () => {
    setError("");
    if (topic.trim().length < 8) {
      setError("Please enter a debate topic (at least 8 characters).");
      return;
    }
    setBusy(true);
    try {
      const res = await fetch("/api/debate/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic: topic.trim(),
          model_id: modelId,
          pro_personality: proPersonality,
          con_personality: conPersonality,
          rounds: Number(rounds),
          use_web_research: !matsOnly,
        }),
      });
      if (!res.ok) {
        const detail = (await res.json()).detail || res.statusText;
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
    } catch (e) {
      setError(e.message);
      setBusy(false);
    }
  };

  return html`
  <main id="setup-view">
    <section class="card">
      <h2>1 · The Motion</h2>
      <input id="topic" type="text" maxlength="500" value=${topic}
             onInput=${(e) => setTopic(e.target.value)}
             placeholder="e.g. Driverless cars are safe enough for public roads" />
      <p class="hint">Works best phrased as a claim — PRO defends it, CON attacks it.
        Questions are fine too (PRO argues “yes”, CON argues “no”).</p>
      <div class="examples">
        Try:
        ${EXAMPLES.map((ex) => html`
          <button key=${ex.label} class="example-btn"
                  onClick=${() => setTopic(ex.topic)}>${ex.label}</button>`)}
      </div>
    </section>

    <section class="card">
      <h2>2 · The Model</h2>
      <p class="hint">${ram === null
        ? "Detecting available memory…"
        : `${ram} GB RAM available — a quantization that fits in memory will ` +
          `be selected automatically.` +
          (fakeLlm ? "  ⚠ FAKE_LLM mode is on (canned responses)." : "")}</p>
      <div id="model-list">
        ${models.map((m) => html`
          <label key=${m.id}
                 class=${"model-option" + (m.id === modelId ? " selected" : "")}>
            <input type="radio" name="model" value=${m.id}
                   checked=${m.id === modelId}
                   onChange=${() => setModelId(m.id)} />
            <div style="flex:1">
              <div class="model-name">${m.name} <span class="hint">· ${m.params}</span></div>
              <div class="model-desc">${m.description}</div>
            </div>
            <div>
              ${m.uncensored && html`<span class="badge uncensored">no guardrails</span>`}
              ${" "}
              ${m.downloaded
                ? html`<span class="badge downloaded">✓ downloaded · ${m.downloaded_gb} GB</span>`
                : html`<span class="badge will-download">will download</span>`}
            </div>
          </label>`)}
      </div>
    </section>

    <section class="card">
      <h2>3. Set debater personalities (optional) <span class="optional-tag">optional</span></h2>
      <div class="debater-grid">
        <div class="debater-config pro">
          <h3>FOR the motion <span class="side-tag pro-tag">PRO</span></h3>
          <textarea maxlength="1000" rows="3" value=${proPersonality}
                    onInput=${(e) => setProPersonality(e.target.value)}
                    placeholder="e.g. polite and considerate"></textarea>
        </div>
        <div class="debater-config con">
          <h3>AGAINST the motion <span class="side-tag con-tag">CON</span></h3>
          <textarea maxlength="1000" rows="3" value=${conPersonality}
                    onInput=${(e) => setConPersonality(e.target.value)}
                    placeholder="e.g. sassy and sarcastic"></textarea>
        </div>
      </div>
      <div class="rounds-row">
        <label for="rounds">Rebuttal rounds:</label>
        <select id="rounds" value=${rounds}
                onChange=${(e) => setRounds(e.target.value)}>
          ${[1, 2, 3, 4].map((n) => html`<option key=${n} value=${n}>${n}</option>`)}
        </select>
      </div>
    </section>

    <section class="card">
      <h2>4 · Your Research <span class="optional-tag">optional</span></h2>
      <p class="hint">Give the debaters your own source material — PDF, Word
        (.docx), text, Markdown or HTML files. They are indexed alongside the
        automatic Wikipedia ${"&"} web research, and cited by filename.</p>
      <input ref=${fileRef} type="file" multiple hidden
             accept=".pdf,.docx,.txt,.md,.markdown,.rst,.csv,.log,.html,.htm"
             onChange=${onFiles} />
      <button id="material-add" disabled=${pending.length > 0}
              onClick=${() => fileRef.current && fileRef.current.click()}>＋ Add documents</button>
      <ul id="material-list">
        ${materials.map((m) => html`
          <li key=${m.id} class="material-item">
            <span>📄 ${m.filename} (${(m.chars / 1000).toFixed(1)}k chars)</span>
            <button class="material-remove" title="Remove"
                    onClick=${() => removeMaterial(m.id)}>✕</button>
          </li>`)}
        ${pending.map((x) => html`
          <li key=${x.id} class="material-item pending">
            <span>📄 ${x.name}</span>
            <span class="material-status">${x.pct < 100 ? `uploading ${x.pct}%` : "processing…"}</span>
            <div class="material-progress-track">
              <div class=${"material-progress-fill" + (x.pct >= 100 ? " processing" : "")}
                   style=${`width:${x.pct}%`}></div>
            </div>
          </li>`)}
      </ul>
      <label class="hint checkbox-row">
        <input type="checkbox" checked=${matsOnly}
               onChange=${(e) => setMatsOnly(e.target.checked)} />
        Use only my materials (skip Wikipedia ${"&"} web search)
      </label>
      ${matError && html`<p class="error">${matError}</p>`}
    </section>

    <div class="begin-row">
      <button class="primary big" disabled=${busy} onClick=${begin}>Begin Debate</button>
      ${error && html`<p class="error">${error}</p>`}
    </div>
  </main>`;
}

/* ---------------- Arena: chrome ---------------- */

function Stepper({ phase }) {
  const idx = STEP_ORDER.indexOf(phase === "error" ? "model" : phase);
  return html`
  <ol id="stepper">
    ${STEP_ORDER.map((step, i) => {
      const active = step === phase || (phase === "done" && step === "done");
      const complete = idx > i || phase === "done";
      return html`
        <li key=${step}
            class=${(active ? "active" : "") + (complete ? " complete" : "")}>
          ${STEP_LABELS[step]}
        </li>`;
    })}
  </ol>`;
}

function DownloadPanel({ download, phase }) {
  if (!download || phase !== "model") return null;
  const gb = (n) => (n / 1024 ** 3).toFixed(2);
  return html`
  <section id="download-panel" class="card">
    <h2>Downloading model</h2>
    <div class="progress-track">
      <div class="progress-fill" style=${`width:${download.pct}%`}></div>
    </div>
    <p class="hint">${download.filename} — ${gb(download.done)} / ${gb(download.total)} GB (${download.pct}%)</p>
  </section>`;
}

function ResearchPanel({ state }) {
  if (!(state.sources.length > 0 || state.phase === "research")) return null;
  const summary = state.num_chunks
    ? `${state.sources.length} sources collected · ${state.num_chunks} passages indexed` +
      `${state.semantic ? " (hybrid keyword + semantic search)" : ""} for the debaters.`
    : state.phase === "research"
      ? "Collecting articles and encyclopedia entries…"
      : `${state.sources.length} sources collected.`;
  return html`
  <section id="research-panel" class="card">
    <h2>📚 Research</h2>
    <p class="hint">${summary}</p>
    <ul id="sources">
      ${state.sources.map((s, i) => html`
        <li key=${i}>
          ${s.url
            ? html`<a href=${s.url} target="_blank" rel="noopener">
                ${s.title} (${(s.chars / 1000).toFixed(1)}k chars)</a>`
            : html`📄 ${s.title} (${(s.chars / 1000).toFixed(1)}k chars)${" "}
                <span class="badge yours">your material</span>`}
        </li>`)}
    </ul>
  </section>`;
}

/* ---------------- Arena: case prep ---------------- */

function QuoteCard({ q, extra }) {
  const text = q.quote.length > 90 ? q.quote.slice(0, 90) + "…" : q.quote;
  return html`
  <div class=${"quote-card" + (extra ? ` ${extra}` : "")}
       title=${`${q.quote} — ${q.source}`}>“${text}”</div>`;
}

function prepHint(p) {
  if (!p) return "clarifying the clash…";
  if (p.stage === "positions") return "positions set";
  if (p.stage === "mining") return `mining evidence — ${p.quotes.length} quotes so far`;
  if (p.stage === "sorting") return `sorting evidence ${p.sort_done}/${p.quotes.length}`;
  const n = (side) => (p.briefs[side] || []).length;
  return `complete — PRO briefs ${n("pro")} quotes · CON briefs ${n("con")}`;
}

function PrepMining({ p }) {
  const feedRef = useRef();
  useEffect(() => {
    const feed = feedRef.current;
    if (feed) feed.scrollTop = feed.scrollHeight;
  }, [p.quotes.length]);
  return html`
  <div id="prep-mining">
    <div class="prep-scan-row">
      <div class="scan-doc"><div class="scan-line"></div></div>
      <div class="prep-scan-info">
        <p class="hint">${p.window
          ? `Scanning passage ${p.window} of ${p.total_windows} — ${p.source}`
          : "Preparing to scan the source material…"}</p>
        <div class="progress-track">
          <div class="progress-fill"
               style=${`width:${p.total_windows ? (100 * p.window) / p.total_windows : 0}%`}></div>
        </div>
        <p class="hint">${p.quotes.length} verbatim quote${p.quotes.length === 1 ? "" : "s"} verified against the source</p>
      </div>
    </div>
    <div class="quote-feed" ref=${feedRef}>
      ${p.quotes.map((q, i) => html`<${QuoteCard} key=${i} q=${q} extra="pop" />`)}
    </div>
  </div>`;
}

function PrepSorting({ p }) {
  const count = (side) => p.quotes.filter((q) => q.side === side).length;
  const waiting = p.quotes.filter((q) => q.side === null).length;
  return html`
  <div id="prep-sorting">
    <p class="hint">
      Each quote is weighed — reasoning first — and dealt to the side it
      truly supports (${p.sort_done}/${p.quotes.length})
    </p>
    <div class="sort-columns">
      <div class="sort-col pro">
        <h4>PRO's pile <span class="sort-count">${count("pro") || ""}</span></h4>
        <div class="sort-cards">
          ${p.quotes.map((q, i) => q.side === "pro" &&
            html`<${QuoteCard} key=${i} q=${q} extra="fly-left" />`)}
        </div>
      </div>
      <div class="sort-col neutral">
        <h4>In review <span class="sort-count">${waiting ? `${waiting} waiting` : ""}</span></h4>
        <div class="sort-cards">
          ${p.quotes.map((q, i) => (q.side === null || q.side === "neutral") &&
            html`<${QuoteCard} key=${i} q=${q}
                   extra=${q.side === null ? "unsorted" : "neutral-sorted pop"} />`)}
        </div>
      </div>
      <div class="sort-col con">
        <h4>CON's pile <span class="sort-count">${count("con") || ""}</span></h4>
        <div class="sort-cards">
          ${p.quotes.map((q, i) => q.side === "con" &&
            html`<${QuoteCard} key=${i} q=${q} extra="fly-right" />`)}
        </div>
      </div>
    </div>
  </div>`;
}

function PrepBriefs({ p }) {
  return html`
  <div id="prep-briefs">
    <p class="hint">Each side's final evidence brief — their strongest verbatim quotes:</p>
    <div class="brief-columns">
      ${["pro", "con"].map((side) => html`
        <div key=${side} class=${`brief-col ${side}`}>
          <h4>${side.toUpperCase()}'s brief</h4>
          <ol>
            ${(p.briefs[side] || []).map((q, i) => html`
              <li key=${i} title=${q.source}>“${q.quote}”</li>`)}
          </ol>
        </div>`)}
    </div>
  </div>`;
}

function PrepPanel({ prep: p, phase }) {
  if (!p && phase !== "prep") return null;
  return html`
  <section id="prep-panel" class="card">
    <details id="prep-details" open>
      <summary>
        <span class="prep-title">🗂 Case Preparation</span>
        <span class="hint">${prepHint(p)}</span>
      </summary>
      ${!p && html`
        <div class="prep-clarifying">
          <span class="scale-anim">⚖️</span> Clarifying what each side must
          prove<span class="dots"></span>
        </div>`}
      ${p && html`
        <div class="prep-positions">
          <div class="position-card pro"><b>PRO must prove</b><span>${p.positions.pro}</span></div>
          <div class="position-card con"><b>CON must prove</b><span>${p.positions.con}</span></div>
        </div>`}
      ${p && p.stage === "mining" && html`<${PrepMining} p=${p} />`}
      ${p && p.stage === "sorting" && html`<${PrepSorting} p=${p} />`}
      ${p && p.stage === "done" && html`<${PrepBriefs} p=${p} />`}
    </details>
  </section>`;
}

/* ---------------- Arena: transcript ---------------- */

function Turn({ turn, speaking }) {
  return html`
  <div class=${`turn ${turn.speaker}${speaking ? " speaking" : ""}`}>
    <div class="turn-label">${turn.label}</div>
    <div class="turn-text">${turn.text}</div>
  </div>`;
}

/* Placeholder bubble shown while a debater is researching/planning the
   speech it has not started delivering yet. */
function PrepBubble({ p }) {
  return html`
  <div class=${`turn ${p.speaker} preparing`}>
    <div class="turn-label">${p.label}</div>
    <div>
      <span class="prep-msg">
        ${p.queries.length ? "🔍 searching the research library" : "💭 planning this speech"}<span class="dots"></span>
      </span>
      ${p.queries.length > 0 && html`
        <div class="query-chips">
          ${p.queries.map((q, i) => html`<span key=${i} class="query-chip">${q}</span>`)}
        </div>`}
    </div>
  </div>`;
}

function Transcript({ transcript, current, current_prep }) {
  if (!transcript.length && !current && !current_prep) return null;
  return html`
  <section id="debate-panel">
    <div id="transcript">
      ${transcript.map((t) => html`<${Turn} key=${t.label} turn=${t} />`)}
      ${current && html`<${Turn} key=${current.label} turn=${current} speaking=${true} />`}
      ${!current && current_prep &&
        html`<${PrepBubble} key=${`prep-${current_prep.label}`} p=${current_prep} />`}
    </div>
  </section>`;
}

/* ---------------- Arena: judging & verdict ---------------- */

function JudgeCard({ judge: j, criteria }) {
  const partial = j.partial || {};
  const next = criteria.find((c) => !partial[c.key]);
  return html`
  <div class="judge-card">
    <h3>${j.name}</h3>
    ${j.status === "waiting" && html`<div class="judge-status">waiting</div>`}
    ${j.status === "deliberating" && html`
      <div class="judge-status deliberating">
        ${next ? `scoring ${next.label}` : "writing summary"}
      </div>`}
    ${j.status === "done" && html`<div class="judge-status">ballot in</div>`}
    ${j.status !== "waiting" && criteria.map((c) => {
      let p, q, rp, rq;
      if (j.ballot) {
        p = j.ballot.scores.pro[c.key];
        q = j.ballot.scores.con[c.key];
        rp = (j.ballot.reasons?.pro || {})[c.key];
        rq = (j.ballot.reasons?.con || {})[c.key];
      } else if (partial[c.key]) {
        p = partial[c.key].pro.score;
        q = partial[c.key].con.score;
        rp = partial[c.key].pro.reasoning;
        rq = partial[c.key].con.reasoning;
      } else {
        return null;
      }
      return html`
      <div key=${c.key} class="crit-row">
        <div class="crit-name"><span>${c.label}</span><span>${p} · ${q} / ${c.max}</span></div>
        <div class="crit-bars">
          <div class="crit-bar pro"><div style=${`width:${Math.min(100, (100 * p) / c.max)}%`}></div></div>
          <div class="crit-bar con"><div style=${`width:${Math.min(100, (100 * q) / c.max)}%`}></div></div>
        </div>
        ${rp && html`<div class="crit-reason pro-reason"><b>PRO ${p}/${c.max}</b> — ${rp}</div>`}
        ${rq && html`<div class="crit-reason con-reason"><b>CON ${q}/${c.max}</b> — ${rq}</div>`}
      </div>`;
    })}
    ${j.ballot && html`
      <div class="judge-total">
        <span class="pro-score">PRO ${j.ballot.totals.pro}</span>
        <span class="con-score">CON ${j.ballot.totals.con}</span>
      </div>
      ${j.ballot.summary && html`
        <div class="judge-summary"><h4>The Judge's Summary</h4>${j.ballot.summary}</div>`}`}
  </div>`;
}

function JudgingPanel({ judges, criteria }) {
  if (!judges.some((j) => j.status !== "waiting")) return null;
  return html`
  <section id="judging-panel">
    <h2 class="section-title">⚖️ The Judge's Ballot</h2>
    <div id="judges" class="judges-grid">
      ${judges.map((j) => html`<${JudgeCard} key=${j.id} judge=${j} criteria=${criteria} />`)}
    </div>
  </section>`;
}

const capitalize = (s) => s.charAt(0).toUpperCase() + s.slice(1);

function VerdictPanel({ verdict: v, judges }) {
  if (!v) return null;
  const cls = v.winner === "pro" ? "pro-wins" : v.winner === "con" ? "con-wins" : "";
  const banner = v.winner === "pro" ? "🏆 PRO wins — the motion carries!"
    : v.winner === "con" ? "🏆 CON wins — the motion falls!"
    : "🤝 It's a tie!";
  return html`
  <section id="verdict-panel">
    <div id="verdict-banner" class=${cls}>${banner}</div>
    <div id="verdict-detail">
      ${capitalize(v.method)}.<br />
      ${judges.length > 1 && html`Ballots — PRO ${v.ballots_won.pro} · CON ${v.ballots_won.con}  |  `}
      Total points — PRO ${v.totals.pro} · CON ${v.totals.con} (of ${judges.length * 100})
    </div>
  </section>`;
}

/* ---------------- Arena: actions & jump button ---------------- */

function Actions({ state }) {
  const finished = ["done", "error", "idle"].includes(state.phase);
  return html`
  <div class="arena-actions">
    ${!finished && html`
      <button class="danger"
              onClick=${() => fetch("/api/debate/stop", { method: "POST" })}>■ Stop debate</button>`}
    ${state.phase === "done" && state.transcript.length > 0 && html`
      <a class="button" href="/api/export/pdf" download>⬇ Export PDF transcript</a>`}
    ${finished && html`
      <button onClick=${async () => {
        // Clear the finished debate server-side; the broadcast snapshot
        // flips the app back to the setup view.
        try { await fetch("/api/debate/reset", { method: "POST" }); } catch {}
      }}>↻ New debate</button>`}
  </div>`;
}

/* The page never scrolls on its own — the reader scrolls freely while text
   streams in. A floating button offers a jump to the newest text instead. */
function JumpButton({ streaming }) {
  const [, poke] = useState(0);
  useEffect(() => {
    const onScroll = () => poke((x) => x + 1);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  const atBottom = window.innerHeight + window.scrollY >=
    document.body.scrollHeight - 200;
  if (!streaming || atBottom) return null;
  return html`
  <button id="jump-btn"
          onClick=${() => window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" })}>
    ↓ Jump to latest
  </button>`;
}

/* ---------------- App ---------------- */

function Arena({ state, criteria }) {
  return html`
  <main id="arena-view">
    <div id="topic-banner">“${state.topic}”</div>
    <${Stepper} phase=${state.phase} />
    <${DownloadPanel} download=${state.download} phase=${state.phase} />
    <${ResearchPanel} state=${state} />
    <${PrepPanel} prep=${state.prep} phase=${state.phase} />
    <${Transcript} transcript=${state.transcript} current=${state.current}
                   current_prep=${state.current_prep} />
    <${JudgingPanel} judges=${state.judges} criteria=${criteria} />
    <${VerdictPanel} verdict=${state.verdict} judges=${state.judges} />
    ${state.phase === "error" && html`
      <section id="error-panel" class="card error-card">
        <h2>Something went wrong</h2>
        <p id="error-message">${state.error || "Unknown error"}</p>
      </section>`}
    <${Actions} state=${state} />
    <p id="status-line" class="hint">${state.log.length ? state.log[state.log.length - 1] : ""}</p>
  </main>`;
}

function App() {
  const state = useBackendState();
  const [criteria, setCriteria] = useState([]);
  useEffect(() => {
    fetch("/api/system").then((r) => r.json()).then((d) => setCriteria(d.criteria));
  }, []);
  const active = state && state.phase !== "idle";
  return html`
    ${active
      ? html`<${Arena} state=${state} criteria=${criteria} />`
      : html`<${Setup} key=${state ? "ready" : "loading"} />`}
    <${JumpButton} streaming=${!!(state && state.current)} />`;
}

render(html`<${App} />`, document.getElementById("root"));
