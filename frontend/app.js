/* AI Debate Arena — frontend. Mirrors backend state from the WebSocket
   snapshot, then applies incremental events for live streaming. */

const $ = (id) => document.getElementById(id);

let CRITERIA = [];      // [{key, label, max}] from /api/system
let state = null;       // mirror of backend state
let ws = null;

/* ---------------- Setup view ---------------- */

async function loadSetup() {
  const [modelsRes, sysRes] = await Promise.all([
    fetch("/api/models").then((r) => r.json()),
    fetch("/api/system").then((r) => r.json()),
  ]);
  CRITERIA = sysRes.criteria;
  // The WebSocket snapshot can render judges before this fetch resolves.
  if (state && state.phase !== "idle") renderJudges();
  $("ram-info").textContent =
    `${sysRes.available_ram_gb} GB RAM available — a quantization that fits ` +
    `in memory will be selected automatically.` +
    (modelsRes.fake_llm ? "  ⚠ FAKE_LLM mode is on (canned responses)." : "");

  const list = $("model-list");
  list.innerHTML = "";
  for (const m of modelsRes.models) {
    const opt = document.createElement("label");
    opt.className = "model-option" + (m.default ? " selected" : "");
    const badges = [];
    if (m.uncensored) badges.push(`<span class="badge uncensored">no guardrails</span>`);
    badges.push(m.downloaded
      ? `<span class="badge downloaded">✓ downloaded · ${m.downloaded_gb} GB</span>`
      : `<span class="badge will-download">will download</span>`);
    opt.innerHTML = `
      <input type="radio" name="model" value="${m.id}" ${m.default ? "checked" : ""}>
      <div style="flex:1">
        <div class="model-name">${m.name} <span class="hint">· ${m.params}</span></div>
        <div class="model-desc">${m.description}</div>
      </div>
      <div>${badges.join(" ")}</div>`;
    opt.addEventListener("change", () => {
      document.querySelectorAll(".model-option").forEach((el) => el.classList.remove("selected"));
      opt.classList.add("selected");
    });
    list.appendChild(opt);
  }
}

document.querySelectorAll(".example-btn").forEach((b) =>
  b.addEventListener("click", () => { $("topic").value = b.dataset.topic; }));

/* ---------------- User research materials ---------------- */

async function refreshMaterials() {
  const res = await fetch("/api/materials").then((r) => r.json());
  const ul = $("material-list");
  ul.innerHTML = "";
  for (const m of res.materials) {
    const li = document.createElement("li");
    li.className = "material-item";
    const name = document.createElement("span");
    name.textContent = `📄 ${m.filename} (${(m.chars / 1000).toFixed(1)}k chars)`;
    const del = document.createElement("button");
    del.className = "material-remove";
    del.title = "Remove";
    del.textContent = "✕";
    del.addEventListener("click", async () => {
      await fetch(`/api/materials/${m.id}`, { method: "DELETE" });
      refreshMaterials();
    });
    li.append(name, del);
    ul.appendChild(li);
  }
}

$("material-add").addEventListener("click", () => $("material-file").click());

$("material-file").addEventListener("change", async () => {
  const err = $("material-error");
  err.hidden = true;
  const failures = [];
  $("material-add").disabled = true;
  for (const file of $("material-file").files) {
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch("/api/materials", { method: "POST", body: form });
      if (!res.ok) {
        const detail = (await res.json()).detail || res.statusText;
        failures.push(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
    } catch (e) {
      failures.push(`${file.name}: ${e.message}`);
    }
  }
  $("material-add").disabled = false;
  $("material-file").value = "";
  if (failures.length) {
    err.textContent = failures.join(" — ");
    err.hidden = false;
  }
  refreshMaterials();
});

$("begin-btn").addEventListener("click", async () => {
  const topic = $("topic").value.trim();
  const err = $("setup-error");
  err.hidden = true;
  if (topic.length < 8) {
    err.textContent = "Please enter a debate topic (at least 8 characters).";
    err.hidden = false;
    return;
  }
  $("begin-btn").disabled = true;
  try {
    const res = await fetch("/api/debate/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        topic,
        model_id: document.querySelector('input[name="model"]:checked').value,
        pro_personality: $("pro-personality").value,
        con_personality: $("con-personality").value,
        rounds: parseInt($("rounds").value, 10),
        use_web_research: !$("materials-only").checked,
      }),
    });
    if (!res.ok) {
      const detail = (await res.json()).detail || res.statusText;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
  } catch (e) {
    err.textContent = e.message;
    err.hidden = false;
    $("begin-btn").disabled = false;
  }
});

$("reset-btn").addEventListener("click", () => location.reload());
$("stop-btn").addEventListener("click", () => fetch("/api/debate/stop", { method: "POST" }));

/* ---------------- WebSocket ---------------- */

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (msg) => handleEvent(JSON.parse(msg.data));
  ws.onclose = () => setTimeout(connect, 1500);
}
setInterval(() => { if (ws && ws.readyState === 1) ws.send("ping"); }, 25000);

function handleEvent(ev) {
  if (ev.type === "snapshot") {
    state = ev.state;
    renderAll();
    return;
  }
  if (!state) return;
  applyEvent(ev);

  switch (ev.type) {
    case "token": appendToken(ev); break;
    case "turn_start": renderTranscript(); break;
    case "turn_end": renderTranscript(); break;
    case "phase": renderChrome(); break;
    case "status": renderChrome(); break;
    case "download_progress": renderDownload(); break;
    case "research_source": renderResearch(); break;
    case "research_done": renderResearch(); break;
    case "judge_start": renderJudges(); break;
    case "judge_criterion": renderJudges(); break;
    case "judge_result": renderJudges(); break;
    case "verdict": renderVerdict(); break;
    case "error": renderChrome(); break;
  }
}

/* Mirror of backend _apply() */
function applyEvent(ev) {
  const s = state;
  switch (ev.type) {
    case "phase": s.phase = ev.phase; s.log.push(ev.message); break;
    case "status": s.log.push(ev.message); break;
    case "download_progress": s.download = ev; break;
    case "research_source": s.sources.push(ev); break;
    case "research_done": s.num_chunks = ev.num_chunks; break;
    case "turn_start":
      s.current = { speaker: ev.speaker, phase: ev.phase, round: ev.round,
                    label: ev.label, text: "" };
      break;
    case "token": if (s.current) s.current.text += ev.text; break;
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

/* ---------------- Rendering ---------------- */

const STEP_ORDER = ["model", "research", "debate", "judging", "done"];

function renderAll() {
  const active = state.phase !== "idle";
  $("setup-view").hidden = active;
  $("arena-view").hidden = !active;
  if (!active) { loadSetup(); return; }
  $("topic-banner").textContent = `“${state.topic}”`;
  renderChrome();
  renderDownload();
  renderResearch();
  renderTranscript();
  renderJudges();
  renderVerdict();
}

function renderChrome() {
  if (state.phase !== "idle" && $("arena-view").hidden) renderAll();
  const phase = state.phase;
  const idx = STEP_ORDER.indexOf(phase === "error" ? "model" : phase);
  document.querySelectorAll("#stepper li").forEach((li) => {
    const i = STEP_ORDER.indexOf(li.dataset.step);
    li.classList.toggle("active", li.dataset.step === phase ||
      (phase === "done" && li.dataset.step === "done"));
    li.classList.toggle("complete", idx > i || phase === "done");
  });

  $("status-line").textContent = state.log.length ? state.log[state.log.length - 1] : "";
  $("stop-btn").hidden = ["done", "error", "idle"].includes(phase);
  $("pdf-btn").hidden = !(phase === "done" && state.transcript.length);
  $("reset-btn").hidden = !["done", "error", "idle"].includes(phase);
  $("error-panel").hidden = phase !== "error";
  if (phase === "error") $("error-message").textContent = state.error || "Unknown error";
  if (phase === "idle") location.reload();  // debate was cancelled
}

function renderDownload() {
  const d = state.download;
  const active = d && state.phase === "model";
  $("download-panel").hidden = !active;
  if (!active) return;
  $("download-bar").style.width = `${d.pct}%`;
  const gb = (n) => (n / 1024 ** 3).toFixed(2);
  $("download-label").textContent =
    `${d.filename} — ${gb(d.done)} / ${gb(d.total)} GB (${d.pct}%)`;
}

function renderResearch() {
  const show = state.sources.length > 0 || state.phase === "research";
  $("research-panel").hidden = !show;
  if (!show) return;
  $("research-summary").textContent = state.num_chunks
    ? `${state.sources.length} sources collected · ${state.num_chunks} passages indexed for the debaters.`
    : state.phase === "research"
      ? "Collecting articles and encyclopedia entries…"
      : `${state.sources.length} sources collected.`;
  const ul = $("sources");
  ul.innerHTML = "";
  for (const s of state.sources) {
    const li = document.createElement("li");
    const label = `${s.title} (${(s.chars / 1000).toFixed(1)}k chars)`;
    if (s.url) {
      const a = document.createElement("a");
      a.href = s.url; a.target = "_blank"; a.rel = "noopener";
      a.textContent = label;
      li.appendChild(a);
    } else {
      li.textContent = `📄 ${label}`;
      const badge = document.createElement("span");
      badge.className = "badge yours";
      badge.textContent = "your material";
      li.append(" ", badge);
    }
    ul.appendChild(li);
  }
}

function turnDiv(turn, speaking) {
  const div = document.createElement("div");
  div.className = `turn ${turn.speaker}${speaking ? " speaking" : ""}`;
  const label = document.createElement("div");
  label.className = "turn-label";
  label.textContent = turn.label;
  const text = document.createElement("div");
  text.className = "turn-text";
  text.textContent = turn.text;
  div.append(label, text);
  return div;
}

function renderTranscript() {
  const show = state.transcript.length > 0 || state.current;
  $("debate-panel").hidden = !show;
  if (!show) return;
  const box = $("transcript");
  box.innerHTML = "";
  for (const t of state.transcript) box.appendChild(turnDiv(t, false));
  if (state.current) {
    const div = turnDiv(state.current, true);
    div.id = "current-turn";
    box.appendChild(div);
  }
  updateJumpBtn();
}

function appendToken() {
  const div = document.querySelector("#current-turn .turn-text");
  if (div && state.current) {
    div.textContent = state.current.text;
    updateJumpBtn();
  } else {
    renderTranscript();
  }
}

/* The page never scrolls on its own — the reader scrolls freely while text
   streams in. A floating button offers a jump to the newest text instead. */
const jumpBtn = $("jump-btn");
function updateJumpBtn() {
  const streaming = state && state.current;
  const atBottom = window.innerHeight + window.scrollY >=
    document.body.scrollHeight - 200;
  jumpBtn.hidden = !streaming || atBottom;
}
jumpBtn.addEventListener("click", () =>
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" }));
window.addEventListener("scroll", updateJumpBtn, { passive: true });

function renderJudges() {
  const started = state.judges.some((j) => j.status !== "waiting");
  $("judging-panel").hidden = !started;
  if (!started) return;
  const grid = $("judges");
  grid.innerHTML = "";
  for (const j of state.judges) {
    const card = document.createElement("div");
    card.className = "judge-card";
    let html = `<h3>${j.name}</h3>`;
    const partial = j.partial || {};
    if (j.status === "waiting") {
      html += `<div class="judge-status">waiting</div>`;
    } else {
      if (j.status === "deliberating") {
        const next = CRITERIA.find((c) => !partial[c.key]);
        html += `<div class="judge-status deliberating">${
          next ? `scoring ${next.label}` : "writing summary"}</div>`;
      } else {
        html += `<div class="judge-status">ballot in</div>`;
      }
      for (const c of CRITERIA) {
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
          continue;
        }
        html += `
          <div class="crit-row">
            <div class="crit-name"><span>${c.label}</span><span>${p} · ${q} / ${c.max}</span></div>
            <div class="crit-bars">
              <div class="crit-bar pro"><div style="width:${(100 * p) / c.max}%"></div></div>
              <div class="crit-bar con"><div style="width:${(100 * q) / c.max}%"></div></div>
            </div>
            ${rp ? `<div class="crit-reason pro-reason"><b>PRO ${p}/${c.max}</b> — ${escapeHtml(rp)}</div>` : ""}
            ${rq ? `<div class="crit-reason con-reason"><b>CON ${q}/${c.max}</b> — ${escapeHtml(rq)}</div>` : ""}
          </div>`;
      }
      if (j.ballot) {
        const b = j.ballot;
        html += `<div class="judge-total">
          <span class="pro-score">PRO ${b.totals.pro}</span>
          <span class="con-score">CON ${b.totals.con}</span></div>`;
        if (b.summary) {
          html += `<div class="judge-summary">
            <h4>The Judge's Summary</h4>${escapeHtml(b.summary)}</div>`;
        }
      }
    }
    card.innerHTML = html;
    grid.appendChild(card);
  }
}

function renderVerdict() {
  const v = state.verdict;
  $("verdict-panel").hidden = !v;
  if (!v) return;
  const banner = $("verdict-banner");
  banner.className = "";
  if (v.winner === "pro") {
    banner.classList.add("pro-wins");
    banner.textContent = "🏆 PRO wins — the motion carries!";
  } else if (v.winner === "con") {
    banner.classList.add("con-wins");
    banner.textContent = "🏆 CON wins — the motion falls!";
  } else {
    banner.textContent = "🤝 It's a tie!";
  }
  const ballotsLine = state.judges.length > 1
    ? `Ballots — PRO ${v.ballots_won.pro} · CON ${v.ballots_won.con} &nbsp;|&nbsp; `
    : "";
  $("verdict-detail").innerHTML =
    `${escapeHtml(capitalize(v.method))}.<br>` + ballotsLine +
    `Total points — PRO ${v.totals.pro} · CON ${v.totals.con} (of ${state.judges.length * 100})`;
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}
const capitalize = (s) => s.charAt(0).toUpperCase() + s.slice(1);

/* ---------------- Boot ---------------- */
loadSetup();
refreshMaterials();
connect();
