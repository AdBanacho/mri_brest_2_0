/* Configurations view: all valid configurations, every option, shipped config files.
   Data: window.CONFIG_SPACE (config-data.js, generated with the project's own validator). */
(function () {
  "use strict";

  const CS = window.CONFIG_SPACE, W = window.WFV;
  if (!CS || !W) return;
  const esc = W.esc;
  const $ = (sel, root) => (root || document).querySelector(sel);
  const fmt = (n) => Number(n).toLocaleString("en-US");
  function approx(n) {
    n = Number(n);
    if (n >= 1e9) return `≈ ${(n / 1e9).toFixed(1)} billion`;
    if (n >= 1e6) return `≈ ${(n / 1e6).toFixed(1)} million`;
    return fmt(n);
  }

  const dims = CS.dims;
  const rows = CS.rows.map((r, id) => ({
    id, v: r[0], n: r[1], parts: CS.shared[r[2]],
    inputs: CS.shared[r[3]].map((k) => CS.pathKeys[k]), notes: CS.shared[r[4]],
  }));
  const ANY = -1, NA = -2;
  const filter = dims.map(() => ANY);
  const PAGE = 25;
  let page = 0;
  let detailId = null;
  let sub = "explorer";
  let built = false;
  const optState = { query: "", group: "", stage: null, structuralOnly: false };

  // which shipped config rows map to each structural row
  const usedBy = new Map();
  CS.shipped.forEach((f) => f.rows.forEach((r, i) => {
    if (r.structural == null) return;
    if (!usedBy.has(r.structural)) usedBy.set(r.structural, []);
    usedBy.get(r.structural).push(`${f.file} row ${i + 1}`);
  }));

  function value(row, d) { return row.v[d] < 0 ? null : dims[d].values[row.v[d]]; }
  function matches(row, skip) {
    for (let d = 0; d < dims.length; d++) {
      if (d === skip || filter[d] === ANY) continue;
      if (filter[d] === NA ? row.v[d] !== -1 : row.v[d] !== filter[d]) return false;
    }
    return true;
  }
  function matching() { return rows.filter((r) => matches(r, -1)); }
  function rowLabel(row) {
    const bits = [];
    dims.forEach((d, i) => {
      const v = value(row, i);
      if (v != null && v !== d.default) bits.push(`${d.key}=${v}`);
    });
    return `#${row.id + 1}` + (bits.length ? " · " + bits.join(", ") : " · all defaults");
  }

  // ------------------------------------------------------------------ explorer
  function buildExplorer() {
    const root = $("#sub-explorer");
    root.innerHTML = `
      <div class="cx">
        <aside class="facets" aria-label="Filters">
          <div class="facet-head">
            <h2>Filter configurations</h2>
            <button type="button" class="btn" id="cx-reset">Reset</button>
          </div>
          <p class="muted small">Pick values to narrow the list. Numbers show how many valid configurations each choice leaves; greyed choices are impossible together with the current selection.</p>
          ${dims.map((d, i) => `
            <fieldset class="facet" data-dim="${i}">
              <legend><span class="facet-label">${esc(d.label)}</span> <code class="facet-key">${esc(d.key)}</code></legend>
              <p class="facet-help">${esc(d.help)}</p>
              <div class="pills">
                ${pill(i, ANY, "Any")}
                ${d.values.map((v, j) => pill(i, j, v)).join("")}
                ${pill(i, NA, "n/a")}
              </div>
            </fieldset>`).join("")}
        </aside>
        <div class="results">
          <div id="cx-summary" class="cx-summary" aria-live="polite"></div>
          <div id="cx-detail"></div>
          <div id="cx-table"></div>
          <details class="tech" id="cx-rejected">
            <summary>Why ${fmt(CS.enumerated - CS.valid)} of ${fmt(CS.enumerated)} combinations are invalid</summary>
            <div class="inner">
              <p class="muted">Each combination was checked by <code>schema.validate_combinations</code> (<code>src/mribreast/config/schema.py:214</code>), the same function the CLI runs. First error per combination:</p>
              <table class="params"><thead><tr><th>Rule message (from the code)</th><th>Combinations</th></tr></thead>
              <tbody>${CS.rejected.map((r) => `<tr><td>${esc(r.message)}</td><td>${fmt(r.count)}</td></tr>`).join("")}</tbody></table>
            </div>
          </details>
          <details class="tech">
            <summary>How this list was built</summary>
            <div class="inner">
              <ul>${CS.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>
              <p>Free numeric / text settings (not counted, any value in range): ${CS.numericFields.map((k) => `<code>${esc(k)}</code>`).join(" ")}</p>
              <p class="muted">Sources: ${CS.source.map((s) => `<code>${esc(s)}</code>`).join(" ")}. Regenerate with <code>python sync_data.py</code>.</p>
            </div>
          </details>
        </div>
      </div>`;
    function pill(d, j, label) {
      const id = `f-${d}-${j < 0 ? (j === ANY ? "any" : "na") : j}`;
      return `<span class="pill"><input type="radio" name="f-${d}" id="${id}" value="${j}" ${filter[d] === j ? "checked" : ""}>` +
        `<label for="${id}"${j === NA ? ' title="not applicable: the setting has no effect in this configuration"' : ""}>${esc(label)} <span class="cnt" data-cnt="${d}:${j}"></span></label></span>`;
    }
    root.querySelectorAll(".facet input").forEach((inp) => inp.addEventListener("change", () => {
      const d = Number(inp.name.slice(2));
      filter[d] = Number(inp.value);
      page = 0;
      updateExplorer();
    }));
    $("#cx-reset").addEventListener("click", () => {
      filter.fill(ANY); page = 0; detailId = null;
      root.querySelectorAll(".facet input").forEach((inp) => { inp.checked = Number(inp.value) === ANY; });
      updateExplorer();
    });
    root.addEventListener("click", onExplorerClick);
  }

  function updateExplorer() {
    // facet counts (each facet counted against the other filters)
    dims.forEach((d, i) => {
      const counts = new Map();
      let any = 0;
      rows.forEach((r) => {
        if (!matches(r, i)) return;
        any++;
        const k = r.v[i] === -1 ? NA : r.v[i];
        counts.set(k, (counts.get(k) || 0) + 1);
      });
      const fs = $(`.facet[data-dim="${i}"]`);
      fs.querySelectorAll(".cnt").forEach((c) => {
        const j = Number(c.getAttribute("data-cnt").split(":")[1]);
        const n = j === ANY ? any : (counts.get(j) || 0);
        c.textContent = fmt(n);
        const input = c.closest(".pill").querySelector("input");
        input.disabled = n === 0 && filter[i] !== j;
        c.closest(".pill").hidden = j === NA && n === 0 && filter[i] !== NA;
      });
      const naOnly = counts.size === 1 && counts.has(NA);
      fs.classList.toggle("na-only", naOnly);
    });
    const list = matching();
    const variants = list.reduce((s, r) => s + r.n, 0);
    const fixed = [];
    dims.forEach((d, i) => {
      if (!list.length) return;
      const first = list[0].v[i];
      if (list.every((r) => r.v[i] === first)) fixed.push(`${d.key} = ${first === -1 ? "n/a" : d.values[first]}`);
    });
    $("#cx-summary").innerHTML = `
      <div class="big-numbers">
        <div><span class="bn">${fmt(list.length)}</span><span class="bl">of ${fmt(CS.valid)} valid pipeline configurations match</span></div>
        <div><span class="bn">${approx(variants)}</span><span class="bl">configurations once the independent options are included (model type, feature groups, optimiser…)</span></div>
        <div><span class="bn">&infin;</span><span class="bl">with free numeric settings (${CS.numericFields.length} fields such as epochs, learning rate, sizes)</span></div>
      </div>
      ${fixed.length && list.length < CS.valid ? `<p class="muted small"><b>Same in every match:</b> ${fixed.map(esc).join(" · ")}</p>` : ""}`;
    const pages = Math.max(1, Math.ceil(list.length / PAGE));
    page = Math.min(page, pages - 1);
    const varying = dims.map((d, i) => i).filter((i) => list.some((r) => r.v[i] !== list[0].v[i]));
    const shown = list.slice(page * PAGE, page * PAGE + PAGE);
    $("#cx-table").innerHTML = list.length ? `
      <div class="table-wrap"><table class="cx-table">
        <thead><tr><th>#</th>${varying.map((i) => `<th title="${esc(dims[i].help)}">${esc(dims[i].label)}</th>`).join("")}<th>Variants</th><th>Stages off</th></tr></thead>
        <tbody>${shown.map((r) => `<tr class="${r.id === detailId ? "sel" : ""}">
          <td><button type="button" class="linkbtn" data-row="${r.id}" aria-label="Open configuration ${r.id + 1}">#${r.id + 1}</button></td>
          ${varying.map((i) => { const v = value(r, i); return `<td>${v == null ? '<span class="muted">n/a</span>' : esc(v)}</td>`; }).join("")}
          <td class="num">${fmt(r.n)}</td>
          <td class="num">${Object.values(r.notes).filter((x) => x.startsWith("off:")).length}</td></tr>`).join("")}
        </tbody></table></div>
      <div class="pager">
        <button type="button" class="btn" data-page="-1" ${page === 0 ? "disabled" : ""}>&larr; Previous</button>
        <span class="muted">Page ${page + 1} of ${pages} · rows ${fmt(page * PAGE + 1)}–${fmt(Math.min(list.length, page * PAGE + PAGE))}</span>
        <button type="button" class="btn" data-page="1" ${page >= pages - 1 ? "disabled" : ""}>Next &rarr;</button>
      </div>` : `<p class="muted">No valid configuration matches these filters.</p>`;
    renderDetail();
  }

  function onExplorerClick(ev) {
    const r = ev.target.closest("[data-row]");
    if (r) { openDetail(Number(r.getAttribute("data-row"))); return; }
    const p = ev.target.closest("[data-page]");
    if (p) { page += Number(p.getAttribute("data-page")); updateExplorer(); $("#cx-table").scrollIntoView({ block: "nearest" }); return; }
    const s = ev.target.closest("[data-stage]");
    if (s) { W.showStage(s.getAttribute("data-stage")); return; }
    const a = ev.target.closest("[data-action]");
    if (!a) return;
    const row = rows[detailId];
    if (a.getAttribute("data-action") === "close") { detailId = null; updateExplorer(); }
    if (a.getAttribute("data-action") === "pipeline") showInPipeline(row);
    if (a.getAttribute("data-action") === "copy") copyText(a.getAttribute("data-copy"), a);
  }

  function openDetail(id) {
    detailId = id;
    updateExplorer();
    const d = $("#cx-detail");
    d.scrollIntoView({ block: "start", behavior: "smooth" });
    const h = $("h3", d);
    if (h) h.focus();
  }

  function cliArgs(row) {
    const args = [];
    dims.forEach((d, i) => {
      const v = value(row, i);
      if (v != null && v !== d.default) args.push([d.key, v]);
    });
    return args;
  }

  function renderDetail() {
    const box = $("#cx-detail");
    if (detailId == null) { box.innerHTML = ""; return; }
    const row = rows[detailId];
    const args = cliArgs(row);
    const cmd = "python -m mribreast train" + args.map(([k, v]) => ` --${k} ${v}`).join("") + " ...";
    const conf = args.length ? "declare -A TRAIN=(\n" + args.map(([k, v]) => `  [${k}]=${/\s/.test(v) ? `'${v}'` : v}`).join("\n") + "\n  ...\n)" : "# all structural settings at their defaults";
    const notes = Object.entries(row.notes);
    const off = notes.filter(([, v]) => v.startsWith("off:"));
    const changed = notes.filter(([, v]) => !v.startsWith("off:"));
    const stageName = (id) => (W.stages.get(id) || { name: id }).name;
    const used = usedBy.get(row.id) || [];
    box.innerHTML = `
      <article class="cx-detail" aria-labelledby="cx-detail-title">
        <div class="cx-detail-head">
          <h3 id="cx-detail-title" tabindex="-1">Configuration #${row.id + 1}</h3>
          <div>
            <button type="button" class="btn primary" data-action="pipeline">Show in pipeline</button>
            <button type="button" class="btn" data-action="close">Close</button>
          </div>
        </div>
        <div class="cx-grid">
          <section>
            <h4>Settings</h4>
            <dl class="kv">${dims.map((d, i) => { const v = value(row, i); return `<dt title="${esc(d.help)}">${esc(d.label)}</dt><dd>${v == null ? '<span class="muted">n/a — no effect here</span>' : `<code>${esc(v)}</code>${v === d.default ? ' <span class="muted">(default)</span>' : ""}`}</dd>`; }).join("")}</dl>
          </section>
          <section>
            <h4>What changes in the pipeline</h4>
            ${off.length ? `<p class="small"><b>Not used:</b></p><ul class="small">${off.map(([id, v]) => `<li><button type="button" class="linkbtn" data-stage="${esc(id)}">${esc(stageName(id))}</button> <span class="muted">${esc(v.slice(5))}</span></li>`).join("")}</ul>` : `<p class="small">Every core stage runs.</p>`}
            ${changed.length ? `<p class="small"><b>Behaves differently:</b></p><ul class="small">${changed.map(([id, v]) => `<li><button type="button" class="linkbtn" data-stage="${esc(id)}">${esc(stageName(id))}</button> <span class="muted">${esc(v.replace(/^(changed|on): /, ""))}</span></li>`).join("")}</ul>` : ""}
            <h4>Input files needed</h4>
            <p class="small">${row.inputs.map((k) => `<code>${esc(k)}</code>`).join(" ")}</p>
            <p class="muted small">From <code>cli.required_inputs</code> with feature_groups=clinical; kinetic / morphology / heterogeneity add <code>imaging_features_file</code>, density adds <code>density_file</code>, mricore adds <code>mricore_features_file</code>.</p>
          </section>
          <section>
            <h4>Independent options: ${fmt(row.n)} variants</h4>
            <ul class="parts">${row.parts.map((p) => `<li><span><code>${esc(p.label)}</code>${p.note ? `<span class="muted small part-note">${esc(p.note)}</span>` : ""}</span><b class="num">&times;${fmt(p.n)}</b></li>`).join("")}</ul>
            <p class="muted small">Plus free numeric settings (not counted).</p>
            ${used.length ? `<h4>Used by shipped configs</h4><ul class="small">${used.map((u) => `<li><code>${esc(u)}</code></li>`).join("")}</ul>` : ""}
          </section>
        </div>
        <h4>Run it</h4>
        <div class="code-row"><pre class="code">${esc(cmd)}</pre><button type="button" class="btn icon" data-action="copy" data-copy="${esc(cmd)}" aria-label="Copy command">Copy</button></div>
        <div class="code-row"><pre class="code">${esc(conf)}</pre><button type="button" class="btn icon" data-action="copy" data-copy="${esc(conf)}" aria-label="Copy experiment.conf snippet">Copy</button></div>
        <p class="muted small">Only settings that differ from the defaults are listed; the path keys above go in <code>PATHS</code>.</p>
      </article>`;
  }

  function showInPipeline(row) {
    W.setOverlay({ label: rowLabel(row), notes: row.notes, onBack: () => { setSub("explorer"); openDetail(row.id); } });
    W.showView("pipeline");
  }

  function copyText(text, btn) {
    const done = () => { const t = btn.textContent; btn.textContent = "Copied"; setTimeout(() => { btn.textContent = t; }, 1200); };
    if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).then(done, () => fallback());
    else fallback();
    function fallback() {
      const ta = document.createElement("textarea");
      ta.value = text; ta.setAttribute("readonly", ""); ta.style.position = "fixed"; ta.style.opacity = "0";
      document.body.appendChild(ta); ta.select();
      try { document.execCommand("copy"); done(); } catch (e) { /* ignore */ }
      ta.remove();
    }
  }

  // ------------------------------------------------------------------ every option
  const groups = [];
  CS.fields.forEach((f) => { if (!groups.includes(f.group)) groups.push(f.group); });

  function buildOptions() {
    const root = $("#sub-options");
    root.innerHTML = `
      <div class="opt-controls">
        <label class="field">Search <input type="search" id="opt-q" placeholder="key or description…" autocomplete="off"></label>
        <label class="field">Group
          <select id="opt-group"><option value="">All groups</option>${groups.map((g) => `<option>${esc(g)}</option>`).join("")}</select>
        </label>
        <label class="check"><input type="checkbox" id="opt-struct"> Only settings that change the pipeline's shape</label>
        <span id="opt-stage"></span>
        <span id="opt-count" class="muted small" aria-live="polite"></span>
      </div>
      <p class="muted small">All ${CS.fields.length} keys of <code>src/mribreast/config/fields.tsv</code>. Every key is accepted as <code>--key value</code> by <code>python -m mribreast</code> and as an entry in <code>experiment.conf</code>. <span class="badge id">train_id</span> = part of the experiment identity (changing it means a new training run); <span class="badge st">shape</span> = enumerated in “All valid configurations”.</p>
      <div id="opt-list"></div>`;
    $("#opt-q").addEventListener("input", (e) => { optState.query = e.target.value.trim().toLowerCase(); renderOptions(); });
    $("#opt-group").addEventListener("change", (e) => { optState.group = e.target.value; renderOptions(); });
    $("#opt-struct").addEventListener("change", (e) => { optState.structuralOnly = e.target.checked; renderOptions(); });
    root.addEventListener("click", (ev) => {
      const s = ev.target.closest("[data-stage]");
      if (s) { W.showStage(s.getAttribute("data-stage")); return; }
      if (ev.target.closest("#opt-stage-clear")) { optState.stage = null; renderOptions(); }
    });
  }

  function renderOptions() {
    const q = optState.query;
    const list = CS.fields.filter((f) => (!optState.group || f.group === optState.group)
      && (!optState.stage || f.stage === optState.stage)
      && (!optState.structuralOnly || f.structural)
      && (!q || f.key.toLowerCase().includes(q) || (f.help || "").toLowerCase().includes(q) || f.choices.some((c) => c.toLowerCase().includes(q))));
    const st = optState.stage && W.stages.get(optState.stage);
    $("#opt-stage").innerHTML = st ? `<span class="chip">Stage: ${esc(st.name)} <button type="button" class="linkbtn" id="opt-stage-clear" aria-label="Remove stage filter">&times;</button></span>` : "";
    $("#opt-count").textContent = `${list.length} of ${CS.fields.length} options`;
    const byGroup = groups.map((g) => [g, list.filter((f) => f.group === g)]).filter(([, l]) => l.length);
    $("#opt-list").innerHTML = byGroup.length ? byGroup.map(([g, l]) => {
      const stage = W.stages.get(l[0].stage);
      return `<section class="opt-group">
        <h3>${esc(g)} <span class="muted small">(${l.length})</span> ${stage ? `<button type="button" class="linkbtn" data-stage="${esc(stage.id)}">configures: ${esc(stage.name)}</button>` : ""}</h3>
        <div class="table-wrap"><table class="params opt-table">
          <thead><tr><th>Key</th><th>Type</th><th>Default</th><th>Possible values</th><th>Meaning</th></tr></thead>
          <tbody>${l.map((f) => `<tr>
            <td><code>${esc(f.key)}</code><br>${f.identity ? '<span class="badge id">train_id</span>' : ""}${f.structural ? ' <span class="badge st">shape</span>' : ""}</td>
            <td>${esc(f.type)}</td>
            <td>${f.default === "" ? '<span class="muted">(empty)</span>' : `<code>${esc(f.default)}</code>`}</td>
            <td>${choiceText(f)}</td>
            <td>${esc(f.help || "")}</td></tr>`).join("")}</tbody></table></div></section>`;
    }).join("") : `<p class="muted">No option matches.</p>`;
  }
  function choiceText(f) {
    if (f.choices.length) {
      const multi = f.type === "csv" ? '<span class="muted small">one or more, comma-separated:</span><br>' : "";
      return multi + f.choices.map((c) => `<span class="chip">${esc(c)}</span>`).join(" ");
    }
    if (f.type === "bool") return '<span class="chip">true</span> <span class="chip">false</span>';
    if (f.type === "path") return '<span class="muted">a file or directory path</span>';
    if (f.type === "json") return '<span class="muted">a JSON object</span>';
    return `<span class="muted">any ${esc(f.type)} value (rules in schema.py)</span>`;
  }

  // ------------------------------------------------------------------ shipped configs
  function buildShipped() {
    const root = $("#sub-shipped");
    root.innerHTML = `
      <p class="muted small">Each file is read the way the controller reads it: <code>TRAIN</code> fixed values plus every combination of <code>TRAIN_GRID</code> (keys sorted, last key varies fastest), and validation tasks = training rows &times; <code>VALIDATE_GRID</code>. Every row was resolved with <code>schema.resolve</code>.</p>
      ${CS.shipped.map((f, fi) => `
        <section class="shipped">
          <h3><code>${esc(f.file)}</code></h3>
          <p class="small">${fmt(f.trainTasks)} training task(s) · ${fmt(f.validateTasks)} validation task(s) · <b>${new Set(f.rows.map((r) => r.structural)).size} distinct pipeline configuration(s)</b>${Object.keys(f.validateGrid).length ? ` · validate grid: ${Object.entries(f.validateGrid).map(([k, v]) => `<code>${esc(k)}</code> = ${v.map(esc).join(" | ")}`).join(", ")}` : ""}</p>
          <p class="small"><b>Fixed:</b> ${Object.entries(f.fixed).map(([k, v]) => `<code>${esc(k)}=${esc(v)}</code>`).join(" ") || '<span class="muted">defaults</span>'}</p>
          <div class="table-wrap"><table class="params">
            <thead><tr><th>Row</th>${f.axes.map((a) => `<th><code>${esc(a)}</code></th>`).join("")}<th>Valid?</th><th>Configuration</th></tr></thead>
            <tbody>${f.rows.map((r, i) => `<tr>
              <td class="num">${i + 1}</td>
              ${f.axes.map((a) => `<td>${esc(r.values[a])}</td>`).join("")}
              <td>${r.error ? `<span class="bad">✗ ${esc(r.error)}</span>` : '<span class="ok">✓</span>'}</td>
              <td>${r.structural != null ? `<button type="button" class="linkbtn" data-open="${r.structural}">#${r.structural + 1}</button>` : "—"}</td></tr>`).join("")}</tbody>
          </table></div>
        </section>`).join("")}`;
    root.addEventListener("click", (ev) => {
      const o = ev.target.closest("[data-open]");
      if (o) { setSub("explorer"); openDetail(Number(o.getAttribute("data-open"))); }
    });
  }

  // ------------------------------------------------------------------ sub tabs
  function setSub(name) {
    sub = name;
    document.querySelectorAll(".subtab").forEach((t) => {
      const on = t.getAttribute("data-sub") === name;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
    });
    ["explorer", "options", "shipped"].forEach((n) => { $(`#sub-${n}`).hidden = n !== name; });
    if (name === "options") renderOptions();
  }
  const tabs = Array.from(document.querySelectorAll(".subtab"));
  tabs.forEach((t, i) => {
    t.addEventListener("click", () => setSub(t.getAttribute("data-sub")));
    t.addEventListener("keydown", (ev) => {
      const d = ev.key === "ArrowRight" ? 1 : ev.key === "ArrowLeft" ? -1 : 0;
      if (!d) return;
      ev.preventDefault();
      const n = tabs[(i + d + tabs.length) % tabs.length];
      setSub(n.getAttribute("data-sub")); n.focus();
    });
  });

  function ensureBuilt() {
    if (built) return;
    built = true;
    buildExplorer(); buildOptions(); buildShipped();
    updateExplorer();
    const h = new URLSearchParams(location.hash.replace(/^#/, ""));
    if (["options", "shipped"].includes(h.get("sub"))) setSub(h.get("sub"));
    if (h.get("config")) openDetail(Math.max(0, Math.min(rows.length - 1, Number(h.get("config")) - 1)));
  }
  W.onShowConfigs = ensureBuilt;
  if (!document.getElementById("config-view").hidden) ensureBuilt(); // view was opened before this script ran
  const startHash = new URLSearchParams(location.hash.replace(/^#/, ""));
  if (startHash.get("overlay")) {                       // #overlay=<configuration #>: show it on the diagram
    const id = Number(startHash.get("overlay")) - 1;
    if (rows[id]) showInPipeline(rows[id]);
  }
  W.onShowOptions = (stageId) => { ensureBuilt(); optState.stage = stageId; setSub("options"); };
})();
