/* Workflow viewer. Vanilla JS, no dependencies. Data: window.WORKFLOW (from workflow-data.js). */
(function () {
  "use strict";

  const WF = window.WORKFLOW;
  const SVG_NS = "http://www.w3.org/2000/svg";
  const $ = (sel) => document.querySelector(sel);

  if (!WF || !Array.isArray(WF.stages)) {
    document.body.innerHTML = "<p style='padding:2rem'>workflow-data.js is missing or invalid. Run <code>python sync_data.py</code> in workflow-viewer/.</p>";
    return;
  }

  // ------------------------------------------------------------------ geometry
  const G = { padX: 24, pitch: 210, nodeW: 152, nodeH: 66, rowPitch: 100, top: 84, bottomPad: 26 };
  const maxCol = Math.max(...WF.stages.map((s) => s.col));
  const maxRow = Math.max(...WF.stages.map((s) => s.row));
  const W = G.padX * 2 + maxCol * G.pitch + G.nodeW;
  const H = G.top + maxRow * G.rowPitch + G.nodeH + G.bottomPad;

  const stageById = new Map(WF.stages.map((s) => [s.id, s]));
  const laneById = new Map(WF.lanes.map((l) => [l.id, l]));
  const order = WF.playOrder.slice();
  const orderIndex = new Map(order.map((id, i) => [id, i]));
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function box(s) {
    const x = G.padX + s.col * G.pitch;
    const y = G.top + s.row * G.rowPitch;
    return { x, y, w: G.nodeW, h: G.nodeH, r: x + G.nodeW, b: y + G.nodeH, cx: x + G.nodeW / 2, cy: y + G.nodeH / 2 };
  }
  WF.stages.forEach((s) => { s._box = box(s); });

  // ------------------------------------------------------------------ helpers
  function esc(v) {
    return String(v == null ? "" : v).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function el(name, attrs, parent) {
    const n = document.createElementNS(SVG_NS, name);
    for (const k in attrs || {}) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function laneColor(laneId) { return `var(--lane-${laneId})`; }
  function storageGet(k) { try { return window.localStorage.getItem(k); } catch (e) { return null; } }
  function storageSet(k, v) { try { window.localStorage.setItem(k, v); } catch (e) { /* ignore */ } }

  // ------------------------------------------------------------------ edge routing
  function routeType(e) {
    const a = stageById.get(e.from), b = stageById.get(e.to);
    if (e.kind === "loop") return "loop";
    if (a.col === b.col && b.row > a.row) return "down";
    if (b.col > a.col + 1) return "over";
    return "curve";
  }

  // Assign port positions so several edges on one side of a node do not overlap.
  function assignPorts() {
    const sides = new Map(); // key nodeId|side -> [{edge, end, other}]
    const add = (id, side, entry) => {
      const k = id + "|" + side;
      if (!sides.has(k)) sides.set(k, []);
      sides.get(k).push(entry);
    };
    WF.edges.forEach((e) => {
      const t = routeType(e);
      e._route = t;
      const a = stageById.get(e.from), b = stageById.get(e.to);
      if (t === "down") { add(a.id, "bottom", { e, end: "from", other: b._box.cx }); add(b.id, "top", { e, end: "to", other: a._box.cx }); }
      if (t === "curve") { add(a.id, "right", { e, end: "from", other: b._box.cy }); add(b.id, "left", { e, end: "to", other: a._box.cy }); }
      if (t === "over") { add(a.id, "right", { e, end: "from", other: b._box.cy }); add(b.id, "top", { e, end: "to", other: a._box.cx }); }
      if (t === "loop") { add(a.id, "right", { e, end: "from", loop: true }); add(b.id, "right", { e, end: "to", loop: true }); }
    });
    sides.forEach((list, key) => {
      const [id, side] = key.split("|");
      const bx = stageById.get(id)._box;
      const normal = list.filter((x) => !x.loop).sort((p, q) => p.other - q.other);
      const hasLoop = list.some((x) => x.loop);
      const lo = 0.3, hi = hasLoop ? 0.58 : 0.7;
      normal.forEach((x, i) => {
        const t = normal.length === 1 ? (hasLoop ? 0.45 : 0.5) : lo + (hi - lo) * i / (normal.length - 1);
        x.e["_" + x.end] = portPoint(bx, side, t);
      });
      list.filter((x) => x.loop).forEach((x) => {
        x.e["_" + x.end] = portPoint(bx, side, x.end === "from" ? 0.8 : 0.22);
      });
    });
  }
  function portPoint(bx, side, t) {
    if (side === "right") return { x: bx.r, y: bx.y + bx.h * t };
    if (side === "left") return { x: bx.x, y: bx.y + bx.h * t };
    if (side === "top") return { x: bx.x + bx.w * t, y: bx.y };
    return { x: bx.x + bx.w * t, y: bx.b };
  }

  function roundedPolyline(pts, r) {
    let d = `M ${pts[0].x} ${pts[0].y}`;
    for (let i = 1; i < pts.length - 1; i++) {
      const p0 = pts[i - 1], p = pts[i], p1 = pts[i + 1];
      const d0 = Math.hypot(p.x - p0.x, p.y - p0.y), d1 = Math.hypot(p1.x - p.x, p1.y - p.y);
      const rr = Math.min(r, d0 / 2, d1 / 2);
      const a = { x: p.x - (p.x - p0.x) / d0 * rr, y: p.y - (p.y - p0.y) / d0 * rr };
      const b = { x: p.x + (p1.x - p.x) / d1 * rr, y: p.y + (p1.y - p.y) / d1 * rr };
      d += ` L ${a.x} ${a.y} Q ${p.x} ${p.y} ${b.x} ${b.y}`;
    }
    const last = pts[pts.length - 1];
    return d + ` L ${last.x} ${last.y}`;
  }

  function edgePath(e) {
    const p1 = e._from, p2 = e._to;
    const a = stageById.get(e.from);
    switch (e._route) {
      case "down": {
        if (Math.abs(p1.x - p2.x) < 1) return `M ${p1.x} ${p1.y} L ${p2.x} ${p2.y - 2}`;
        const my = (p1.y + p2.y) / 2;
        return `M ${p1.x} ${p1.y} C ${p1.x} ${my}, ${p2.x} ${my}, ${p2.x} ${p2.y - 2}`;
      }
      case "curve": {
        const dx = Math.max(40, (p2.x - p1.x) / 2);
        return `M ${p1.x} ${p1.y} C ${p1.x + dx} ${p1.y}, ${p2.x - dx} ${p2.y}, ${p2.x - 2} ${p2.y}`;
      }
      case "over": {
        const gx = a._box.r + (G.pitch - G.nodeW) / 2;
        const ty = G.top - 16;
        return roundedPolyline([p1, { x: gx, y: p1.y }, { x: gx, y: ty }, { x: p2.x, y: ty }, { x: p2.x, y: p2.y - 2 }], 10);
      }
      case "loop": {
        const bx = a._box.r + 22;
        return roundedPolyline([p1, { x: bx, y: p1.y }, { x: bx, y: p2.y }, { x: p2.x + 2, y: p2.y }], 10);
      }
    }
    return "";
  }

  // ------------------------------------------------------------------ render diagram
  const svg = $("#diagram");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const defs = el("defs", {}, svg);
  [["arrow", "var(--edge)"], ["arrow-loop", "var(--loop)"], ["arrow-active", "var(--edge-active)"]].forEach(([id, fill]) => {
    const m = el("marker", { id, viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse" }, defs);
    el("path", { d: "M 0 0 L 10 5 L 0 10 z", style: `fill:${fill}` }, m);
  });

  const gBands = el("g", { class: "bands" }, svg);
  const gEdges = el("g", { class: "edges" }, svg);
  const gLabels = el("g", { class: "edge-labels" }, svg);
  const gNodes = el("g", { class: "nodes" }, svg);
  const gTokens = el("g", { class: "tokens", "aria-hidden": "true" }, svg);

  WF.lanes.forEach((l) => {
    const x = G.padX + l.col * G.pitch - 12;
    const w = (l.span - 1) * G.pitch + G.nodeW + 24;
    el("rect", { class: "lane-band", x, y: 4, width: w, height: H - 8, rx: 12, style: `fill:${laneColor(l.id)}` }, gBands);
    const t = el("text", { class: "lane-title", x: x + 12, y: 28, style: `fill:${laneColor(l.id)}` }, gBands);
    t.textContent = l.title;
    const st = el("text", { class: "lane-sub", x: x + 12, y: 45 }, gBands);
    st.textContent = l.subtitle || "";
  });

  assignPorts();
  const edgeEls = new Map(); // edge index -> {g, path, label}
  WF.edges.forEach((e, i) => {
    const g = el("g", { class: `edge kind-${e.kind}`, "data-from": e.from, "data-to": e.to }, gEdges);
    const path = el("path", { class: "line", d: edgePath(e), "marker-end": e.kind === "loop" ? "url(#arrow-loop)" : "url(#arrow)" }, g);
    const title = el("title", {}, g);
    title.textContent = `${stageById.get(e.from).name} → ${stageById.get(e.to).name}: ${e.label}`;
    const lg = el("g", { class: `edge kind-${e.kind}`, "data-edge": i }, gLabels);
    edgeEls.set(i, { g, lg, path, e });
  });

  WF.stages.slice().sort((a, b) => orderIndex.get(a.id) - orderIndex.get(b.id)).forEach((s) => {
    const b = s._box;
    const label = `${s.name}. ${s.kind === "core" ? "" : s.kind + " step. "}${(s.uncertain || []).length ? "Has unverified points. " : ""}Step ${orderIndex.get(s.id) + 1} of ${order.length}.`;
    const g = el("g", { class: `node kind-${s.kind}`, tabindex: "0", role: "button", "data-id": s.id, "aria-label": label }, gNodes);
    el("rect", { class: "focus-ring", x: b.x - 5, y: b.y - 5, width: b.w + 10, height: b.h + 10, rx: 13 }, g);
    el("rect", { class: "box", x: b.x, y: b.y, width: b.w, height: b.h, rx: 9, style: `stroke:${laneColor(s.lane)}` }, g);
    // number badge
    el("circle", { class: "num-bg", cx: b.x + 2, cy: b.y + 2, r: 10 }, g);
    const num = el("text", { class: "num", x: b.x + 2, y: b.y + 6.5, "text-anchor": "middle" }, g);
    num.textContent = String(orderIndex.get(s.id) + 1);
    // name (wrapped to 2 lines)
    const lines = wrap(s.name, 17);
    const startY = b.y + (lines.length > 1 ? 28 : 36);
    lines.forEach((ln, i) => {
      const t = el("text", { class: "name", x: b.x + 12, y: startY + i * 17 }, g);
      t.textContent = ln;
    });
    el("circle", { class: "chg-dot", cx: b.r - 10, cy: b.cy, r: 5 }, g);
    if (s.kind !== "core") {
      const tag = el("text", { class: "tag", x: b.r - 8, y: b.b - 7, "text-anchor": "end" }, g);
      tag.textContent = s.kind;
    }
    if ((s.uncertain || []).length) {
      el("circle", { class: "q-bg", cx: b.r - 2, cy: b.y + 2, r: 9 }, g);
      const q = el("text", { class: "q", x: b.r - 2, y: b.y + 6, "text-anchor": "middle" }, g);
      q.textContent = "?";
      const tt = el("title", {}, g);
      tt.textContent = "Uncertain / unverified: " + s.uncertain.join(" ");
    }
    if (s.loop) {
      const lp = el("text", { class: "tag", x: b.x + 12, y: b.b - 7 }, g);
      lp.textContent = "↻ repeats";
    }
    g.addEventListener("click", () => select(s.id, { focusPanel: false }));
    g.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); select(s.id, { focusPanel: false }); }
      const dir = { ArrowLeft: "l", ArrowRight: "r", ArrowUp: "u", ArrowDown: "d" }[ev.key];
      if (dir) { ev.preventDefault(); moveFocus(s.id, dir); }
    });
    g.addEventListener("focus", () => { focusedId = s.id; });
  });

  function wrap(text, max) {
    const words = text.split(" ");
    const lines = [""];
    words.forEach((w) => {
      const cur = lines[lines.length - 1];
      if ((cur + " " + w).trim().length > max && cur) lines.push(w);
      else lines[lines.length - 1] = (cur + " " + w).trim();
    });
    return lines.slice(0, 2);
  }

  // edge labels with simple collision avoidance against nodes and other labels
  function layoutLabels() {
    const taken = WF.stages.map((s) => ({ x: s._box.x - 2, y: s._box.y - 2, w: s._box.w + 4, h: s._box.h + 4 }));
    const hit = (r) => taken.some((t) => r.x < t.x + t.w && r.x + r.w > t.x && r.y < t.y + t.h && r.y + r.h > t.y);
    edgeEls.forEach(({ lg, path, e }) => {
      lg.textContent = "";
      const bg = el("rect", { class: "lbl-bg", rx: 4 }, lg);
      const txt = el("text", { class: "lbl", "text-anchor": "middle", "dominant-baseline": "central" }, lg);
      txt.textContent = e.label;
      const len = path.getTotalLength();
      const tw = txt.getComputedTextLength() + 8, th = 15;
      const isLoop = e._route === "loop";
      const tries = isLoop ? [0.5] : [0.5, 0.38, 0.62, 0.28, 0.72, 0.2, 0.8];
      let chosen = null;
      for (const t of tries) {
        const p = path.getPointAtLength(len * t);
        const r = isLoop ? { x: p.x - th / 2, y: p.y - tw / 2, w: th, h: tw } : { x: p.x - tw / 2, y: p.y - th / 2, w: tw, h: th };
        if (!hit(r)) { chosen = { p, r }; break; }
      }
      // No free spot: keep the label at the midpoint but only show it while the edge is active (Play mode).
      lg.classList.toggle("lbl-hidden", !chosen);
      if (!chosen) {
        const p = path.getPointAtLength(len * 0.5);
        chosen = { p, r: { x: p.x - tw / 2, y: p.y - th / 2, w: tw, h: th } };
      } else taken.push(chosen.r);
      const { p } = chosen;
      txt.setAttribute("x", p.x); txt.setAttribute("y", p.y);
      bg.setAttribute("x", p.x - tw / 2); bg.setAttribute("y", p.y - th / 2);
      bg.setAttribute("width", tw); bg.setAttribute("height", th);
      if (isLoop) lg.setAttribute("transform", `rotate(-90 ${p.x} ${p.y})`);
    });
  }
  layoutLabels();

  // ------------------------------------------------------------------ zoom
  const wrapEl = $("#diagram-wrap");
  let zoomMode = "fit";
  let scale = 1;
  function applyZoom() {
    if (zoomMode === "fit") {
      const avail = wrapEl.clientWidth - 24;
      scale = Math.max(0.45, Math.min(1.5, avail / W));
    }
    svg.style.width = Math.round(W * scale) + "px";
    svg.style.height = Math.round(H * scale) + "px";
  }
  function zoomBy(f) { zoomMode = "manual"; scale = Math.max(0.4, Math.min(2.5, scale * f)); applyZoom(); }
  $("#btn-zoom-in").addEventListener("click", () => zoomBy(1.15));
  $("#btn-zoom-out").addEventListener("click", () => zoomBy(1 / 1.15));
  $("#btn-zoom-fit").addEventListener("click", () => { zoomMode = "fit"; applyZoom(); });
  window.addEventListener("resize", () => { if (zoomMode === "fit") applyZoom(); });
  applyZoom();

  $("#chk-labels").addEventListener("change", (ev) => svg.classList.toggle("hide-labels", !ev.target.checked));

  // ------------------------------------------------------------------ keyboard focus movement
  let focusedId = null;
  function nodeEl(id) { return gNodes.querySelector(`[data-id="${id}"]`); }
  function moveFocus(fromId, dir) {
    const a = stageById.get(fromId)._box;
    let best = null, bestScore = Infinity;
    WF.stages.forEach((s) => {
      if (s.id === fromId) return;
      const b = s._box;
      const dx = b.cx - a.cx, dy = b.cy - a.cy;
      const primary = { l: -dx, r: dx, u: -dy, d: dy }[dir];
      const secondary = dir === "l" || dir === "r" ? Math.abs(dy) : Math.abs(dx);
      if (primary <= 1) return;
      const score = primary + secondary * 2.2;
      if (score < bestScore) { bestScore = score; best = s.id; }
    });
    if (best) nodeEl(best).focus();
  }

  // ------------------------------------------------------------------ panel
  const panel = $("#panel");
  const panelBody = $("#panel-body");
  let selectedId = null;
  let techOpen = storageGet("wfv-tech") === "1";

  function stageLink(id) {
    const s = stageById.get(id);
    return `<button type="button" class="linkbtn" data-goto="${esc(id)}">${esc(orderIndex.get(id) + 1)}. ${esc(s.name)}</button>`;
  }
  function srcHref(file) { return "../" + file.split("/").map(encodeURIComponent).join("/"); }

  // Configuration overlay (set from the Configurations view): stage id -> "off: ..." | "changed: ..." | "on: ...".
  let overlay = null;
  function isOff(id) { return !!(overlay && overlay.notes[id] && overlay.notes[id].startsWith("off:")); }
  function applyOverlayClasses() {
    gNodes.querySelectorAll(".node").forEach((n) => {
      const note = overlay && overlay.notes[n.getAttribute("data-id")];
      n.classList.toggle("off", !!note && note.startsWith("off:"));
      n.classList.toggle("changed", !!note && !note.startsWith("off:"));
    });
    edgeEls.forEach((x) => {
      const off = isOff(x.e.from) || isOff(x.e.to);
      x.g.classList.toggle("off", off); x.lg.classList.toggle("off", off);
    });
    const banner = $("#overlay-banner");
    if (overlay) {
      const offCount = Object.values(overlay.notes).filter((v) => v.startsWith("off:")).length;
      banner.hidden = false;
      banner.innerHTML = `<span><b>Showing configuration:</b> ${esc(overlay.label)} <span class="muted">· ${offCount} stage(s) not used (faded); stages that behave differently are marked with a dot</span></span>
        <span class="ov-actions"><button type="button" class="btn" id="ov-back">Back to configuration</button> <button type="button" class="btn" id="ov-clear">Clear</button></span>`;
      $("#ov-clear").addEventListener("click", () => setOverlay(null));
      $("#ov-back").addEventListener("click", () => { const back = overlay && overlay.onBack; showView("configs"); if (back) back(); });
    } else {
      banner.hidden = true; banner.textContent = "";
    }
  }
  function setOverlay(o) {
    overlay = o;
    applyOverlayClasses();
    select(selectedId);
  }
  const fieldCount = new Map();
  (window.CONFIG_SPACE ? window.CONFIG_SPACE.fields : []).forEach((f) => fieldCount.set(f.stage, (fieldCount.get(f.stage) || 0) + 1));
  const NOTE_PREFIX = { off: "not used — ", on: "used — ", changed: "behaves differently — " };
  function overlayText(note) { return note.replace(/^(off|on|changed): /, (m, k) => NOTE_PREFIX[k]); }

  function renderStage(s) {
    const lane = laneById.get(s.lane);
    const incoming = WF.edges.filter((e) => e.to === s.id);
    const outgoing = WF.edges.filter((e) => e.from === s.id);
    const idx = orderIndex.get(s.id);
    const flowList = (list, end) => list.length ? `<ul>${list.map((e) => `<li>${stageLink(e[end])} <span class="muted">— ${esc(e.label)}${e.kind !== "data" ? ` (${esc(e.kind)})` : ""}</span></li>`).join("")}</ul>` : `<p class="muted">—</p>`;
    const io = (list) => list && list.length ? `<dl class="io">${list.map((x) => `<dt>${esc(x.name)}</dt><dd>${esc(x.description || "")}</dd>`).join("")}</dl>` : `<p class="muted">—</p>`;
    const params = (s.parameters || []).length ? `<table class="params"><thead><tr><th>Key</th><th>Default</th><th>Note</th></tr></thead><tbody>${s.parameters.map((p) => `<tr><td><code>${esc(p.key)}</code></td><td>${esc(p.default)}</td><td>${esc(p.note || "")}</td></tr>`).join("")}</tbody></table>` : `<p class="muted">No parameters.</p>`;
    const sources = `<ul class="src">${(s.source || []).map((r) => `<li><a href="${esc(srcHref(r.file))}" target="_blank" rel="noopener"><code>${esc(r.file)}${r.line ? ":" + esc(r.line) : ""}</code></a> <span class="muted">${esc(r.symbol || "")}</span></li>`).join("")}</ul>`;
    const prev = idx > 0 ? order[idx - 1] : null, next = idx < order.length - 1 ? order[idx + 1] : null;

    panelBody.innerHTML = `
      <div class="chips">
        <span class="chip lane" style="border-color:${laneColor(s.lane)}">${esc(lane.title)}</span>
        <span class="chip">Step ${idx + 1} of ${order.length}</span>
        ${s.kind !== "core" ? `<span class="chip">${esc(s.kind)}</span>` : ""}
      </div>
      <h2 id="panel-title">${esc(s.name)}</h2>
      <div class="plain"><strong>In plain words</strong>${esc(s.summary)}</div>
      ${(s.uncertain || []).length ? `<div class="callout" role="note"><b>Uncertain / not verified</b><ul>${s.uncertain.map((u) => `<li>${esc(u)}</li>`).join("")}</ul></div>` : ""}
      ${overlay && overlay.notes[s.id] ? `<div class="callout cfg-note" role="note"><b>In the selected configuration:</b> ${esc(overlayText(overlay.notes[s.id]))}</div>` : ""}
      ${s.loop ? `<p class="loop-note"><b>Loop:</b> ${esc(s.loop)}</p>` : ""}
      <h3>Inputs</h3>${io(s.inputs)}
      <h3>Outputs</h3>${io(s.outputs)}
      <h3>Receives data from</h3>${flowList(incoming, "from")}
      <h3>Sends data to</h3>${flowList(outgoing, "to")}
      ${s.example ? `<h3>Example</h3><div class="example"><div class="ph">Illustrative placeholder — not real data</div>${esc(s.example)}</div>` : ""}
      <details class="tech" ${techOpen ? "open" : ""}>
        <summary>Technical details</summary>
        <div class="inner">
          <h3>Purpose</h3><p>${esc(s.purpose)}</p>
          <h3>Command / entry</h3><code class="cmd">${esc(s.command)}</code>
          <h3>Key parameters</h3>${params}
          <h3>Libraries / models</h3><p>${(s.libraries || []).map(esc).join(", ") || "—"}</p>
          ${(s.details || []).length ? `<h3>How it works</h3><ul>${s.details.map((d) => `<li>${esc(d)}</li>`).join("")}</ul>` : ""}
          <h3>Source</h3>${sources}
          ${fieldCount.get(s.id) ? `<h3>Configuration options</h3><p><button type="button" class="linkbtn" data-options="${esc(s.id)}">See the ${fieldCount.get(s.id)} options that configure this stage</button></p>` : ""}
        </div>
      </details>
      <div class="pager">
        ${prev ? `<button type="button" class="btn" data-goto="${esc(prev)}">&larr; ${esc(stageById.get(prev).name)}</button>` : "<span></span>"}
        ${next ? `<button type="button" class="btn" data-goto="${esc(next)}">${esc(stageById.get(next).name)} &rarr;</button>` : ""}
      </div>`;
    panelBody.querySelector("details.tech").addEventListener("toggle", (ev) => {
      techOpen = ev.target.open; storageSet("wfv-tech", techOpen ? "1" : "0");
    });
  }

  function renderOverview() {
    const m = WF.meta;
    panelBody.innerHTML = `
      <div class="chips"><span class="chip">Overview</span><span class="chip">${WF.stages.length} stages</span><span class="chip">analysed ${esc(m.analysedOn)}</span></div>
      <h2 id="panel-title">${esc(m.title)}</h2>
      <div class="plain"><strong>What it does</strong>${esc(m.oneLiner)}</div>
      <p>${esc(m.howToRead)}</p>
      <div class="callout" role="note">${esc(m.labelNote)}</div>
      <p class="muted">${esc(m.placeholderNote)}</p>
      ${window.CONFIG_SPACE ? `<p><button type="button" class="btn primary" data-view="configs">Explore all ${Number(window.CONFIG_SPACE.valid).toLocaleString("en-US")} valid configurations &rarr;</button></p>` : ""}
      <h3>Stages in order</h3>
      <div class="links">${order.map(stageLink).join("")}</div>
      <h3>Branches, options and loops</h3>
      <ul>${WF.branchesAndOptions.map((b) => `<li><b>${esc(b.title)}:</b> ${esc(b.text)}</li>`).join("")}</ul>
      <h3>Uncertain / not verified</h3>
      <ul class="overview-list">${WF.uncertainties.map((u) => `<li>${stageLink(u.stage)} ${esc(u.text)}</li>`).join("")}</ul>
      <details class="tech" ${techOpen ? "open" : ""}>
        <summary>Entry points and side tools</summary>
        <div class="inner">
          <h3>Entry points</h3>
          <ul>${m.entryPoints.map((p) => `<li><code>${esc(p.command)}</code><br><span class="muted">${esc(p.role)} — <code>${esc(p.file)}</code></span></li>`).join("")}</ul>
          <h3>Side tools (not in the main flow)</h3>
          <ul>${WF.sideTools.map((t) => `<li><b>${esc(t.name)}</b>: ${esc(t.text)}<br><code>${esc(t.command)}</code> <span class="muted">${esc(t.source)}</span></li>`).join("")}</ul>
          <h3>Analysed sources</h3>
          <p>${m.sources.map((x) => `<code>${esc(x)}</code>`).join(" ")}</p>
        </div>
      </details>`;
    panelBody.querySelector("details.tech").addEventListener("toggle", (ev) => {
      techOpen = ev.target.open; storageSet("wfv-tech", techOpen ? "1" : "0");
    });
  }

  panelBody.addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-goto]");
    if (b) { select(b.getAttribute("data-goto"), { focusNode: true }); return; }
    const o = ev.target.closest("[data-options]");
    if (o) { showView("configs"); if (window.WFV.onShowOptions) window.WFV.onShowOptions(o.getAttribute("data-options")); return; }
    const v = ev.target.closest("[data-view]");
    if (v) showView(v.getAttribute("data-view"));
  });

  function select(id, opts) {
    opts = opts || {};
    selectedId = id;
    gNodes.querySelectorAll(".node").forEach((n) => n.classList.toggle("selected", n.getAttribute("data-id") === id));
    if (id) {
      renderStage(stageById.get(id));
      panel.scrollTop = 0;
      if (opts.focusNode) nodeEl(id).focus({ preventScroll: false });
      if (opts.focusPanel) panel.focus();
      scrollNodeIntoView(id);
    } else {
      renderOverview();
    }
  }
  function scrollNodeIntoView(id) {
    const n = nodeEl(id);
    if (!n || !n.getBoundingClientRect) return;
    const r = n.getBoundingClientRect(), w = wrapEl.getBoundingClientRect();
    if (r.left < w.left || r.right > w.right || r.top < w.top || r.bottom > w.bottom) {
      n.scrollIntoView({ block: "nearest", inline: "nearest", behavior: reducedMotion ? "auto" : "smooth" });
    }
  }

  // ------------------------------------------------------------------ player
  let step = -1;
  let playing = false;
  let timer = null;
  let speed = 1;
  let animations = [];
  const narration = $("#narration");
  const btnPlay = $("#btn-play");

  function setStep(i, opts) {
    opts = opts || {};
    step = Math.max(-1, Math.min(order.length - 1, i));
    cancelTokens();
    const activeId = step >= 0 ? order[step] : null;
    gNodes.querySelectorAll(".node").forEach((n) => {
      const idx = orderIndex.get(n.getAttribute("data-id"));
      n.classList.toggle("active", step >= 0 && idx === step);
      n.classList.toggle("visited", step >= 0 && idx < step);
      n.classList.toggle("dim", step >= 0 && idx > step);
    });
    const activeEdges = [];
    edgeEls.forEach((x) => {
      const fromIdx = orderIndex.get(x.e.from), toIdx = orderIndex.get(x.e.to);
      let active = false;
      if (activeId) {
        if (x.e.kind === "loop") active = x.e.from === activeId;
        else active = x.e.to === activeId && fromIdx < step;
      }
      const future = activeId && x.e.kind !== "loop" && (toIdx > step || fromIdx > step);
      [x.g, x.lg].forEach((g) => { g.classList.toggle("active", active); g.classList.toggle("dim", !!future && !active); });
      x.path.setAttribute("marker-end", active ? "url(#arrow-active)" : x.e.kind === "loop" ? "url(#arrow-loop)" : "url(#arrow)");
      if (active) activeEdges.push(x);
    });
    $("#step-count").textContent = step >= 0 ? `Step ${step + 1} / ${order.length}` : `${order.length} steps`;
    $("#btn-back").disabled = nextIndex(step, -1) < 0;
    $("#btn-next").disabled = nextIndex(step, 1) >= order.length;
    renderNarration(activeId, activeEdges);
    if (activeId) {
      select(activeId, { focusNode: !!opts.focusNode });
      animateTokens(activeEdges);
    }
  }

  function renderNarration(id, activeEdges) {
    if (!id) {
      narration.innerHTML = `<p class="muted">Press <kbd>Play</kbd> or <kbd>Next</kbd> to walk through the pipeline one step at a time, or click any box to read about it.</p>`;
      return;
    }
    const s = stageById.get(id);
    const ins = activeEdges.filter((x) => x.e.kind !== "loop").map((x) => `${esc(x.e.label)} <span class="muted">from</span> ${esc(stageById.get(x.e.from).name)}`);
    const loops = activeEdges.filter((x) => x.e.kind === "loop").map((x) => `↻ ${esc(x.e.label)} → ${esc(stageById.get(x.e.to).name)}`);
    const outs = WF.edges.filter((e) => e.from === id && e.kind !== "loop").map((e) => `${esc(e.label)} → ${esc(stageById.get(e.to).name)}`);
    narration.innerHTML = `
      <p><span class="n-title">${step + 1}. ${esc(s.name)}</span> <span class="muted">· ${esc(laneById.get(s.lane).title)}${s.kind !== "core" ? " · " + esc(s.kind) : ""}</span></p>
      <p>${esc(s.summary)}</p>
      <p class="n-flow">${ins.length ? "<b>Data in:</b> " + ins.join("; ") : "<b>Data in:</b> starting point"}${outs.length ? " &nbsp;|&nbsp; <b>Data out:</b> " + outs.join("; ") : ""}${loops.length ? " &nbsp;|&nbsp; " + loops.join("; ") : ""}</p>`;
  }

  function cancelTokens() {
    animations.forEach((a) => { a.cancelled = true; });
    animations = [];
    gTokens.textContent = "";
  }
  function animateTokens(list) {
    if (reducedMotion || !list.length) return;
    const dur = 1500 / speed;
    const normal = list.filter((x) => x.e.kind !== "loop"), loops = list.filter((x) => x.e.kind === "loop");
    normal.forEach((x) => runToken(x.path, dur, 0, false));
    loops.forEach((x) => runToken(x.path, dur, normal.length ? dur * 0.9 : 0, true));
  }
  function runToken(path, dur, delay, isLoop) {
    const len = path.getTotalLength();
    const tk = el("g", { class: "token" }, gTokens);
    const trail = el("circle", { r: 4, opacity: 0.35 }, tk);
    const dot = el("circle", { r: 6.5 }, tk);
    tk.style.opacity = "0";
    const anim = { cancelled: false };
    animations.push(anim);
    const start = performance.now() + delay;
    function frame(now) {
      if (anim.cancelled) return;
      if (now < start) { requestAnimationFrame(frame); return; }
      tk.style.opacity = "1";
      const t = Math.min(1, (now - start) / dur);
      const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      const p = path.getPointAtLength(len * ease);
      const q = path.getPointAtLength(Math.max(0, len * ease - 14));
      dot.setAttribute("cx", p.x); dot.setAttribute("cy", p.y);
      trail.setAttribute("cx", q.x); trail.setAttribute("cy", q.y);
      if (t < 1) requestAnimationFrame(frame);
      else setTimeout(() => { if (!anim.cancelled) tk.remove(); }, isLoop ? 250 : 500);
    }
    requestAnimationFrame(frame);
  }

  function stepDelay() { return Math.round(4200 / speed); }
  function schedule() {
    clearTimeout(timer);
    if (!playing) return;
    timer = setTimeout(() => {
      const i = nextIndex(step, 1);
      if (i >= order.length) { setPlaying(false); return; }
      setStep(i);
      schedule();
    }, step < 0 ? 50 : stepDelay());
  }
  function setPlaying(v) {
    playing = v;
    btnPlay.innerHTML = playing ? "&#x23F8; Pause" : "&#x25B6; Play";
    btnPlay.setAttribute("aria-pressed", String(playing));
    if (playing && step >= order.length - 1) setStep(-1);
    schedule();
  }
  function nextIndex(from, dir) {
    let i = from + dir;
    while (i >= 0 && i < order.length && isOff(order[i])) i += dir;
    return i;
  }
  function next() { setPlaying(false); const i = nextIndex(step, 1); if (i < order.length) setStep(i); }
  function back() { setPlaying(false); const i = nextIndex(step, -1); if (i >= 0) setStep(i); }
  function reset() { setPlaying(false); setStep(nextIndex(-1, 1)); }

  $("#btn-next").addEventListener("click", next);
  $("#btn-back").addEventListener("click", back);
  $("#btn-reset").addEventListener("click", reset);
  btnPlay.addEventListener("click", () => setPlaying(!playing));
  const speedInput = $("#speed");
  function setSpeed(v) {
    speed = Math.max(0.5, Math.min(3, v));
    speedInput.value = String(speed);
    $("#speed-val").textContent = speed + "×";
    if (playing) schedule();
  }
  speedInput.addEventListener("input", () => setSpeed(parseFloat(speedInput.value)));

  // ------------------------------------------------------------------ theme
  const root = document.documentElement;
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  function effectiveTheme() { return root.getAttribute("data-theme") || (mq.matches ? "dark" : "light"); }
  function setTheme(t) { root.setAttribute("data-theme", t); storageSet("wfv-theme", t); updateThemeLabel(); }
  function updateThemeLabel() { $("#btn-theme .theme-label").textContent = effectiveTheme() === "dark" ? "Light mode" : "Dark mode"; }
  const savedTheme = storageGet("wfv-theme");
  if (savedTheme === "dark" || savedTheme === "light") root.setAttribute("data-theme", savedTheme);
  updateThemeLabel();
  mq.addEventListener && mq.addEventListener("change", updateThemeLabel);
  $("#btn-theme").addEventListener("click", () => setTheme(effectiveTheme() === "dark" ? "light" : "dark"));

  // ------------------------------------------------------------------ help + global keys
  const help = $("#help-dialog");
  $("#btn-help").addEventListener("click", () => help.showModal());
  $("#btn-overview").addEventListener("click", () => select(null));

  document.addEventListener("keydown", (ev) => {
    if (help.open) return; // dialog handles Esc itself
    const t = ev.target;
    const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (typing && t.type !== "checkbox" && t.type !== "radio") return;
    const k = ev.key;
    const handled = () => ev.preventDefault();
    if (k === "v" || k === "V") { handled(); showView(currentView === "pipeline" ? "configs" : "pipeline"); return; }
    if (currentView !== "pipeline") {
      if (k === "t" || k === "T") { handled(); setTheme(effectiveTheme() === "dark" ? "light" : "dark"); }
      else if (k === "?") { handled(); help.showModal(); }
      return;
    }
    if (k === "n" || k === "N" || k === ".") { handled(); next(); }
    else if (k === "b" || k === "B" || k === ",") { handled(); back(); }
    else if (k === "p" || k === "P") { handled(); setPlaying(!playing); }
    else if (k === "Home") { handled(); reset(); }
    else if (k === "[") { handled(); setSpeed(speed - 0.25); }
    else if (k === "]") { handled(); setSpeed(speed + 0.25); }
    else if (k === "+" || k === "=") { handled(); zoomBy(1.15); }
    else if (k === "-" || k === "_") { handled(); zoomBy(1 / 1.15); }
    else if (k === "0") { handled(); zoomMode = "fit"; applyZoom(); }
    else if (k === "t" || k === "T") { handled(); setTheme(effectiveTheme() === "dark" ? "light" : "dark"); }
    else if (k === "o" || k === "O") { handled(); select(null); }
    else if (k === "?") { handled(); help.showModal(); }
    else if (k === "d" || k === "D") {
      handled();
      const d = panelBody.querySelector("details.tech");
      if (d) d.open = !d.open;
    }
    else if (k === "Escape") { handled(); setPlaying(false); if (selectedId) { const id = selectedId; select(null); nodeEl(id).focus(); } }
    else if (k.startsWith("Arrow") && !(t && t.closest && t.closest(".node"))) {
      if (t === document.body || t === wrapEl || (t && t.id === "diagram")) {
        handled();
        nodeEl(selectedId || focusedId || order[0]).focus();
      }
    }
  });

  // ------------------------------------------------------------------ header + footer + drift check
  $("#app-title").textContent = WF.meta.title;
  $("#app-oneliner").textContent = WF.meta.oneLiner;
  $("#foot-meta").textContent = `${WF.meta.project} · analysed ${WF.meta.analysedOn} · ${WF.stages.length} stages, ${WF.edges.length} connections`;

  if (/^https?:$/.test(location.protocol)) {
    fetch("workflow.json", { cache: "no-store" }).then((r) => (r.ok ? r.json() : null)).then((live) => {
      if (live && JSON.stringify(live) !== JSON.stringify(WF)) {
        const b = $("#drift-banner");
        b.hidden = false;
        b.textContent = "workflow.json has changed since workflow-data.js was generated. Run: python sync_data.py (in workflow-viewer/) and reload.";
      }
    }).catch(() => { /* offline or file:// - nothing to compare */ });
  }

  setSpeed(1);
  setStep(-1);
  select(null);

  // ------------------------------------------------------------------ views
  let currentView = "pipeline";
  function showView(name) {
    currentView = name === "configs" && window.CONFIG_SPACE ? "configs" : "pipeline";
    const pipe = currentView === "pipeline";
    $("#pipeline-view").hidden = !pipe;
    $("#config-view").hidden = pipe;
    [["#tab-pipeline", pipe], ["#tab-configs", !pipe]].forEach(([sel, on]) => {
      const tab = $(sel);
      tab.setAttribute("aria-selected", String(on));
      tab.tabIndex = on ? 0 : -1;
    });
    if (pipe) { if (zoomMode === "fit") applyZoom(); } else { setPlaying(false); if (window.WFV.onShowConfigs) window.WFV.onShowConfigs(); }
  }
  $("#tab-pipeline").addEventListener("click", () => showView("pipeline"));
  $("#tab-configs").addEventListener("click", () => showView("configs"));
  $(".views").addEventListener("keydown", (ev) => {
    if (ev.key === "ArrowLeft" || ev.key === "ArrowRight") {
      ev.preventDefault(); showView(currentView === "pipeline" ? "configs" : "pipeline");
      (currentView === "pipeline" ? $("#tab-pipeline") : $("#tab-configs")).focus();
    }
  });
  if (!window.CONFIG_SPACE) $("#tab-configs").hidden = true;

  window.WFV = {
    stages: stageById, lanes: laneById, order, esc,
    showView,
    showStage(id) { showView("pipeline"); select(id, { focusNode: true }); },
    setOverlay,
    onShowConfigs: null, onShowOptions: null,
  };

  // Deep links: #stage=<id>, #step=<1..N>, #theme=light|dark, #view=configs[&sub=options|shipped] (combine with &).
  const hash = new URLSearchParams(location.hash.replace(/^#/, ""));
  if (hash.get("theme") === "light" || hash.get("theme") === "dark") root.setAttribute("data-theme", hash.get("theme")), updateThemeLabel();
  if (hash.get("step")) setStep(parseInt(hash.get("step"), 10) - 1);
  else if (stageById.has(hash.get("stage"))) select(hash.get("stage"));
  // DOMContentLoaded fires after every parser-blocking script (configs.js) has run; a timer could fire earlier.
  if (hash.get("view") === "configs") document.addEventListener("DOMContentLoaded", () => showView("configs"));
})();
