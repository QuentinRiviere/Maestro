"use strict";
// Spark model hub UI. Plain DOM, no innerHTML with remote data.

const S = { csrf: "", status: null, tab: "catalog", jobsTimer: null };
const $ = (id) => document.getElementById(id);

function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k in el && typeof v !== "string") el[k] = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

const gb = (bytes) => (bytes / 1e9).toFixed(bytes >= 10e9 ? 1 : 2) + " GB";
const date = (s) => (s ? String(s).slice(0, 10) : "");
const num = (n) => (n == null ? "" : Number(n).toLocaleString());
const badge = (text, cls) => h("span", { class: "badge " + (cls || text) }, text);

class ApiError extends Error {}

async function api(path, opts = {}) {
  const init = { method: opts.method || "GET", credentials: "same-origin", headers: {} };
  if (init.method !== "GET") init.headers["X-CSRF-Token"] = S.csrf;
  if (opts.body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(opts.body);
  }
  const res = await fetch(path, init);
  let data = null;
  try { data = await res.json(); } catch (_) { data = null; }
  if (res.status === 401 && !opts.allow401) { showLogin(); throw new ApiError("login required"); }
  if (!res.ok && !opts.allowError) {
    const detail = data && (data.detail || data.error);
    throw new ApiError(typeof detail === "string" ? detail : res.status + " " + res.statusText);
  }
  return data;
}

function report(err) { if (!(err instanceof ApiError && err.message === "login required")) alert(err.message || err); }

// -- login / tabs -----------------------------------------------------------
function showLogin() {
  $("login").hidden = false;
  $("tabs").hidden = true;
  $("logout").hidden = true;
  document.querySelectorAll(".tab").forEach((t) => (t.hidden = true));
  $("status").textContent = "";
}

async function boot() {
  const s = await api("/api/session", { allow401: true, allowError: true });
  if (!s || !s.authenticated) return showLogin();
  S.csrf = s.csrf;
  $("login").hidden = true;
  $("tabs").hidden = false;
  $("logout").hidden = false;
  await refreshStatus();
  showTab(S.tab);
}

$("login-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  $("login-error").textContent = "";
  try {
    const r = await api("/api/login", { method: "POST", body: { token: $("token").value }, allow401: true, allowError: true });
    if (!r || !r.authenticated) { $("login-error").textContent = (r && r.detail) || "Login failed"; return; }
    $("token").value = "";
    await boot();
  } catch (e) { $("login-error").textContent = e.message; }
});

$("logout").addEventListener("click", async () => {
  try { await api("/api/logout", { method: "POST" }); } catch (_) { /* ignore */ }
  S.csrf = "";
  showLogin();
});

document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

function showTab(name) {
  S.tab = name;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => (t.hidden = t.id !== "tab-" + name));
  clearInterval(S.jobsTimer);
  ({ catalog: loadCatalog, explore: initExplore, downloads: loadJobs, installed: loadInstalled, config: loadConfig })[name]();
  if (name === "downloads") S.jobsTimer = setInterval(loadJobs, 1500);
}

async function refreshStatus() {
  try {
    S.status = await api("/api/status");
    const st = S.status, b = st.budget;
    $("status").textContent = `disk free ${gb(st.disk.free)} · memory ${b.total_gb} GB − ${b.reserved_gb} reserved − ` +
      `${b.resident_gb.toFixed(1)} resident = ${b.available_gb.toFixed(1)} GB for on-demand models` +
      (st.base_error ? " · base config error: " + st.base_error : "");
  } catch (e) { report(e); }
}

// -- catalog -----------------------------------------------------------------
let catalogItems = [];
$("catalog-category").addEventListener("change", renderCatalog);

async function loadCatalog() {
  try {
    const data = await api("/api/catalog");
    catalogItems = data.items;
    $("catalog-budget").textContent = `Fit badges against ${data.budget.available_gb.toFixed(1)} GB available`;
    renderCatalog();
  } catch (e) { report(e); }
}

function renderCatalog() {
  const cat = $("catalog-category").value;
  const list = $("catalog-list");
  list.replaceChildren(...catalogItems.filter((i) => !cat || i.category === cat).map((i) => h("div", { class: "card" },
    h("div", { class: "row between" }, h("strong", {}, i.name), badge(i.category, "")),
    h("code", { class: "small" }, i.repo),
    h("div", { class: "row small" }, `${i.size_gb.toFixed(1)} GB download`, "·", `~${i.memory_gb} GB in memory`, badge(i.fit),
      i.installed ? badge("installed", "ok") : null, i.serve && i.serve.mtp ? badge("MTP", "") : null,
      i.serve && i.serve.mmproj ? badge("vision", "") : null),
    h("div", { class: "muted small" }, i.notes),
    h("div", { class: "muted small" }, "pinned ", h("code", {}, i.revision.slice(0, 12))),
    h("div", { class: "row" }, h("button", { class: "primary", onclick: () => openRepo(i.repo, i.revision, i) }, "Open / download")))));
}

// -- explore -----------------------------------------------------------------
function initExplore() {
  if (S.status) $("trusted-note").textContent = "Trusted publishers: " + S.status.trusted_authors.join(", ");
}
$("search-all").addEventListener("change", () => ($("search-warning").hidden = !$("search-all").checked));
$("search-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const params = new URLSearchParams({ q: $("search-q").value, sort: $("search-sort").value, limit: "40", all: $("search-all").checked });
  const table = $("search-results");
  table.replaceChildren(h("tr", {}, h("td", { class: "muted" }, "Searching...")));
  try {
    const data = await api("/api/search?" + params);
    table.replaceChildren(
      h("tr", {}, h("th", {}, "Repository"), h("th", { class: "num" }, "Downloads"), h("th", { class: "num" }, "Likes"), h("th", {}, "Updated"), h("th", {}, "")),
      ...data.results.map((r) => h("tr", {},
        h("td", {}, h("button", { class: "link", onclick: () => openRepo(r.repo, "main") }, r.repo), " ",
          r.trusted ? badge("trusted") : badge("untrusted"), r.gated ? badge("gated", "") : null),
        h("td", { class: "num" }, num(r.downloads)), h("td", { class: "num" }, num(r.likes)),
        h("td", {}, date(r.last_modified)), h("td", {}, h("button", { onclick: () => openRepo(r.repo, "main") }, "Open")))));
    if (!data.results.length) table.replaceChildren(h("tr", {}, h("td", { class: "muted" }, "No GGUF repository found.")));
  } catch (e) { table.replaceChildren(h("tr", {}, h("td", { class: "error" }, e.message))); }
});

// -- repo detail ---------------------------------------------------------------
$("repo-close").addEventListener("click", () => ($("repo").hidden = true));

async function openRepo(repo, revision, catalogItem, ctx) {
  $("repo").hidden = false;
  $("repo-title").textContent = repo;
  $("repo-meta").textContent = "Loading...";
  $("repo-body").replaceChildren();
  const preset = catalogItem && catalogItem.serve;
  ctx = ctx || (preset && preset.ctx_size) || 32768;
  let d;
  try {
    d = await api(`/api/repo/${repo}?` + new URLSearchParams({ revision, ctx }));
  } catch (e) { $("repo-meta").textContent = ""; $("repo-body").replaceChildren(h("p", { class: "error" }, e.message)); return; }
  $("repo-meta").replaceChildren(
    h("div", { class: "row" }, "commit ", h("code", {}, d.commit), d.pinned ? badge("pinned", "ok") : badge("resolved from " + d.revision, "tight"),
      d.trusted ? badge("trusted") : badge("untrusted"), d.gated ? badge("gated", "tight") : null,
      d.mtp.mtp ? badge("MTP" + (d.mtp.draft_n_max ? " n=" + d.mtp.draft_n_max : ""), "") : null,
      d.license ? badge(d.license, "") : null),
    h("div", { class: "small" }, `updated ${date(d.last_modified)} · ${num(d.downloads)} downloads · ${num(d.likes)} likes · `,
      h("a", { href: `https://huggingface.co/${d.repo}`, target: "_blank", rel: "noopener noreferrer" }, "model card")),
    d.refused.length ? h("div", { class: "warning small" }, "Refused (unsafe formats, never downloaded): " + d.refused.join(", ")) : null);

  const models = d.ggufs.filter((g) => g.kind === "model");
  const mmprojs = d.ggufs.filter((g) => g.kind === "mmproj");
  const others = d.ggufs.filter((g) => g.kind !== "model" && g.kind !== "mmproj");
  const wantedFirst = preset ? preset.model : null;
  let selected = models.find((g) => g.files.includes(wantedFirst)) || models.find((g) => g.fit === "fits") || models[0];

  const ctxInput = h("input", { type: "number", min: 0, step: 1024, value: d.ctx });
  const ctxRow = h("div", { class: "row small" }, "Memory estimate for context ", ctxInput, " tokens ",
    h("button", { onclick: () => openRepo(repo, revision, catalogItem, Number(ctxInput.value) || 0) }, "Recompute"),
    h("span", { class: "muted" }, ` · ${d.budget.available_gb.toFixed(1)} GB available (resident: ${d.budget.resident.join(", ") || "none"})`));

  const rows = models.map((g) => h("tr", {},
    h("td", {}, h("input", { type: "radio", name: "gguf", checked: g === selected, onchange: () => { selected = g; form.setModel(g); } })),
    h("td", {}, h("code", {}, g.key), g.parts > 1 ? h("span", { class: "muted small" }, ` (${g.parts} parts${g.complete ? "" : ", INCOMPLETE"})`) : null),
    h("td", {}, g.quant), h("td", { class: "num" }, gb(g.size)),
    h("td", { class: "num" }, g.memory ? `~${g.memory.total.toFixed(1)} GB` : ""),
    h("td", {}, g.fit ? badge(g.fit) : null, g.installed ? badge("installed", "ok") : null)));
  const table = h("table", { class: "list" },
    h("tr", {}, h("th", {}, ""), h("th", {}, "File"), h("th", {}, "Quant"), h("th", { class: "num" }, "Size"), h("th", { class: "num" }, "Memory"), h("th", {}, "")), ...rows);

  const extra = [...mmprojs, ...others].map((g) => h("li", { class: "small" }, h("code", {}, g.key), ` ${g.kind} · ${gb(g.size)}`,
    g.kind === "mtp" || g.kind === "draft" ? h("span", { class: "muted" }, " (draft model, not wired automatically)") : null));

  const form = serveForm({
    repo: d.repo, models, mmprojs, mtp: d.mtp,
    serve: preset || { id: suggestId(d.repo), ctx_size: d.ctx, parallel: 1, ttl: 600, mtp: d.mtp.mtp, mtp_draft_n_max: d.mtp.draft_n_max || 6 },
    defaultMmproj: preset ? preset.mmproj || "" : d.default_mmproj || "",
  });
  if (selected) form.setModel(selected, true);

  const mapCheck = h("input", { type: "checkbox", checked: true });
  const out = h("div", {});
  const downloadBtn = h("button", { class: "primary", onclick: async () => {
    if (!selected) return;
    const files = [...selected.files];
    const serve = mapCheck.checked ? form.value() : null;
    if (serve && serve.mmproj) files.push(serve.mmproj);
    downloadBtn.disabled = true;
    try {
      await api("/api/downloads", { method: "POST", body: { repo: d.repo, revision: d.commit, files, serve } });
      $("repo").hidden = true;
      showTab("downloads");
    } catch (e) { out.replaceChildren(h("p", { class: "error" }, e.message)); }
    downloadBtn.disabled = false;
  } }, "Download (verified)");
  const previewBtn = h("button", { onclick: () => preview(d.repo, form.value(), out) }, "Preview llama-swap.yaml diff");

  $("repo-body").replaceChildren(
    ctxRow, models.length ? table : h("p", { class: "muted" }, "No loadable GGUF model in this repository."),
    extra.length ? h("ul", {}, extra) : null,
    h("h3", {}, "Serve with llama-swap"),
    h("label", { class: "inline" }, mapCheck, "Map into llama-swap after the download"),
    form.el,
    h("div", { class: "row" }, downloadBtn, previewBtn), out);
}

function suggestId(repo) {
  return repo.split("/")[1].toLowerCase().replace(/-gguf$/, "").replace(/[^a-z0-9._-]+/g, "-").slice(0, 60);
}

async function preview(repo, serve, out) {
  try {
    const r = await api("/api/serve/preview", { method: "POST", body: { repo, serve }, allowError: true });
    if (r.error) { out.replaceChildren(h("p", { class: "error" }, r.error)); return; }
    out.replaceChildren(...(r.warnings || []).map((w) => h("p", { class: "warning small" }, w)),
      r.diff ? diffView(r.diff) : h("p", { class: "muted" }, "No change to llama-swap.yaml."));
  } catch (e) { out.replaceChildren(h("p", { class: "error" }, e.message)); }
}

function diffView(text) {
  return h("pre", { class: "code" }, ...text.split("\n").map((l) =>
    h("div", { class: l.startsWith("+") && !l.startsWith("+++") ? "diff-add" : l.startsWith("-") && !l.startsWith("---") ? "diff-del" : "" }, l)));
}

// Serve settings editor. models: [{key, first, files}], mmprojs: same shape.
function serveForm({ models, mmprojs, serve, defaultMmproj }) {
  const s = Object.assign({ parallel: 1, ttl: 600, ctx_size: 32768, mtp: false, mtp_draft_n_max: 6, preload: false, enabled: true, aliases: [], extra_args: [] }, serve || {});
  const groups = (S.status && S.status.groups) || [];
  const group = s.group || (S.status && S.status.default_group) || "on-demand";
  const f = {
    id: h("input", { value: s.id || "", required: true }),
    name: h("input", { value: s.name || "" }),
    model: h("select", {}, ...models.map((g) => h("option", { value: g.first }, g.key))),
    mmproj: h("select", {}, h("option", { value: "" }, "none"), ...mmprojs.map((g) => h("option", { value: g.first }, g.key))),
    ctx_size: h("input", { type: "number", min: 0, step: 1024, value: s.ctx_size }),
    parallel: h("input", { type: "number", min: 1, max: 64, value: s.parallel }),
    ttl: h("input", { type: "number", min: -1, value: s.ttl }),
    group: h("input", { value: group, list: "group-list" }),
    aliases: h("input", { value: (s.aliases || []).join(", "), placeholder: "comma separated" }),
    extra_args: h("input", { value: (s.extra_args || []).join(" "), placeholder: "e.g. --temp 0.6 --top-k 20" }),
    mtp: h("input", { type: "checkbox", checked: !!s.mtp }),
    mtp_draft_n_max: h("input", { type: "number", min: 1, max: 32, value: s.mtp_draft_n_max || 6 }),
    preload: h("input", { type: "checkbox", checked: !!s.preload }),
    enabled: h("input", { type: "checkbox", checked: s.enabled !== false }),
  };
  if (s.model) f.model.value = s.model;
  f.mmproj.value = s.mmproj !== undefined ? s.mmproj || "" : defaultMmproj || "";
  const lab = (text, el, title) => h("label", { title }, text, el);
  const chk = (text, el, title) => h("label", { class: "check", title }, el, text);
  const el = h("div", {},
    h("datalist", { id: "group-list" }, ...groups.map((g) => h("option", { value: g }))),
    h("div", { class: "form" },
      lab("Model id (llama-swap)", f.id), lab("Display name", f.name), lab("Model file", f.model), lab("Vision projector (mmproj)", f.mmproj),
      lab("Context size (--ctx-size, 0 = model default)", f.ctx_size), lab("Parallel slots (--parallel)", f.parallel),
      lab("TTL seconds (0 = never unload, -1 = global)", f.ttl), lab("Group (from llama-swap.base.yaml)", f.group),
      lab("Aliases", f.aliases), lab("Extra llama-server args", f.extra_args),
      lab("MTP draft tokens (--spec-draft-n-max)", f.mtp_draft_n_max)),
    h("div", { class: "row" }, chk("MTP speculative decoding (--spec-type draft-mtp)", f.mtp), chk("Preload at llama-swap start", f.preload),
      chk("Enabled", f.enabled)));
  return {
    el,
    setModel(g, keepId) { f.model.value = g.first; if (!keepId && !f.id.value) f.id.value = g.first.replace(/\.gguf$/i, "").toLowerCase(); },
    value() {
      const split = (v, re) => v.split(re).map((x) => x.trim()).filter(Boolean);
      const out = {
        id: f.id.value.trim(), model: f.model.value, ctx_size: Number(f.ctx_size.value), parallel: Number(f.parallel.value),
        ttl: Number(f.ttl.value), group: f.group.value.trim(), aliases: split(f.aliases.value, /,/), extra_args: split(f.extra_args.value, /\s+/),
        mtp: f.mtp.checked, preload: f.preload.checked, enabled: f.enabled.checked,
      };
      if (f.name.value.trim()) out.name = f.name.value.trim();
      if (f.mmproj.value) out.mmproj = f.mmproj.value;
      if (out.mtp) out.mtp_draft_n_max = Number(f.mtp_draft_n_max.value);
      return out;
    },
  };
}

// -- downloads ---------------------------------------------------------------
async function loadJobs() {
  let data;
  try { data = await api("/api/downloads"); } catch (e) { clearInterval(S.jobsTimer); return report(e); }
  const box = $("jobs");
  if (!data.jobs.length) { box.replaceChildren(h("p", { class: "muted" }, "No download yet. Open a model from the Catalog or Explore tab.")); return; }
  box.replaceChildren(...data.jobs.map((j) => {
    const pct = j.total_bytes ? j.done_bytes / j.total_bytes : 0;
    const eta = j.eta_seconds != null ? ` · ETA ${fmtDuration(j.eta_seconds)}` : "";
    const rate = j.rate ? ` · ${(j.rate / 1e6).toFixed(1)} MB/s` : "";
    return h("div", { class: "card job" },
      h("div", { class: "row between" }, h("strong", {}, j.repo), badge(j.state, j.state === "done" ? "ok" : j.state)),
      h("div", { class: "row small" }, h("progress", { max: 1, value: pct }),
        `${gb(j.done_bytes)} / ${gb(j.total_bytes)}${rate}${eta}`, j.phase === "verify" ? " · verifying hash" : ""),
      h("div", { class: "muted small" }, j.current ? "current: " + j.current : j.files.join(", ")),
      j.message ? h("div", { class: "small" }, j.message) : null,
      j.error ? h("div", { class: "error small" }, j.error) : null,
      h("div", { class: "row" },
        j.state === "queued" || j.state === "running" ? h("button", { class: "danger", onclick: () => jobAction(j.id, "cancel") }, "Cancel") : null,
        j.state === "cancelled" || j.state === "failed" ? h("button", { onclick: () => jobAction(j.id, "resume") }, "Resume") : null));
  }));
}

async function jobAction(id, action) {
  try { await api(`/api/downloads/${id}/${action}`, { method: "POST" }); loadJobs(); } catch (e) { report(e); }
}

function fmtDuration(s) {
  s = Math.round(s);
  if (s < 60) return s + " s";
  if (s < 3600) return Math.floor(s / 60) + " min " + (s % 60) + " s";
  return Math.floor(s / 3600) + " h " + Math.floor((s % 3600) / 60) + " min";
}

// -- installed ---------------------------------------------------------------
async function loadInstalled() {
  let data;
  try { data = await api("/api/installed"); } catch (e) { return report(e); }
  $("disk").textContent = `${gb(data.disk.free)} free of ${gb(data.disk.total)}`;
  const box = $("installed-list");
  if (!data.repos.length) box.replaceChildren(h("p", { class: "muted" }, "Nothing installed yet."));
  else box.replaceChildren(...data.repos.map((r) => {
    const v = r.verify;
    return h("div", { class: "card job" },
      h("div", { class: "row between" }, h("strong", {}, r.repo),
        h("span", { class: "row" }, r.serve ? (r.mapped ? badge("mapped as " + r.serve.id, "mapped") : badge("not mapped", "tight")) : badge("no serve settings", ""),
          v ? badge("hash " + v.state, v.state) : badge("hash verified at download", "ok"))),
      h("div", { class: "small muted" }, `${gb(r.bytes)} · commit `, h("code", {}, (r.commit || "").slice(0, 12)), r.in_models_yml ? "" : " · not in models.yml"),
      h("ul", { class: "small" }, ...r.files.map((f) => h("li", {}, h("code", {}, f.name), ` ${gb(f.size)}`, f.present ? "" : " MISSING",
        v && v.files && v.files[f.name] === false ? h("span", { class: "error" }, " hash mismatch") : null))),
      h("div", { class: "row" },
        h("button", { onclick: () => verifyRepo(r.repo) }, "Re-verify hashes"),
        h("button", { onclick: () => editServe(r) }, r.serve ? "Edit mapping" : "Map into llama-swap"),
        h("button", { class: "danger", disabled: r.downloading, onclick: () => deleteRepo(r.repo) }, "Delete")),
      h("div", { id: "edit-" + r.repo }));
  }));
  loadRunning();
}

async function loadRunning() {
  try {
    const r = await api("/api/llama-swap/running");
    $("running").replaceChildren(!r.reachable ? h("span", {}, "llama-swap not reachable: " + r.error)
      : !r.running.length ? h("span", {}, "No model loaded.")
        : h("ul", {}, ...r.running.map((m) => h("li", {}, h("code", {}, m.model), " ", badge(m.state, m.state === "ready" ? "ok" : "running"), m.ttl ? ` ttl ${m.ttl}s` : ""))));
  } catch (e) { $("running").textContent = e.message; }
}

async function verifyRepo(repo) {
  try { await api("/api/installed/verify", { method: "POST", body: { repo } }); } catch (e) { return report(e); }
  setTimeout(loadInstalled, 1500);
}

async function deleteRepo(repo) {
  const typed = prompt(`Delete every file of ${repo} and its llama-swap mapping?\nType the repo id to confirm:`);
  if (typed === null) return;
  if (typed.trim() !== repo) { alert("The repo id did not match: nothing deleted."); return; }
  try {
    const r = await api(`/api/installed/${repo}?` + new URLSearchParams({ confirm: repo }), { method: "DELETE" });
    alert(`Deleted ${r.deleted}, freed ${gb(r.freed)}.`);
  } catch (e) { report(e); }
  loadInstalled();
}

function editServe(r) {
  const ggufFiles = r.files.map((f) => f.name).filter((n) => /\.gguf$/i.test(n));
  const firstParts = ggufFiles.filter((n) => !/-\d{5}-of-\d{5}\.gguf$/i.test(n) || /-00001-of-\d{5}\.gguf$/i.test(n));
  const toEntry = (n) => ({ key: n, first: n, files: [n] });
  const models = firstParts.filter((n) => !/mmproj/i.test(n)).map(toEntry);
  const mmprojs = firstParts.filter((n) => /mmproj/i.test(n)).map(toEntry);
  const form = serveForm({ models, mmprojs, serve: r.serve || { id: suggestId(r.repo) }, defaultMmproj: "" });
  const out = h("div", {});
  const box = document.getElementById("edit-" + r.repo);
  box.replaceChildren(form.el, h("div", { class: "row" },
    h("button", { onclick: () => preview(r.repo, form.value(), out) }, "Preview diff"),
    h("button", { class: "primary", onclick: async () => {
      try {
        const res = await api("/api/serve", { method: "POST", body: { repo: r.repo, serve: form.value() } });
        out.replaceChildren(...(res.warnings || []).map((w) => h("p", { class: "warning small" }, w)),
          h("p", { class: "small" }, res.changed ? "Saved; llama-swap.yaml updated." : "Saved; llama-swap.yaml unchanged."));
        setTimeout(loadInstalled, 800);
      } catch (e) { out.replaceChildren(h("p", { class: "error" }, e.message)); }
    } }, "Save mapping"),
    r.serve ? h("button", { onclick: async () => {
      try { await api("/api/serve", { method: "POST", body: { repo: r.repo, serve: null } }); loadInstalled(); } catch (e) { report(e); }
    } }, "Remove mapping") : null), out);
}

// -- config ------------------------------------------------------------------
async function loadConfig() {
  try {
    const c = await api("/api/config");
    $("config-msg").textContent = c.error ? "Render error: " + c.error : (c.warnings || []).join(" · ");
    $("config-diff").replaceChildren(c.pending_diff ? diffView(c.pending_diff) : document.createTextNode("None: llama-swap.yaml matches the base file and models.yml."));
    $("config-rendered").textContent = c.rendered || "(not rendered yet)";
  } catch (e) { report(e); }
}

$("rerender").addEventListener("click", async () => {
  try {
    const r = await api("/api/render", { method: "POST" });
    $("config-msg").textContent = r.changed ? "llama-swap.yaml written." : "Unchanged.";
    loadConfig();
  } catch (e) { $("config-msg").textContent = e.message; }
});

boot().catch(report);
