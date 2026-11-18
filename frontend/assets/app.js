/**
 * Clinician search UI - vanilla JS, no build step beyond Tailwind.
 *
 * Talks to the FastAPI backend:
 *   POST /search?resource_type=&date_from=&date_to=   { query, top_k }
 *   GET  /patients/{mrn}       demographics, AI summary, linked FHIR resources
 *   GET  /filters              filter options      GET /health   index status
 *
 * All server text is inserted with textContent (never innerHTML) so record
 * content cannot inject markup.
 */

const API_BASE = document.querySelector('meta[name="api-base"]')?.content?.replace(/\/$/, "") ?? "";
const DEBOUNCE_MS = 250;
const MIN_QUERY = 2;
const TOP_K = 5;

const TYPE_STYLES = {
  DocumentReference: { label: "Clinical note", cls: "bg-doc-soft text-doc" },
  DiagnosticReport: { label: "Diagnostic report", cls: "bg-dx-soft text-dx" },
  ClinicalSummary: { label: "AI summary", cls: "bg-ai-soft text-ai" },
};
const TYPE_OPTION_LABELS = {
  DocumentReference: "Clinical notes (DocumentReference)",
  DiagnosticReport: "Imaging & labs (DiagnosticReport)",
  ClinicalSummary: "AI summaries",
};
const SUGGESTIONS = [
  "blood clot in the lungs",
  "elevated troponin chest pain",
  "suspicious breast mass",
  "worsening kidney function",
  "fracture after a fall",
];

const $ = (sel) => document.querySelector(sel);
const els = {
  form: $("#search-form"),
  query: $("#query"),
  type: $("#filter-type"),
  from: $("#filter-from"),
  to: $("#filter-to"),
  clear: $("#clear-filters"),
  filterError: $("#filter-error"),
  body: $("#results-body"),
  meta: $("#results-meta"),
  srStatus: $("#sr-status"),
  indexStatus: $("#index-status"),
  drawer: $("#drawer"),
  drawerPanel: $("#drawer-panel"),
  drawerBackdrop: $("#drawer-backdrop"),
  drawerTitle: $("#drawer-title"),
  drawerSubtitle: $("#drawer-subtitle"),
  drawerBody: $("#drawer-body"),
  drawerClose: $("#drawer-close"),
  skeleton: $("#tpl-skeleton"),
};

const state = {
  controller: null,
  debounce: null,
  lastQuery: "",
  returnFocus: null,
  patientController: null,
};

/* ------------------------------------------------------------------ DOM helpers */

/** Create an element: el("p", { class: "x", text: "hi" }, child1, child2) */
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

/** Text with query terms wrapped in <mark>, built from text nodes (XSS-safe). */
function highlight(text, query) {
  const frag = document.createDocumentFragment();
  const terms = [...new Set(query.toLowerCase().match(/[a-z0-9]{3,}/g) ?? [])];
  if (!text || terms.length === 0) {
    frag.append(text ?? "");
    return frag;
  }
  const escaped = terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp(`(${escaped.join("|")})`, "gi");
  let last = 0;
  for (const match of text.matchAll(re)) {
    frag.append(text.slice(last, match.index), el("mark", { text: match[0] }));
    last = match.index + match[0].length;
  }
  frag.append(text.slice(last));
  return frag;
}

function formatDate(iso) {
  if (!iso) return "Undated";
  const d = new Date(`${iso.slice(0, 10)}T00:00:00`);
  return Number.isNaN(d.getTime())
    ? iso
    : d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function ageFrom(birthDate) {
  if (!birthDate) return null;
  const dob = new Date(`${birthDate}T00:00:00`);
  const now = new Date();
  let age = now.getFullYear() - dob.getFullYear();
  if (now < new Date(now.getFullYear(), dob.getMonth(), dob.getDate())) age -= 1;
  return age;
}

function typeBadge(resourceType, kind) {
  const style = TYPE_STYLES[resourceType] ?? { label: resourceType, cls: "bg-paper text-muted" };
  const label = resourceType === "DiagnosticReport" && kind ? kind : style.label;
  return el("span", {
    class: `inline-flex items-center rounded px-2 py-0.5 text-xs font-medium ${style.cls}`,
    title: resourceType,
    text: label,
  });
}

/* ------------------------------------------------------------------------ API */

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function api(path, { method = "GET", body, signal } = {}) {
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method,
      signal,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new ApiError("The search service can't be reached. Check that `ehr-intel serve` is running.", 0);
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status}).`;
    try {
      const data = await res.json();
      if (typeof data.detail === "string") detail = data.detail;
      else if (Array.isArray(data.detail)) detail = data.detail.map((d) => d.msg).join("; ");
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail, res.status);
  }
  return res.json();
}

/* --------------------------------------------------------------- Filters & URL */

function currentFilters() {
  return { type: els.type.value, from: els.from.value, to: els.to.value };
}

function validateFilters({ from, to }) {
  if (from && to && from > to) return "The start date is after the end date. Change one of them.";
  return "";
}

function syncUrl(query, filters) {
  const params = new URLSearchParams();
  if (query) params.set("q", query);
  if (filters.type) params.set("type", filters.type);
  if (filters.from) params.set("from", filters.from);
  if (filters.to) params.set("to", filters.to);
  const qs = params.toString();
  history.replaceState(null, "", qs ? `?${qs}` : location.pathname);
}

function restoreFromUrl() {
  const params = new URLSearchParams(location.search);
  els.query.value = params.get("q") ?? "";
  els.from.value = params.get("from") ?? "";
  els.to.value = params.get("to") ?? "";
  return params.get("type") ?? "";
}

function updateClearButton() {
  const f = currentFilters();
  els.clear.disabled = !(f.type || f.from || f.to);
}

async function loadFilterOptions(selectedType) {
  try {
    const data = await api("/filters");
    for (const type of data.resource_types) {
      els.type.append(el("option", { value: type, text: TYPE_OPTION_LABELS[type] ?? type }));
    }
    els.type.value = selectedType;
    for (const input of [els.from, els.to]) {
      if (data.date_min) input.min = data.date_min.slice(0, 10);
      if (data.date_max) input.max = data.date_max.slice(0, 10);
    }
  } catch {
    /* health check reports connectivity problems */
  }
  updateClearButton();
}

async function loadHealth() {
  try {
    const h = await api("/health");
    els.indexStatus.textContent =
      h.status === "ok"
        ? `${h.indexed_documents} records indexed for ${h.patients} patients`
        : "Search index is empty";
    els.indexStatus.title = `Embeddings: ${h.embedder}. Summaries: ${h.summary_provider}.`;
  } catch {
    els.indexStatus.textContent = "Search service offline";
  }
}

/* ---------------------------------------------------------------------- Search */

function scheduleSearch() {
  clearTimeout(state.debounce);
  state.debounce = setTimeout(runSearch, DEBOUNCE_MS);
}

async function runSearch() {
  clearTimeout(state.debounce);
  const query = els.query.value.trim();
  const filters = currentFilters();
  updateClearButton();

  const filterProblem = validateFilters(filters);
  els.filterError.hidden = !filterProblem;
  els.filterError.textContent = filterProblem;
  if (filterProblem) return;

  syncUrl(query, filters);
  if (query.length < MIN_QUERY) {
    state.controller?.abort();
    renderIntro();
    return;
  }

  state.controller?.abort();
  const controller = new AbortController();
  state.controller = controller;
  state.lastQuery = query;
  renderLoading();

  const params = new URLSearchParams();
  if (filters.type) params.append("resource_type", filters.type);
  if (filters.from) params.set("date_from", filters.from);
  if (filters.to) params.set("date_to", filters.to);
  const qs = params.toString();

  try {
    const data = await api(`/search${qs ? `?${qs}` : ""}`, {
      method: "POST",
      body: { query, top_k: TOP_K },
      signal: controller.signal,
    });
    if (controller !== state.controller) return; // a newer search superseded this one
    renderResults(data, filters);
  } catch (err) {
    if (err.name === "AbortError") return;
    renderError(err);
  } finally {
    if (controller === state.controller) els.body.setAttribute("aria-busy", "false");
  }
}

/* ------------------------------------------------------------------- Rendering */

function setStatus(meta, announcement = meta) {
  els.meta.textContent = meta;
  els.srStatus.textContent = announcement;
}

function renderLoading() {
  els.body.setAttribute("aria-busy", "true");
  els.body.replaceChildren(...Array.from({ length: 3 }, () => els.skeleton.content.cloneNode(true)));
  setStatus("Searching…");
}

function renderIntro() {
  setStatus("");
  els.body.replaceChildren(
    el("div", { class: "rounded-lg border border-dashed border-rule bg-white px-5 py-6" },
      el("p", { class: "text-[15px] font-medium", text: "Describe what you're looking for." }),
      el("p", {
        class: "mt-1 max-w-prose text-sm text-muted",
        text: "Lay terms work: the search maps them to clinical language. Try one of these:",
      }),
      el("ul", { class: "mt-4 flex flex-wrap gap-2", "aria-label": "Example searches" },
        SUGGESTIONS.map((s) =>
          el("li", {},
            el("button", {
              type: "button",
              class: "rounded-full border border-rule bg-paper px-3 py-1.5 text-sm hover:border-accent hover:text-accent",
              text: s,
              onclick: () => {
                els.query.value = s;
                runSearch();
              },
            })))),
    ),
  );
}

function renderError(err) {
  const notReady = err.status === 503;
  setStatus("", `Search failed. ${err.message}`);
  els.body.replaceChildren(
    el("div", { class: "rounded-lg border border-alarm/30 bg-alarm-soft px-5 py-4", role: "alert" },
      el("p", { class: "font-medium text-alarm", text: notReady ? "The search index isn't built yet." : "Search didn't complete." }),
      el("p", {
        class: "mt-1 text-sm",
        text: notReady ? "Run `ehr-intel run-all` on the server, then search again." : err.message,
      }),
      el("button", {
        type: "button",
        class: "mt-3 rounded-md bg-white px-3 py-1.5 text-sm font-medium text-alarm ring-1 ring-alarm/30 hover:bg-alarm-soft",
        text: "Try again",
        onclick: runSearch,
      }),
    ),
  );
}

function renderResults(data, filters) {
  const { results } = data;
  const filtered = Boolean(filters.type || filters.from || filters.to);

  if (results.length === 0) {
    setStatus("No matches", `No records match "${data.query}".`);
    els.body.replaceChildren(
      el("div", { class: "rounded-lg border border-dashed border-rule bg-white px-5 py-6" },
        el("p", { class: "text-[15px] font-medium", text: `No records match "${data.query}".` }),
        el("p", {
          class: "mt-1 text-sm text-muted",
          text: filtered
            ? "The filters may be too narrow. Clear them to search every record type and date."
            : "Try a broader description, a diagnosis, or a test name.",
        }),
        filtered &&
          el("button", {
            type: "button",
            class: "mt-3 rounded-md px-3 py-1.5 text-sm font-medium text-accent ring-1 ring-accent/30 hover:bg-accent-soft",
            text: "Clear filters",
            onclick: clearFilters,
          }),
      ),
    );
    return;
  }

  const plural = results.length === 1 ? "match" : "matches";
  setStatus(
    `${results.length} ${plural} in ${Math.round(data.took_ms)} ms`,
    `${results.length} ${plural} for "${data.query}". Use the arrow keys to move between results.`,
  );
  const top = Math.max(...results.map((r) => r.score), 1e-6);
  const list = el("ol", { class: "divide-y divide-rule rounded-lg border border-rule bg-white", id: "result-list" });
  // Show each patient's AI summary once (on their highest-ranked result) to avoid repetition.
  const seenPatients = new Set();
  results.forEach((r, i) => {
    const firstForPatient = !seenPatients.has(r.mrn);
    seenPatients.add(r.mrn);
    list.append(resultItem(r, i, top, data.query, firstForPatient));
  });
  els.body.replaceChildren(list);
}

function resultItem(r, index, topScore, query, showSummary) {
  const pct = Math.max(4, Math.round((r.score / topScore) * 100));
  const titleId = `result-title-${index}`;
  return el("li", { class: "relative px-4 py-4 hover:bg-paper focus-within:bg-paper sm:px-5" },
    el("div", { class: "flex flex-wrap items-center gap-x-3 gap-y-1" },
      typeBadge(r.resource_type, r.kind),
      el("span", { class: "text-sm text-muted", text: formatDate(r.date) }),
      el("span", { class: "ml-auto flex items-center gap-2 text-xs text-muted" },
        el("span", {
          class: "match-meter",
          role: "img",
          "aria-label": `Relevance ${r.score.toFixed(2)}, ${pct}% of the best match`,
        }, el("span", { style: `width:${pct}%` })),
        el("span", { class: "w-9 text-right font-medium text-ink", "aria-hidden": "true", text: r.score.toFixed(2) }),
      ),
    ),
    el("h3", { class: "mt-2 text-[15px] font-semibold leading-snug", id: titleId },
      el("button", {
        type: "button",
        class: "result-open text-left after:absolute after:inset-0 after:content-[''] focus-visible:outline-none",
        dataset: { mrn: r.mrn, index: String(index) },
        "aria-describedby": `result-desc-${index}`,
        onclick: (e) => openPatient(r.mrn, e.currentTarget),
      },
        el("span", { text: r.patient_name ?? "Unknown patient" }),
        el("span", { class: "ml-2 font-normal text-muted", text: `MRN ${r.mrn}` }),
      ),
    ),
    r.patient_flags?.length > 0 &&
      el("p", { class: "mt-1 inline-flex items-center gap-1.5 rounded bg-caution-soft px-2 py-0.5 text-xs font-medium text-caution" },
        "Identity needs review: see patient details"),
    el("p", { class: "text-sm text-ink/90", text: r.title }),
    el("p", { class: "mt-1.5 max-w-prose text-sm leading-relaxed text-muted", id: `result-desc-${index}` },
      highlight(r.snippet, query)),
    showSummary && r.summary_snippet &&
      el("p", { class: "mt-2 max-w-prose border-l-2 border-ai/40 pl-3 text-sm leading-relaxed" },
        el("span", { class: "font-medium text-ai", text: "AI summary: " }),
        r.summary_snippet,
        r.summary_confidence === "low" &&
          el("span", { class: "ml-2 rounded bg-caution-soft px-1.5 py-0.5 text-xs text-caution", text: "Low confidence" }),
      ),
  );
}

/* --------------------------------------------------------------- Keyboard nav */

function resultButtons() {
  return [...document.querySelectorAll("#result-list .result-open")];
}

function onKeydown(e) {
  if (!els.drawer.classList.contains("hidden")) {
    if (e.key === "Escape") closeDrawer();
    if (e.key === "Tab") trapFocus(e);
    return;
  }
  const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName);
  if (e.key === "/" && !typing) {
    e.preventDefault();
    els.query.focus();
    els.query.select();
    return;
  }
  const buttons = resultButtons();
  if (!buttons.length) return;
  const current = buttons.indexOf(document.activeElement);
  if (e.key === "ArrowDown" && (document.activeElement === els.query || current >= 0)) {
    e.preventDefault();
    buttons[Math.min(current + 1, buttons.length - 1)].focus();
  } else if (e.key === "ArrowUp" && current >= 0) {
    e.preventDefault();
    if (current === 0) els.query.focus();
    else buttons[current - 1].focus();
  } else if ((e.key === "Home" || e.key === "End") && current >= 0) {
    e.preventDefault();
    buttons[e.key === "Home" ? 0 : buttons.length - 1].focus();
  }
}

function trapFocus(e) {
  const focusable = [...els.drawerPanel.querySelectorAll(
    'button, a[href], [tabindex]:not([tabindex="-1"]), summary',
  )].filter((n) => !n.disabled && n.offsetParent !== null);
  if (!focusable.length) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (e.shiftKey && document.activeElement === first) {
    e.preventDefault();
    last.focus();
  } else if (!e.shiftKey && document.activeElement === last) {
    e.preventDefault();
    first.focus();
  }
}

/* --------------------------------------------------------------------- Drawer */

function showDrawer() {
  els.drawer.classList.remove("hidden");
  els.drawer.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
  requestAnimationFrame(() => {
    els.drawerPanel.classList.remove("translate-x-full");
    els.drawerBackdrop.classList.remove("opacity-0");
  });
  els.drawerClose.focus();
}

function closeDrawer() {
  state.patientController?.abort();
  els.drawerPanel.classList.add("translate-x-full");
  els.drawerBackdrop.classList.add("opacity-0");
  els.drawer.setAttribute("aria-hidden", "true");
  document.body.style.overflow = "";
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  setTimeout(() => els.drawer.classList.add("hidden"), reduced ? 0 : 220);
  state.returnFocus?.focus();
}

async function openPatient(mrn, trigger) {
  state.returnFocus = trigger ?? document.activeElement;
  state.patientController?.abort();
  const controller = new AbortController();
  state.patientController = controller;

  els.drawerTitle.textContent = "Loading patient…";
  els.drawerSubtitle.textContent = `MRN ${mrn}`;
  els.drawerBody.setAttribute("aria-busy", "true");
  els.drawerBody.replaceChildren(...Array.from({ length: 3 }, () => els.skeleton.content.cloneNode(true)));
  showDrawer();

  try {
    const p = await api(`/patients/${encodeURIComponent(mrn)}`, { signal: controller.signal });
    renderPatient(p);
  } catch (err) {
    if (err.name === "AbortError") return;
    els.drawerTitle.textContent = "Patient unavailable";
    els.drawerBody.replaceChildren(
      el("p", { class: "text-sm text-alarm", role: "alert", text: err.message }),
    );
  } finally {
    els.drawerBody.setAttribute("aria-busy", "false");
  }
}

function section(title, ...children) {
  const id = `sec-${title.toLowerCase().replace(/\W+/g, "-")}`;
  return el("section", { class: "mt-6 first:mt-0", "aria-labelledby": id },
    el("h3", { id, class: "text-sm font-semibold", text: title }),
    ...children);
}

function renderPatient(p) {
  els.drawerTitle.textContent = p.name ?? `MRN ${p.mrn}`;
  const age = ageFrom(p.birth_date);
  const bits = [`MRN ${p.mrn}`];
  if (p.birth_date) bits.push(`DOB ${formatDate(p.birth_date)}${age !== null ? ` (${age} y)` : ""}`);
  if (p.gender) bits.push(p.gender[0].toUpperCase() + p.gender.slice(1));
  els.drawerSubtitle.textContent = bits.join(", ");

  const s = p.summary;
  const parts = [];

  if (p.identity_flags?.length) {
    parts.push(
      el("div", { class: "mb-5 rounded-lg border border-caution/30 bg-caution-soft px-4 py-3 text-sm", role: "alert" },
        el("p", { class: "font-medium text-caution", text: "Confirm patient identity before acting on these records" }),
        el("ul", { class: "mt-1.5 list-disc space-y-1 pl-5 text-ink/85" },
          p.identity_flags.map((f) => el("li", { text: f }))),
      ),
    );
  }

  if (s) {
    const lowConfidence = s.confidence === "low";
    parts.push(
      el("div", { class: `rounded-lg px-4 py-3 text-sm ${lowConfidence ? "bg-caution-soft" : "bg-ai-soft"}`, role: "note" },
        el("p", { class: "font-medium text-ai", text: `AI-generated summary, ${s.confidence} confidence` }),
        el("p", { class: "mt-1 leading-relaxed text-ink/80", text: s.disclaimer }),
        s.confidence_rationale && el("p", { class: "mt-1 text-xs text-muted", text: s.confidence_rationale }),
      ),
      section("Chief concern", el("p", { class: "mt-1 text-[15px] leading-relaxed", text: s.chief_concern })),
      section("Key diagnoses",
        s.key_diagnoses.length
          ? el("ul", { class: "mt-1 list-disc space-y-1 pl-5 text-[15px]" }, s.key_diagnoses.map((d) => el("li", { text: d })))
          : el("p", { class: "mt-1 text-sm text-muted", text: "None documented." })),
      section("Flagged anomalies",
        s.flagged_anomalies.length
          ? el("ul", { class: "mt-2 space-y-1.5" },
              s.flagged_anomalies.map((a) =>
                el("li", { class: "rounded-md bg-alarm-soft px-3 py-2 text-sm text-alarm", text: a })))
          : el("p", { class: "mt-1 text-sm text-muted", text: "Nothing flagged." })),
      section("Recent imaging and labs",
        el("ul", { class: "mt-2 divide-y divide-rule rounded-lg border border-rule" },
          s.recent_media.map((m) =>
            el("li", { class: "px-3 py-2.5" },
              el("p", { class: "flex flex-wrap items-baseline gap-x-2 text-sm" },
                el("span", { class: "font-medium", text: m.title }),
                el("span", { class: "text-muted", text: `${m.kind}, ${formatDate(m.date)}` })),
              el("p", { class: "mt-0.5 text-sm text-muted", text: m.finding }))))),
    );
    const generated = new Date(s.generated_at).toLocaleString();
    parts.push(el("p", { class: "mt-3 text-xs text-muted",
      text: `Generated ${generated} by ${s.provider} (${s.model}) from ${s.source_resource_count} ${s.source_resource_count === 1 ? "record" : "records"}.` }));
  } else {
    parts.push(el("p", { class: "rounded-lg bg-paper px-4 py-3 text-sm text-muted",
      text: "No AI summary yet. Run `ehr-intel summarize` to generate one." }));
  }

  parts.push(linkedResources(p));
  els.drawerBody.replaceChildren(...parts);
}

function linkedResources(p) {
  const groups = new Map();
  for (const r of p.resources) {
    if (!groups.has(r.resource_type)) groups.set(r.resource_type, []);
    groups.get(r.resource_type).push(r);
  }
  const order = ["DocumentReference", "DiagnosticReport", "Observation", "Encounter"];
  const sorted = [...groups.keys()].sort((a, b) => order.indexOf(a) - order.indexOf(b));

  return section("Linked FHIR resources",
    el("p", { class: "mt-1 text-sm text-muted", text: `${p.resources.length} resources in this patient's bundle.` }),
    ...sorted.map((type, i) =>
      el("details", { class: "group mt-2 rounded-lg border border-rule", open: i < 2 },
        el("summary", { class: "flex cursor-pointer items-center justify-between px-3 py-2 text-sm font-medium" },
          el("span", { text: type }),
          el("span", { class: "text-muted", text: `${groups.get(type).length}` })),
        el("ul", { class: "divide-y divide-rule border-t border-rule" },
          groups.get(type).map((r) =>
            el("li", { class: "flex items-baseline justify-between gap-3 px-3 py-2 text-sm" },
              el("a", {
                href: `${API_BASE}/resources/${encodeURIComponent(r.resource_id)}`,
                target: "_blank",
                rel: "noopener",
                class: "min-w-0 truncate text-accent underline-offset-2 hover:underline",
                title: `Open ${type} as FHIR JSON`,
                text: r.title || r.kind || type,
              }),
              el("span", { class: "shrink-0 text-muted", text: formatDate(r.date) })))))),
    el("a", {
      href: `${API_BASE}/patients/${encodeURIComponent(p.mrn)}/bundle`,
      target: "_blank",
      rel: "noopener",
      class: "mt-4 inline-block text-sm font-medium text-accent underline-offset-2 hover:underline",
      text: "Open full FHIR bundle (JSON)",
    }),
  );
}

/* ----------------------------------------------------------------------- Init */

function clearFilters() {
  els.type.value = "";
  els.from.value = "";
  els.to.value = "";
  runSearch();
  els.query.focus();
}

function init() {
  els.form.addEventListener("submit", (e) => {
    e.preventDefault();
    runSearch();
  });
  els.query.addEventListener("input", scheduleSearch);
  for (const input of [els.type, els.from, els.to]) input.addEventListener("change", runSearch);
  els.clear.addEventListener("click", clearFilters);
  els.drawerClose.addEventListener("click", closeDrawer);
  els.drawerBackdrop.addEventListener("click", closeDrawer);
  document.addEventListener("keydown", onKeydown);

  const selectedType = restoreFromUrl();
  loadHealth();
  loadFilterOptions(selectedType).then(() => {
    if (els.query.value.trim().length >= MIN_QUERY) runSearch();
    else renderIntro();
  });
}

init();
