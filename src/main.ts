import "@fontsource/jetbrains-mono/300.css";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/600.css";
import "@fontsource/jetbrains-mono/700.css";
import "@fontsource/space-grotesk/400.css";
import "@fontsource/space-grotesk/500.css";
import "@fontsource/space-grotesk/600.css";
import "@fontsource/ibm-plex-sans-arabic/400.css";
import "@fontsource/ibm-plex-sans-arabic/600.css";
import "./style.css";

import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open, save, ask as askDialog } from "@tauri-apps/plugin-dialog";
import { marked } from "marked";

// ───────── types (mirror mahdar-core) ─────────
type Status = "transcribing" | "ready" | "failed";
interface Chunk { start_s: number; end_s: number; text: string; flag: string; speaker: number | null; edited: boolean }
interface ChatMsg { role: "user" | "assistant"; text: string; created: string; provider: string }
interface Output { id: string; template_id: string; template_name: string; provider: string; created: string; text: string }
interface Meeting {
  id: string; title: string; created: string; source: string; duration_s: number; language: string;
  status: Status; error: string | null; transcribe_secs: number | null; chunks: Chunk[]; outputs: Output[];
  speakers: Record<string, string>; chat: ChatMsg[]; notes: string;
}
interface MeetingSummary { id: string; title: string; created: string; duration_s: number; status: Status; n_outputs: number }
interface Template { id: string; name: string; prompt: string; builtin: boolean }
type Provider =
  | { kind: "claude_cli"; model: string }
  | { kind: "claude_api"; model: string; effort: string }
  | { kind: "cursor_cli"; model: string }
  | { kind: "local"; base_url: string; model: string };
interface Settings {
  model_path: string; backend: "vulkan" | "cpu" | "auto"; language: string; output_language: string;
  provider: Provider; custom_templates: Template[];
  watch_folder: string | null; glossary: string; speakers: boolean; diarizer_path: string;
  remote_enabled: boolean; remote_port: number; remote_token: string; watchdog: boolean;
  brain_provider: Provider | null;
  default_template: string; auto_output: string; auto_index: boolean;
  local: { temperature: number; context: number; autostart: boolean; timeout_min: number };
  auto_update: boolean;
}
interface UpdateInfo {
  current: string; state: string; version: string; notes: string; date: string; percent: number;
  last_check: string; error: string; waiting_for: string;
}
interface BrainStatus {
  meetings: number; indexed: number; out_of_date: number; untagged: number; busy: boolean;
  model_present: boolean; model: string; model_mb: number; states: Record<string, string>;
}
interface EntityRow { key: string; kind: string; name: string; aliases: string[]; role: string; meetings: number; last: string }
interface ItemRow { kind: string; text: string; owner: string; due: string; ts: number | null }
interface MentionRow { meeting_id: string; title: string; created: string; summary: string; timestamps: number[]; items: ItemRow[] }
interface EntityPage { entity: EntityRow; mentions: MentionRow[] }
interface Citation { n: number; meeting_id: string; title: string; start_s: number; speaker: string | null }
interface AskRow { id: number; question: string; answer: string; citations: Citation[]; provider: string; scope: string; created: string; thread: number }
interface LogLine { time: string; level: "INFO" | "WARN" | "ERROR"; source: string; msg: string }
interface Monitor {
  version: string; pid: number; started: string; uptime_s: number; speech: string; speakers: string;
  writer: string; local_server: boolean; remote: string;
  jobs: { id: string; title: string; stage: string; done: number; total: number }[];
  api_key: string; spend_today: number; spend_month: number; spend_total: number;
  watchdog: { enabled: boolean; running: boolean; state: string; last_check: string; restarts_24h: number; last_restart: string | null };
  log: LogLine[];
}
interface EngineStatus {
  model_path: string; model_present: boolean; model_bytes_expected: number;
  loaded_backend: string | null; ffmpeg: boolean; data_dir: string; diarizer_present: boolean;
}
interface Progress { id: string; stage: string; done: number; total: number }
interface RemoteInfo { enabled: boolean; port: number; urls: [string, string][]; pair_url: string; qr_svg: string }

const AUDIO_EXT = ["m4a", "mp3", "wav", "aac", "ogg", "opus", "flac", "mp4", "mov", "webm", "amr", "caf", "wma", "mkv"];
const LOCAL_URL = "http://127.0.0.1:11434/v1";
const LOCAL_MODEL = "gemma4-it:12b";
let cursorModel = ""; // remembered (in localStorage) while another engine is selected

/** "Cohere Transcribe Arabic · Q5_K_M" from the configured model file. */
function speechModelName() {
  const file = settings.model_path.split(/[\\/]/).pop() ?? "";
  const quant = file.match(/-((?:Q\d\w*)|F16|BF16|F32)\.gguf$/i)?.[1];
  return file.toLowerCase().startsWith("cohere-transcribe-arabic") ? `Cohere Transcribe Arabic${quant ? ` · ${quant}` : ""}` : file || "speech model";
}
const API_NAMES: Record<string, string> = { "claude-haiku-5-5": "Haiku 5.5", "claude-sonnet-5-5": "Sonnet 5.5", "claude-opus-5-5": "Opus 5.5" };
const providerValue = (p: Provider) =>
  p.kind === "local" ? "local" : p.kind === "cursor_cli" ? "cursor" : p.kind === "claude_api" ? `api:${p.model}` : `claude:${p.model}`;
/** The Settings engine choice for a writer: local | api | cursor | claude. */
const engineKind = (p: Provider) => (p.kind === "local" ? "local" : p.kind === "claude_api" ? "api" : p.kind === "cursor_cli" ? "cursor" : "claude");
function providerName(p: Provider = settings.provider) {
  if (p.kind === "local") return `${p.model} (local)`;
  if (p.kind === "claude_api") return `Claude ${API_NAMES[p.model] ?? p.model}${p.effort ? ` · ${p.effort}` : ""} (API)`;
  if (p.kind === "cursor_cli") return `Cursor${p.model ? ` ${p.model}` : ""} (cloud)`;
  return `Claude ${p.model} (cloud)`;
}

// ───────── helpers ─────────
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
const pad = (n: number) => String(n).padStart(2, "0");
const ts = (s: number) => {
  s = Math.max(0, Math.floor(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h}:${pad(m)}:${pad(s % 60)}` : `${pad(m)}:${pad(s % 60)}`;
};
const dateStr = (iso: string) => {
  const d = new Date(iso);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
};
/** Drop the model's non-speech tags, e.g. "@@@ضوضاء" ("noise"). Mirrors asr::clean_text. */
const cleanText = (t: string) => (t.includes("@") ? t.replace(/ضوضاء/g, "").replace(/@/g, "").replace(/\s+/g, " ").trim() : t.trim());
const store = {
  get: (k: string) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};

marked.use({ renderer: { html: ({ text }) => esc(text) } }); // never render raw HTML from model output

let toastTimer = 0;
function toast(msg: string, isError = false) {
  const t = $("toast");
  t.textContent = msg;
  t.className = `toast${isError ? " error" : ""}`;
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => t.classList.add("hidden"), isError ? 8000 : 3000);
}

// ───────── state ─────────
let meetings: MeetingSummary[] = [];
let current: Meeting | null = null;
let templates: Template[] = [];
let settings: Settings;
let status: EngineStatus;
let activeOutputId: string | null = null;
let follow = store.get("follow") === "1";
let mode: "outputs" | "ask" | "notes" = "outputs";
const asking = new Set<string>(); // meeting ids with a question in flight
const spkName = (m: Meeting, n: number) => m.speakers[String(n)] || `Speaker ${n}`;
const progress = new Map<string, Progress>();
const generating = new Map<string, string>(); // meetingId → template name
const genStage = new Map<string, string>(); // meetingId → "starting local model" | "writing"
const retrying = new Map<string, Set<number>>(); // meetingId → chunk indices being re-run

// ───────── views ─────────
function showView(name: string) {
  document.querySelectorAll<HTMLElement>(".step").forEach((s) => s.classList.toggle("active", s.dataset.view === name));
  document.querySelectorAll<HTMLElement>(".view").forEach((v) => v.classList.toggle("active", v.id === `v-${name}`));
  if (name === "settings") renderSettings();
  if (name === "library") refreshLibrary();
  clearTimeout(monTimer);
  if (name === "monitor") loadMonitor();
  if (name === "brain") loadBrain();
}

// ───────── brain ─────────
let brainStatus: BrainStatus | null = null;
let brainEntities: EntityRow[] = [];
let brainEntity: string | null = null; // open entity page / ask scope
let brainShowing: "home" | "entity" | "answer" = "home";
let brainAsking = false;
let brainThread: number | null = null; // open conversation (id of its first question)
let brainRows: AskRow[] = []; // its questions and answers, oldest first
let brainPending = ""; // question being answered in the open conversation
const KIND_LABEL: Record<string, string> = { person: "People", org: "Organisations", initiative: "Initiatives", topic: "Topics" };
const shortTitle = (t: string) => (t.length > 26 ? t.slice(0, 25) + "…" : t);

async function refreshBrainStatus() {
  try { brainStatus = await invoke<BrainStatus>("brain_status"); } catch (e) { toast(String(e), true); }
  renderBrainStatus();
  if (current) renderBrainButton();
}
function brainEngineValue() {
  const p = settings.brain_provider;
  return !p ? "same" : p.kind === "local" ? "local" : p.kind === "claude_api" ? `api:${p.model}` : p.kind === "cursor_cli" ? "cursor" : "same";
}
async function saveBrainEngine() {
  const v = $<HTMLSelectElement>("b-engine").value;
  const local = settings.provider.kind === "local" ? settings.provider : { kind: "local" as const, base_url: LOCAL_URL, model: LOCAL_MODEL };
  const effort = settings.provider.kind === "claude_api" ? settings.provider.effort : "high";
  const cursor = { kind: "cursor_cli" as const, model: settings.provider.kind === "cursor_cli" ? settings.provider.model : (store.get("cursorModel") ?? cursorModel) };
  settings.brain_provider = v === "same" ? null : v === "local" ? local : v === "cursor" ? cursor : { kind: "claude_api", model: v.slice(4), effort };
  try { await invoke("save_settings", { settings }); toast(`Brain engine: ${settings.brain_provider ? providerName(settings.brain_provider) : "same as minutes"}`); } catch (e) { toast(String(e), true); }
}
function renderBrainStatus() {
  const s = brainStatus;
  if (!s) return;
  $<HTMLSelectElement>("b-engine").value = brainEngineValue();
  const p = progress.get("brain-index");
  const parts = [`${s.indexed} of ${s.meetings} meetings indexed`];
  if (s.out_of_date) parts.push(`${s.out_of_date} out of date`);
  if (s.untagged) parts.push(`${s.untagged} not tagged`);
  if (!s.model_present) parts.push(`search model (${s.model_mb} MB) downloads on first index`);
  $("b-status-text").textContent = "// " + (s.busy && p ? `${p.stage}${p.total ? ` · ${p.done + 1} of ${p.total}` : ""}…` : parts.join(" · "));
  const todo = s.meetings - s.indexed + s.out_of_date + s.untagged;
  const btn = $<HTMLButtonElement>("b-index-all");
  btn.disabled = s.busy || todo <= 0;
  btn.textContent = s.busy ? "INDEXING…" : todo > 0 ? `INDEX ALL (${todo})` : "ALL INDEXED";
}
function renderBrainButton() {
  const b = $<HTMLButtonElement>("btn-brain-index");
  const st = brainStatus?.states[current!.id];
  b.classList.toggle("hidden", !st);
  b.disabled = !!brainStatus?.busy;
  b.textContent = st === "indexed" ? "IN BRAIN ✓" : st === "out_of_date" ? "BRAIN: RE-INDEX" : st === "untagged" ? "BRAIN: RE-TAG" : "ADD TO BRAIN";
}
async function indexBrain(ids: string[] | null) {
  try {
    const n = await invoke<number>("brain_index", { ids });
    toast(`Indexing ${n} meeting${n === 1 ? "" : "s"} for the Brain`);
    await refreshBrainStatus();
  } catch (e) { toast(String(e), true); }
}
async function loadBrain() {
  await refreshBrainStatus();
  try { brainEntities = await invoke<EntityRow[]>("brain_entities"); } catch (e) { toast(String(e), true); }
  renderBrainList();
  if (brainShowing === "home") await renderBrainHome();
}
function renderBrainList() {
  const q = $<HTMLInputElement>("b-filter").value.trim().toLowerCase();
  const rows = brainEntities.filter((e) => !q || `${e.name} ${e.aliases.join(" ")} ${e.role}`.toLowerCase().includes(q));
  $("b-entities").innerHTML = rows.length
    ? Object.keys(KIND_LABEL).map((k) => {
        const g = rows.filter((e) => e.kind === k);
        return g.length ? `<div class="b-group">${KIND_LABEL[k]}</div>` + g.map((e) =>
          `<button class="b-ent${e.key === brainEntity && brainShowing === "entity" ? " on" : ""}" data-k="${esc(e.key)}" title="${esc(e.aliases.join(", "))}"><span dir="auto">${esc(e.name)}</span><span class="n">${e.meetings}</span></button>`).join("") : "";
      }).join("")
    : `<div class="field-note">${brainEntities.length ? "// no match" : "// people, organisations and topics appear here after indexing"}</div>`;
  $("b-entities").querySelectorAll<HTMLButtonElement>(".b-ent").forEach((b) => (b.onclick = () => openEntity(b.dataset.k!)));
}
async function renderBrainHome() {
  brainShowing = "home";
  let asks: AskRow[] = [];
  try { asks = await invoke<AskRow[]>("brain_asks"); } catch {}
  // Newest first; one row per conversation, titled by its first question.
  const threads = new Map<number, AskRow[]>();
  for (const a of asks) threads.set(a.thread, [...(threads.get(a.thread) ?? []), a]);
  const rows = [...threads.entries()].slice(0, 20).map(([t, list]) => {
    const first = list[list.length - 1];
    const n = list.length > 1 ? ` <span class="tag">${list.length} questions</span>` : "";
    return `<button class="b-recent" data-t="${t}">${esc(first.question)}${n}${first.scope ? ` <span class="tag">${esc(first.scope)}</span>` : ""}</button>`;
  });
  $("b-main").innerHTML = `<div class="inner">${
    rows.length
      ? `<div class="section-title">Recent conversations</div>` + rows.join("")
      : `<div class="placeholder">// Ask a question above: Mahdar searches every indexed meeting on this laptop, then sends only the best ~20 excerpts to the writer (${esc(brainEngineName())}).<br>// Then keep going: follow up, or ask what it thinks and what it suggests.<br>// Click a person, organisation or topic on the left for its timeline across meetings.</div>`
  }</div>`;
  $("b-main").querySelectorAll<HTMLButtonElement>(".b-recent").forEach((b) => (b.onclick = () => openThread(Number(b.dataset.t))));
}
/** The engine the Brain answers with. */
const brainEngineName = () => providerName(settings.brain_provider ?? settings.provider);
function setScope(key: string | null) {
  brainEntity = key;
  const e = brainEntities.find((x) => x.key === key);
  $("b-scope").classList.toggle("hidden", !e);
  $("b-scope").textContent = e ? `ONLY: ${e.name}` : "";
}
async function openEntity(key: string) {
  let p: EntityPage;
  try { p = await invoke<EntityPage>("brain_entity", { key }); } catch (e) { return toast(String(e), true); }
  brainShowing = "entity";
  setScope(key);
  renderBrainList();
  const e = p.entity;
  $("b-main").innerHTML = `<div class="inner">
    <div class="b-name" dir="auto">${esc(e.name)}</div>
    <div class="b-meta">${esc(KIND_LABEL[e.kind]?.replace(/s$/, "") ?? e.kind)}${e.role ? ` · ${esc(e.role)}` : ""} · ${e.meetings} meeting${e.meetings === 1 ? "" : "s"}${e.aliases.length ? ` · also: <span dir="auto">${esc(e.aliases.join(", "))}</span>` : ""}</div>
    ${p.mentions.map((m) => `<div class="b-tl"><div class="d">${dateStr(m.created)}</div><div>
      <div class="t" dir="auto" data-m="${esc(m.meeting_id)}">${esc(m.title)}</div>
      ${m.summary ? `<div class="s" dir="auto">${esc(m.summary)}</div>` : ""}
      ${m.items.map((it) => `<div class="b-item"><b>${it.kind === "decision" ? "DECISION" : "ACTION"}</b>${esc(it.text)}${it.owner ? ` · ${esc(it.owner)}` : ""}${it.due ? ` · due ${esc(it.due)}` : ""}</div>`).join("")}
      ${m.timestamps.length ? `<div class="ts-row">${m.timestamps.map((t) => `<span class="b-cite" data-m="${esc(m.meeting_id)}" data-t="${t}">${ts(t)}</span>`).join("")}</div>` : ""}
    </div></div>`).join("")}
  </div>`;
  wireBrainLinks();
}
function wireBrainLinks() {
  $("b-main").querySelectorAll<HTMLElement>("[data-m]").forEach((el) => (el.onclick = () => openAt(el.dataset.m!, Number(el.dataset.t ?? 0))));
}
function turnHtml(a: AskRow) {
  const byN = new Map(a.citations.map((c) => [c.n, c]));
  let html = marked.parse(a.answer) as string;
  html = html.replace(/\[#(\d+)\]/g, (_all, n) => {
    const c = byN.get(Number(n));
    return c ? `<span class="b-cite" data-m="${esc(c.meeting_id)}" data-t="${c.start_s}" title="${esc(c.title)}">${esc(shortTitle(c.title))} · ${ts(c.start_s)}</span>` : "";
  });
  return `<div class="b-turn">
    <div class="b-q" dir="auto">${esc(a.question)}</div>
    <div class="b-meta">// ${esc(a.provider)}${a.scope ? ` · only ${esc(a.scope)}` : ""} · ${dateStr(a.created)}</div>
    <div class="md" dir="auto">${html}</div>
    ${a.citations.length ? `<details class="b-srcs"><summary class="section-title">Sources · ${a.citations.length}</summary>${a.citations.map((c) => `<span class="b-src" data-m="${esc(c.meeting_id)}" data-t="${c.start_s}">${esc(c.title)} · ${ts(c.start_s)}${c.speaker ? ` · ${esc(c.speaker)}` : ""}</span>`).join("")}</details>` : ""}
  </div>`;
}
/** The open conversation: every question and answer, then a box to follow up. */
function renderThread() {
  brainShowing = "answer";
  renderBrainList();
  const draft = document.querySelector<HTMLTextAreaElement>("#b-follow")?.value ?? "";
  const pending = brainPending
    ? `<div class="b-turn"><div class="b-q" dir="auto">${esc(brainPending)}</div><div class="placeholder"><span class="spin-dots">// searching your meetings, then asking ${esc(brainEngineName())}…</span></div></div>`
    : "";
  $("b-main").innerHTML = `<div class="inner">
    <div class="b-thread-head"><button class="hbtn" id="b-back">← ALL CONVERSATIONS</button>${brainThread !== null ? `<button class="hbtn danger" id="b-del-thread">DELETE</button>` : ""}</div>
    ${brainRows.map(turnHtml).join("")}${pending}
    ${brainThread !== null ? `<div class="b-follow">
      <textarea id="b-follow" rows="2" dir="auto" placeholder="// follow up: what do you think? any suggestions? who owns that?"></textarea>
      <button id="b-follow-go" class="go">${brainAsking ? "THINKING…" : "ASK"}</button>
    </div>` : ""}
  </div>`;
  wireBrainLinks();
  $("b-back").onclick = () => { brainThread = null; brainRows = []; renderBrainHome(); };
  const del = document.getElementById("b-del-thread");
  if (del) del.onclick = async () => {
    if (brainThread === null || !(await askDialog("Delete this conversation?"))) return;
    try { await invoke("brain_delete_thread", { thread: brainThread }); } catch (e) { return toast(String(e), true); }
    brainThread = null; brainRows = [];
    await renderBrainHome();
  };
  const box = document.querySelector<HTMLTextAreaElement>("#b-follow");
  if (box) {
    box.value = draft;
    box.disabled = brainAsking;
    ($("b-follow-go") as HTMLButtonElement).disabled = brainAsking;
    box.onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); brainAsk(true); } };
    $("b-follow-go").onclick = () => brainAsk(true);
    if (!brainAsking) box.focus();
  }
  $("b-main").scrollTop = $("b-main").scrollHeight;
}
async function openThread(thread: number) {
  try { brainRows = await invoke<AskRow[]>("brain_thread", { thread }); } catch (e) { return toast(String(e), true); }
  brainThread = thread;
  renderThread();
}
/** Ask from the top box (a new conversation) or the follow-up box (the open one). */
async function brainAsk(followUp: boolean) {
  const input = followUp ? document.querySelector<HTMLTextAreaElement>("#b-follow") : $<HTMLInputElement>("b-q");
  const q = input?.value.trim() ?? "";
  if (!q || brainAsking) return;
  const thread = followUp ? brainThread : null;
  if (!followUp) { brainThread = null; brainRows = []; }
  brainAsking = true;
  brainPending = q;
  input!.value = "";
  const btn = $<HTMLButtonElement>("b-ask");
  btn.disabled = true;
  btn.textContent = "THINKING…";
  renderThread();
  let failed = false;
  try {
    const row = await invoke<AskRow>("brain_ask", { question: q, entity: brainEntity, thread });
    brainThread = row.thread;
    brainRows = await invoke<AskRow[]>("brain_thread", { thread: row.thread }).catch(() => [...brainRows, row]);
  } catch (e) {
    toast(String(e), true);
    failed = true;
  }
  brainAsking = false;
  brainPending = "";
  btn.disabled = false;
  btn.textContent = "ASK";
  if (brainShowing !== "answer") return; // moved on to something else; the answer is in Recent
  if (!brainRows.length) {
    await renderBrainHome();
    if (failed) $<HTMLInputElement>("b-q").value = q;
    return;
  }
  renderThread();
  const box = document.querySelector<HTMLTextAreaElement>("#b-follow");
  if (failed && box) box.value = q;
}
/** Open a meeting at a moment: transcript scrolled there, player ready to play from it. */
async function openAt(id: string, t: number) {
  await openMeeting(id);
  for (let i = 0; i < 30 && !audio().src; i++) await new Promise((r) => setTimeout(r, 100));
  if (audio().src) seek(t, false);
  const idx = current?.chunks.findIndex((c) => t >= c.start_s && t < c.end_s) ?? -1;
  const el = document.querySelector<HTMLElement>(`.line[data-i="${idx}"]`);
  if (el) {
    document.querySelectorAll(".line.now").forEach((l) => l.classList.remove("now"));
    el.classList.add("now");
    const box = $("transcript");
    const r = el.getBoundingClientRect(), b = box.getBoundingClientRect();
    box.scrollTo({ top: box.scrollTop + r.top - b.top - box.clientHeight / 3 });
  }
}

// ───────── monitor ─────────
let monTimer = 0;
let monData: Monitor | null = null;
const money = (x: number) => "$" + (x >= 100 ? x.toFixed(0) : x.toFixed(2));
const dur = (s: number) => (s < 3600 ? `${Math.max(1, Math.round(s / 60))} min` : s < 86400 ? `${(s / 3600).toFixed(1)} h` : `${(s / 86400).toFixed(1)} d`);
const ago = (iso: string) => {
  if (!iso) return "never";
  const s = Math.round((Date.now() - new Date(iso).getTime()) / 1000);
  return s < 60 ? `${s} s ago` : s < 3600 ? `${Math.round(s / 60)} min ago` : `${Math.round(s / 3600)} h ago`;
};
async function loadMonitor() {
  clearTimeout(monTimer);
  try { monData = await invoke<Monitor>("monitor_snapshot"); renderMonitor(); } catch (e) { toast(String(e), true); }
  if ($("v-monitor").classList.contains("active")) monTimer = window.setTimeout(loadMonitor, 3000);
}
function renderMonitor() {
  const d = monData;
  if (!d) return;
  $("mon-uptime").textContent = dur(d.uptime_s);
  $("mon-jobs-n").textContent = String(d.jobs.length);
  $("mon-writer").textContent = d.writer.includes("(local)") ? "LOCAL" : "CLOUD";
  $("mon-errors").textContent = String(d.log.filter((l) => l.level === "ERROR").length);
  const spend = d.spend_total > 0 ? ` · spent ${money(d.spend_today)} today, ${money(d.spend_month)} this month, ${money(d.spend_total)} total` : "";
  const w = d.watchdog;
  const items: [string, string, string, string][] = [
    ["green", "Mahdar", `v${d.version} · pid ${d.pid} · started ${dateStr(d.started)}`, "RUNNING"],
    [d.speech === "busy" ? "orange" : "green", "Speech model", d.speech, d.speech === "busy" ? "BUSY" : d.speech.startsWith("loaded") ? "LOADED" : "IDLE"],
    ["green", "Speakers", d.speakers, d.speakers.toUpperCase().split(" ")[0]],
    [d.writer.includes("(local)") ? "green" : "orange", "Minutes writer", d.writer + (d.local_server ? " · local server running" : ""), d.writer.includes("(local)") ? "LOCAL" : "CLOUD"],
    [d.api_key === "none" ? "gray" : "green", "Claude API key", d.api_key === "none" ? "none · Settings → Writing" : d.api_key + spend, d.api_key === "none" ? "NONE" : "SAVED"],
    [d.remote === "off" ? "gray" : d.remote.startsWith("listening") ? "green" : "red", "Phone server", d.remote, d.remote === "off" ? "OFF" : d.remote.startsWith("listening") ? "ON" : "ERROR"],
    [!w.enabled ? "gray" : w.running ? "green" : "red", "Watchdog",
      !w.enabled ? "off · Settings → Watchdog" : `${w.running ? w.state : "not running"} · checked ${ago(w.last_check)}${w.restarts_24h ? ` · ${w.restarts_24h} restart(s) in 24 h` : ""}`,
      !w.enabled ? "OFF" : w.running ? "ON" : "DOWN"],
  ];
  $("mon-status").innerHTML = items
    .map(([dot, name, desc, badge]) => `<div class="sys-item"><span class="sys-dot ${dot}"></span><div class="sys-info"><div class="sys-name">${name}</div><div class="sys-desc" title="${esc(desc)}">${esc(desc)}</div></div><span class="sys-badge">${esc(badge)}</span></div>`)
    .join("");
  $("mon-jobs").innerHTML = d.jobs.length
    ? d.jobs.map((j) => `<div class="sys-item"><span class="sys-dot orange"></span><div class="sys-info"><div class="sys-name" dir="auto">${esc(j.title)}</div><div class="sys-desc">${esc(j.stage)}${j.total ? ` · ${j.done} / ${j.total}` : ""}</div></div><span class="sys-badge">${j.stage.startsWith("failed") ? "FAILED" : "LIVE"}</span></div>`).join("")
    : `<div class="field-note">// idle: nothing transcribing or writing</div>`;
  const f = getFmt("mon-filter");
  const q = $<HTMLInputElement>("mon-search").value.trim().toLowerCase();
  const lines = d.log
    .filter((l) => f === "all" || (f === "warn" ? l.level !== "INFO" : l.level === "ERROR"))
    .filter((l) => !q || `${l.source} ${l.msg}`.toLowerCase().includes(q))
    .reverse();
  $("mon-rows").innerHTML = lines.length
    ? lines.map((l) => `<tr class="lv-${l.level}"><td class="mono">${esc(l.time.slice(0, 19))}</td><td class="lv-${l.level}">${l.level}</td><td class="mono">${esc(l.source)}</td><td class="msg" dir="auto">${esc(l.msg)}</td></tr>`).join("")
    : `<tr class="empty-row"><td colspan="4">// nothing here</td></tr>`;
}

// ───────── updates ─────────
let upd: UpdateInfo | null = null;
function renderUpdate() {
  const u = upd;
  if (!u) return;
  const tag = $<HTMLButtonElement>("upd-tag");
  tag.hidden = u.state !== "ready";
  tag.textContent = `UPDATE v${u.version} · RESTART`;
  const [dot, desc, badge] =
    u.state === "ready" ? ["orange", `v${u.version} downloaded${u.date ? ` (released ${u.date})` : ""} · ${u.waiting_for ? `installs after ${u.waiting_for} finishes` : "restart to install"}`, "READY"]
    : u.state === "downloading" ? ["orange", `downloading v${u.version} · ${u.percent}%`, "DOWNLOADING"]
    : u.state === "checking" ? ["gray", "checking GitHub Releases…", "CHECKING"]
    : u.state === "error" ? ["red", `check failed: ${u.error}`, "ERROR"]
    : u.state === "latest" ? ["green", `up to date · checked ${ago(u.last_check)}`, "LATEST"]
    : ["gray", u.last_check ? `checked ${ago(u.last_check)}` : "not checked yet", "IDLE"];
  $("upd-status").innerHTML = `<div class="sys-item"><span class="sys-dot ${dot}"></span><div class="sys-info"><div class="sys-name">Mahdar v${esc(u.current)}</div><div class="sys-desc">${esc(desc)}</div></div><span class="sys-badge">${badge}</span></div>`;
  $<HTMLButtonElement>("btn-upd-install").disabled = u.state !== "ready";
  $<HTMLButtonElement>("btn-upd-check").disabled = u.state === "checking" || u.state === "downloading";
  const notes = $("upd-notes");
  notes.hidden = !(u.notes && (u.state === "ready" || u.state === "downloading"));
  if (!notes.hidden) notes.innerHTML = `<strong>What's new in v${esc(u.version)}</strong>` + (marked.parse(u.notes) as string);
}
async function installUpdate() {
  try { await invoke("update_install"); } catch (e) { toast(String(e), true); }
}

// ───────── Claude API key ─────────
async function renderApiKey() {
  const k = await invoke<{ masked: string; source: string }>("api_key_status");
  $("api-status").innerHTML = `<div class="sys-item"><span class="sys-dot ${k.masked ? "green" : "gray"}"></span><div class="sys-info"><div class="sys-name">${k.masked ? esc(k.masked) : "No key saved"}</div><div class="sys-desc">${
    k.masked ? (k.source === "Mahdar" ? "saved in Mahdar" : `shared from ${esc(k.source)}`) : "paste a key below"}</div></div><span class="sys-badge">${k.masked ? "SAVED" : "NONE"}</span></div>`;
  $<HTMLButtonElement>("btn-api-test").disabled = !k.masked;
  $<HTMLButtonElement>("btn-api-del").disabled = k.source !== "Mahdar";
}

function applyTheme(theme: string) {
  document.documentElement.dataset.theme = theme;
  $("btn-theme").textContent = theme === "dark" ? "LIGHT" : "DARK";
  store.set("theme", theme);
}
function applyZoom(z: number) {
  z = Math.min(1.4, Math.max(0.8, Math.round(z * 10) / 10));
  document.documentElement.style.setProperty("--zoom", String(z));
  store.set("zoom", String(z));
}

// ───────── library ─────────
async function refreshLibrary() {
  [meetings, status] = await Promise.all([invoke<MeetingSummary[]>("list_meetings"), invoke<EngineStatus>("engine_status")]);
  renderLibrary();
}

function renderLibrary() {
  $("mt-count").textContent = String(meetings.length);
  $("mt-hours").textContent = (meetings.reduce((a, m) => a + m.duration_s, 0) / 3600).toFixed(1);
  $("mt-outputs").textContent = String(meetings.reduce((a, m) => a + m.n_outputs, 0));
  $("mt-engine").textContent = status.loaded_backend ? (status.loaded_backend.toLowerCase().includes("vulkan") ? "GPU" : "CPU") : settings.backend === "cpu" ? "CPU" : "GPU";

  const p = settings.provider;
  const sys: [string, string, string, string][] = [
    [status.model_present ? "green" : "red", "Speech model", status.model_present ? speechModelName() : "not downloaded", status.model_present ? "READY" : "DOWNLOAD"],
    ["green", "Compute", status.loaded_backend ? `loaded on ${status.loaded_backend}` : `${settings.backend.toUpperCase()} · loads on first transcription`, status.loaded_backend ? "ACTIVE" : "IDLE"],
    [status.diarizer_present && settings.speakers ? "green" : "gray", "Speakers",
      !status.diarizer_present ? "speaker model not downloaded · Settings" : settings.speakers ? "Sortformer · up to 4 speakers" : "detection off",
      status.diarizer_present ? (settings.speakers ? "ON" : "OFF") : "GET"],
    [settings.watch_folder ? "green" : "gray", "Watch folder", settings.watch_folder || "off · Settings", settings.watch_folder ? "WATCHING" : "OFF"],
    [status.ffmpeg ? "green" : "orange", "ffmpeg", status.ffmpeg ? "audio import available" : "missing · winget install Gyan.FFmpeg", status.ffmpeg ? "OK" : "MISSING"],
    [p.kind === "local" ? "green" : "orange", "Minutes writer",
      p.kind === "local" ? `${p.model} · local, private` : p.kind === "cursor_cli" ? `${providerName(p)} · via agent -p`
        : p.kind === "claude_api" ? `${providerName(p)} · cloud, your API credits` : `Claude ${p.model} · cloud via claude -p`,
      p.kind === "local" ? "LOCAL" : "CLOUD"],
  ];
  $("sys").innerHTML = sys
    .map(([dot, name, desc, badge]) => `<div class="sys-item"><span class="sys-dot ${dot}"></span><div class="sys-info"><div class="sys-name">${name}</div><div class="sys-desc" title="${esc(desc)}">${esc(desc)}</div></div>${
      badge === "DOWNLOAD" ? `<button class="sys-badge" id="btn-download">${badge}</button>` : `<span class="sys-badge">${badge}</span>`}</div>`)
    .join("");
  $("btn-download")?.addEventListener("click", startDownload);

  const q = $<HTMLInputElement>("search").value.trim().toLowerCase();
  const rows = meetings.filter((m) => !q || m.title.toLowerCase().includes(q));
  $("meeting-rows").innerHTML = rows.length
    ? rows.map((m, i) => {
        const pr = progress.get(m.id);
        const tag = retrying.has(m.id) ? `<span class="tag live">FIXING</span>` : m.status === "transcribing"
          ? `<span class="tag live">${pr && pr.total ? `${Math.round((100 * pr.done) / pr.total)}%` : "QUEUED"}</span>`
          : m.status === "failed" ? `<span class="tag err">FAILED</span>` : `<span class="tag ${m.n_outputs ? "ok" : "ready"}">READY</span>`;
        return `<tr data-id="${m.id}"><td class="num-col">${pad(rows.length - i)}</td><td class="t" dir="auto">${esc(m.title)}</td>
          <td class="mono">${dateStr(m.created)}</td><td class="r">${m.duration_s ? ts(m.duration_s) : "—"}</td>
          <td class="r">${m.n_outputs || "—"}</td><td>${tag}</td><td class="link">OPEN →</td></tr>`;
      }).join("")
    : `<tr class="empty-row"><td colspan="7">${q ? "// no matches" : "// no meetings yet: drop a recording above"}</td></tr>`;
  $("meeting-rows").querySelectorAll<HTMLTableRowElement>("tr[data-id]").forEach((r) => (r.onclick = () => openMeeting(r.dataset.id!)));

  const warn: string[] = [];
  if (!status.model_present) warn.push("// speech model missing: click DOWNLOAD under System (2.4 GB, one time)");
  if (!status.ffmpeg) warn.push("// ffmpeg missing: audio import needs it → winget install Gyan.FFmpeg, then restart Mahdar");
  $("setup-warning").innerHTML = warn.join("<br>");
  $("setup-warning").classList.toggle("hidden", !warn.length);
}

async function importPaths(paths: string[]) {
  const ok = paths.filter((p) => AUDIO_EXT.includes(p.split(".").pop()!.toLowerCase()));
  if (paths.length && !ok.length) return toast("Not an audio/video file", true);
  let last = "";
  for (const path of ok) {
    try { last = await invoke<string>("import_audio", { path }); } catch (e) { toast(String(e), true); }
  }
  if (last) {
    toast(ok.length > 1 ? `${ok.length} recordings queued` : "Transcribing…");
    await openMeeting(last);
  }
}

// Files copied in File Explorer (Ctrl+C), pasted with the button or Ctrl+V.
async function pasteFiles() {
  try {
    const paths = await invoke<string[]>("clipboard_files");
    if (!paths.length) return toast("Nothing to paste: copy a recording in File Explorer first (Ctrl+C)", true);
    await importPaths(paths);
  } catch (e) { toast(String(e), true); }
}

async function pickFiles() {
  const picked = await open({ multiple: true, filters: [{ name: "Audio / video", extensions: AUDIO_EXT }] });
  await importPaths(Array.isArray(picked) ? picked : picked ? [picked] : []);
}

async function startDownload(ev: Event) {
  (ev.target as HTMLButtonElement).disabled = true;
  (ev.target as HTMLButtonElement).textContent = "0%";
  await invoke("download_model");
}

// ───────── recording ─────────
let recTimer = 0;
let recElapsed = 0;
async function startRecording() {
  try { await invoke("start_recording"); } catch (e) { return toast(String(e), true); }
  $<HTMLInputElement>("rec-title").value = $<HTMLInputElement>("rec-title-quick").value;
  recElapsed = 0;
  $("rec-overlay").classList.remove("hidden");
  $("rec-notes").focus();
  recTimer = window.setInterval(async () => {
    const r = await invoke<[number, number] | null>("recording_level");
    if (!r) return;
    recElapsed = r[1];
    $("rec-time").textContent = ts(r[1]);
    $("rec-level").style.width = `${Math.min(100, Math.sqrt(r[0]) * 100)}%`;
  }, 120);
}
async function stopRecording() {
  clearInterval(recTimer);
  $("rec-overlay").classList.add("hidden");
  try {
    const notes = $<HTMLTextAreaElement>("rec-notes");
    const id = await invoke<string>("stop_recording", { title: $<HTMLInputElement>("rec-title").value, notes: notes.value });
    $<HTMLInputElement>("rec-title-quick").value = "";
    notes.value = "";
    await openMeeting(id);
  } catch (e) { toast(String(e), true); }
}

// ───────── meeting ─────────
const audio = () => $<HTMLAudioElement>("audio");

async function openMeeting(id: string) {
  const switching = current?.id !== id;
  if (switching) await flushNotes();
  current = await invoke<Meeting>("get_meeting", { id });
  if (!switching && notesTimer) current.notes = $<HTMLTextAreaElement>("notes-input").value; // typing not saved yet
  if (switching) {
    activeOutputId = current.outputs[0]?.id ?? null;
    $<HTMLInputElement>("t-search").value = "";
    loadAudio();
  } else if (!audio().src && current.duration_s > 0) {
    loadAudio();
  }
  $("no-meeting").classList.add("hidden");
  $("meeting-layout").classList.remove("hidden");
  renderMeeting();
  showView("meeting");
}

function loadAudio() {
  const a = audio();
  a.pause();
  a.removeAttribute("src");
  a.load();
  $("p-msg").textContent = "";
  updatePlayer();
  const m = current;
  if (!m || m.duration_s <= 0) {
    $("p-msg").textContent = m?.status === "transcribing" ? "// preparing audio…" : "";
    return;
  }
  invoke<string>("audio_path", { id: m.id }).then((p) => {
    if (current?.id !== m.id) return;
    a.src = convertFileSrc(p);
    a.load();
  });
}

function renderMeeting() {
  const m = current!;
  $("m-fname").textContent = m.title;
  renderBrainButton();
  const title = $<HTMLInputElement>("m-title");
  if (document.activeElement !== title) title.value = m.title;
  const parts = m.chunks.length;
  $("m-info").textContent = [m.duration_s ? ts(m.duration_s) : "", parts ? `${parts} parts` : "", m.language === "ar" ? "AR+EN" : "EN"].filter(Boolean).join(" · ");
  const flagged = m.chunks.filter((c) => c.flag).length;
  $("m-flags").textContent = flagged ? `${flagged} unclear` : "";
  $("m-flags").title = flagged ? "The model got stuck on these parts (⚠). Text there may be cut short." : "";
  const fix = $<HTMLButtonElement>("btn-fix");
  fix.classList.toggle("hidden", !flagged && !retrying.has(m.id));
  fix.disabled = retrying.has(m.id) || m.status !== "ready";
  fix.textContent = retrying.has(m.id) ? "FIXING…" : `FIX UNCLEAR (${flagged})`;
  $("btn-follow").classList.toggle("on", follow);
  renderTranscript();
  renderControls();
  renderTimeline();
  renderResults();
  renderChat();
  renderNotes();
}

function renderTranscript() {
  const m = current!;
  const q = $<HTMLInputElement>("t-search").value.trim();
  const box = $("transcript");
  if (!m.chunks.length) {
    box.innerHTML = `<div class="t-empty">${m.status === "transcribing" ? "// transcribing: text appears when done" : m.status === "failed" ? "// transcription failed: see the timeline →" : "// no transcript"}</div>`;
    return;
  }
  const hasSpk = m.chunks.some((c) => c.speaker);
  let prevSpk: number | null | undefined = undefined;
  box.classList.toggle("has-spk", hasSpk);
  const html = m.chunks
    .map((c, i) => ({ c, i }))
    // Hide empty (silence/noise) lines unless they're flagged for a re-run.
    .filter(({ c }) => (q ? cleanText(c.text).includes(q) : cleanText(c.text) !== "" || !!c.flag))
    .map(({ c, i }) => {
      const clean = cleanText(c.text);
      let text = clean ? esc(clean) : `<span class="noise">// silence or noise</span>`;
      if (q) text = text.split(esc(q)).join(`<mark>${esc(q)}</mark>`);
      const busy = retrying.get(m.id)?.has(i);
      const redo = m.status === "ready" ? `<button class="redo" data-i="${i}" title="Re-run this part on smaller pieces">↻ RE-RUN</button>` : "";
      // Label only where the speaker changes, like a chat log.
      const same = prevSpk !== undefined && c.speaker === prevSpk;
      prevSpk = c.speaker;
      const spk = !hasSpk ? "" : same ? `<span></span>` : c.speaker
        ? `<button class="spk s${c.speaker}" data-i="${i}" title="Name this speaker or reassign the line">${esc(spkName(m, c.speaker))}</button>`
        : `<button class="spk none" data-i="${i}">—</button>`;
      return `<div class="line${c.flag ? " flagged" : ""}${busy ? " busy" : ""}${c.edited ? " edited" : ""}" data-i="${i}"><button class="stamp" data-t="${c.start_s}">${ts(c.start_s)}</button>${spk}<p dir="auto" title="Double-click to edit">${text}</p>${redo}</div>`;
    })
    .join("");
  box.innerHTML = html || `<div class="t-empty">// no matches</div>`;
  box.querySelectorAll<HTMLButtonElement>(".stamp").forEach((b) => (b.onclick = () => seek(Number(b.dataset.t), true)));
  box.querySelectorAll<HTMLButtonElement>(".redo").forEach((b) => (b.onclick = () => retry([Number(b.dataset.i)])));
  box.querySelectorAll<HTMLButtonElement>(".spk").forEach((b) => (b.onclick = (e) => { e.stopPropagation(); openSpeakerPop(b, Number(b.dataset.i)); }));
  box.querySelectorAll<HTMLElement>(".line p").forEach((p) => (p.ondblclick = () => editLine(p)));
  highlightNow();
}

function renderControls() {
  const m = current!;
  const sel = $<HTMLSelectElement>("o-template");
  const prev = sel.value;
  sel.innerHTML = templates.map((t) => `<option value="${t.id}">${esc(t.name)}${t.builtin ? "" : " · custom"}</option>`).join("");
  if (templates.some((t) => t.id === prev)) sel.value = prev;
  else if (templates.some((t) => t.id === settings.default_template)) sel.value = settings.default_template;
  $<HTMLSelectElement>("o-lang").value = settings.output_language;
  const p = settings.provider;
  $<HTMLSelectElement>("o-provider").value = providerValue(p);
  const busy = generating.has(m.id);
  const ready = m.status === "ready" && m.chunks.length > 0;
  const btn = $<HTMLButtonElement>("btn-generate");
  btn.disabled = !ready || busy;
  btn.textContent = busy ? "WRITING…" : `GENERATE ${(templates.find((t) => t.id === sel.value)?.name ?? "").toUpperCase()}`;
  const note = $("o-note");
  note.className = `field-note${p.kind === "local" ? "" : " warn"}`;
  note.innerHTML = p.kind === "local"
    ? `// private: runs on ${esc(p.model)} at ${esc(p.base_url)}`
    : p.kind === "cursor_cli"
      ? "// cloud: the transcript is sent to Cursor via your Cursor login"
      : p.kind === "claude_api"
        ? "// cloud: the transcript is sent to Anthropic with your API key (billed to your credits; see // 04 Monitor)"
        : "// cloud: the transcript is sent to Anthropic via your Claude Code login";
  if (m.notes.trim()) note.innerHTML += "<br>// your notes are included";
}

function renderTimeline() {
  const m = current!;
  const pr = progress.get(m.id);
  const steps = ["Decode audio", "Detect speakers", "Load speech model", "Transcribe", "Ready"];
  let activeIdx = 4, pct = 100, statusText = "";
  const meta: string[] = ["", "", "", "", ""];
  const stageIdx: Record<string, number> = { decoding: 0, "detecting speakers": 1, "loading model": 2, transcribing: 3 };
  const errIdx = m.status === "failed" ? (m.chunks.length ? 4 : pr ? stageIdx[pr.stage] ?? 0 : m.duration_s ? 2 : 0) : -1;
  const nSpk = new Set(m.chunks.map((c) => c.speaker).filter(Boolean)).size;

  if (m.status === "transcribing") {
    activeIdx = !pr ? 0 : stageIdx[pr.stage] ?? 0;
    pct = pr && pr.total ? (100 * pr.done) / pr.total : activeIdx * 4;
    if (pr && pr.total) meta[3] = `${pr.done} / ${pr.total} parts`;
    if (activeIdx > 1 && !(pr?.stage === "detecting speakers")) meta[1] = settings.speakers && status.diarizer_present ? "" : "off";
    statusText = `// ${steps[activeIdx].toLowerCase()}…`;
  } else if (m.status === "ready") {
    meta[0] = m.duration_s ? ts(m.duration_s) : "";
    meta[1] = nSpk ? `${nSpk} speaker${nSpk > 1 ? "s" : ""}` : "off";
    meta[3] = `${m.chunks.length} parts${m.transcribe_secs ? ` in ${Math.round(m.transcribe_secs)} s` : ""}`;
    const flagged = m.chunks.filter((c) => c.flag).length;
    meta[4] = flagged ? `${flagged} unclear part${flagged > 1 ? "s" : ""} marked ⚠` : "clean";
    statusText = `// transcript ready · ${dateStr(m.created)}`;
  } else {
    pct = 0;
    statusText = "// failed";
  }
  const gen = generating.get(m.id);
  const events = steps.map((label, i) => {
    let cls = "";
    if (errIdx >= 0) cls = i < errIdx ? "done" : i === errIdx ? "error" : "";
    else if (m.status === "ready") cls = "done";
    else cls = i < activeIdx ? "done" : i === activeIdx ? "active" : "";
    const mt = i === errIdx ? esc(m.error ?? "") : esc(meta[i]);
    return `<div class="tl-ev ${cls}"><div class="tl-dot"></div><div><div class="tl-label">${label}</div>${mt ? `<div class="tl-meta">${mt}</div>` : ""}</div></div>`;
  });
  if (retrying.has(m.id)) {
    const n = retrying.get(m.id)!.size;
    const meta = pr?.stage === "retrying" && pr.total ? `${pr.done} / ${pr.total} parts` : `${n} part${n > 1 ? "s" : ""}`;
    events.push(`<div class="tl-ev active"><div class="tl-dot"></div><div><div class="tl-label">Re-run unclear parts</div><div class="tl-meta">${meta} · smaller pieces</div></div></div>`);
    if (pr?.stage === "retrying" && pr.total) pct = (100 * pr.done) / pr.total;
    statusText = "// re-running unclear parts…";
  }
  if (gen) {
    const starting = genStage.get(m.id) === "starting local model";
    events.push(`<div class="tl-ev active"><div class="tl-dot"></div><div><div class="tl-label">${starting ? "Starting local model" : `Writing ${esc(gen)}`}</div><div class="tl-meta">${starting ? "FastFlowLM · first start takes ~35 s" : esc(providerName())}</div></div></div>`);
  }
  $("tl-status").textContent = gen ? `// ${genStage.get(m.id) === "starting local model" ? "starting local model" : `writing ${gen.toLowerCase()}`}…` : statusText;
  $("tl-fill").style.width = `${gen ? 100 : pct}%`;
  $("tl-events").innerHTML = events.join("");
}

// "Minutes of Meeting · 13:39" when a meeting has several outputs of the same kind.
function outLabel(all: Output[], o: Output) {
  if (all.filter((x) => x.template_name === o.template_name).length < 2) return o.template_name;
  const d = new Date(o.created);
  return `${o.template_name} · ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function renderResults() {
  const m = current!;
  $("res-empty").classList.toggle("hidden", m.outputs.length > 0);
  $("res-body").classList.toggle("hidden", !m.outputs.length);
  if (!m.outputs.length) return;
  if (!m.outputs.some((o) => o.id === activeOutputId)) activeOutputId = m.outputs[0].id;
  const o = m.outputs.find((x) => x.id === activeOutputId)!;
  $("res-tabs").innerHTML = m.outputs
    .map((x) => `<button class="rtab${x.id === o.id ? " active" : ""}" data-oid="${x.id}">${esc(outLabel(m.outputs, x))}</button>`)
    .join("");
  $("res-tabs").querySelectorAll<HTMLButtonElement>(".rtab").forEach((b) => (b.onclick = () => { activeOutputId = b.dataset.oid!; renderResults(); }));
  $("res-title").textContent = `✓ ${o.template_name}`;
  $("res-meta").textContent = `// ${dateStr(o.created)} · ${o.provider}`;
  $("res-text").innerHTML = marked.parse(o.text) as string;
}

async function generate() {
  const m = current!;
  const templateId = $<HTMLSelectElement>("o-template").value;
  const t = templates.find((x) => x.id === templateId)!;
  await flushNotes();
  generating.set(m.id, t.name);
  renderControls();
  renderTimeline();
  try {
    const out = await invoke<Output>("generate_output", { id: m.id, templateId });
    if (current?.id === m.id) {
      activeOutputId = out.id;
      current = await invoke<Meeting>("get_meeting", { id: m.id });
    }
  } catch (e) {
    toast(String(e), true);
  } finally {
    generating.delete(m.id);
    genStage.delete(m.id);
    if (current?.id === m.id) renderMeeting();
  }
}

async function saveProviderFromPanel() {
  const v = $<HTMLSelectElement>("o-provider").value;
  const prevLocal = settings.provider.kind === "local" ? settings.provider : null;
  cursorModel = settings.provider.kind === "cursor_cli" ? settings.provider.model : (store.get("cursorModel") ?? cursorModel);
  settings.provider = v === "local"
    ? { kind: "local", base_url: prevLocal?.base_url ?? LOCAL_URL, model: prevLocal?.model ?? LOCAL_MODEL }
    : v === "cursor"
      ? { kind: "cursor_cli", model: cursorModel }
      : v.startsWith("api:")
        ? { kind: "claude_api", model: v.slice(4), effort: settings.provider.kind === "claude_api" ? settings.provider.effort : (store.get("apiEffort") ?? "") }
        : { kind: "claude_cli", model: v.split(":")[1] };
  settings.output_language = $<HTMLSelectElement>("o-lang").value;
  await invoke("save_settings", { settings });
  renderControls();
}

async function retry(indices: number[] | null) {
  const m = current!;
  if (retrying.has(m.id)) return;
  const targets = indices ?? m.chunks.map((c, i) => (c.flag ? i : -1)).filter((i) => i >= 0);
  if (!targets.length) return;
  retrying.set(m.id, new Set(targets));
  renderMeeting();
  try {
    await invoke("retry_chunks", { id: m.id, indices });
  } catch (e) {
    retrying.delete(m.id);
    toast(String(e), true);
    renderMeeting();
  }
}

// ───────── editing & speakers ─────────
function editLine(p: HTMLElement) {
  const m = current;
  if (!m || m.status !== "ready" || p.isContentEditable) return;
  const i = Number(p.closest<HTMLElement>(".line")!.dataset.i);
  p.textContent = cleanText(m.chunks[i].text);
  p.contentEditable = "true";
  p.focus();
  const done = async (saveIt: boolean) => {
    p.onblur = null;
    p.onkeydown = null;
    p.contentEditable = "false";
    const text = (p.textContent ?? "").trim();
    if (saveIt && text !== cleanText(m.chunks[i].text)) {
      try {
        await invoke("update_chunk", { id: m.id, index: i, text });
        current = await invoke<Meeting>("get_meeting", { id: m.id });
        toast("Line updated");
      } catch (e) { toast(String(e), true); }
    }
    if (current?.id === m.id) renderMeeting();
  };
  p.onkeydown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); done(true); }
    if (e.key === "Escape") { e.preventDefault(); done(false); }
  };
  p.onblur = () => done(true);
}

let popLine = -1;
function openSpeakerPop(anchor: HTMLElement, i: number) {
  const m = current!;
  popLine = i;
  const c = m.chunks[i];
  const pop = $("spk-pop");
  const name = $<HTMLInputElement>("spk-name");
  name.value = c.speaker ? m.speakers[String(c.speaker)] ?? "" : "";
  name.disabled = !c.speaker;
  name.placeholder = c.speaker ? `// name for Speaker ${c.speaker}` : "// assign a speaker first";
  const ids = [1, 2, 3, 4];
  $("spk-assign").innerHTML = ids.map((n) => `<button class="spk s${n}${c.speaker === n ? " cur" : ""}" data-n="${n}">${esc(spkName(m, n))}</button>`).join("") + `<button class="spk none" data-n="0">NONE</button>`;
  $("spk-assign").querySelectorAll<HTMLButtonElement>("button").forEach((b) => (b.onclick = async () => {
    const n = Number(b.dataset.n);
    await invoke("set_chunk_speaker", { id: m.id, index: i, speaker: n || null });
    closeSpeakerPop(false);
    await openMeeting(m.id);
  }));
  const r = anchor.getBoundingClientRect();
  pop.style.left = `${Math.min(r.left, window.innerWidth - 250)}px`;
  pop.style.top = `${Math.min(r.bottom + 6, window.innerHeight - 200)}px`;
  pop.classList.remove("hidden");
  if (c.speaker) name.focus();
}
async function closeSpeakerPop(saveName: boolean) {
  const pop = $("spk-pop");
  if (pop.classList.contains("hidden")) return;
  pop.classList.add("hidden");
  const m = current;
  const c = m?.chunks[popLine];
  if (saveName && m && c?.speaker) {
    const name = $<HTMLInputElement>("spk-name").value.trim();
    if (name !== (m.speakers[String(c.speaker)] ?? "")) {
      await invoke("set_speaker_name", { id: m.id, speaker: c.speaker, name });
      await openMeeting(m.id);
      toast(name ? `Speaker ${c.speaker} → ${name}` : `Speaker ${c.speaker} name cleared`);
    }
  }
}

// ───────── transcript / outputs divider ─────────
/** Drag the line between the transcript and the right panel; double-click resets it. */
function initSplit() {
  const layout = $("meeting-layout"), split = $("split");
  const MIN_RIGHT = 300, MIN_LEFT = 360, DEFAULT = 400;
  let width = Number(store.get("right-w")) || DEFAULT;
  const apply = (w: number) => {
    const max = Math.max(MIN_RIGHT, layout.clientWidth - MIN_LEFT);
    width = Math.round(Math.min(Math.max(w, MIN_RIGHT), layout.clientWidth ? max : w));
    layout.style.setProperty("--right-w", `${width}px`);
  };
  apply(width);
  split.onpointerdown = (e) => {
    e.preventDefault();
    split.setPointerCapture(e.pointerId);
    split.classList.add("drag");
    document.body.classList.add("resizing");
    const right = layout.getBoundingClientRect().right;
    split.onpointermove = (ev) => apply(right - ev.clientX);
    split.onpointerup = () => {
      split.onpointermove = split.onpointerup = null;
      split.classList.remove("drag");
      document.body.classList.remove("resizing");
      store.set("right-w", String(width));
    };
  };
  split.ondblclick = () => { apply(DEFAULT); store.set("right-w", String(width)); };
  split.onkeydown = (e) => {
    const step = e.shiftKey ? 80 : 20;
    if (e.key === "ArrowLeft") apply(width + step);
    else if (e.key === "ArrowRight") apply(width - step);
    else return;
    e.preventDefault();
    store.set("right-w", String(width));
  };
  window.addEventListener("resize", () => apply(width));
}

// ───────── ask ─────────
function setMode(next: typeof mode) {
  mode = next;
  document.querySelectorAll<HTMLElement>("#mode-row .fmt-opt").forEach((o) => o.classList.toggle("active", o.dataset.mode === mode));
  $("ask-panel").classList.toggle("hidden", mode !== "ask");
  $("notes-panel").classList.toggle("hidden", mode !== "notes");
  $("timeline-panel").classList.toggle("hidden", mode !== "outputs");
  $("gen-controls").classList.toggle("hidden", mode !== "outputs");
  // Notes keep only the engine row (it writes the fill-ins); Ask also keeps the title.
  document.querySelectorAll<HTMLElement>(".extract-controls > .field").forEach((f, i) => f.classList.toggle("hidden", (mode === "ask" && i === 1) || (mode === "notes" && i <= 1)));
  $("o-note").classList.toggle("hidden", mode === "notes");
  if (mode === "ask") { renderChat(); $("ask-input").focus(); }
  if (mode === "notes") renderNotes();
  if (mode === "outputs" && current) renderControls();
}

function linkTimestamps(html: string) {
  // [12:34], [1:02:03] and lists like [04:15, 04:23] → each time clickable.
  return html.replace(/\[((?:\d{1,2}:\d{2}(?::\d{2})?)(?:\s*[,،–-]\s*\d{1,2}:\d{2}(?::\d{2})?)*)\]/g, (_all, inner: string) => {
    const linked = inner.replace(/\d{1,2}:\d{2}(?::\d{2})?/g, (t) => {
      const p = t.split(":").map(Number);
      const secs = p.length === 3 ? p[0] * 3600 + p[1] * 60 + p[2] : p[0] * 60 + p[1];
      return `<span class="ts-link" data-t="${secs}">${t}</span>`;
    });
    return `[${linked}]`;
  });
}

function renderChat() {
  const m = current;
  if (!m) return;
  const box = $("chat");
  const msgs = m.chat.map((c) => `<div class="msg ${c.role}"><div class="who">${c.role === "user" ? "You" : esc(c.provider || "Answer")}</div><div class="body${c.role === "assistant" ? " md" : ""}" dir="auto">${c.role === "assistant" ? linkTimestamps(marked.parse(c.text) as string) : esc(c.text)}</div></div>`);
  if (asking.has(m.id)) msgs.push(`<div class="msg assistant"><div class="who">${esc(providerName())}</div><div class="body"><span class="spin-dots">// ${genStage.get(m.id) === "starting local model" ? "starting local model (~35 s)" : "reading the transcript"}…</span></div></div>`);
  box.innerHTML = msgs.join("") || `<div class="res-empty">// ask anything about this meeting<br>// answers cite [timestamps]: click one to play that moment</div>`;
  box.querySelectorAll<HTMLElement>(".ts-link").forEach((s) => (s.onclick = () => seek(Number(s.dataset.t), true)));
  box.scrollTop = box.scrollHeight;
  const ready = m.status === "ready" && m.chunks.length > 0;
  $<HTMLButtonElement>("btn-ask").disabled = !ready || asking.has(m.id);
  $<HTMLButtonElement>("btn-ask").textContent = asking.has(m.id) ? "THINKING…" : "ASK";
}

async function ask() {
  const m = current;
  const input = $<HTMLTextAreaElement>("ask-input");
  const question = input.value.trim();
  if (!m || !question || asking.has(m.id)) return;
  await flushNotes();
  asking.add(m.id);
  m.chat.push({ role: "user", text: question, created: new Date().toISOString(), provider: "" });
  input.value = "";
  renderChat();
  try {
    await invoke<ChatMsg>("ask_meeting", { id: m.id, question });
  } catch (e) {
    toast(String(e), true);
    input.value = question;
  } finally {
    asking.delete(m.id);
    genStage.delete(m.id);
    if (current?.id === m.id) {
      current = await invoke<Meeting>("get_meeting", { id: m.id });
      renderChat();
    }
  }
}


// ───────── notes ─────────
let notesTimer = 0;
const peekOpen = new Map<string, boolean>(); // note line → transcript preview shown (default: open for gaps)
const filling = new Set<string>(); // `${meetingId}\n${line}` being filled in
const PEEK_S = 90;

/** Arabic-Indic digits → ASCII, so "دقيقة ٨" reads as minute 8. */
const asciiDigits = (s: string) => s.replace(/[٠-٩۰-۹]/g, (d) => String(d.charCodeAt(0) & 0xf));
const hms = (a: string, b?: string, c?: string) => (c !== undefined ? +a * 3600 + +b! * 60 + +c : b !== undefined ? +a * 60 + +b : +a * 60);

/** The recording time a note line points to, in seconds: [08:12], @8, @8:30, ~8m, min 8, 8 min, دقيقة 8, or a time at the start. */
function noteTime(line: string): number | null {
  const l = asciiDigits(line);
  let m = l.match(/\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]/);
  if (m) return hms(m[1], m[2], m[3]);
  m = l.match(/[@~]\s*(\d{1,3})(?::(\d{2}))?(?::(\d{2}))?/);
  if (m) return hms(m[1], m[2], m[3]);
  m = l.match(/(?:^|[^\p{L}])(?:min(?:ute)?s?\.?|(?:ال)?دقيقة)\s*(\d{1,3})(?::(\d{2}))?/iu);
  if (m) return hms(m[1], m[2]);
  m = l.match(/(?:^|[^\p{L}\d:])(\d{1,3})(?::(\d{2}))?\s*(?:mins?|minutes?)(?![\p{L}])/iu);
  if (m) return hms(m[1], m[2]);
  m = l.match(/^\s*(?:[-*•?]\s*)*(\d{1,2}):(\d{2})(?::(\d{2}))?(?!\d)/);
  if (m) return hms(m[1], m[2], m[3]);
  return null;
}
/** A note that says something was missed or unclear. */
const isGap = (line: string) => /[?؟]|\bmissed\b|didn'?t (?:catch|get|hear)|unclear|فاتني|فاتتني|ما سمعت|مش واضح|مو واضح/i.test(line);
const isFill = (line: string) => line.trimStart().startsWith("↳");

interface Mark { line: string; t: number; gap: boolean; answer: string }
function noteMarks(notes: string, duration: number): Mark[] {
  const lines = notes.split("\n");
  const marks: Mark[] = [];
  lines.forEach((line, i) => {
    if (!line.trim() || isFill(line)) return;
    const t = noteTime(line);
    if (t === null || (duration > 0 && t > duration + 5)) return;
    const next = lines[i + 1] ?? "";
    marks.push({ line, t, gap: isGap(line), answer: isFill(next) ? next.trim().replace(/^↳\s*/, "") : "" });
  });
  return marks;
}

function renderNotes() {
  const m = current;
  if (!m) return;
  $("mode-notes").textContent = m.notes.trim() ? "NOTES ●" : "NOTES";
  if (mode !== "notes") return;
  const box = $<HTMLTextAreaElement>("notes-input");
  if (document.activeElement !== box && !notesTimer && box.value !== m.notes) box.value = m.notes;
  renderMarks();
}

function renderMarks() {
  const m = current!;
  const marks = noteMarks($<HTMLTextAreaElement>("notes-input").value, m.duration_s);
  const ready = m.status === "ready" && m.chunks.length > 0;
  const html = marks.map((k, i) => {
    const open = peekOpen.get(k.line) ?? k.gap;
    const busy = filling.has(`${m.id}\n${k.line}`);
    const lines = open ? m.chunks.filter((c) => c.end_s > k.t - PEEK_S && c.start_s < k.t + PEEK_S && cleanText(c.text)) : [];
    const peek = !open ? "" : `<div class="n-peek">${lines.map((c) => `<div class="n-pline${c.start_s <= k.t && k.t < c.end_s ? " at" : ""}"><button class="stamp" data-t="${c.start_s}">${ts(c.start_s)}</button><span dir="auto">${c.speaker ? `<b>${esc(spkName(m, c.speaker))}:</b> ` : ""}${esc(cleanText(c.text))}</span></div>`).join("") || `<div class="res-empty">// nothing transcribed around ${ts(k.t)}</div>`}</div>`;
    return `<div class="n-mark${k.gap ? " gap" : ""}" data-i="${i}">
      <div class="n-mark-head"><button class="n-ts" data-t="${k.t}" title="Play from here">${ts(k.t)}</button><span class="n-text" dir="auto">${esc(k.line.trim())}</span></div>
      ${k.answer ? `<div class="n-answer md" dir="auto">↳ ${linkTimestamps(marked.parseInline(k.answer) as string)}</div>` : ""}
      <div class="n-acts"><button class="hbtn n-peek-btn">${open ? "HIDE TRANSCRIPT" : "TRANSCRIPT"}</button><button class="hbtn${k.gap && !k.answer ? " accent" : ""} n-fill" ${ready && !busy ? "" : "disabled"}>${busy ? "FILLING IN…" : k.answer ? "FILL IN AGAIN" : "FILL IN"}</button></div>
      ${peek}
    </div>`;
  }).join("");
  const box = $("n-marks");
  box.innerHTML = marks.length ? `<div class="section-title">Time marks · ${marks.length}</div>${html}` : "";
  box.querySelectorAll<HTMLElement>(".stamp, .n-ts, .ts-link").forEach((b) => (b.onclick = () => seek(Number(b.dataset.t), true)));
  box.querySelectorAll<HTMLElement>(".n-mark").forEach((el) => {
    const k = marks[Number(el.dataset.i)];
    el.querySelector<HTMLButtonElement>(".n-peek-btn")!.onclick = () => { peekOpen.set(k.line, !(peekOpen.get(k.line) ?? k.gap)); renderMarks(); };
    el.querySelector<HTMLButtonElement>(".n-fill")!.onclick = () => fillNote(k);
  });
}

function notesInput() {
  clearTimeout(notesTimer);
  $("n-saved").textContent = "";
  notesTimer = window.setTimeout(flushNotes, 800);
  renderMarks();
}

async function flushNotes() {
  if (!notesTimer || !current) return;
  clearTimeout(notesTimer);
  notesTimer = 0;
  const m = current, notes = $<HTMLTextAreaElement>("notes-input").value;
  try {
    await invoke("save_notes", { id: m.id, notes });
    m.notes = notes;
    if (current?.id === m.id) { $("n-saved").textContent = "// saved"; $("mode-notes").textContent = notes.trim() ? "NOTES ●" : "NOTES"; }
  } catch (e) { toast(`Notes not saved: ${e}`, true); }
}

function setNotes(value: string, caret?: number) {
  const box = $<HTMLTextAreaElement>("notes-input");
  const typing = document.activeElement === box, [a, b] = [box.selectionStart, box.selectionEnd];
  box.value = value;
  if (caret !== undefined) { box.focus(); box.setSelectionRange(caret, caret); }
  else if (typing) box.setSelectionRange(a, b);
  notesInput();
}

/** Start a note at the player's position: a new line beginning "[mm:ss] ". */
function stampNote(atCaret: boolean) {
  if (!current) return;
  if (mode !== "notes") setMode("notes");
  const box = $<HTMLTextAreaElement>("notes-input");
  const v = box.value;
  const pos = atCaret && document.activeElement === box ? box.selectionStart : v.length;
  const before = v.slice(0, pos).replace(/[ \t]+$/, "");
  const lineStart = before === "" || before.endsWith("\n");
  const ins = `${lineStart ? "" : "\n"}[${ts(audio().currentTime || 0)}] `;
  setNotes(before + ins + v.slice(pos), before.length + ins.length);
}

async function fillNote(k: Mark) {
  const m = current;
  const key = `${m?.id}\n${k.line}`;
  if (!m || filling.has(key)) return;
  await flushNotes();
  filling.add(key);
  renderMarks();
  try {
    const answer = await invoke<string>("fill_note", { id: m.id, at: k.t, note: k.line.trim() });
    const fill = `    ↳ ${answer.replace(/\s*\n+\s*/g, " ")}`;
    // Put the answer under its note (replacing an earlier one); the notes may have changed meanwhile.
    const base = current?.id === m.id ? $<HTMLTextAreaElement>("notes-input").value : m.notes;
    const lines = base.split("\n");
    const at = lines.indexOf(k.line);
    if (at < 0) lines.push(k.line, fill);
    else if (isFill(lines[at + 1] ?? "")) lines[at + 1] = fill;
    else lines.splice(at + 1, 0, fill);
    if (current?.id === m.id) setNotes(lines.join("\n"));
    else await invoke("save_notes", { id: m.id, notes: lines.join("\n") });
  } catch (e) {
    toast(String(e), true);
  } finally {
    filling.delete(key);
    if (current?.id === m.id && mode === "notes") renderMarks();
  }
}

async function importNotes() {
  if (!current) return;
  const path = await open({ multiple: false, filters: [{ name: "Notes", extensions: ["txt", "md", "docx"] }] });
  if (typeof path !== "string") return;
  try {
    const text = await invoke<string>("read_notes_file", { path });
    if (!text) return toast("That file has no text", true);
    const v = $<HTMLTextAreaElement>("notes-input").value.trimEnd();
    setNotes(v ? `${v}\n\n${text}` : text);
    toast("Notes added");
  } catch (e) { toast(String(e), true); }
}

/** While recording, each new line of notes starts with the recording time. */
function stampRecNotes(e: InputEvent) {
  const box = e.target as HTMLTextAreaElement;
  if (e.inputType !== "insertText" || !e.data || e.isComposing) return;
  const { selectionStart: a, selectionEnd: b, value: v } = box;
  const lineStart = v.lastIndexOf("\n", a - 1) + 1;
  const lineEnd = v.indexOf("\n", b);
  if (v.slice(lineStart, lineEnd < 0 ? v.length : lineEnd).trim()) return;
  e.preventDefault();
  box.setRangeText(`[${ts(recElapsed)}] ${e.data}`, lineStart, b, "end");
}

// ───────── player ─────────
function seek(t: number, play = false) {
  const a = audio();
  if (!a.src) return toast("Audio not available yet", true);
  a.currentTime = t;
  if (play) a.play().catch((e) => toast(`Can't play audio: ${e}`, true));
}

function updatePlayer() {
  const a = audio();
  const d = isFinite(a.duration) ? a.duration : current?.duration_s ?? 0;
  $("p-cur").textContent = ts(a.currentTime || 0);
  $("p-dur").textContent = ts(d);
  $("p-fill").style.width = d ? `${(100 * (a.currentTime || 0)) / d}%` : "0";
  $("p-play").textContent = a.paused ? "▶" : "❚❚";
  ($("p-play") as HTMLButtonElement).disabled = !a.src;
}

let lastNow = -1;
function highlightNow() {
  const m = current;
  if (!m) return;
  const t = audio().currentTime;
  const idx = audio().src && t > 0 ? m.chunks.findIndex((c) => t >= c.start_s && t < c.end_s) : -1;
  if (idx === lastNow && document.querySelector(".line.now")) return;
  lastNow = idx;
  document.querySelectorAll(".line.now").forEach((l) => l.classList.remove("now"));
  const el = document.querySelector<HTMLElement>(`.line[data-i="${idx}"]`);
  if (el) {
    el.classList.add("now");
    if (follow && !audio().paused) {
      // Scroll only the transcript pane; scrollIntoView would also shift the fixed-layout ancestors.
      const box = $("transcript");
      const r = el.getBoundingClientRect(), b = box.getBoundingClientRect();
      box.scrollTo({ top: box.scrollTop + r.top - b.top - box.clientHeight / 2 + r.height / 2, behavior: "smooth" });
    }
  }
}

// ───────── settings ─────────
async function renderSettings() {
  [settings, status] = await Promise.all([invoke<Settings>("get_settings"), invoke<EngineStatus>("engine_status")]);
  $("model-status").innerHTML = `<div class="sys-item"><span class="sys-dot ${status.model_present ? "green" : "red"}"></span>
    <div class="sys-info"><div class="sys-name">${esc(speechModelName())}</div><div class="sys-desc">${status.model_present ? (status.loaded_backend ? `loaded on ${esc(status.loaded_backend)}` : "ready") : "not downloaded (2.4 GB)"}</div></div>
    ${status.model_present ? `<span class="sys-badge">READY</span>` : `<button class="sys-badge" id="btn-download-s">DOWNLOAD</button>`}</div>`;
  $("btn-download-s")?.addEventListener("click", startDownload);
  $("diar-status").innerHTML = `<div class="sys-item"><span class="sys-dot ${status.diarizer_present ? "green" : "red"}"></span>
    <div class="sys-info"><div class="sys-name">Sortformer 4spk v2.1 · F16</div><div class="sys-desc">${status.diarizer_present ? "ready" : "not downloaded (237 MB)"}</div></div>
    ${status.diarizer_present ? `<span class="sys-badge">READY</span>` : `<button class="sys-badge" id="btn-download-diar">DOWNLOAD</button>`}</div>`;
  $("btn-download-diar")?.addEventListener("click", async (ev) => {
    (ev.target as HTMLButtonElement).disabled = true;
    (ev.target as HTMLButtonElement).textContent = "0%";
    await invoke("download_model", { which: "diarizer" });
  });
  setFmt("s-speakers", settings.speakers ? "on" : "off");
  setFmt("s-remote", settings.remote_enabled ? "on" : "off");
  setFmt("s-watchdog", settings.watchdog ? "on" : "off");
  setFmt("s-effort", settings.provider.kind === "claude_api" ? settings.provider.effort : (store.get("apiEffort") ?? ""));
  await renderApiKey();
  $<HTMLInputElement>("s-remote-port").value = String(settings.remote_port || 8787);
  await renderRemote();
  $<HTMLInputElement>("s-watch").value = settings.watch_folder ?? "";
  $<HTMLTextAreaElement>("s-glossary").value = settings.glossary ?? "";
  $<HTMLInputElement>("s-model").value = settings.model_path;
  setFmt("s-backend", settings.backend);
  setFmt("s-language", settings.language);
  const p = settings.provider;
  $<HTMLInputElement>("s-local-url").value = p.kind === "local" ? p.base_url : LOCAL_URL;
  $<HTMLInputElement>("s-local-model").value = p.kind === "local" ? p.model : LOCAL_MODEL;
  cursorModel = p.kind === "cursor_cli" ? p.model : (store.get("cursorModel") ?? cursorModel);
  $<HTMLInputElement>("s-cursor-model").value = cursorModel;
  if (p.kind !== "local") {
    $<HTMLInputElement>("s-local-url").value = store.get("localUrl") ?? LOCAL_URL;
    $<HTMLInputElement>("s-local-model").value = store.get("localModel") ?? LOCAL_MODEL;
  }
  setFmt("s-engine", engineKind(p));
  setFmt("s-api-model", p.kind === "claude_api" ? p.model : (store.get("apiModel") ?? "claude-sonnet-5-5"));
  setFmt("s-claude-model", p.kind === "claude_cli" ? p.model : (store.get("claudeModel") ?? "haiku"));
  $<HTMLSelectElement>("s-outlang").value = settings.output_language;
  const opts = templates.map((t) => `<option value="${t.id}">${esc(t.name)}${t.builtin ? "" : " · custom"}</option>`).join("");
  $("s-default-template").innerHTML = opts;
  $<HTMLSelectElement>("s-default-template").value = templates.some((t) => t.id === settings.default_template) ? settings.default_template : templates[0]?.id ?? "";
  $("s-auto-output").innerHTML = `<option value="">Nothing</option>${opts}`;
  $<HTMLSelectElement>("s-auto-output").value = templates.some((t) => t.id === settings.auto_output) ? settings.auto_output : "";
  setFmt("s-auto-index", settings.auto_index ? "on" : "off");
  const bp = settings.brain_provider;
  setFmt("s-brain-engine", bp ? engineKind(bp) : "same");
  $<HTMLInputElement>("s-local-temp").value = String(settings.local.temperature);
  $<HTMLInputElement>("s-local-ctx").value = String(settings.local.context);
  $<HTMLInputElement>("s-local-timeout").value = String(settings.local.timeout_min);
  setFmt("s-local-autostart", settings.local.autostart ? "on" : "off");
  setFmt("s-auto-update", settings.auto_update ? "on" : "off");
  upd = await invoke<UpdateInfo>("update_status");
  renderUpdate();
  renderCustomTemplates();
  $("s-data-dir").textContent = `// data: ${status.data_dir}`;
  markSettingsDirty(false);
}

function markSettingsDirty(dirty = true) {
  const n = $("s-dirty");
  n.textContent = dirty ? "// unsaved changes" : "// changes apply when saved";
  n.classList.toggle("dirty", dirty);
}

/** Settings side index: jump to a group, and follow the scroll position. */
function initSettingsNav() {
  const box = document.querySelector<HTMLElement>("#v-settings .settings")!;
  const links = [...document.querySelectorAll<HTMLAnchorElement>("#s-nav a")];
  const mark = (id: string) => links.forEach((a) => a.classList.toggle("on", a.dataset.to === id));
  // Scroll only the settings pane (scrollIntoView would also shift the window).
  links.forEach((a) => (a.onclick = () => { box.scrollTo({ top: $(a.dataset.to!).offsetTop - 32, behavior: "smooth" }); mark(a.dataset.to!); }));
  box.addEventListener("scroll", () => {
    const top = box.getBoundingClientRect().top + 40;
    let cur = links[0].dataset.to!;
    for (const a of links) if ($(a.dataset.to!).getBoundingClientRect().top <= top) cur = a.dataset.to!;
    if (box.scrollTop + box.clientHeight >= box.scrollHeight - 4) cur = links[links.length - 1].dataset.to!;
    mark(cur);
  });
  box.addEventListener("input", () => markSettingsDirty());
  box.addEventListener("click", (e) => { if ((e.target as HTMLElement).classList.contains("fmt-opt")) markSettingsDirty(); });
}

function setFmt(id: string, v: string) {
  $(id).querySelectorAll<HTMLElement>(".fmt-opt").forEach((o) => o.classList.toggle("active", o.dataset.v === v));
}
const getFmt = (id: string) => $(id).querySelector<HTMLElement>(".fmt-opt.active")?.dataset.v ?? "";

function renderCustomTemplates() {
  const box = $("custom-templates");
  box.innerHTML = settings.custom_templates
    .map((t) => `<div class="ct"><div><div class="ct-name">${esc(t.name)}</div><div class="ct-desc">${esc(t.prompt)}</div></div><button class="hbtn danger" data-id="${t.id}">REMOVE</button></div>`)
    .join("");
  box.querySelectorAll<HTMLButtonElement>("button").forEach((b) => (b.onclick = () => {
    settings.custom_templates = settings.custom_templates.filter((t) => t.id !== b.dataset.id);
    renderCustomTemplates();
    markSettingsDirty();
  }));
}

async function saveSettings() {
  settings.model_path = $<HTMLInputElement>("s-model").value.trim();
  settings.backend = getFmt("s-backend") as Settings["backend"];
  settings.language = getFmt("s-language");
  settings.speakers = getFmt("s-speakers") === "on";
  settings.remote_enabled = getFmt("s-remote") === "on";
  settings.watchdog = getFmt("s-watchdog") === "on";
  settings.remote_port = Number($<HTMLInputElement>("s-remote-port").value) || 8787;
  settings.watch_folder = $<HTMLInputElement>("s-watch").value.trim() || null;
  settings.glossary = $<HTMLTextAreaElement>("s-glossary").value;
  const localUrl = $<HTMLInputElement>("s-local-url").value.trim() || LOCAL_URL;
  const localModel = $<HTMLInputElement>("s-local-model").value.trim() || LOCAL_MODEL;
  store.set("localUrl", localUrl);
  store.set("localModel", localModel);
  cursorModel = $<HTMLInputElement>("s-cursor-model").value.trim();
  store.set("cursorModel", cursorModel);
  const local = { kind: "local" as const, base_url: localUrl, model: localModel };
  const effort = getFmt("s-effort");
  const apiModel = getFmt("s-api-model") || "claude-sonnet-5-5";
  const claudeModel = getFmt("s-claude-model") || "haiku";
  store.set("apiEffort", effort);
  store.set("apiModel", apiModel);
  store.set("claudeModel", claudeModel);
  const engines: Record<string, Provider> = {
    local,
    cursor: { kind: "cursor_cli", model: cursorModel },
    api: { kind: "claude_api", model: apiModel, effort },
    claude: { kind: "claude_cli", model: claudeModel },
  };
  settings.provider = engines[getFmt("s-engine")] ?? local;
  settings.brain_provider = engines[getFmt("s-brain-engine")] ?? null;
  settings.output_language = $<HTMLSelectElement>("s-outlang").value;
  settings.default_template = $<HTMLSelectElement>("s-default-template").value;
  settings.auto_output = $<HTMLSelectElement>("s-auto-output").value;
  settings.auto_index = getFmt("s-auto-index") === "on";
  settings.auto_update = getFmt("s-auto-update") !== "off";
  const num = (id: string, def: number, lo: number, hi: number) => {
    const v = Number($<HTMLInputElement>(id).value);
    return Number.isFinite(v) && $<HTMLInputElement>(id).value.trim() !== "" ? Math.min(hi, Math.max(lo, v)) : def;
  };
  settings.local = {
    temperature: num("s-local-temp", 0.2, 0, 1.5),
    context: Math.round(num("s-local-ctx", 65536, 4096, 262144)),
    timeout_min: Math.round(num("s-local-timeout", 30, 1, 240)),
    autostart: getFmt("s-local-autostart") !== "off",
  };
  try {
    await invoke("save_settings", { settings });
    templates = await invoke<Template[]>("list_templates");
    toast("Settings saved");
    await renderSettings();
  } catch (e) { toast(String(e), true); }
}

async function renderRemote() {
  const r = await invoke<RemoteInfo>("remote_info");
  const label: Record<string, string> = { https: "Tailscale · HTTPS (recording works)", tailscale: "Tailscale", wifi: "Same Wi‑Fi", lan: "Other network adapter" };
  $("remote-urls").innerHTML = r.enabled
    ? (r.urls.length
        ? r.urls.map(([k, u]) => `<div class="sys-item"><span class="sys-dot ${k === "https" || k === "tailscale" ? "green" : k === "wifi" ? "orange" : "gray"}"></span><div class="sys-info"><div class="sys-name">${esc(label[k] ?? k)}</div><div class="sys-desc sel">${esc(u)}</div></div></div>`).join("")
        : `<div class="field-note warn">// no network address found: connect to Wi‑Fi or log in to Tailscale</div>`)
    : `<div class="field-note">// off: turn ON and press SAVE SETTINGS</div>`;
  $("remote-qr").innerHTML = r.enabled && r.qr_svg ? r.qr_svg : `<div class="off">// QR code appears when phone access is ON</div>`;
}

// ───────── wiring ─────────
function wireResultButtons() {
  $("res-copy").onclick = async () => {
    const o = current?.outputs.find((x) => x.id === activeOutputId);
    if (o) { await navigator.clipboard.writeText(o.text); toast("Copied"); }
  };
  $("res-save").onclick = async () => {
    const m = current, o = m?.outputs.find((x) => x.id === activeOutputId);
    if (!m || !o) return;
    const path = await save({ defaultPath: `${m.title} - ${o.template_name}.md`, filters: [{ name: "Markdown", extensions: ["md"] }] });
    if (path) { await invoke("save_text_file", { path, text: `# ${m.title}: ${o.template_name}\n\n${o.text}\n` }); toast("Saved"); }
  };
  $("res-word").onclick = async () => {
    const m = current, o = m?.outputs.find((x) => x.id === activeOutputId);
    if (!m || !o) return;
    const path = await save({ defaultPath: `${m.title} - ${o.template_name}.docx`, filters: [{ name: "Word", extensions: ["docx"] }] });
    if (!path) return;
    try { await invoke("export_docx", { id: m.id, outputId: o.id, path }); toast("Saved Word document"); } catch (e) { toast(String(e), true); }
  };
  $("res-pdf").onclick = () => {
    const m = current, o = m?.outputs.find((x) => x.id === activeOutputId);
    if (!m || !o) return;
    $("print-area").innerHTML = `<h1 class="pt" dir="auto">${esc(m.title)}</h1><div class="ps">${esc(o.template_name)} · ${dateStr(o.created)}</div><div class="md" dir="auto">${marked.parse(o.text) as string}</div>`;
    toast("Choose “Save as PDF” as the printer");
    setTimeout(() => window.print(), 50);
  };
  $("res-del").onclick = async () => {
    const m = current, o = m?.outputs.find((x) => x.id === activeOutputId);
    if (!m || !o || !(await askDialog(`Delete this ${o.template_name}?`, { kind: "warning" }))) return;
    await invoke("delete_output", { id: m.id, outputId: o.id });
    activeOutputId = null;
    await openMeeting(m.id);
  };
}

async function init() {
  applyTheme(store.get("theme") ?? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  applyZoom(Number(store.get("zoom") ?? 1));
  [settings, templates] = await Promise.all([invoke<Settings>("get_settings"), invoke<Template[]>("list_templates")]);
  await refreshLibrary();

  // top bar
  document.querySelectorAll<HTMLElement>(".step").forEach((s) => (s.onclick = () => showView(s.dataset.view!)));
  document.querySelectorAll<HTMLElement>("[data-view-link]").forEach((a) => (a.onclick = () => showView(a.dataset.viewLink!)));
  $("btn-theme").onclick = () => applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  $("zoom-in").onclick = () => applyZoom(Number(store.get("zoom") ?? 1) + 0.1);
  $("zoom-out").onclick = () => applyZoom(Number(store.get("zoom") ?? 1) - 0.1);

  // library
  $("drop").onclick = pickFiles;
  $("btn-paste").onclick = (e) => { e.stopPropagation(); pasteFiles(); };
  document.addEventListener("keydown", (e) => {
    const typing = ["INPUT", "TEXTAREA", "SELECT"].includes((e.target as HTMLElement).tagName) || (e.target as HTMLElement).isContentEditable;
    if (e.ctrlKey && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "v" && !typing && $("v-library").classList.contains("active")) {
      e.preventDefault();
      pasteFiles();
    }
  });
  $("btn-record").onclick = startRecording;
  $("btn-stop").onclick = stopRecording;
  $("search").oninput = renderLibrary;
  const drop = $("drop");
  await getCurrentWebview().onDragDropEvent((ev) => {
    const t = ev.payload.type;
    drop.classList.toggle("over", t === "enter" || t === "over");
    if (t === "drop") importPaths(ev.payload.paths);
  });

  // meeting
  $("t-search").oninput = () => current && renderTranscript();
  $("o-template").onchange = renderControls;
  $("o-provider").onchange = saveProviderFromPanel;
  $("o-lang").onchange = saveProviderFromPanel;
  $("btn-generate").onclick = generate;
  $("btn-fix").onclick = () => retry(null);
  document.querySelectorAll<HTMLElement>("#mode-row .fmt-opt").forEach((o) => (o.onclick = () => setMode(o.dataset.mode as typeof mode)));
  $("notes-input").oninput = notesInput;
  $("notes-input").onblur = flushNotes;
  $("n-stamp").onclick = () => stampNote(true);
  $("n-import").onclick = importNotes;
  $("rec-notes").addEventListener("beforeinput", (e) => stampRecNotes(e as InputEvent));
  window.addEventListener("beforeunload", () => { flushNotes(); });
  $("btn-ask").onclick = ask;
  $("ask-input").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } };
  $("btn-clear-chat").onclick = async () => {
    if (!current || !current.chat.length || !(await askDialog("Clear this conversation?"))) return;
    await invoke("clear_chat", { id: current.id });
    current = await invoke<Meeting>("get_meeting", { id: current.id });
    renderChat();
  };
  $("btn-export-transcript").onclick = async () => {
    const m = current;
    if (!m || !m.chunks.length) return;
    const path = await save({ defaultPath: `${m.title} - Transcript.docx`, filters: [{ name: "Word", extensions: ["docx"] }] });
    if (!path) return;
    try { await invoke("export_docx", { id: m.id, outputId: null, path }); toast("Saved transcript"); } catch (e) { toast(String(e), true); }
  };
  $("spk-name").onkeydown = (e) => {
    if (e.key === "Enter") { e.preventDefault(); closeSpeakerPop(true); }
    if (e.key === "Escape") closeSpeakerPop(false);
  };
  document.addEventListener("mousedown", (e) => {
    if (!$("spk-pop").contains(e.target as Node) && !(e.target as HTMLElement).classList.contains("spk")) closeSpeakerPop(true);
  });
  window.addEventListener("afterprint", () => ($("print-area").innerHTML = ""));
  $("btn-watch-pick").onclick = async () => {
    const dir = await open({ directory: true, multiple: false, defaultPath: $<HTMLInputElement>("s-watch").value || undefined });
    if (typeof dir === "string") { $<HTMLInputElement>("s-watch").value = dir; markSettingsDirty(); }
  };
  $("btn-watch-clear").onclick = () => { $<HTMLInputElement>("s-watch").value = ""; markSettingsDirty(); };
  $("btn-remote-token").onclick = async () => {
    if (!(await askDialog("Make a new pairing code? Phones paired before will need to scan the QR code again.", { kind: "warning" }))) return;
    await invoke("remote_new_token");
    await renderSettings();
    toast("New pairing code: scan again on the phone");
  };
  $("btn-remote-https").onclick = async () => {
    try {
      const url = await invoke<string>("remote_setup_https");
      toast(`HTTPS ready: ${url}`);
      await renderRemote();
    } catch (e) { toast(String(e), true); }
  };
  $("btn-follow").onclick = () => { follow = !follow; store.set("follow", follow ? "1" : "0"); $("btn-follow").classList.toggle("on", follow); };
  const title = $<HTMLInputElement>("m-title");
  const saveTitle = async () => {
    const t = title.value.trim();
    if (current && t && t !== current.title) {
      await invoke("rename_meeting", { id: current.id, title: t });
      current.title = t;
      $("m-fname").textContent = t;
    }
  };
  title.onchange = saveTitle;
  title.onkeydown = (e) => { if (e.key === "Enter") title.blur(); };
  $("btn-delete").onclick = async () => {
    if (!current || !(await askDialog(`Delete "${current.title}" and its recording?`, { kind: "warning" }))) return;
    audio().pause();
    audio().removeAttribute("src");
    await invoke("delete_meeting", { id: current.id });
    current = null;
    $("meeting-layout").classList.add("hidden");
    $("no-meeting").classList.remove("hidden");
    showView("library");
  };
  $("btn-retranscribe").onclick = async () => {
    if (!current || !(await askDialog("Transcribe this meeting again with the current settings? The existing transcript is replaced.", { kind: "warning" }))) return;
    try { await invoke("retranscribe", { id: current.id }); await openMeeting(current.id); } catch (e) { toast(String(e), true); }
  };
  wireResultButtons();

  // player
  const a = audio();
  $("p-play").onclick = () => (a.paused ? a.play().catch((e) => toast(`Can't play audio: ${e}`, true)) : a.pause());
  $("p-track").onclick = (e) => {
    const r = $("p-track").getBoundingClientRect();
    const d = isFinite(a.duration) ? a.duration : current?.duration_s ?? 0;
    seek(((e.clientX - r.left) / r.width) * d);
  };
  const speeds = [1, 1.25, 1.5, 2];
  $("p-speed").onclick = () => {
    a.playbackRate = speeds[(speeds.indexOf(a.playbackRate) + 1) % speeds.length];
    $("p-speed").textContent = `${a.playbackRate}×`;
  };
  ["timeupdate", "play", "pause", "loadedmetadata", "emptied"].forEach((ev) => a.addEventListener(ev, () => { updatePlayer(); highlightNow(); }));
  a.addEventListener("error", () => { if (a.src) $("p-msg").textContent = "// audio could not be loaded"; });
  a.addEventListener("loadedmetadata", () => ($("p-msg").textContent = ""));
  document.addEventListener("keydown", (e) => {
    const typing = ["INPUT", "TEXTAREA", "SELECT"].includes((e.target as HTMLElement).tagName);
    if (e.code === "Space" && !typing && $("v-meeting").classList.contains("active") && a.src) {
      e.preventDefault();
      a.paused ? a.play() : a.pause();
    }
    if (e.code === "KeyM" && !e.ctrlKey && !e.metaKey && !e.altKey && !typing && $("v-meeting").classList.contains("active") && current) {
      e.preventDefault();
      stampNote(false);
    }
  });

  // settings
  document.querySelectorAll<HTMLElement>(".fmt-row").forEach((row) =>
    row.querySelectorAll<HTMLElement>(".fmt-opt").forEach((o) => (o.onclick = () => setFmt(row.id, o.dataset.v!))));
  $("btn-save-settings").onclick = saveSettings;
  $("b-engine").onchange = saveBrainEngine;
  $("btn-api-save").onclick = async () => {
    const input = $<HTMLInputElement>("s-api-key");
    const btn = $<HTMLButtonElement>("btn-api-save");
    if (!input.value.trim()) return toast("Paste a key first", true);
    btn.disabled = true;
    btn.textContent = "CHECKING…";
    try {
      toast(`Saved · ${await invoke<string>("api_key_save", { key: input.value })}`);
      input.value = "";
      await renderApiKey();
    } catch (e) { toast(String(e), true); }
    btn.disabled = false;
    btn.textContent = "SAVE KEY";
  };
  $("btn-api-test").onclick = async () => {
    try { toast(await invoke<string>("api_key_test")); } catch (e) { toast(String(e), true); }
  };
  $("btn-api-del").onclick = async () => {
    if (!(await askDialog("Remove the Claude API key from Mahdar?", { kind: "warning" }))) return;
    try { await invoke("api_key_delete"); await renderApiKey(); toast("Key removed"); } catch (e) { toast(String(e), true); }
  };
  $("mon-filter").addEventListener("click", () => setTimeout(renderMonitor));
  $("b-ask").onclick = () => brainAsk(false);
  $<HTMLInputElement>("b-q").onkeydown = (e) => { if (e.key === "Enter") brainAsk(false); };
  initSplit();
  $("b-filter").oninput = renderBrainList;
  $("b-scope").onclick = () => setScope(null);
  $("b-index-all").onclick = () => indexBrain(null);
  $("btn-brain-index").onclick = () => { if (current) indexBrain([current.id]); };
  refreshBrainStatus();
  $("mon-search").oninput = renderMonitor;
  $("btn-log-folder").onclick = () => invoke("open_log_folder").catch((e) => toast(String(e), true));
  $("btn-log-folder-s").onclick = () => invoke("open_log_folder").catch((e) => toast(String(e), true));
  $("btn-data-folder").onclick = () => invoke("open_data_folder").catch((e) => toast(String(e), true));
  initSettingsNav();
  $("upd-tag").onclick = installUpdate;
  $("btn-upd-install").onclick = installUpdate;
  $("btn-upd-check").onclick = async () => {
    try { upd = await invoke<UpdateInfo>("update_check"); renderUpdate(); if (upd.state === "latest") toast(`Mahdar v${upd.current} is up to date`); } catch (e) { toast(String(e), true); }
  };
  $("btn-upd-releases").onclick = () => invoke("update_open_releases").catch((e) => toast(String(e), true));
  invoke<UpdateInfo>("update_status").then((u) => { upd = u; renderUpdate(); });
  $("btn-add-template").onclick = () => {
    const name = $<HTMLInputElement>("ct-name").value.trim();
    const prompt = $<HTMLTextAreaElement>("ct-prompt").value.trim();
    if (!name || !prompt) return toast("Give the output a name and instructions", true);
    settings.custom_templates.push({ id: `custom-${Date.now()}`, name, prompt, builtin: false });
    $<HTMLInputElement>("ct-name").value = "";
    $<HTMLTextAreaElement>("ct-prompt").value = "";
    renderCustomTemplates();
    markSettingsDirty();
    toast("Added: press SAVE SETTINGS to keep it");
  };

  // backend events
  await listen<UpdateInfo>("update", (ev) => {
    const was = upd?.state;
    upd = ev.payload;
    renderUpdate();
    if (upd.state === "ready" && was !== "ready") toast(`Mahdar v${upd.version} is ready · RESTART to update`);
  });
  await listen("brain-updated", async () => {
    await refreshBrainStatus();
    if (!brainStatus?.busy) progress.delete("brain-index");
    renderBrainStatus();
    if ($("v-brain").classList.contains("active")) {
      brainEntities = await invoke<EntityRow[]>("brain_entities");
      renderBrainList();
      if (brainShowing === "entity" && brainEntity) await openEntity(brainEntity);
    }
  });
  await listen<Progress>("progress", (ev) => {
    progress.set(ev.payload.id, ev.payload);
    if (ev.payload.id === "brain-index") renderBrainStatus();
    if (current?.id === ev.payload.id) renderTimeline();
    if ($("v-library").classList.contains("active")) renderLibrary();
  });
  await listen<[string, string]>("gen-stage", (ev) => {
    genStage.set(ev.payload[0], ev.payload[1]);
    if (current?.id === ev.payload[0]) { renderTimeline(); renderChat(); }
  });
  await listen("settings-changed", async () => {
    settings = await invoke<Settings>("get_settings");
    if (current) renderControls();
    if ($("v-library").classList.contains("active")) renderLibrary();
  });
  await listen<[string, string]>("watch-imported", async (ev) => {
    toast(`New recording found: ${ev.payload[1].split(/[\\/]/).pop()}`);
    meetings = await invoke<MeetingSummary[]>("list_meetings");
    if ($("v-library").classList.contains("active")) renderLibrary();
  });
  await listen<[string, string]>("retry-done", (ev) => {
    retrying.delete(ev.payload[0]);
    toast(ev.payload[1], ev.payload[1].startsWith("re-run failed"));
  });
  await listen<string>("meeting-audio", async (ev) => {
    if (current?.id === ev.payload) {
      current = await invoke<Meeting>("get_meeting", { id: ev.payload });
      loadAudio();
      renderMeeting();
    }
  });
  await listen<string>("meeting-updated", async (ev) => {
    progress.delete(ev.payload);
    meetings = await invoke<MeetingSummary[]>("list_meetings");
    status = await invoke<EngineStatus>("engine_status");
    if ($("v-library").classList.contains("active")) renderLibrary();
    const m = meetings.find((x) => x.id === ev.payload);
    if (current?.id === ev.payload) await openMeeting(ev.payload);
    if (m?.status === "ready") toast(`Transcribed: ${m.title}`);
    if (m?.status === "failed") toast(`Failed: ${m.title}`, true);
  });
  await listen<[number, number]>("model-download", (ev) => {
    const pct = `${Math.floor((100 * ev.payload[0]) / ev.payload[1])}%`;
    document.querySelectorAll<HTMLButtonElement>("#btn-download, #btn-download-s, #btn-download-diar").forEach((b) => { if (b.disabled) b.textContent = pct; });
  });
  await listen<string | null>("model-download-done", async (ev) => {
    toast(ev.payload ? `Download failed: ${ev.payload}` : "Speech model ready", !!ev.payload);
    await refreshLibrary();
    if ($("v-settings").classList.contains("active")) await renderSettings();
  });
}

init().catch((e) => toast(String(e), true));
