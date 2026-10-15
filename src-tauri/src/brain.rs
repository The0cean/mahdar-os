//! Mahdar Brain in the app: indexing meetings on demand, questions across meetings, and
//! people/organisation/topic pages. The index lives in `<data>/brain.db`; the embedding model
//! in `<data>/models` (downloaded on first use). See the `mahdar-brain` crate.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use mahdar_brain::{ask, embed, tags, text, AskRow, Brain, Doc, EntityPage, EntityRow, Filter};
use mahdar_core::log;
use mahdar_core::store::{Meeting, Status};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};

use crate::{clear_progress, err, report, AppState, CmdResult, Progress};

/// Progress key of the indexing job (shown in Monitor and polled by the phone).
pub const INDEX_JOB: &str = "brain-index";
/// Progress key of a question across meetings (phone jobs).
pub const ASK_JOB: &str = "brain-ask";
/// Excerpts sent with a question.
const TOP_K: usize = 20;

#[derive(Default)]
pub struct BrainState {
    db: Mutex<Option<Arc<Brain>>>,
    embedder: Mutex<Option<embed::Embedder>>,
    busy: AtomicBool,
}

impl BrainState {
    /// Indexing is running.
    pub(crate) fn is_busy(&self) -> bool {
        self.busy.load(Ordering::SeqCst)
    }
}

fn models_root(app: &AppHandle) -> PathBuf {
    app.state::<AppState>().store.root().join("models")
}

fn db(app: &AppHandle) -> anyhow::Result<Arc<Brain>> {
    let state = app.state::<AppState>();
    let mut g = state.brain.db.lock().unwrap();
    if g.is_none() {
        *g = Some(Arc::new(Brain::open(&state.store.root().join("brain.db"))?));
    }
    Ok(g.as_ref().unwrap().clone())
}

/// Download the embedding model if it isn't there yet, reporting progress under `job`.
fn ensure_model(app: &AppHandle, job: &str) -> anyhow::Result<()> {
    let root = models_root(app);
    if embed::model_present(&root) {
        return Ok(());
    }
    log::info("brain", format!("downloading the search model ({}, {} MB)", embed::MODEL_NAME, embed::model_bytes() / 1_000_000));
    let dir = embed::model_dir(&root);
    let total = embed::model_bytes();
    let mut before = 0u64;
    for (url, name, bytes) in embed::model_files() {
        let dest = dir.join(name);
        if !dest.is_file() {
            let mut last = 0u64;
            mahdar_core::download(&url, &dest, bytes, |done, _| {
                if done - last > 4 << 20 {
                    last = done;
                    report(app, Progress { id: job.into(), stage: "downloading search model".into(), done: ((before + done) >> 20) as usize, total: (total >> 20) as usize });
                }
            })?;
        }
        before += bytes;
    }
    log::info("brain", "search model ready");
    Ok(())
}

/// Lock the embedder, loading it first if needed.
fn with_embedder<T>(app: &AppHandle, job: &str, f: impl FnOnce(&mut embed::Embedder) -> anyhow::Result<T>) -> anyhow::Result<T> {
    ensure_model(app, job)?;
    let state = app.state::<AppState>();
    let mut g = state.brain.embedder.lock().unwrap();
    if g.is_none() {
        *g = Some(embed::Embedder::load(&models_root(app))?);
    }
    f(g.as_mut().unwrap())
}

fn lines(m: &Meeting) -> Vec<text::Line> {
    m.chunks
        .iter()
        .map(|c| text::Line {
            start_s: c.start_s,
            end_s: c.end_s,
            speaker: c.speaker.map(|n| m.speaker_name(n)),
            text: mahdar_core::asr::clean_text(&c.text),
        })
        .collect()
}

fn transcript_hash(m: &Meeting) -> String {
    text::hash(&m.transcript_text())
}

/// Index one meeting: passages + embeddings (local), then tags (writer). A tagging failure
/// leaves the search index in place and is recorded, so the meeting shows "not tagged".
fn index_meeting(app: &AppHandle, id: &str) -> anyhow::Result<bool> {
    let state = app.state::<AppState>();
    let m = state.store.load_meeting(id)?;
    anyhow::ensure!(m.status == Status::Ready && !m.chunks.is_empty(), "“{}” has no transcript yet", m.title);
    let brain = db(app)?;
    let t0 = std::time::Instant::now();
    let passages = text::passages(&lines(&m));
    let texts: Vec<String> = passages.iter().map(|p| p.text.clone()).collect();
    let embeddings = with_embedder(app, INDEX_JOB, |e| e.passages(&texts))?;
    let doc = Doc { id: m.id.clone(), title: m.title.clone(), created: m.created.to_rfc3339(), hash: transcript_hash(&m) };
    brain.index(&doc, &passages, &embeddings)?;
    log::info("brain", format!("indexed “{}”: {} passages in {:.1} s", m.title, passages.len(), t0.elapsed().as_secs_f64()));

    let settings = state.store.load_settings();
    let prompt = tags::prompt(&m.title, &m.created.format("%d %b %Y").to_string(), &m.transcript_text(), &settings.glossary, &brain.entities_all()?);
    let what = format!("Brain tags for “{}”", m.title);
    let tagged = crate::write_with(app, &settings.brain_engine(), "brain-tag", &what, &prompt, false)
        .map_err(anyhow::Error::msg)
        .and_then(|(reply, _)| tags::parse(&reply))
        .and_then(|t| {
            let n = t.entities.len();
            brain.set_tags(&m.id, &t)?;
            log::info("brain", format!("tagged “{}”: {n} entities, {} decisions, {} actions", m.title, t.decisions.len(), t.actions.len()));
            Ok(())
        });
    if let Err(e) = tagged {
        log::warn("brain", format!("“{}” is searchable but not tagged: {e:#}", m.title));
        brain.set_tag_error(&m.id, &format!("{e:#}"))?;
        return Ok(false);
    }
    Ok(true)
}

/// Where each meeting stands in the Brain.
fn meeting_state(m_hash: &str, idx: Option<&mahdar_brain::IndexedDoc>) -> &'static str {
    match idx {
        None => "not_indexed",
        Some(d) if d.transcript_hash != m_hash => "out_of_date",
        Some(d) if !d.tagged => "untagged",
        Some(_) => "indexed",
    }
}

#[derive(Serialize)]
pub struct BrainStatus {
    pub meetings: usize,
    pub indexed: usize,
    pub out_of_date: usize,
    pub untagged: usize,
    pub busy: bool,
    pub model_present: bool,
    pub model: String,
    pub model_mb: u64,
    /// meeting id → "indexed" | "untagged" | "out_of_date" | "not_indexed" (ready meetings only)
    pub states: HashMap<String, String>,
}

pub fn status(app: &AppHandle) -> anyhow::Result<BrainStatus> {
    let state = app.state::<AppState>();
    let brain = db(app)?;
    let ready: Vec<Meeting> = state
        .store
        .list_meetings()?
        .into_iter()
        .filter(|s| s.status == Status::Ready)
        .filter_map(|s| state.store.load_meeting(&s.id).ok())
        .filter(|m| !m.chunks.is_empty())
        .collect();
    brain.sync_titles(&ready.iter().map(|m| (m.id.clone(), m.title.clone())).collect::<Vec<_>>())?;
    let idx: HashMap<String, mahdar_brain::IndexedDoc> = brain.indexed()?.into_iter().map(|d| (d.meeting_id.clone(), d)).collect();
    let mut states = HashMap::new();
    for m in &ready {
        states.insert(m.id.clone(), meeting_state(&transcript_hash(m), idx.get(&m.id)).to_string());
    }
    let count = |s: &str| states.values().filter(|v| v.as_str() == s).count();
    Ok(BrainStatus {
        meetings: ready.len(),
        indexed: count("indexed") + count("untagged"),
        out_of_date: count("out_of_date"),
        untagged: count("untagged"),
        busy: state.brain.busy.load(Ordering::SeqCst),
        model_present: embed::model_present(&models_root(app)),
        model: embed::MODEL_NAME.into(),
        model_mb: embed::model_bytes() / 1_000_000,
        states,
    })
}

/// Index the given meetings, or every ready meeting that isn't indexed, is out of date or
/// isn't tagged yet. Runs in the background; progress under [`INDEX_JOB`].
pub fn start_index(app: &AppHandle, ids: Option<Vec<String>>) -> CmdResult<usize> {
    let st = status(app).map_err(err)?;
    let targets: Vec<String> = match ids {
        Some(ids) => ids.into_iter().filter(|id| st.states.contains_key(id)).collect(),
        None => st.states.iter().filter(|(_, s)| s.as_str() != "indexed").map(|(id, _)| id.clone()).collect(),
    };
    if targets.is_empty() {
        return Err("nothing to index: every transcribed meeting is already in the Brain".into());
    }
    let state = app.state::<AppState>();
    if state.brain.busy.swap(true, Ordering::SeqCst) {
        return Err("the Brain is already indexing".into());
    }
    let n = targets.len();
    log::info("brain", format!("indexing {n} meeting{}", if n == 1 { "" } else { "s" }));
    let app = app.clone();
    std::thread::spawn(move || {
        let state = app.state::<AppState>();
        let (mut ok, mut failed) = (0, 0);
        for (i, id) in targets.iter().enumerate() {
            let title = state.store.load_meeting(id).map(|m| m.title).unwrap_or_default();
            report(&app, Progress { id: INDEX_JOB.into(), stage: format!("indexing “{title}”"), done: i, total: n });
            match index_meeting(&app, id) {
                Ok(_) => ok += 1,
                Err(e) => {
                    failed += 1;
                    log::error("brain", format!("could not index “{title}”: {e:#}"));
                }
            }
            let _ = app.emit("brain-updated", ());
        }
        clear_progress(&app, INDEX_JOB);
        state.brain.busy.store(false, Ordering::SeqCst);
        log::info("brain", format!("indexing finished: {ok} done{}", if failed > 0 { format!(", {failed} failed") } else { String::new() }));
        let _ = app.emit("brain-updated", ());
    });
    Ok(n)
}

#[derive(Deserialize, Default)]
pub struct AskBody {
    pub question: String,
    #[serde(default)]
    pub entity: Option<String>,
    #[serde(default)]
    pub from: Option<String>,
    #[serde(default)]
    pub to: Option<String>,
    /// Follow up this conversation (id of its first question).
    #[serde(default)]
    pub thread: Option<i64>,
}

/// Earlier turns sent with a follow-up.
const HISTORY: usize = 6;

/// Answer a question from the best excerpts across all indexed meetings (blocking). With a
/// thread, earlier questions and answers go along, so follow-ups like "what do you think?" work.
pub fn ask(app: &AppHandle, b: &AskBody) -> CmdResult<AskRow> {
    let q = b.question.trim();
    if q.is_empty() {
        return Err("type a question".into());
    }
    let brain = db(app).map_err(err)?;
    if brain.indexed().map_err(err)?.is_empty() {
        return Err("nothing is indexed yet: press INDEX ALL in Brain first".into());
    }
    let history: Vec<(String, String)> = match b.thread {
        Some(t) => {
            let rows = brain.thread(t).map_err(err)?;
            if rows.is_empty() {
                return Err("that conversation was deleted: ask it as a new question".into());
            }
            let skip = rows.len().saturating_sub(HISTORY);
            rows.into_iter().skip(skip).map(|r| (r.question, r.answer)).collect()
        }
        None => Vec::new(),
    };
    // A follow-up ("and the deadline?", "what do you think?") is searched together with the
    // question before it, so it finds the same meetings.
    let search_q = match history.last() {
        Some((prev, _)) => format!("{prev}\n{q}"),
        None => q.to_string(),
    };
    let qv = with_embedder(app, ASK_JOB, |e| e.query(&search_q)).map_err(err)?;
    let filter = Filter { entity: b.entity.clone().filter(|s| !s.is_empty()), from: b.from.clone(), to: b.to.clone() };
    let hits = brain.search(&search_q, Some(&qv), &filter, TOP_K).map_err(err)?;
    if hits.is_empty() {
        return Err("no indexed meeting matches that filter".into());
    }
    let meetings = hits.iter().map(|h| &h.doc_id).collect::<std::collections::HashSet<_>>().len();
    let what = format!("Brain question over {} excerpts from {meetings} meeting{}", hits.len(), if meetings == 1 { "" } else { "s" });
    let engine = app.state::<crate::AppState>().store.load_settings().brain_engine();
    let (reply, provider) = crate::write_with(app, &engine, ASK_JOB, &what, &ask::prompt(q, &hits, &history), false)?;
    let (answer, cites) = ask::cite(&reply, &hits);
    let scope = match &filter.entity {
        Some(k) => brain.entity_page(k).ok().flatten().map(|p| p.entity.name).unwrap_or_default(),
        None => String::new(),
    };
    let id = brain.save_ask(q, &answer, &cites, &provider, &scope, b.thread).map_err(err)?;
    let thread = b.thread.unwrap_or(id);
    Ok(AskRow { id, question: q.into(), answer, citations: cites, provider, scope, created: chrono::Local::now().to_rfc3339(), thread })
}

pub fn entities(app: &AppHandle) -> CmdResult<Vec<EntityRow>> {
    db(app).and_then(|b| b.entities()).map_err(err)
}

pub fn entity(app: &AppHandle, key: &str) -> CmdResult<EntityPage> {
    db(app).and_then(|b| b.entity_page(key)).map_err(err)?.ok_or_else(|| "not found".to_string())
}

pub fn asks(app: &AppHandle) -> CmdResult<Vec<AskRow>> {
    db(app).and_then(|b| b.asks(60)).map_err(err)
}

/// A deleted meeting leaves the Brain too.
pub fn forget(app: &AppHandle, id: &str) {
    if let Ok(b) = db(app) {
        if let Err(e) = b.remove(id) {
            log::warn("brain", format!("could not remove {id} from the Brain: {e:#}"));
        }
    }
}

// ---------- Tauri commands ----------

#[tauri::command]
pub async fn brain_status(app: AppHandle) -> CmdResult<BrainStatus> {
    tauri::async_runtime::spawn_blocking(move || status(&app).map_err(err)).await.map_err(err)?
}

#[tauri::command]
pub fn brain_index(app: AppHandle, ids: Option<Vec<String>>) -> CmdResult<usize> {
    start_index(&app, ids)
}

#[tauri::command]
pub fn brain_entities(app: AppHandle) -> CmdResult<Vec<EntityRow>> {
    entities(&app)
}

#[tauri::command]
pub fn brain_entity(app: AppHandle, key: String) -> CmdResult<EntityPage> {
    entity(&app, &key)
}

#[tauri::command]
pub async fn brain_ask(app: AppHandle, question: String, entity: Option<String>, thread: Option<i64>) -> CmdResult<AskRow> {
    tauri::async_runtime::spawn_blocking(move || ask(&app, &AskBody { question, entity, thread, ..Default::default() })).await.map_err(err)?
}

#[tauri::command]
pub fn brain_thread(app: AppHandle, thread: i64) -> CmdResult<Vec<AskRow>> {
    db(&app).and_then(|b| b.thread(thread)).map_err(err)
}

#[tauri::command]
pub fn brain_delete_thread(app: AppHandle, thread: i64) -> CmdResult<()> {
    db(&app).and_then(|b| b.delete_thread(thread)).map_err(err)
}

#[tauri::command]
pub fn brain_asks(app: AppHandle) -> CmdResult<Vec<AskRow>> {
    asks(&app)
}
