mod brain;
mod remote;
mod watch;
pub mod watchdog;
mod update;

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use chrono::Local;
use mahdar_core::audio::{self, Recorder};
use mahdar_core::log;
use mahdar_core::store::{fmt_ts, ChatMsg, Meeting, MeetingSummary, Output, Settings, Status, Store};
use mahdar_core::summarize::{self, Provider, Template};
use mahdar_core::{Backend, Engine};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};

pub(crate) struct AppState {
    pub(crate) store: Store,
    native_dir: PathBuf,
    engine: Mutex<Option<Engine>>,
    /// Speaker diarizer (Sortformer), loaded on first use.
    diarizer: Mutex<Option<Engine>>,
    recorder: Mutex<Option<(Recorder, Instant)>>,
    /// FastFlowLM server started by Mahdar; stopped when the app exits.
    flm: Mutex<Option<std::process::Child>>,
    /// Mahdar Remote server task (phone access), when enabled.
    remote: Mutex<Option<tauri::async_runtime::JoinHandle<()>>>,
    /// Latest progress per meeting (also sent as events), so the phone can poll it.
    pub(crate) progress: Mutex<std::collections::HashMap<String, Progress>>,
    /// Phone server state for the Monitor, e.g. "listening on port 8787".
    pub(crate) remote_state: Mutex<String>,
    started: chrono::DateTime<Local>,
    /// Mahdar Brain: index, embedder, indexing flag.
    pub(crate) brain: brain::BrainState,
    /// Auto-update: status and the downloaded update.
    pub(crate) update: update::UpdateState,
}

type CmdResult<T> = Result<T, String>;

fn err<E: std::fmt::Display>(e: E) -> String {
    format!("{e:#}")
}

#[derive(Serialize, Clone)]
pub(crate) struct Progress {
    id: String,
    stage: String,
    done: usize,
    total: usize,
}

#[derive(Serialize)]
struct EngineStatus {
    model_path: String,
    model_present: bool,
    model_bytes_expected: u64,
    loaded_backend: Option<String>,
    ffmpeg: bool,
    data_dir: String,
    diarizer_present: bool,
}

/// Record and broadcast a job's progress.
fn report(app: &AppHandle, p: Progress) {
    app.state::<AppState>().progress.lock().unwrap().insert(p.id.clone(), p.clone());
    let _ = app.emit("progress", p);
}

fn clear_progress(app: &AppHandle, id: &str) {
    app.state::<AppState>().progress.lock().unwrap().remove(id);
}

/// Run a writing job (minutes or a question) in the background for the phone.
/// Progress shows "writing" until it ends; a failure stays visible as "failed: …" for 2 minutes.
pub(crate) fn spawn_job(app: &AppHandle, id: &str, job: impl FnOnce(&AppHandle) -> CmdResult<()> + Send + 'static) {
    report(app, Progress { id: id.to_string(), stage: "writing".into(), done: 0, total: 0 });
    let (app, id) = (app.clone(), id.to_string());
    std::thread::spawn(move || {
        if let Err(e) = job(&app) {
            let stage = format!("failed: {e}");
            report(&app, Progress { id: id.clone(), stage: stage.clone(), done: 0, total: 0 });
            std::thread::sleep(std::time::Duration::from_secs(120));
            let state = app.state::<AppState>();
            let mut map = state.progress.lock().unwrap();
            if map.get(&id).map(|p| p.stage == stage).unwrap_or(false) {
                map.remove(&id);
            }
        } else {
            clear_progress(&app, &id);
        }
        let _ = app.emit("meeting-updated", id);
    });
}

fn which(cmd: &str) -> bool {
    let mut c = std::process::Command::new(cmd);
    c.arg("-version").stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000);
    }
    c.status().is_ok()
}

// ---------- settings & status ----------

#[tauri::command]
fn get_settings(state: State<AppState>) -> Settings {
    state.store.load_settings()
}

#[tauri::command]
fn save_settings(app: AppHandle, state: State<AppState>, settings: Settings) -> CmdResult<()> {
    let old = state.store.load_settings();
    let remote_changed = (old.remote_enabled, old.remote_port, &old.remote_token)
        != (settings.remote_enabled, settings.remote_port, &settings.remote_token);
    if old.model_path != settings.model_path || old.backend != settings.backend {
        *state.engine.lock().unwrap() = None; // reload on next use
    }
    if old.diarizer_path != settings.diarizer_path || old.backend != settings.backend {
        *state.diarizer.lock().unwrap() = None;
    }
    state.store.save_settings(&settings).map_err(err)?;
    if old.provider.label() != settings.provider.label() {
        log::info("settings", format!("minutes writer: {}", settings.provider.label()));
    }
    if remote_changed {
        restart_remote(&app);
    }
    if old.watchdog != settings.watchdog {
        log::info("settings", format!("watchdog turned {}", if settings.watchdog { "on" } else { "off" }));
        watchdog::sync(state.store.root(), settings.watchdog);
    }
    Ok(())
}

// ---------- Claude API key ----------

#[derive(Serialize)]
struct ApiKeyStatus {
    /// Masked key, e.g. "sk-ant-…WXYZ"; empty when none.
    masked: String,
    /// "Mahdar", "Mizan" (shared from Mizan's Settings) or "environment".
    source: String,
}

#[tauri::command]
fn api_key_status() -> ApiKeyStatus {
    match mahdar_core::secrets::anthropic_key() {
        Some((k, src)) => ApiKeyStatus { masked: mahdar_core::secrets::mask(&k), source: src.into() },
        None => ApiKeyStatus { masked: String::new(), source: String::new() },
    }
}

#[tauri::command]
async fn api_key_save(key: String) -> CmdResult<String> {
    let key = key.trim().to_string();
    if !key.starts_with("sk-ant-") {
        return Err("that doesn't look like a Claude API key (they start with sk-ant-)".into());
    }
    // Check it before saving, so a typo isn't stored.
    let k = key.clone();
    let msg = tauri::async_runtime::spawn_blocking(move || summarize::test_api_key(&k)).await.map_err(err)?.map_err(err)?;
    mahdar_core::secrets::set_anthropic_key(&key).map_err(err)?;
    log::info("settings", format!("Claude API key saved ({})", mahdar_core::secrets::mask(&key)));
    Ok(msg)
}

#[tauri::command]
async fn api_key_test() -> CmdResult<String> {
    let (k, _) = mahdar_core::secrets::anthropic_key().ok_or("no key saved")?;
    tauri::async_runtime::spawn_blocking(move || summarize::test_api_key(&k)).await.map_err(err)?.map_err(err)
}

#[tauri::command]
fn api_key_delete() -> CmdResult<()> {
    mahdar_core::secrets::delete_anthropic_key().map_err(err)?;
    log::info("settings", "Claude API key removed from Mahdar");
    Ok(())
}

/// Files copied in Explorer (Ctrl+C), for PASTE / Ctrl+V on the library screen.
#[tauri::command]
fn clipboard_files() -> CmdResult<Vec<String>> {
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::DataExchange::{CloseClipboard, GetClipboardData, OpenClipboard};
        use windows_sys::Win32::UI::Shell::DragQueryFileW;
        const CF_HDROP: u32 = 15;
        // Another app may hold the clipboard for a moment.
        let mut opened = false;
        for _ in 0..5 {
            if OpenClipboard(std::ptr::null_mut()) != 0 {
                opened = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(30));
        }
        if !opened {
            return Err("the clipboard is busy, try again".into());
        }
        let h = GetClipboardData(CF_HDROP);
        let mut out = Vec::new();
        if !h.is_null() {
            let n = DragQueryFileW(h, u32::MAX, std::ptr::null_mut(), 0);
            for i in 0..n {
                let len = DragQueryFileW(h, i, std::ptr::null_mut(), 0);
                let mut buf = vec![0u16; len as usize + 1];
                let got = DragQueryFileW(h, i, buf.as_mut_ptr(), buf.len() as u32);
                out.push(String::from_utf16_lossy(&buf[..got as usize]));
            }
        }
        CloseClipboard();
        Ok(out)
    }
    #[cfg(not(windows))]
    Ok(Vec::new())
}

// ---------- Monitor ----------

#[derive(Serialize)]
pub(crate) struct JobInfo {
    id: String,
    title: String,
    stage: String,
    done: usize,
    total: usize,
}

#[derive(Serialize)]
pub(crate) struct Monitor {
    version: &'static str,
    pid: u32,
    started: String,
    uptime_s: i64,
    speech: String,
    speakers: String,
    writer: String,
    local_server: bool,
    remote: String,
    jobs: Vec<JobInfo>,
    api_key: String,
    spend_today: f64,
    spend_month: f64,
    spend_total: f64,
    watchdog: watchdog::Status,
    log: Vec<log::Line>,
}

/// Everything the Monitor screens show (desktop and phone).
pub(crate) fn monitor(app: &AppHandle, lines: usize) -> Monitor {
    let state = app.state::<AppState>();
    let s = state.store.load_settings();
    // try_lock: a transcription holds the engine for minutes; the Monitor must not wait for it.
    let speech = match state.engine.try_lock() {
        Ok(g) => g.as_ref().map(|e| format!("loaded on {}", e.backend_name())).unwrap_or_else(|| "not loaded (loads on first use)".into()),
        Err(_) => "busy".into(),
    };
    let speakers = if !s.speakers {
        "off".into()
    } else {
        match state.diarizer.try_lock() {
            Ok(g) if g.is_some() => "loaded".into(),
            Ok(_) => "not loaded".into(),
            Err(_) => "busy".into(),
        }
    };
    let local_server = state.flm.lock().unwrap().as_mut().map(|c| matches!(c.try_wait(), Ok(None))).unwrap_or(false);
    let jobs = state
        .progress
        .lock()
        .unwrap()
        .values()
        .map(|p| JobInfo {
            id: p.id.clone(),
            title: match p.id.as_str() {
                brain::INDEX_JOB => "Brain · indexing".into(),
                brain::ASK_JOB => "Brain · question".into(),
                "brain-tag" => "Brain · tagging".into(),
                id => match id.strip_suffix("#note") {
                    Some(mid) => state.store.load_meeting(mid).map(|m| format!("{} · note", m.title)).unwrap_or_else(|_| p.id.clone()),
                    None => state.store.load_meeting(id).map(|m| m.title).unwrap_or_else(|_| p.id.clone()),
                },
            },
            stage: p.stage.clone(),
            done: p.done,
            total: p.total,
        })
        .collect();
    let api_key = match mahdar_core::secrets::anthropic_key() {
        Some((k, src)) => format!("{} · saved in {src}", mahdar_core::secrets::mask(&k)),
        None => "none".into(),
    };
    let spend = state.store.spend();
    Monitor {
        version: env!("CARGO_PKG_VERSION"),
        pid: std::process::id(),
        started: state.started.to_rfc3339(),
        uptime_s: (Local::now() - state.started).num_seconds(),
        speech,
        speakers,
        writer: s.provider.label(),
        local_server,
        remote: if s.remote_enabled { state.remote_state.lock().unwrap().clone() } else { "off".into() },
        jobs,
        api_key,
        spend_today: spend.today(),
        spend_month: spend.this_month(),
        spend_total: spend.total,
        watchdog: watchdog::status(state.store.root(), s.watchdog),
        log: log::tail(lines),
    }
}

#[tauri::command]
fn monitor_snapshot(app: AppHandle) -> Monitor {
    monitor(&app, 400)
}

#[tauri::command]
fn open_data_folder(state: State<AppState>) -> CmdResult<()> {
    std::process::Command::new("explorer").arg(state.store.root()).spawn().map_err(err)?;
    Ok(())
}

#[tauri::command]
fn open_log_folder(state: State<AppState>) -> CmdResult<()> {
    let dir = log::log_dir(state.store.root());
    std::process::Command::new("explorer").arg(dir).spawn().map_err(err)?;
    Ok(())
}

// ---------- Mahdar Remote (phone) ----------

fn new_token() -> String {
    use rand::Rng;
    rand::thread_rng().sample_iter(&rand::distributions::Alphanumeric).take(24).map(char::from).collect()
}

/// Stop the phone server and start it again if enabled (creating a token on first use).
fn restart_remote(app: &AppHandle) {
    let state = app.state::<AppState>();
    if let Some(h) = state.remote.lock().unwrap().take() {
        h.abort();
    }
    let mut s = state.store.load_settings();
    if !s.remote_enabled {
        return;
    }
    if s.remote_token.is_empty() {
        s.remote_token = new_token();
        let _ = state.store.save_settings(&s);
    }
    *state.remote.lock().unwrap() = Some(remote::start(app.clone(), s.remote_port, s.remote_token.clone()));
}

#[derive(Serialize)]
struct RemoteInfo {
    enabled: bool,
    port: u16,
    /// (kind, url) for each reachable address: "wifi", "tailscale", "https"
    urls: Vec<(String, String)>,
    /// Best URL for the phone, with the token (encoded in the QR code).
    pair_url: String,
    qr_svg: String,
}

#[tauri::command]
fn remote_info(state: State<AppState>) -> RemoteInfo {
    let s = state.store.load_settings();
    let (addrs, dns) = remote::addresses();
    let mut urls: Vec<(String, String)> = addrs
        .into_iter()
        .map(|(kind, ip)| {
            let ip_only = ip.split_whitespace().next().unwrap_or("").to_string();
            (kind, format!("http://{ip_only}:{}", s.remote_port))
        })
        .collect();
    if let Some(d) = dns {
        urls.insert(0, ("https".into(), format!("https://{d}")));
    }
    let best = urls
        .iter()
        .find(|(k, _)| k == "https")
        .or_else(|| urls.iter().find(|(k, _)| k == "tailscale"))
        .or_else(|| urls.first())
        .map(|(_, u)| u.clone())
        .unwrap_or_default();
    let pair_url = if best.is_empty() || s.remote_token.is_empty() { String::new() } else { format!("{best}/?t={}", s.remote_token) };
    RemoteInfo { enabled: s.remote_enabled, port: s.remote_port, qr_svg: remote::qr_svg(&pair_url), urls, pair_url }
}

#[tauri::command]
fn remote_new_token(app: AppHandle, state: State<AppState>) -> CmdResult<()> {
    let mut s = state.store.load_settings();
    s.remote_token = new_token();
    state.store.save_settings(&s).map_err(err)?;
    log::info("phone", "new pairing code made; phones must scan the QR code again");
    restart_remote(&app);
    Ok(())
}

#[tauri::command]
async fn remote_setup_https(app: AppHandle) -> CmdResult<String> {
    let port = app.state::<AppState>().store.load_settings().remote_port;
    tauri::async_runtime::spawn_blocking(move || remote::setup_https(port)).await.map_err(err)?
}

#[tauri::command]
fn engine_status(state: State<AppState>) -> EngineStatus {
    let s = state.store.load_settings();
    let loaded_backend = state.engine.lock().unwrap().as_ref().map(|e| e.backend_name().to_string());
    EngineStatus {
        model_present: Path::new(&s.model_path).is_file(),
        model_path: s.model_path,
        model_bytes_expected: mahdar_core::MODEL_BYTES,
        loaded_backend,
        ffmpeg: which("ffmpeg"),
        data_dir: state.store.root().to_string_lossy().into_owned(),
        diarizer_present: Path::new(&s.diarizer_path).is_file(),
    }
}

/// Download the speech model (default) or the speaker model (`which` = "diarizer").
#[tauri::command]
fn download_model(app: AppHandle, state: State<AppState>, which: Option<String>) -> CmdResult<()> {
    let s = state.store.load_settings();
    let (url, dest, bytes) = if which.as_deref() == Some("diarizer") {
        (mahdar_core::DIARIZER_URL, PathBuf::from(s.diarizer_path), 236_606_560u64)
    } else {
        (mahdar_core::MODEL_URL, PathBuf::from(s.model_path), mahdar_core::MODEL_BYTES)
    };
    std::thread::spawn(move || {
        let mut last = 0u64;
        let res = mahdar_core::download(url, &dest, bytes, |done, total| {
            if done - last > 4 << 20 || done == total {
                last = done;
                let _ = app.emit("model-download", (done, total));
            }
        });
        let _ = app.emit("model-download-done", res.err().map(err));
    });
    Ok(())
}

// ---------- meetings ----------

#[tauri::command]
fn list_meetings(state: State<AppState>) -> CmdResult<Vec<MeetingSummary>> {
    state.store.list_meetings().map_err(err)
}

#[tauri::command]
fn get_meeting(state: State<AppState>, id: String) -> CmdResult<Meeting> {
    state.store.load_meeting(&id).map_err(err)
}

#[tauri::command]
fn rename_meeting(state: State<AppState>, id: String, title: String) -> CmdResult<()> {
    let mut m = state.store.load_meeting(&id).map_err(err)?;
    m.title = title.trim().to_string();
    state.store.save_meeting(&m).map_err(err)
}

#[tauri::command]
fn delete_meeting(app: AppHandle, state: State<AppState>, id: String) -> CmdResult<()> {
    state.store.delete_meeting(&id).map_err(err)?;
    brain::forget(&app, &id);
    Ok(())
}

#[tauri::command]
fn audio_path(state: State<AppState>, id: String) -> String {
    state.store.audio_path(&id).to_string_lossy().into_owned()
}

#[tauri::command]
fn save_text_file(path: String, text: String) -> CmdResult<()> {
    std::fs::write(path, text).map_err(err)
}

fn new_meeting(state: &AppState, title: String, source: String) -> CmdResult<Meeting> {
    let s = state.store.load_settings();
    let m = Meeting {
        id: format!("{}-{}", Local::now().format("%Y%m%d-%H%M%S"), &uuid::Uuid::new_v4().to_string()[..6]),
        title,
        created: Local::now(),
        source,
        duration_s: 0.0,
        language: s.language,
        status: Status::Transcribing,
        error: None,
        transcribe_secs: None,
        chunks: Vec::new(),
        outputs: Vec::new(),
        speakers: Default::default(),
        chat: Vec::new(),
        notes: String::new(),
    };
    state.store.save_meeting(&m).map_err(err)?;
    Ok(m)
}

fn load_engine(state: &AppState, model: &Path, backend: Backend) -> anyhow::Result<Engine> {
    match Engine::load(&state.native_dir, model, backend) {
        Ok(e) => Ok(e),
        Err(e) if backend != Backend::Cpu => {
            log::warn("speech", format!("GPU load failed for {} ({e:#}); using the CPU", model.display()));
            Engine::load(&state.native_dir, model, Backend::Cpu)
        }
        Err(e) => Err(e),
    }
}

/// Lock the speech engine, loading the model first if needed (GPU with CPU fallback).
fn loaded_engine(state: &AppState) -> anyhow::Result<std::sync::MutexGuard<'_, Option<Engine>>> {
    let mut guard = state.engine.lock().unwrap();
    if guard.is_none() {
        let s = state.store.load_settings();
        let model = PathBuf::from(&s.model_path);
        if !model.is_file() {
            anyhow::bail!("speech model not found at {}. Open Settings to download it.", model.display());
        }
        *guard = Some(load_engine(state, &model, s.backend)?);
    }
    Ok(guard)
}

/// Lock the speaker diarizer, loading it if needed. None when speakers are off or the model is missing.
fn loaded_diarizer(state: &AppState) -> anyhow::Result<Option<std::sync::MutexGuard<'_, Option<Engine>>>> {
    let s = state.store.load_settings();
    let path = PathBuf::from(&s.diarizer_path);
    if !s.speakers || !path.is_file() {
        return Ok(None);
    }
    let mut guard = state.diarizer.lock().unwrap();
    if guard.is_none() {
        *guard = Some(load_engine(state, &path, s.backend)?);
    }
    Ok(Some(guard))
}

/// Background job: get PCM (decode file or use recorded samples), save audio.wav,
/// detect speakers (when enabled), transcribe.
fn spawn_transcription(app: AppHandle, id: String, input: Result<PathBuf, Vec<f32>>) {
    std::thread::spawn(move || {
        let state = app.state::<AppState>();
        let emit = |stage: &str, done: usize, total: usize| {
            report(&app, Progress { id: id.clone(), stage: stage.into(), done, total });
        };
        let title = state.store.load_meeting(&id).map(|m| m.title).unwrap_or_default();
        let t0 = Instant::now();
        log::info("transcribe", format!("started \u{201c}{title}\u{201d}"));
        let run = || -> anyhow::Result<()> {
            emit("decoding", 0, 0);
            let pcm = match &input {
                Ok(path) => audio::decode_to_pcm(path)?,
                Err(samples) => samples.clone(),
            };
            // Write the audio first, then publish the duration: the UI loads the player once duration > 0.
            audio::write_wav(&state.store.audio_path(&id), &pcm)?;
            // Phone copy in the background, so playback on the phone starts instantly.
            let wav = state.store.audio_path(&id);
            std::thread::spawn(move || {
                let _ = audio::phone_copy(&wav);
            });
            let mut m = state.store.load_meeting(&id)?;
            m.duration_s = pcm.len() as f64 / audio::SAMPLE_RATE as f64;
            state.store.save_meeting(&m)?;
            let _ = app.emit("meeting-audio", id.clone());

            // Speakers first, so transcription chunks follow speaker turns.
            let spans = match loaded_diarizer(&state)? {
                Some(mut d) => {
                    emit("detecting speakers", 0, 0);
                    let turns = d.as_mut().unwrap().diarize(&pcm)?;
                    mahdar_core::speaker_spans(&pcm, &turns)
                }
                None => mahdar_core::plain_spans(&pcm),
            };

            emit("loading model", 0, 0);
            let mut guard = loaded_engine(&state)?;
            let engine = guard.as_mut().unwrap();
            let (chunks, secs) = mahdar_core::transcribe_spans(engine, &pcm, &spans, &m.language, |done, total, _| {
                emit("transcribing", done, total)
            })?;
            drop(guard);

            // Re-load: the title may have been edited meanwhile.
            let mut m = state.store.load_meeting(&id)?;
            if !chunks.iter().any(|c| c.speaker.is_some()) {
                m.speakers.clear();
            }
            let unclear = chunks.iter().filter(|c| !c.flag.is_empty()).count();
            log::info(
                "transcribe",
                format!(
                    "done \u{201c}{title}\u{201d}: {} of audio in {} · {} lines{}{}",
                    fmt_ts(m.duration_s),
                    fmt_ts(t0.elapsed().as_secs_f64()),
                    chunks.len(),
                    if unclear > 0 { format!(" · {unclear} unclear") } else { String::new() },
                    if chunks.iter().any(|c| c.speaker.is_some()) { " · with speakers" } else { "" }
                ),
            );
            m.chunks = chunks;
            m.transcribe_secs = Some(secs);
            m.status = Status::Ready;
            state.store.save_meeting(&m)?;
            Ok(())
        };
        let ok = match run() {
            Ok(()) => true,
            Err(e) => {
                log::error("transcribe", format!("failed \u{201c}{title}\u{201d}: {e:#}"));
                if let Ok(mut m) = state.store.load_meeting(&id) {
                    m.status = Status::Failed;
                    m.error = Some(format!("{e:#}"));
                    let _ = state.store.save_meeting(&m);
                }
                false
            }
        };
        clear_progress(&app, &id);
        let _ = app.emit("meeting-updated", id.clone());
        if ok {
            after_transcription(&app, &id);
        }
    });
}

/// Settings → Automation: write an output and/or add the meeting to the Brain once its
/// transcript is ready. Failures are logged; the transcript itself is already saved.
fn after_transcription(app: &AppHandle, id: &str) {
    let s = app.state::<AppState>().store.load_settings();
    if !s.auto_output.is_empty() {
        if let Err(e) = do_generate(app, id, &s.auto_output) {
            log::warn("writer", format!("automatic output skipped: {e}"));
        }
    }
    if s.auto_index {
        if let Err(e) = brain::start_index(app, Some(vec![id.to_string()])) {
            log::warn("brain", format!("automatic indexing skipped: {e}"));
        }
    }
}

/// Create a meeting for an audio file and start transcribing it.
pub(crate) fn import_path(app: &AppHandle, p: &Path) -> CmdResult<String> {
    let state = app.state::<AppState>();
    let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    // Phone shortcuts often reuse names like "input.m4a": title those by date instead.
    let lower = stem.to_lowercase();
    let generic = ["", "input", "input_audio", "audio", "recording"].contains(&lower.as_str()) || lower.starts_with("new recording");
    let title = if generic {
        let when: chrono::DateTime<Local> =
            std::fs::metadata(p).and_then(|m| m.modified()).map(Into::into).unwrap_or_else(|_| Local::now());
        format!("Recording {}", when.format("%d %b %Y %H:%M"))
    } else {
        stem
    };
    let m = new_meeting(&state, title, p.display().to_string())?;
    let size = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    log::info(
        "import",
        format!("{} ({:.1} MB) → \u{201c}{}\u{201d}", p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(), size as f64 / 1e6, m.title),
    );
    spawn_transcription(app.clone(), m.id.clone(), Ok(p.to_path_buf()));
    Ok(m.id)
}

#[tauri::command]
fn import_audio(app: AppHandle, path: String) -> CmdResult<String> {
    import_path(&app, Path::new(&path))
}

#[tauri::command]
fn retranscribe(app: AppHandle, id: String) -> CmdResult<()> {
    start_retranscribe(app, id)
}

/// Transcribe a meeting again (speakers + text) with the current settings.
pub(crate) fn start_retranscribe(app: AppHandle, id: String) -> CmdResult<()> {
    let input = {
        let state = app.state::<AppState>();
        let mut m = state.store.load_meeting(&id).map_err(err)?;
        let audio = state.store.audio_path(&id);
        let input = if audio.is_file() { audio } else { PathBuf::from(&m.source) };
        m.status = Status::Transcribing;
        m.error = None;
        m.language = state.store.load_settings().language;
        state.store.save_meeting(&m).map_err(err)?;
        input
    };
    spawn_transcription(app, id, Ok(input));
    Ok(())
}

/// Re-run unclear parts (all flagged chunks when `indices` is None) on smaller pieces.
#[tauri::command]
fn retry_chunks(app: AppHandle, id: String, indices: Option<Vec<usize>>) -> CmdResult<()> {
    start_retry(app, id, indices)
}

/// Re-run chosen chunks, or all flagged ones, on smaller pieces in the background.
pub(crate) fn start_retry(app: AppHandle, id: String, indices: Option<Vec<usize>>) -> CmdResult<()> {
    let indices = {
        let m = app.state::<AppState>().store.load_meeting(&id).map_err(err)?;
        indices.unwrap_or_else(|| (0..m.chunks.len()).filter(|&i| !m.chunks[i].flag.is_empty()).collect())
    };
    if indices.is_empty() {
        return Err("nothing to re-run".into());
    }
    std::thread::spawn(move || {
        let state = app.state::<AppState>();
        let emit = |done: usize, total: usize| {
            report(&app, Progress { id: id.clone(), stage: "retrying".into(), done, total });
        };
        emit(0, indices.len());
        let run = || -> anyhow::Result<usize> {
            let pcm = audio::read_wav(&state.store.audio_path(&id))?;
            let mut guard = loaded_engine(&state)?;
            let mut m = state.store.load_meeting(&id)?;
            let fixed = mahdar_core::retry_chunks(guard.as_mut().unwrap(), &pcm, &mut m.chunks, &indices, &m.language, emit)?;
            drop(guard);
            // Merge into a fresh copy so a rename or new output made meanwhile isn't lost.
            let mut latest = state.store.load_meeting(&id)?;
            for &i in &indices {
                if let (Some(dst), Some(src)) = (latest.chunks.get_mut(i), m.chunks.get(i)) {
                    dst.text = src.text.clone();
                    dst.flag = src.flag.clone();
                }
            }
            state.store.save_meeting(&latest)?;
            Ok(fixed)
        };
        let msg = match run() {
            Ok(fixed) => format!("{fixed} of {} part{} now clean", indices.len(), if indices.len() > 1 { "s" } else { "" }),
            Err(e) => format!("re-run failed: {e:#}"),
        };
        log::info("transcribe", format!("re-run in {id}: {msg}"));
        let _ = app.emit("retry-done", (id.clone(), msg));
        clear_progress(&app, &id);
        let _ = app.emit("meeting-updated", id.clone());
    });
    Ok(())
}

// ---------- editing ----------

#[tauri::command]
fn update_chunk(state: State<AppState>, id: String, index: usize, text: String) -> CmdResult<()> {
    let mut m = state.store.load_meeting(&id).map_err(err)?;
    let c = m.chunks.get_mut(index).ok_or("no such line")?;
    c.text = text.trim().to_string();
    c.edited = true;
    c.flag.clear();
    state.store.save_meeting(&m).map_err(err)
}

#[tauri::command]
fn set_speaker_name(state: State<AppState>, id: String, speaker: u32, name: String) -> CmdResult<()> {
    let mut m = state.store.load_meeting(&id).map_err(err)?;
    let name = name.trim();
    if name.is_empty() {
        m.speakers.remove(&speaker);
    } else {
        m.speakers.insert(speaker, name.to_string());
    }
    state.store.save_meeting(&m).map_err(err)
}

/// Change the speaker of one line (fixes a wrong label).
#[tauri::command]
fn set_chunk_speaker(state: State<AppState>, id: String, index: usize, speaker: Option<u32>) -> CmdResult<()> {
    let mut m = state.store.load_meeting(&id).map_err(err)?;
    m.chunks.get_mut(index).ok_or("no such line")?.speaker = speaker;
    state.store.save_meeting(&m).map_err(err)
}

/// Export an output, or the transcript when `output_id` is None, as .docx.
#[tauri::command]
fn export_docx(state: State<AppState>, id: String, output_id: Option<String>, path: String) -> CmdResult<()> {
    let m = state.store.load_meeting(&id).map_err(err)?;
    let date = m.created.format("%d %B %Y").to_string();
    let bytes = match output_id {
        Some(oid) => {
            let o = m.outputs.iter().find(|o| o.id == oid).ok_or("output not found")?;
            mahdar_core::export::markdown_to_docx(&m.title, &format!("{} · {date}", o.template_name), &o.text)
        }
        None => {
            let lines: Vec<(String, String, String)> = m
                .chunks
                .iter()
                .map(|c| (c, mahdar_core::asr::clean_text(&c.text)))
                .filter(|(_, t)| !t.is_empty())
                .map(|(c, t)| (fmt_ts(c.start_s), c.speaker.map(|n| m.speaker_name(n)).unwrap_or_default(), t))
                .collect();
            mahdar_core::export::transcript_to_docx(&m.title, &format!("Transcript · {date}"), &lines)
        }
    }
    .map_err(err)?;
    std::fs::write(path, bytes).map_err(err)
}

// ---------- writing (outputs and questions) ----------

/// Send a prompt to the configured writer, starting the local server if needed.
/// `what` names the job in the log, e.g. "Minutes of Meeting for “<title>”".
/// Returns (text, provider label).
pub(crate) fn write_with_provider(app: &AppHandle, id: &str, what: &str, prompt: &str, reuse: bool) -> CmdResult<(String, String)> {
    let provider = app.state::<AppState>().store.load_settings().provider;
    write_with(app, &provider, id, what, prompt, reuse)
}

/// Like [`write_with_provider`], with an explicit engine (the Brain has its own setting).
pub(crate) fn write_with(app: &AppHandle, provider: &Provider, id: &str, what: &str, prompt: &str, reuse: bool) -> CmdResult<(String, String)> {
    let state = app.state::<AppState>();
    let local = state.store.load_settings().local;
    let label = provider.label();
    log::info("writer", format!("{what} · {label} · {} k chars", prompt.chars().count() / 1000));
    let t0 = Instant::now();
    if let Provider::Local { base_url, model } = provider {
        let child = summarize::ensure_local_server(base_url, model, &local, || {
            log::info("writer", format!("starting local model {model}"));
            report(app, Progress { id: id.to_string(), stage: "starting local model".into(), done: 0, total: 0 });
            let _ = app.emit("gen-stage", (id.to_string(), "starting local model"));
        })
        .map_err(|e| {
            clear_progress(app, id);
            log::error("writer", format!("{what}: {e:#}"));
            err(e)
        })?;
        if let Some(c) = child {
            if let Some(mut old) = state.flm.lock().unwrap().replace(c) {
                let _ = old.kill();
            }
        }
        let _ = app.emit("gen-stage", (id.to_string(), "writing"));
    }
    report(app, Progress { id: id.to_string(), stage: "writing".into(), done: 0, total: 0 });
    let res = summarize::generate(provider, prompt, reuse, &local);
    clear_progress(app, id);
    let reply = res.map_err(|e| {
        log::error("writer", format!("{what} failed after {} s: {e:#}", t0.elapsed().as_secs()));
        err(e)
    })?;
    let cost = match reply.cost_usd {
        Some(usd) => {
            let _ = state.store.add_spend(usd);
            format!(" · ${usd:.3} · {}", reply.usage)
        }
        None => String::new(),
    };
    log::info("writer", format!("{what} done in {} s{cost}", t0.elapsed().as_secs()));
    Ok((reply.text, label))
}

#[tauri::command]
fn list_templates(state: State<AppState>) -> Vec<Template> {
    let mut t = summarize::builtin_templates();
    t.extend(state.store.load_settings().custom_templates);
    t
}

#[tauri::command]
async fn generate_output(app: AppHandle, id: String, template_id: String) -> CmdResult<Output> {
    tauri::async_runtime::spawn_blocking(move || do_generate(&app, &id, &template_id)).await.map_err(err)?
}

/// Write an output for a meeting (blocking). Shared by the desktop UI and Mahdar Remote.
pub(crate) fn do_generate(app: &AppHandle, id: &str, template_id: &str) -> CmdResult<Output> {
    let (app, id, template_id) = (app.clone(), id.to_string(), template_id.to_string());
    {
        let state = app.state::<AppState>();
        let settings = state.store.load_settings();
        let template = summarize::builtin_templates()
            .into_iter()
            .chain(settings.custom_templates.clone())
            .find(|t| t.id == template_id)
            .ok_or("template not found")?;
        let m = state.store.load_meeting(&id).map_err(err)?;
        if m.chunks.is_empty() {
            return Err("this meeting has no transcript yet".to_string());
        }
        let prompt =
            summarize::build_prompt(&template, &settings.output_language, &m.title, &m.transcript_text(), &settings.glossary, &m.notes);
        let what = format!("{} for \u{201c}{}\u{201d}", template.name, m.title);
        let (text, provider) = write_with_provider(&app, &id, &what, &prompt, false)?;
        let out = Output {
            id: uuid::Uuid::new_v4().to_string(),
            template_id: template.id,
            template_name: template.name,
            provider,
            created: Local::now(),
            text,
        };
        // Re-load: the meeting may have changed while the model was writing.
        let mut m = state.store.load_meeting(&id).map_err(err)?;
        m.outputs.insert(0, out.clone());
        state.store.save_meeting(&m).map_err(err)?;
        let _ = app.emit("meeting-updated", id.clone());
        Ok(out)
    }
}

#[tauri::command]
fn delete_output(state: State<AppState>, id: String, output_id: String) -> CmdResult<()> {
    let mut m = state.store.load_meeting(&id).map_err(err)?;
    m.outputs.retain(|o| o.id != output_id);
    state.store.save_meeting(&m).map_err(err)
}

#[tauri::command]
async fn ask_meeting(app: AppHandle, id: String, question: String) -> CmdResult<ChatMsg> {
    tauri::async_runtime::spawn_blocking(move || do_ask(&app, &id, &question)).await.map_err(err)?
}

/// Answer a question about a meeting (blocking). Shared by the desktop UI and Mahdar Remote.
pub(crate) fn do_ask(app: &AppHandle, id: &str, question: &str) -> CmdResult<ChatMsg> {
    let (app, id, question) = (app.clone(), id.to_string(), question.to_string());
    {
        let state = app.state::<AppState>();
        let settings = state.store.load_settings();
        let m = state.store.load_meeting(&id).map_err(err)?;
        if m.chunks.is_empty() {
            return Err("this meeting has no transcript yet".to_string());
        }
        let history: Vec<(String, String)> =
            m.chat.iter().rev().take(10).rev().map(|c| (c.role.clone(), c.text.clone())).collect();
        let prompt =
            summarize::build_question(&m.title, &m.transcript_text(), &settings.glossary, &m.notes, &history, question.trim());
        let what = format!("question about \u{201c}{}\u{201d}", m.title);
        let (text, provider) = write_with_provider(&app, &id, &what, &prompt, true)?;
        let q = ChatMsg { role: "user".into(), text: question.trim().into(), created: Local::now(), provider: String::new() };
        let a = ChatMsg { role: "assistant".into(), text, created: Local::now(), provider };
        let mut m = state.store.load_meeting(&id).map_err(err)?;
        m.chat.push(q);
        m.chat.push(a.clone());
        state.store.save_meeting(&m).map_err(err)?;
        Ok(a)
    }
}

#[tauri::command]
fn clear_chat(state: State<AppState>, id: String) -> CmdResult<()> {
    let mut m = state.store.load_meeting(&id).map_err(err)?;
    m.chat.clear();
    state.store.save_meeting(&m).map_err(err)
}

// ---------- notes ----------

#[tauri::command]
fn save_notes(state: State<AppState>, id: String, notes: String) -> CmdResult<()> {
    let mut m = state.store.load_meeting(&id).map_err(err)?;
    if m.notes != notes {
        m.notes = notes;
        state.store.save_meeting(&m).map_err(err)?;
    }
    Ok(())
}

#[tauri::command]
fn read_notes_file(path: String) -> CmdResult<String> {
    mahdar_core::notes::read_file(Path::new(&path)).map_err(err)
}

/// Seconds of transcript either side of a note's time that the fill-in sees.
const FILL_WINDOW_S: f64 = 180.0;

#[tauri::command]
async fn fill_note(app: AppHandle, id: String, at: f64, note: String) -> CmdResult<String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let settings = state.store.load_settings();
        let m = state.store.load_meeting(&id).map_err(err)?;
        let excerpt = m.transcript_between(at - FILL_WINDOW_S, at + FILL_WINDOW_S);
        if excerpt.is_empty() {
            return Err(format!("nothing was transcribed around {}", fmt_ts(at)));
        }
        let prompt = summarize::build_fill(&m.title, &excerpt, &settings.glossary, at, &note);
        let what = format!("note at {} in \u{201c}{}\u{201d}", fmt_ts(at), m.title);
        // Own job id, so the OUTPUTS timeline doesn't show this as an output being written.
        write_with_provider(&app, &format!("{id}#note"), &what, &prompt, false).map(|(text, _)| text.trim().to_string())
    })
    .await
    .map_err(err)?
}

// ---------- recording ----------

#[tauri::command]
fn start_recording(state: State<AppState>) -> CmdResult<()> {
    let mut rec = state.recorder.lock().unwrap();
    if rec.is_some() {
        return Err("already recording".into());
    }
    *rec = Some((Recorder::start().map_err(err)?, Instant::now()));
    Ok(())
}

#[tauri::command]
fn recording_level(state: State<AppState>) -> Option<(f32, f64)> {
    state.recorder.lock().unwrap().as_ref().map(|(r, t)| (r.level(), t.elapsed().as_secs_f64()))
}

#[tauri::command]
fn stop_recording(app: AppHandle, state: State<AppState>, title: Option<String>, notes: Option<String>) -> CmdResult<String> {
    let (rec, _) = state.recorder.lock().unwrap().take().ok_or("not recording")?;
    let pcm = rec.stop().map_err(err)?;
    let title = title
        .filter(|t| !t.trim().is_empty())
        .unwrap_or_else(|| format!("Recording {}", Local::now().format("%d %b %Y %H:%M")));
    let mut m = new_meeting(&state, title, "microphone".into())?;
    if let Some(n) = notes.filter(|n| !n.trim().is_empty()) {
        m.notes = n;
        state.store.save_meeting(&m).map_err(err)?;
    }
    log::info("record", format!("recorded {} → \u{201c}{}\u{201d}", fmt_ts(pcm.len() as f64 / audio::SAMPLE_RATE as f64), m.title));
    spawn_transcription(app, m.id.clone(), Err(pcm));
    Ok(m.id)
}

fn native_dir(app: &AppHandle) -> PathBuf {
    let bundled = app.path().resource_dir().map(|d| d.join("native")).ok();
    match bundled {
        Some(d) if d.join("transcribe.dll").is_file() => d,
        _ => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("native"),
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            let data_dir = app.path().app_data_dir()?;
            log::init(&data_dir);
            let recovered = std::env::args().any(|a| a == "--recovered");
            log::info(
                "app",
                format!("Mahdar {} started (pid {}){}", env!("CARGO_PKG_VERSION"), std::process::id(), if recovered { " by the watchdog after a crash" } else { "" }),
            );
            watchdog::write_app_file(&data_dir);
            let store = Store::new(data_dir)?;
            let s = store.load_settings();
            store.save_settings(&s)?;
            // Jobs don't survive a restart: mark interrupted transcriptions as failed (re-transcribe
            // fixes them), or, after a crash restart by the watchdog, start them again.
            let mut resume = Vec::new();
            for summary in store.list_meetings()? {
                if summary.status == Status::Transcribing {
                    if recovered {
                        log::warn("app", format!("resuming interrupted transcription \u{201c}{}\u{201d}", summary.title));
                        resume.push(summary.id.clone());
                    } else if let Ok(mut m) = store.load_meeting(&summary.id) {
                        log::warn("app", format!("\u{201c}{}\u{201d} was interrupted while transcribing", m.title));
                        m.status = Status::Failed;
                        m.error = Some("Mahdar was closed while transcribing. Use Re-transcribe.".into());
                        store.save_meeting(&m)?;
                    }
                }
            }
            let watchdog_on = s.watchdog;
            let native_dir = native_dir(app.handle());
            app.manage(AppState {
                store,
                native_dir,
                engine: Mutex::new(None),
                diarizer: Mutex::new(None),
                recorder: Mutex::new(None),
                flm: Mutex::new(None),
                remote: Mutex::new(None),
                progress: Mutex::new(Default::default()),
                remote_state: Mutex::new("starting".into()),
                brain: Default::default(),
                update: Default::default(),
                started: Local::now(),
            });
            for id in resume {
                let _ = start_retranscribe(app.handle().clone(), id);
            }
            if watchdog_on {
                watchdog::sync(&app.path().app_data_dir()?, true);
            }
            watch::spawn(app.handle().clone());
            // Make phone copies for meetings that don't have one yet (one at a time, in the background).
            let h = app.handle().clone();
            std::thread::spawn(move || {
                let state = h.state::<AppState>();
                for m in state.store.list_meetings().unwrap_or_default() {
                    let wav = state.store.audio_path(&m.id);
                    if wav.is_file() {
                        let _ = audio::phone_copy(&wav);
                    }
                }
            });
            restart_remote(app.handle());
            update::spawn(app.handle().clone());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_settings,
            save_settings,
            engine_status,
            download_model,
            list_meetings,
            get_meeting,
            rename_meeting,
            delete_meeting,
            audio_path,
            save_text_file,
            import_audio,
            clipboard_files,
            retranscribe,
            retry_chunks,
            update_chunk,
            set_speaker_name,
            set_chunk_speaker,
            export_docx,
            start_recording,
            recording_level,
            stop_recording,
            list_templates,
            generate_output,
            delete_output,
            ask_meeting,
            clear_chat,
            save_notes,
            read_notes_file,
            fill_note,
            remote_info,
            remote_new_token,
            remote_setup_https,
            api_key_status,
            api_key_save,
            api_key_test,
            api_key_delete,
            monitor_snapshot,
            brain::brain_status,
            brain::brain_index,
            brain::brain_entities,
            brain::brain_entity,
            brain::brain_ask,
            brain::brain_asks,
            brain::brain_thread,
            brain::brain_delete_thread,
            open_log_folder,
            open_data_folder,
            update::update_status,
            update::update_check,
            update::update_install,
            update::update_open_releases,
        ])
        .build(tauri::generate_context!())
        .expect("error while building Mahdar")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                let state = app.state::<AppState>();
                if let Some(mut child) = state.flm.lock().unwrap().take() {
                    let _ = child.kill();
                }
                // A normal close: the watchdog must not restart Mahdar.
                watchdog::remove_app_file(state.store.root());
                log::info("app", "Mahdar closed");
                update::install_on_exit(app);
            }
        });
}
