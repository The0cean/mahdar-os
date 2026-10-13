//! Mahdar Remote: a small token-protected HTTP API + phone web UI, so the iPhone
//! (Safari now, the native app later) can upload recordings, read transcripts and
//! minutes, generate outputs and ask questions. Reachable on the LAN and over
//! Tailscale; `tailscale serve` adds HTTPS, which Safari needs for the microphone.

use std::net::SocketAddr;
use std::path::PathBuf;

use axum::body::Bytes;
use axum::extract::{ConnectInfo, DefaultBodyLimit, Path, Query, Request, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use tauri::{AppHandle, Emitter, Manager};
use tower::ServiceExt;
use tower_http::compression::predicate::{DefaultPredicate, NotForContentType, Predicate};
use tower_http::compression::CompressionLayer;
use tower_http::services::ServeFile;

use crate::AppState;
use mahdar_core::log;

const PAGE: &str = include_str!("../remote/index.html");
const COOKIE: &str = "mahdar_token";

#[derive(Clone)]
struct Ctx {
    app: AppHandle,
    token: String,
}

type ApiResult<T> = Result<Json<T>, (StatusCode, String)>;

fn bad(e: impl std::fmt::Display) -> (StatusCode, String) {
    (StatusCode::BAD_REQUEST, e.to_string())
}

fn set_state(app: &AppHandle, msg: String) {
    *app.state::<AppState>().remote_state.lock().unwrap() = msg.clone();
    let _ = app.emit("remote-state", msg);
}

/// Start the server; the returned task is aborted to stop it. If the port can't be opened or
/// the server stops, it tries again every 30 s.
pub fn start(app: AppHandle, port: u16, token: String) -> tauri::async_runtime::JoinHandle<()> {
    tauri::async_runtime::spawn(async move {
        let ctx = Ctx { app: app.clone(), token };
        let api = Router::new()
            .route("/api/status", get(status))
            .route("/api/meetings", get(list))
            .route("/api/meetings/{id}", get(meeting).delete(delete))
            .route("/api/meetings/{id}/audio", get(audio))
            .route("/api/meetings/{id}/audio-m4a", get(audio_m4a))
            .route("/api/meetings/{id}/docx", get(docx))
            .route("/api/meetings/{id}/retry", post(retry))
            .route("/api/meetings/{id}/retranscribe", post(retranscribe))
            .route("/api/meetings/{id}/chunks/{i}/text", post(chunk_text))
            .route("/api/meetings/{id}/chunks/{i}/speaker", post(chunk_speaker))
            .route("/api/meetings/{id}/outputs/{oid}", axum::routing::delete(delete_output))
            .route("/api/progress", get(progress))
            .route("/api/monitor", get(monitor))
            .route("/api/brain", get(brain_status))
            .route("/api/brain/index", post(brain_index))
            .route("/api/brain/entities", get(brain_entities))
            .route("/api/brain/entity", get(brain_entity))
            .route("/api/brain/ask", post(brain_ask))
            .route("/api/brain/asks", get(brain_asks))
            .route("/api/log", post(phone_log))
            .route("/api/settings", get(get_settings).post(set_settings))
            .route("/api/meetings/{id}/generate", post(generate))
            .route("/api/meetings/{id}/ask", post(ask))
            .route("/api/meetings/{id}/rename", post(rename))
            .route("/api/meetings/{id}/speaker", post(speaker))
            .route("/api/templates", get(templates))
            .route("/api/upload", post(upload))
            .layer(DefaultBodyLimit::max(2 << 30)) // recordings up to 2 GB
            .route("/", get(page))
            .layer(middleware::from_fn_with_state(ctx.clone(), auth))
            // Unauthenticated: "ok" for the watchdog, and the app manifest (browsers fetch it
            // without cookies; it holds only the name and colours).
            .route("/api/health", get(|| async { "ok" }))
            .route("/manifest.webmanifest", get(manifest))
            .layer(middleware::from_fn(access_log))
            .layer(
                CompressionLayer::new().compress_when(
                    DefaultPredicate::new()
                        .and(NotForContentType::const_new("audio/"))
                        .and(NotForContentType::const_new("application/vnd")),
                ),
            )
            .with_state(ctx);
        let addr = SocketAddr::from(([0, 0, 0, 0], port));
        let mut logged_fail = false;
        loop {
            match tokio::net::TcpListener::bind(addr).await {
                Ok(listener) => {
                    logged_fail = false;
                    log::info("phone", format!("server listening on port {port}"));
                    set_state(&app, format!("listening on port {port}"));
                    let svc = api.clone().into_make_service_with_connect_info::<SocketAddr>();
                    match axum::serve(listener, svc).await {
                        Ok(()) => log::warn("phone", "server stopped; restarting in 30 s"),
                        Err(e) => log::error("phone", format!("server stopped: {e}; restarting in 30 s")),
                    }
                    set_state(&app, "stopped, restarting".into());
                }
                Err(e) => {
                    if !logged_fail {
                        log::error("phone", format!("could not open port {port}: {e} (retrying every 30 s)"));
                        logged_fail = true;
                    }
                    set_state(&app, format!("could not open port {port}: {e}"));
                }
            }
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        }
    })
}

/// Log what the phone does: page loads, changes, downloads, errors and slow requests. Routine
/// polling and audio range requests are left out unless they fail.
async fn access_log(ConnectInfo(addr): ConnectInfo<SocketAddr>, req: Request, next: Next) -> Response {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let t0 = std::time::Instant::now();
    let resp = next.run(req).await;
    let ms = t0.elapsed().as_millis();
    let status = resp.status();
    let who = if addr.ip().is_loopback() { "via Tailscale HTTPS".to_string() } else { addr.ip().to_string() };
    let audio = path.ends_with("/audio") || path.ends_with("/audio-m4a");
    let routine = method == axum::http::Method::GET && path != "/" && !path.ends_with("/docx");
    let line = format!("{method} {path} → {} in {ms} ms ({who})", status.as_u16());
    if status == StatusCode::UNAUTHORIZED {
        if path != "/favicon.ico" {
            log::warn("phone", format!("{line}: not paired"));
        }
    } else if status.is_server_error() {
        log::error("phone", line);
    } else if status.is_client_error() {
        log::warn("phone", line);
    } else if ms > 3000 && !audio && path != "/api/health" {
        log::warn("phone", format!("slow: {line}"));
    } else if !routine {
        log::info("phone", line);
    }
    resp
}

/// Every request needs the token: `?t=` (sets a cookie), the cookie, or an `x-mahdar-token` header.
async fn auth(State(ctx): State<Ctx>, req: Request, next: Next) -> Response {
    let q_token = req.uri().query().and_then(|q| {
        q.split('&').find_map(|kv| kv.strip_prefix("t=").map(|v| v.to_string()))
    });
    let headers = req.headers();
    let header_token = headers.get("x-mahdar-token").and_then(|v| v.to_str().ok()).map(str::to_string);
    let cookie_token = headers
        .get(header::COOKIE)
        .and_then(|v| v.to_str().ok())
        .and_then(|c| c.split(';').find_map(|kv| kv.trim().strip_prefix(&format!("{COOKIE}=")).map(str::to_string)));
    let ok = |t: &Option<String>| t.as_deref() == Some(ctx.token.as_str());
    if ok(&header_token) || ok(&cookie_token) {
        return next.run(req).await;
    }
    if ok(&q_token) {
        let mut resp = next.run(req).await;
        let cookie = format!("{COOKIE}={}; Path=/; Max-Age=31536000; HttpOnly; SameSite=Strict", ctx.token);
        resp.headers_mut().insert(header::SET_COOKIE, cookie.parse().unwrap());
        return resp;
    }
    (StatusCode::UNAUTHORIZED, Html("<h3 style='font-family:monospace'>MAHDAR // scan the QR code in Mahdar → Settings → Phone</h3>"))
        .into_response()
}

async fn page() -> impl IntoResponse {
    ([(header::CACHE_CONTROL, "no-store")], Html(PAGE))
}

async fn manifest() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/manifest+json")],
        r##"{"name":"Mahdar","short_name":"Mahdar","start_url":"/","display":"standalone","background_color":"#0a0a0a","theme_color":"#0a0a0a"}"##,
    )
}

fn state(ctx: &Ctx) -> tauri::State<'_, AppState> {
    ctx.app.state::<AppState>()
}

async fn status(State(ctx): State<Ctx>) -> ApiResult<serde_json::Value> {
    let s = state(&ctx).store.load_settings();
    Ok(Json(serde_json::json!({ "app": "mahdar", "version": env!("CARGO_PKG_VERSION"), "provider": s.provider.label() })))
}

async fn list(State(ctx): State<Ctx>) -> ApiResult<Vec<mahdar_core::store::MeetingSummary>> {
    state(&ctx).store.list_meetings().map(Json).map_err(bad)
}

async fn meeting(State(ctx): State<Ctx>, Path(id): Path<String>) -> ApiResult<mahdar_core::Meeting> {
    state(&ctx).store.load_meeting(&id).map(Json).map_err(bad)
}

async fn delete(State(ctx): State<Ctx>, Path(id): Path<String>) -> ApiResult<bool> {
    state(&ctx).store.delete_meeting(&id).map_err(bad)?;
    crate::brain::forget(&ctx.app, &id);
    let _ = ctx.app.emit("meeting-updated", id);
    Ok(Json(true))
}

async fn audio(State(ctx): State<Ctx>, Path(id): Path<String>, req: Request) -> Response {
    let path = state(&ctx).store.audio_path(&id);
    match ServeFile::new(path).oneshot(req).await {
        Ok(r) => r.into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

/// Compressed copy for phones; made after transcription, or here on first request.
async fn audio_m4a(State(ctx): State<Ctx>, Path(id): Path<String>, req: Request) -> Response {
    let wav = state(&ctx).store.audio_path(&id);
    let made = tokio::task::spawn_blocking(move || mahdar_core::audio::phone_copy(&wav)).await;
    match made {
        Ok(Ok(m4a)) => match ServeFile::new_with_mime(m4a, &"audio/mp4".parse().unwrap()).oneshot(req).await {
            Ok(r) => r.into_response(),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        },
        // Fall back to the WAV rather than failing playback.
        _ => audio(State(ctx), Path(id), req).await,
    }
}

#[derive(Deserialize)]
struct DocxQuery {
    output: Option<String>,
}

/// Word document of an output (`?output=<id>`) or of the transcript.
async fn docx(State(ctx): State<Ctx>, Path(id): Path<String>, Query(q): Query<DocxQuery>) -> Response {
    let m = match state(&ctx).store.load_meeting(&id) {
        Ok(m) => m,
        Err(e) => return bad(e).into_response(),
    };
    let date = m.created.format("%d %B %Y").to_string();
    let (bytes, name) = match q.output {
        Some(oid) => match m.outputs.iter().find(|o| o.id == oid) {
            Some(o) => (
                mahdar_core::export::markdown_to_docx(&m.title, &format!("{} · {date}", o.template_name), &o.text),
                format!("{} - {}", m.title, o.template_name),
            ),
            None => return bad("output not found").into_response(),
        },
        None => {
            let lines: Vec<(String, String, String)> = m
                .chunks
                .iter()
                .map(|c| (c, mahdar_core::asr::clean_text(&c.text)))
                .filter(|(_, t)| !t.is_empty())
                .map(|(c, t)| (mahdar_core::store::fmt_ts(c.start_s), c.speaker.map(|n| m.speaker_name(n)).unwrap_or_default(), t))
                .collect();
            (
                mahdar_core::export::transcript_to_docx(&m.title, &format!("Transcript · {date}"), &lines),
                format!("{} - Transcript", m.title),
            )
        }
    };
    match bytes {
        Ok(b) => {
            let safe: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || " -_().".contains(c) { c } else { '_' }).collect();
            let disposition = format!("attachment; filename=\"{safe}.docx\"");
            (
                [
                    (header::CONTENT_TYPE, "application/vnd.openxmlformats-officedocument.wordprocessingml.document".to_string()),
                    (header::CONTENT_DISPOSITION, disposition),
                ],
                b,
            )
                .into_response()
        }
        Err(e) => bad(e).into_response(),
    }
}

#[derive(Deserialize)]
struct RetryBody {
    indices: Option<Vec<usize>>,
}

async fn retry(State(ctx): State<Ctx>, Path(id): Path<String>, Json(b): Json<RetryBody>) -> ApiResult<bool> {
    crate::start_retry(ctx.app.clone(), id, b.indices).map_err(bad)?;
    Ok(Json(true))
}

async fn retranscribe(State(ctx): State<Ctx>, Path(id): Path<String>) -> ApiResult<bool> {
    crate::start_retranscribe(ctx.app.clone(), id).map_err(bad)?;
    Ok(Json(true))
}

#[derive(Deserialize)]
struct TextBody {
    text: String,
}

async fn chunk_text(State(ctx): State<Ctx>, Path((id, i)): Path<(String, usize)>, Json(b): Json<TextBody>) -> ApiResult<bool> {
    let st = state(&ctx);
    let mut m = st.store.load_meeting(&id).map_err(bad)?;
    let c = m.chunks.get_mut(i).ok_or_else(|| bad("no such line"))?;
    c.text = b.text.trim().to_string();
    c.edited = true;
    c.flag.clear();
    st.store.save_meeting(&m).map_err(bad)?;
    let _ = ctx.app.emit("meeting-updated", id);
    Ok(Json(true))
}

#[derive(Deserialize)]
struct ChunkSpeakerBody {
    speaker: Option<u32>,
}

async fn chunk_speaker(
    State(ctx): State<Ctx>,
    Path((id, i)): Path<(String, usize)>,
    Json(b): Json<ChunkSpeakerBody>,
) -> ApiResult<bool> {
    let st = state(&ctx);
    let mut m = st.store.load_meeting(&id).map_err(bad)?;
    m.chunks.get_mut(i).ok_or_else(|| bad("no such line"))?.speaker = b.speaker;
    st.store.save_meeting(&m).map_err(bad)?;
    let _ = ctx.app.emit("meeting-updated", id);
    Ok(Json(true))
}

async fn delete_output(State(ctx): State<Ctx>, Path((id, oid)): Path<(String, String)>) -> ApiResult<bool> {
    let st = state(&ctx);
    let mut m = st.store.load_meeting(&id).map_err(bad)?;
    m.outputs.retain(|o| o.id != oid);
    st.store.save_meeting(&m).map_err(bad)?;
    let _ = ctx.app.emit("meeting-updated", id);
    Ok(Json(true))
}

/// Errors from the phone page's scripts, into the activity log.
async fn phone_log(body: String) -> ApiResult<bool> {
    let msg: String = body.chars().take(2000).collect();
    if !msg.trim().is_empty() {
        log::warn("phone", format!("page: {msg}"));
    }
    Ok(Json(true))
}

async fn monitor(State(ctx): State<Ctx>) -> ApiResult<crate::Monitor> {
    let app = ctx.app.clone();
    tokio::task::spawn_blocking(move || crate::monitor(&app, 200)).await.map(Json).map_err(bad)
}

async fn progress(State(ctx): State<Ctx>) -> ApiResult<std::collections::HashMap<String, crate::Progress>> {
    Ok(Json(state(&ctx).progress.lock().unwrap().clone()))
}

/// The settings a phone may see and change (no paths, ports or tokens).
#[derive(serde::Serialize, Deserialize)]
struct PhoneSettings {
    provider: mahdar_core::summarize::Provider,
    output_language: String,
    language: String,
    speakers: bool,
    glossary: String,
    #[serde(default)]
    brain_provider: Option<mahdar_core::summarize::Provider>,
}

async fn get_settings(State(ctx): State<Ctx>) -> ApiResult<PhoneSettings> {
    let s = state(&ctx).store.load_settings();
    Ok(Json(PhoneSettings {
        provider: s.provider,
        output_language: s.output_language,
        language: s.language,
        speakers: s.speakers,
        glossary: s.glossary,
        brain_provider: s.brain_provider,
    }))
}

async fn set_settings(State(ctx): State<Ctx>, Json(b): Json<PhoneSettings>) -> ApiResult<bool> {
    let st = state(&ctx);
    let mut s = st.store.load_settings();
    s.provider = b.provider;
    s.output_language = b.output_language;
    s.language = b.language;
    s.speakers = b.speakers;
    s.glossary = b.glossary;
    s.brain_provider = b.brain_provider;
    st.store.save_settings(&s).map_err(bad)?;
    let _ = ctx.app.emit("settings-changed", ());
    Ok(Json(true))
}

#[derive(Deserialize)]
struct GenerateBody {
    template_id: String,
}

/// Starts writing in the background and returns at once; the phone polls /api/progress.
async fn generate(State(ctx): State<Ctx>, Path(id): Path<String>, Json(b): Json<GenerateBody>) -> ApiResult<bool> {
    let jid = id.clone();
    crate::spawn_job(&ctx.app, &id, move |app| crate::do_generate(app, &jid, &b.template_id).map(|_| ()));
    Ok(Json(true))
}

#[derive(Deserialize)]
struct AskBody {
    question: String,
}

/// Starts answering in the background and returns at once; the phone polls /api/progress.
async fn ask(State(ctx): State<Ctx>, Path(id): Path<String>, Json(b): Json<AskBody>) -> ApiResult<bool> {
    let jid = id.clone();
    crate::spawn_job(&ctx.app, &id, move |app| crate::do_ask(app, &jid, &b.question).map(|_| ()));
    Ok(Json(true))
}

#[derive(Deserialize)]
struct RenameBody {
    title: String,
}

async fn rename(State(ctx): State<Ctx>, Path(id): Path<String>, Json(b): Json<RenameBody>) -> ApiResult<bool> {
    let st = state(&ctx);
    let mut m = st.store.load_meeting(&id).map_err(bad)?;
    m.title = b.title.trim().to_string();
    st.store.save_meeting(&m).map_err(bad)?;
    let _ = ctx.app.emit("meeting-updated", id);
    Ok(Json(true))
}

#[derive(Deserialize)]
struct SpeakerBody {
    speaker: u32,
    name: String,
}

async fn speaker(State(ctx): State<Ctx>, Path(id): Path<String>, Json(b): Json<SpeakerBody>) -> ApiResult<bool> {
    let st = state(&ctx);
    let mut m = st.store.load_meeting(&id).map_err(bad)?;
    if b.name.trim().is_empty() {
        m.speakers.remove(&b.speaker);
    } else {
        m.speakers.insert(b.speaker, b.name.trim().to_string());
    }
    st.store.save_meeting(&m).map_err(bad)?;
    Ok(Json(true))
}

async fn templates(State(ctx): State<Ctx>) -> ApiResult<Vec<mahdar_core::summarize::Template>> {
    let mut t = mahdar_core::summarize::builtin_templates();
    t.extend(state(&ctx).store.load_settings().custom_templates);
    Ok(Json(t))
}

#[derive(Deserialize)]
struct UploadQuery {
    name: Option<String>,
    title: Option<String>,
}

/// Raw audio in the request body; `name` gives the file extension, `title` the meeting title.
async fn upload(State(ctx): State<Ctx>, Query(q): Query<UploadQuery>, headers: HeaderMap, body: Bytes) -> ApiResult<String> {
    if body.len() < 1024 {
        return Err(bad("empty recording"));
    }
    let name = q.name.unwrap_or_else(|| "recording.m4a".into());
    let ext = std::path::Path::new(&name)
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .filter(|e| crate::watch::AUDIO_EXT.contains(&e.as_str()))
        .or_else(|| {
            let ct = headers.get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("");
            Some(if ct.contains("webm") { "webm" } else if ct.contains("wav") { "wav" } else { "m4a" }.to_string())
        })
        .unwrap();
    let st = state(&ctx);
    let dir: PathBuf = st.store.root().join("uploads");
    std::fs::create_dir_all(&dir).map_err(bad)?;
    let stem = q.title.clone().filter(|t| !t.trim().is_empty()).unwrap_or_else(|| {
        format!("Phone recording {}", chrono::Local::now().format("%d %b %Y %H-%M"))
    });
    let safe: String = stem.chars().map(|c| if r#"\/:*?"<>|"#.contains(c) { '-' } else { c }).collect();
    let path = dir.join(format!("{}-{}.{ext}", chrono::Local::now().format("%Y%m%d-%H%M%S"), safe.trim()));
    std::fs::write(&path, &body).map_err(bad)?;
    let id = crate::import_path(&ctx.app, &path).map_err(bad)?;
    // Title: the one typed on the phone, else "Phone recording 11 Oct 2026 21:00" (not the file name).
    let title = q.title.filter(|t| !t.trim().is_empty()).map(|t| t.trim().to_string())
        .unwrap_or_else(|| format!("Phone recording {}", chrono::Local::now().format("%d %b %Y %H:%M")));
    if let Ok(mut m) = st.store.load_meeting(&id) {
        m.title = title;
        let _ = st.store.save_meeting(&m);
    }
    let _ = ctx.app.emit("watch-imported", (id.clone(), format!("from phone: {name}")));
    Ok(Json(id))
}

/// Reachable addresses as (kind, "ip  (interface)"), best first: Tailscale, then the
/// adapter that carries the default route ("wifi"), then other LAN adapters ("lan").
/// Virtual switches (WSL/Hyper-V) are skipped. Also returns the tailnet DNS name.
pub fn addresses() -> (Vec<(String, String)>, Option<String>) {
    let primary = primary_lan_ip().or_else(|| local_ip_address::local_ip().ok());
    let mut out: Vec<(u8, String, String)> = Vec::new();
    if let Ok(list) = local_ip_address::list_afinet_netifas() {
        for (name, ip) in list {
            let std::net::IpAddr::V4(v4) = ip else { continue };
            if v4.is_loopback() || v4.is_link_local() || name.starts_with("vEthernet") {
                continue;
            }
            let o = v4.octets();
            let (rank, kind) = if o[0] == 100 && (64..128).contains(&o[1]) {
                (0, "tailscale")
            } else if Some(ip) == primary {
                (1, "wifi")
            } else if v4.is_private() {
                (2, "lan")
            } else {
                continue;
            };
            out.push((rank, kind.to_string(), format!("{v4}  ({name})")));
        }
    }
    out.sort_by_key(|(r, _, _)| *r);
    (out.into_iter().map(|(_, k, v)| (k, v)).collect(), tailscale_dns())
}

/// The private IPv4 of the adapter with the best default route (lowest route + interface
/// metric). VPNs like Radmin also carry a default route, so "any default route" isn't enough.
fn primary_lan_ip() -> Option<std::net::IpAddr> {
    if !cfg!(windows) {
        return None;
    }
    let script = "Get-NetRoute -DestinationPrefix 0.0.0.0/0 | Sort-Object { $_.RouteMetric + $_.InterfaceMetric } |         ForEach-Object { (Get-NetIPAddress -InterfaceIndex $_.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue).IPAddress }";
    let mut cmd = std::process::Command::new("powershell");
    cmd.args(["-NoProfile", "-NonInteractive", "-Command", script]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let out = cmd.output().ok()?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().parse::<std::net::Ipv4Addr>().ok())
        .find(|ip| ip.is_private())
        .map(std::net::IpAddr::V4)
}

fn tailscale_exe() -> String {
    let p = r"C:\Program Files\Tailscale\tailscale.exe";
    if std::path::Path::new(p).is_file() { p.into() } else { "tailscale".into() }
}

fn tailscale_dns() -> Option<String> {
    let mut cmd = std::process::Command::new(tailscale_exe());
    cmd.args(["status", "--json"]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let out = cmd.output().ok()?;
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    if v["BackendState"] != "Running" {
        return None;
    }
    v["Self"]["DNSName"].as_str().map(|s| s.trim_end_matches('.').to_string()).filter(|s| !s.is_empty())
}

/// Point Tailscale HTTPS (port 443 on the tailnet name) at the local server.
pub fn setup_https(port: u16) -> Result<String, String> {
    let dns = tailscale_dns().ok_or("Tailscale isn't logged in on this laptop. Open Tailscale and log in first.")?;
    let mut cmd = std::process::Command::new(tailscale_exe());
    cmd.args(["serve", "--bg", &port.to_string()]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let out = cmd.output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        let msg = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        return Err(format!("tailscale serve failed: {}", msg.trim()));
    }
    Ok(format!("https://{dns}"))
}

/// QR code (SVG) for a URL.
pub fn qr_svg(data: &str) -> String {
    match qrcode::QrCode::new(data.as_bytes()) {
        Ok(code) => code
            .render::<qrcode::render::svg::Color>()
            .min_dimensions(220, 220)
            .dark_color(qrcode::render::svg::Color("#000000"))
            .light_color(qrcode::render::svg::Color("#ffffff"))
            .build(),
        Err(_) => String::new(),
    }
}

// ---------- Brain ----------

async fn brain_status(State(ctx): State<Ctx>) -> ApiResult<crate::brain::BrainStatus> {
    let app = ctx.app.clone();
    tokio::task::spawn_blocking(move || crate::brain::status(&app)).await.map_err(bad)?.map(Json).map_err(bad)
}

#[derive(Deserialize)]
struct BrainIndexBody {
    ids: Option<Vec<String>>,
}

async fn brain_index(State(ctx): State<Ctx>, Json(b): Json<BrainIndexBody>) -> ApiResult<usize> {
    let app = ctx.app.clone();
    tokio::task::spawn_blocking(move || crate::brain::start_index(&app, b.ids)).await.map_err(bad)?.map(Json).map_err(bad)
}

async fn brain_entities(State(ctx): State<Ctx>) -> ApiResult<Vec<mahdar_brain::EntityRow>> {
    crate::brain::entities(&ctx.app).map(Json).map_err(bad)
}

#[derive(Deserialize)]
struct KeyQuery {
    key: String,
}

async fn brain_entity(State(ctx): State<Ctx>, Query(q): Query<KeyQuery>) -> ApiResult<mahdar_brain::EntityPage> {
    crate::brain::entity(&ctx.app, &q.key).map(Json).map_err(bad)
}

/// Starts the question in the background; the phone polls /api/progress ("brain-ask"), then /api/brain/asks.
async fn brain_ask(State(ctx): State<Ctx>, Json(b): Json<crate::brain::AskBody>) -> ApiResult<bool> {
    crate::spawn_job(&ctx.app, crate::brain::ASK_JOB, move |app| crate::brain::ask(app, &b).map(|_| ()));
    Ok(Json(true))
}

async fn brain_asks(State(ctx): State<Ctx>) -> ApiResult<Vec<mahdar_brain::AskRow>> {
    crate::brain::asks(&ctx.app).map(Json).map_err(bad)
}
