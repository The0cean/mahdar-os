//! Auto-update from GitHub Releases, like Cursor or Teams.
//!
//! Every release carries a signed installer and `latest.json` (see README › Releasing).
//! Mahdar checks a minute after it starts and then every 6 hours, downloads a newer version
//! quietly, and shows UPDATE READY. Installing waits until nothing is running (recording,
//! transcribing, writing, Brain indexing). RESTART installs and reopens Mahdar; closing
//! Mahdar with an update waiting installs it without reopening. The installer only accepts
//! files signed with Mahdar's update key, so a public download URL is safe.

use std::sync::Mutex;
use std::time::Duration;

use mahdar_core::log;
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_updater::{Update, UpdaterExt};

use crate::{err, watchdog, AppState, CmdResult};

const FIRST_CHECK: Duration = Duration::from_secs(60);
const CHECK_EVERY: Duration = Duration::from_secs(6 * 3600);

#[derive(Serialize, Clone, Default)]
pub struct UpdateInfo {
    pub current: String,
    /// "idle" | "checking" | "downloading" | "ready" | "latest" | "error"
    pub state: String,
    /// The newer version, when one was found.
    pub version: String,
    /// Release notes (Markdown).
    pub notes: String,
    pub date: String,
    /// Download progress, 0–100.
    pub percent: u8,
    pub last_check: String,
    pub error: String,
    /// What has to finish before the update can install; empty = nothing.
    pub waiting_for: String,
}

#[derive(Default)]
pub struct UpdateState {
    info: Mutex<UpdateInfo>,
    /// The downloaded update, ready to install.
    ready: Mutex<Option<(Update, Vec<u8>)>>,
    checking: Mutex<bool>,
}

/// Status for Settings, the top bar and the phone.
pub fn info(app: &AppHandle) -> UpdateInfo {
    let mut i = app.state::<AppState>().update.info.lock().unwrap().clone();
    i.current = app.package_info().version.to_string();
    if i.state.is_empty() {
        i.state = "idle".into();
    }
    i.waiting_for = busy_reason(app).unwrap_or_default();
    i
}

fn set(app: &AppHandle, f: impl FnOnce(&mut UpdateInfo)) {
    f(&mut app.state::<AppState>().update.info.lock().unwrap());
    let _ = app.emit("update", info(app));
}

/// What would be cut off by installing now, if anything.
pub fn busy_reason(app: &AppHandle) -> Option<String> {
    let state = app.state::<AppState>();
    if state.recorder.lock().unwrap().is_some() {
        return Some("the recording".into());
    }
    if !state.progress.lock().unwrap().is_empty() {
        return Some("transcription or writing".into());
    }
    if state.brain.is_busy() {
        return Some("Brain indexing".into());
    }
    None
}

/// Check now; download a newer version if there is one. `manual`: CHECK NOW in Settings
/// (logs "up to date" too).
pub async fn check(app: AppHandle, manual: bool) -> CmdResult<UpdateInfo> {
    {
        let state = app.state::<AppState>();
        let mut checking = state.update.checking.lock().unwrap();
        if *checking {
            return Ok(info(&app));
        }
        *checking = true;
    }
    let res = check_inner(&app, manual).await;
    *app.state::<AppState>().update.checking.lock().unwrap() = false;
    let now = chrono::Local::now().to_rfc3339();
    match res {
        Ok(()) => set(&app, |i| i.last_check = now),
        Err(e) => {
            log::warn("update", format!("update check failed: {e}"));
            set(&app, |i| {
                i.last_check = now;
                i.state = "error".into();
                i.error = e;
            });
        }
    }
    Ok(info(&app))
}

async fn check_inner(app: &AppHandle, manual: bool) -> CmdResult<()> {
    if app.state::<AppState>().update.ready.lock().unwrap().is_some() {
        return Ok(()); // already downloaded; a newer one is picked up after it installs
    }
    set(app, |i| {
        i.state = "checking".into();
        i.error.clear();
    });
    let data = app.state::<AppState>().store.root().to_path_buf();
    let h = app.clone();
    let updater = app
        .updater_builder()
        .on_before_exit(move || {
            // The installer replaces mahdar.exe, which the watchdog also runs from.
            let state = h.state::<AppState>();
            if let Some(mut child) = state.flm.lock().unwrap().take() {
                let _ = child.kill();
            }
            watchdog::stop_for_update(&data);
            log::info("update", "installing the update; Mahdar closes now");
        })
        .build()
        .map_err(err)?;
    let Some(update) = updater.check().await.map_err(err)? else {
        if manual {
            log::info("update", format!("Mahdar {} is up to date", app.package_info().version));
        }
        set(app, |i| i.state = "latest".into());
        return Ok(());
    };
    log::info("update", format!("Mahdar {} is available; downloading", update.version));
    let notes = update.body.clone().unwrap_or_default();
    let date = update.date.map(|d| d.date().to_string()).unwrap_or_default();
    let version = update.version.clone();
    set(app, |i| {
        i.state = "downloading".into();
        i.version = version.clone();
        i.notes = notes;
        i.date = date;
        i.percent = 0;
    });
    let (mut got, h) = (0usize, app.clone());
    let mut last = 0u8;
    let bytes = update
        .download(
            move |n, total| {
                got += n;
                if let Some(t) = total.filter(|t| *t > 0) {
                    let p = ((got as u64 * 100) / t).min(100) as u8;
                    if p >= last + 5 {
                        last = p;
                        set(&h, |i| i.percent = p);
                    }
                }
            },
            || {},
        )
        .await
        .map_err(err)?;
    log::info("update", format!("Mahdar {version} downloaded; it installs on RESTART or when Mahdar closes"));
    *app.state::<AppState>().update.ready.lock().unwrap() = Some((update, bytes));
    set(app, |i| {
        i.state = "ready".into();
        i.percent = 100;
    });
    Ok(())
}

/// RESTART: install the downloaded update and reopen Mahdar. On success this never returns
/// (the process exits and the installer takes over).
pub fn install_now(app: &AppHandle) -> CmdResult<()> {
    if let Some(r) = busy_reason(app) {
        return Err(format!("Wait until {r} finishes, then restart to update."));
    }
    install(app, true)
}

/// Mahdar is closing: install a waiting update quietly, without reopening.
pub fn install_on_exit(app: &AppHandle) {
    let auto = app.state::<AppState>().store.load_settings().auto_update;
    if auto && busy_reason(app).is_none() && app.state::<AppState>().update.ready.lock().unwrap().is_some() {
        let _ = install(app, false);
    }
}

fn install(app: &AppHandle, reopen: bool) -> CmdResult<()> {
    let (update, bytes) = app.state::<AppState>().update.ready.lock().unwrap().take().ok_or("no update downloaded yet")?;
    log::info("update", format!("installing Mahdar {}{}", update.version, if reopen { "; Mahdar reopens when done" } else { "" }));
    let update = update.restart_after_install(reopen);
    if let Err(e) = update.install(&bytes) {
        log::error("update", format!("could not start the installer: {e}"));
        *app.state::<AppState>().update.ready.lock().unwrap() = Some((update, bytes));
        return Err(err(e));
    }
    Ok(())
}

/// Background checks: a minute after start, then every 6 hours (when auto-update is on).
pub fn spawn(app: AppHandle) {
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(FIRST_CHECK).await;
        loop {
            if app.state::<AppState>().store.load_settings().auto_update && cfg!(not(debug_assertions)) {
                let _ = check(app.clone(), false).await;
            }
            tokio::time::sleep(CHECK_EVERY).await;
        }
    });
}

#[tauri::command]
pub fn update_status(app: AppHandle) -> UpdateInfo {
    info(&app)
}

#[tauri::command]
pub async fn update_check(app: AppHandle) -> CmdResult<UpdateInfo> {
    check(app, true).await
}

/// The Releases page, in the browser.
pub const RELEASES_URL: &str = "https://github.com/The0cean/mahdar-os/releases";

#[tauri::command]
pub fn update_open_releases(app: AppHandle) -> CmdResult<()> {
    use tauri_plugin_opener::OpenerExt;
    app.opener().open_url(RELEASES_URL, None::<&str>).map_err(err)
}

#[tauri::command]
pub fn update_install(app: AppHandle) -> CmdResult<()> {
    install_now(&app)
}
