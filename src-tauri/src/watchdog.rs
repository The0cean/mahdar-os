//! Mahdar's watchdog: the same mahdar.exe started with `--watchdog`, no window.
//!
//! - Restarts Mahdar when it stops without closing normally (a crash, or Windows shut down
//!   while it was open), passing `--recovered` so interrupted transcriptions resume.
//!   Gives up after 3 restarts in 10 minutes, so a file that crashes the app can't loop.
//! - Restarts Mahdar when its phone server stops answering for 5 minutes (a hang).
//!   A server that refuses connections (port taken) is only logged: restarting won't fix it.
//! - With `--at-login` (Windows Run key, set when the watchdog is turned on) it also starts
//!   Mahdar when you log in.
//!
//! The app writes `<data>/app.json` ({pid, exe}) when it starts and removes it when it
//! closes normally; the watchdog writes `<data>/watchdog.json` every check. Both log to
//! `<data>/logs/mahdar.log`. The watchdog stops itself when the setting is turned off.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mahdar_core::log;
use mahdar_core::Store;
use serde::{Deserialize, Serialize};

const CHECK_EVERY: Duration = Duration::from_secs(10);
const HANG_AFTER: Duration = Duration::from_secs(5 * 60);
const MAX_RESTARTS: usize = 3;
const RESTART_WINDOW: Duration = Duration::from_secs(10 * 60);
const RUN_KEY: &str = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run";
const RUN_VALUE: &str = "Mahdar Watchdog";

/// Same folder Tauri uses for app data (`com.mahdar.app` under Roaming AppData).
/// `MAHDAR_DATA_DIR` overrides it (tests).
pub fn data_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("MAHDAR_DATA_DIR") {
        return PathBuf::from(d);
    }
    let base = std::env::var_os("APPDATA").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    base.join("com.mahdar.app")
}

#[derive(Serialize, Deserialize, Clone)]
pub struct AppFile {
    pub pid: u32,
    pub exe: String,
    pub started: String,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct WatchdogFile {
    pub pid: u32,
    pub exe: String,
    pub started: String,
    pub last_check: String,
    /// What the watchdog saw on its last check, e.g. "Mahdar running".
    pub state: String,
    /// Restart times (RFC 3339), newest last, at most 20.
    pub restarts: Vec<String>,
}

fn read<T: for<'de> Deserialize<'de>>(p: &Path) -> Option<T> {
    std::fs::read(p).ok().and_then(|b| serde_json::from_slice(&b).ok())
}

pub fn app_file(data: &Path) -> PathBuf {
    data.join("app.json")
}
fn watchdog_file(data: &Path) -> PathBuf {
    data.join("watchdog.json")
}

/// Called by the app at start: who to watch.
pub fn write_app_file(data: &Path) {
    let f = AppFile {
        pid: std::process::id(),
        exe: std::env::current_exe().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default(),
        started: chrono::Local::now().to_rfc3339(),
    };
    let _ = std::fs::write(app_file(data), serde_json::to_vec_pretty(&f).unwrap_or_default());
}

/// Called by the app when it closes normally, so the watchdog doesn't restart it.
pub fn remove_app_file(data: &Path) {
    let _ = std::fs::remove_file(app_file(data));
}

// ---------- processes (Windows) ----------

#[cfg(windows)]
mod proc {
    use windows_sys::Win32::Foundation::{CloseHandle, STILL_ACTIVE};
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, QueryFullProcessImageNameW, TerminateProcess, PROCESS_NAME_WIN32,
        PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_TERMINATE,
    };

    /// The exe path of a running process, or None if it isn't running.
    pub fn running_exe(pid: u32) -> Option<String> {
        unsafe {
            let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if h.is_null() {
                return None;
            }
            let mut code = 0u32;
            let alive = GetExitCodeProcess(h, &mut code) != 0 && code == STILL_ACTIVE as u32;
            let mut buf = [0u16; 1024];
            let mut len = buf.len() as u32;
            let ok = QueryFullProcessImageNameW(h, PROCESS_NAME_WIN32, buf.as_mut_ptr(), &mut len) != 0;
            CloseHandle(h);
            if !alive {
                return None;
            }
            Some(if ok { String::from_utf16_lossy(&buf[..len as usize]) } else { String::new() })
        }
    }

    pub fn kill(pid: u32) -> bool {
        unsafe {
            let h = OpenProcess(PROCESS_TERMINATE, 0, pid);
            if h.is_null() {
                return false;
            }
            let ok = TerminateProcess(h, 1) != 0;
            CloseHandle(h);
            ok
        }
    }
}

#[cfg(not(windows))]
mod proc {
    pub fn running_exe(_pid: u32) -> Option<String> {
        None
    }
    pub fn kill(_pid: u32) -> bool {
        false
    }
}

/// Is `pid` alive and running `exe`? (Checks the path so a reused pid isn't mistaken for Mahdar.)
fn is_running(pid: u32, exe: &str) -> bool {
    match proc::running_exe(pid) {
        Some(path) => path.is_empty() || exe.is_empty() || path.eq_ignore_ascii_case(exe),
        None => false,
    }
}

fn detached(cmd: &mut std::process::Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0000_0008 | 0x0000_0200); // DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    }
    cmd.stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
}

// ---------- used by the app ----------

#[derive(Serialize)]
pub struct Status {
    pub enabled: bool,
    pub running: bool,
    pub state: String,
    pub last_check: String,
    pub restarts_24h: usize,
    pub last_restart: Option<String>,
}

pub fn status(data: &Path, enabled: bool) -> Status {
    let f: WatchdogFile = read(&watchdog_file(data)).unwrap_or_default();
    let running = f.pid != 0 && is_running(f.pid, &f.exe);
    let day_ago = chrono::Local::now() - chrono::Duration::hours(24);
    let restarts_24h = f
        .restarts
        .iter()
        .filter_map(|t| chrono::DateTime::parse_from_rfc3339(t).ok())
        .filter(|t| *t > day_ago)
        .count();
    Status {
        enabled,
        running,
        state: if running { f.state } else if enabled { "not running".into() } else { "off".into() },
        last_check: f.last_check,
        restarts_24h,
        last_restart: f.restarts.last().cloned(),
    }
}

/// Make the watchdog match the setting: register/unregister it at login, start it if needed.
/// (Turning it off needs no kill: it sees the setting on its next check and stops.)
pub fn sync(data: &Path, enabled: bool) {
    let exe = std::env::current_exe().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
    if cfg!(debug_assertions) {
        // A dev build must not register target\debug\mahdar.exe to run at every login.
        log::info("watchdog", "dev build: not registering at Windows login");
    } else if cfg!(windows) {
        let mut cmd = std::process::Command::new("reg");
        if enabled {
            cmd.args(["add", RUN_KEY, "/v", RUN_VALUE, "/t", "REG_SZ", "/d", &format!("\"{exe}\" --watchdog --at-login"), "/f"]);
        } else {
            cmd.args(["delete", RUN_KEY, "/v", RUN_VALUE, "/f"]);
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }
        let ok = cmd.output().map(|o| o.status.success()).unwrap_or(false);
        if enabled && !ok {
            log::warn("watchdog", "could not register the watchdog to start at Windows login");
        }
    }
    if enabled && !status(data, true).running {
        let mut cmd = std::process::Command::new(&exe);
        cmd.arg("--watchdog");
        detached(&mut cmd);
        match cmd.spawn() {
            Ok(_) => {}
            Err(e) => log::error("watchdog", format!("could not start the watchdog: {e}")),
        }
    }
}

// ---------- the watchdog process ----------

/// Health of the phone server: Ok, refused/other (not a hang), or timed out (a hang).
enum Health {
    Ok,
    Down,
    Hung,
}

fn health(port: u16) -> Health {
    let client = match reqwest::blocking::Client::builder().timeout(Duration::from_secs(10)).build() {
        Ok(c) => c,
        Err(_) => return Health::Down,
    };
    match client.get(format!("http://127.0.0.1:{port}/api/health")).send() {
        Ok(r) if r.status().is_success() => Health::Ok,
        Ok(_) => Health::Down,
        Err(e) if e.is_timeout() => Health::Hung,
        Err(_) => Health::Down,
    }
}

fn launch(exe: &str, recovered: bool) -> bool {
    let mut cmd = std::process::Command::new(exe);
    if recovered {
        cmd.arg("--recovered");
    }
    detached(&mut cmd);
    match cmd.spawn() {
        Ok(_) => true,
        Err(e) => {
            log::error("watchdog", format!("could not start Mahdar ({exe}): {e}"));
            false
        }
    }
}

pub fn run(at_login: bool) {
    let data = data_dir();
    log::init(&data);
    let me = std::process::id();
    let my_exe = std::env::current_exe().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
    let mut file: WatchdogFile = read(&watchdog_file(&data)).unwrap_or_default();
    if file.pid != 0 && file.pid != me && is_running(file.pid, &file.exe) {
        return; // another watchdog is already on duty
    }
    let store = match Store::new(data.clone()) {
        Ok(s) => s,
        Err(e) => {
            log::error("watchdog", format!("can't open Mahdar's data folder: {e:#}"));
            return;
        }
    };
    file.pid = me;
    file.exe = my_exe.clone();
    file.started = chrono::Local::now().to_rfc3339();
    log::info("watchdog", format!("started (pid {me}){}", if at_login { " at Windows login" } else { "" }));

    let mut recent: Vec<Instant> = Vec::new();
    let mut gave_up = false;
    let mut hung_since: Option<Instant> = None;
    let mut down_logged = false;
    let mut first = true;
    loop {
        let settings = store.load_settings();
        if !settings.watchdog {
            log::info("watchdog", "turned off in Settings; stopping");
            let _ = std::fs::remove_file(watchdog_file(&data));
            return;
        }
        let app: Option<AppFile> = read(&app_file(&data));
        file.state = match &app {
            None => {
                if first && at_login {
                    log::info("watchdog", "starting Mahdar at login");
                    launch(&my_exe, false);
                    "starting Mahdar".into()
                } else {
                    "Mahdar closed".into()
                }
            }
            Some(a) if !is_running(a.pid, &a.exe) => {
                recent.retain(|t| t.elapsed() < RESTART_WINDOW);
                if gave_up {
                    "gave up restarting (crashed 3 times in 10 minutes)".into()
                } else if recent.len() >= MAX_RESTARTS {
                    gave_up = true;
                    log::error("watchdog", "Mahdar crashed 3 times in 10 minutes; not restarting it again. Open it yourself and check the log.");
                    let _ = std::fs::remove_file(app_file(&data));
                    "gave up restarting (crashed 3 times in 10 minutes)".into()
                } else {
                    log::error("watchdog", format!("Mahdar (pid {}) stopped without closing normally; restarting it", a.pid));
                    let exe = if a.exe.is_empty() { my_exe.clone() } else { a.exe.clone() };
                    let _ = std::fs::remove_file(app_file(&data));
                    if launch(&exe, true) {
                        recent.push(Instant::now());
                        file.restarts.push(chrono::Local::now().to_rfc3339());
                        let n = file.restarts.len();
                        if n > 20 {
                            file.restarts.drain(..n - 20);
                        }
                    }
                    "restarted Mahdar".into()
                }
            }
            Some(a) => {
                gave_up = false;
                if !settings.remote_enabled {
                    hung_since = None;
                    "Mahdar running".into()
                } else {
                    match health(settings.remote_port) {
                        Health::Ok => {
                            if hung_since.take().is_some() || down_logged {
                                log::info("watchdog", "phone server answering again");
                            }
                            down_logged = false;
                            "Mahdar running · phone server OK".into()
                        }
                        Health::Down => {
                            hung_since = None;
                            if !down_logged {
                                down_logged = true;
                                log::warn("watchdog", format!("phone server isn't answering on port {} (Mahdar keeps retrying)", settings.remote_port));
                            }
                            "Mahdar running · phone server down".into()
                        }
                        Health::Hung => {
                            let since = *hung_since.get_or_insert_with(|| {
                                log::warn("watchdog", "phone server stopped responding; restarting Mahdar if it stays stuck for 5 minutes");
                                Instant::now()
                            });
                            if since.elapsed() >= HANG_AFTER {
                                log::error("watchdog", format!("Mahdar (pid {}) has been stuck for 5 minutes; restarting it", a.pid));
                                hung_since = None;
                                if proc::kill(a.pid) {
                                    // The next check sees it gone and restarts it (with --recovered).
                                    "restarting stuck Mahdar".into()
                                } else {
                                    log::error("watchdog", "could not stop the stuck Mahdar");
                                    "Mahdar stuck".into()
                                }
                            } else {
                                "Mahdar running · phone server not responding".into()
                            }
                        }
                    }
                }
            }
        };
        file.last_check = chrono::Local::now().to_rfc3339();
        let _ = std::fs::write(watchdog_file(&data), serde_json::to_vec_pretty(&file).unwrap_or_default());
        first = false;
        std::thread::sleep(CHECK_EVERY);
    }
}
