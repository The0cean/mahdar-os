//! Folder watcher: imports recordings that appear in the watched folder (e.g. the
//! phone's iCloud folder). Files present when the folder is first watched are
//! recorded as a baseline and skipped. A file counts as new when its path, size
//! or modified time changes, so a shortcut that overwrites the same `input.m4a`
//! is picked up each time. Files are imported once their size stops changing.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use tauri::{AppHandle, Emitter, Manager};

use crate::AppState;

pub const AUDIO_EXT: &[&str] = &["m4a", "mp3", "wav", "aac", "ogg", "opus", "flac", "mp4", "mov", "webm", "amr", "caf", "wma", "mkv"];
const POLL: Duration = Duration::from_secs(20);
const MAX_DEPTH: usize = 4;

fn walk(dir: &Path, depth: usize, out: &mut Vec<(PathBuf, u64, u64)>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        let Ok(md) = e.metadata() else { continue };
        let name = e.file_name().to_string_lossy().to_lowercase();
        if md.is_dir() {
            if depth < MAX_DEPTH && !name.starts_with('.') {
                walk(&p, depth + 1, out);
            }
        } else if AUDIO_EXT.iter().any(|x| name.ends_with(&format!(".{x}"))) && md.len() > 32 * 1024 {
            let mtime = md.modified().ok().and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0);
            out.push((p, md.len(), mtime));
        }
    }
}

fn key(p: &Path, size: u64, mtime: u64) -> String {
    format!("{}|{size}|{mtime}", p.display())
}

pub fn spawn(app: AppHandle) {
    std::thread::spawn(move || {
        // key → size seen on the previous poll (file must be stable for one poll)
        let mut pending: HashMap<String, u64> = HashMap::new();
        loop {
            std::thread::sleep(POLL);
            let state = app.state::<AppState>();
            let Some(folder) = state.store.load_settings().watch_folder.filter(|f| !f.trim().is_empty()) else {
                continue;
            };
            let root = PathBuf::from(&folder);
            if !root.is_dir() {
                continue;
            }
            let mut files = Vec::new();
            walk(&root, 0, &mut files);
            let mut seen: HashSet<String> = state.store.load_seen();
            let baseline = format!("baseline|{}", root.display());
            if !seen.contains(&baseline) {
                seen.extend(files.iter().map(|(p, s, m)| key(p, *s, *m)));
                seen.insert(baseline);
                let _ = state.store.save_seen(&seen);
                continue;
            }
            let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
            for (path, size, mtime) in files {
                let k = key(&path, size, mtime);
                if seen.contains(&k) {
                    continue;
                }
                let stable = pending.get(&k) == Some(&size) && now.saturating_sub(mtime) > 15;
                if !stable {
                    pending.insert(k, size);
                    continue;
                }
                pending.remove(&k);
                seen.insert(k);
                let _ = state.store.save_seen(&seen);
                match crate::import_path(&app, &path) {
                    Ok(id) => {
                        let _ = app.emit("watch-imported", (id, path.display().to_string()));
                    }
                    Err(e) => eprintln!("watch import failed for {}: {e}", path.display()),
                }
            }
        }
    });
}
