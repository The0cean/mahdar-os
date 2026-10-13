//! Mahdar's activity log: `<data>/logs/mahdar.log`, one line per event, shared by the app and
//! its watchdog (both only append). Rotated at 5 MB to `mahdar.1.log`. The Monitor screens
//! read the tail of the file, so events from both processes show up together.
//!
//! Line format: `2026-10-12 21:04:05.123 INFO  writer    text`. Messages never contain
//! transcript text or keys: callers log titles, timings, sizes and errors only.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use serde::Serialize;

const MAX_BYTES: u64 = 5 << 20;

static LOG: OnceLock<Mutex<PathBuf>> = OnceLock::new();

pub fn log_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("logs")
}

/// Start logging to `<data_dir>/logs/mahdar.log`. Safe to call more than once.
pub fn init(data_dir: &Path) {
    let dir = log_dir(data_dir);
    let _ = std::fs::create_dir_all(&dir);
    let _ = LOG.set(Mutex::new(dir.join("mahdar.log")));
}

pub fn path() -> Option<PathBuf> {
    LOG.get().map(|p| p.lock().unwrap().clone())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Info,
    Warn,
    Error,
}

impl Level {
    fn tag(self) -> &'static str {
        match self {
            Level::Info => "INFO ",
            Level::Warn => "WARN ",
            Level::Error => "ERROR",
        }
    }
}

pub fn write(level: Level, source: &str, msg: &str) {
    let Some(lock) = LOG.get() else { return };
    let path = lock.lock().unwrap();
    if std::fs::metadata(&*path).map(|m| m.len() > MAX_BYTES).unwrap_or(false) {
        let _ = std::fs::rename(&*path, path.with_file_name("mahdar.1.log"));
    }
    // One line per event: newlines inside a message would break the reader.
    let msg = msg.replace(['\r', '\n'], " ");
    let line = format!("{} {} {:<9} {}\n", chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f"), level.tag(), source, msg);
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&*path) {
        let _ = f.write_all(line.as_bytes());
    }
    if cfg!(debug_assertions) {
        eprint!("{line}");
    }
}

pub fn info(source: &str, msg: impl AsRef<str>) {
    write(Level::Info, source, msg.as_ref())
}
pub fn warn(source: &str, msg: impl AsRef<str>) {
    write(Level::Warn, source, msg.as_ref())
}
pub fn error(source: &str, msg: impl AsRef<str>) {
    write(Level::Error, source, msg.as_ref())
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Line {
    pub time: String,
    pub level: String,
    pub source: String,
    pub msg: String,
}

fn parse(line: &str) -> Option<Line> {
    // "YYYY-MM-DD HH:MM:SS.mmm LEVEL source    msg"
    let (date, rest) = line.split_once(' ')?;
    let (time, rest) = rest.split_once(' ')?;
    let (level, rest) = rest.trim_start().split_once(' ')?;
    let rest = rest.trim_start();
    let (source, msg) = rest.split_once(' ').unwrap_or((rest, ""));
    if date.len() != 10 || !["INFO", "WARN", "ERROR"].contains(&level) {
        return None;
    }
    Some(Line { time: format!("{date} {time}"), level: level.into(), source: source.into(), msg: msg.trim_start().into() })
}

/// The last `n` events, oldest first (reads at most the last 512 KB of the log).
pub fn tail(n: usize) -> Vec<Line> {
    let Some(path) = path() else { return Vec::new() };
    tail_file(&path, n)
}

pub fn tail_file(path: &Path, n: usize) -> Vec<Line> {
    let Ok(mut f) = std::fs::File::open(path) else { return Vec::new() };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let start = len.saturating_sub(512 << 10);
    if f.seek(SeekFrom::Start(start)).is_err() {
        return Vec::new();
    }
    let mut buf = Vec::new();
    let _ = f.read_to_end(&mut buf);
    let text = String::from_utf8_lossy(&buf);
    let mut lines: Vec<Line> = text.lines().skip(usize::from(start > 0)).filter_map(parse).collect();
    if lines.len() > n {
        lines.drain(..lines.len() - n);
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_written_lines() {
        let l = parse("2026-10-12 21:04:05.123 INFO  writer    Minutes for \"x\" in 12 s").unwrap();
        assert_eq!(l.time, "2026-10-12 21:04:05.123");
        assert_eq!(l.level, "INFO");
        assert_eq!(l.source, "writer");
        assert_eq!(l.msg, "Minutes for \"x\" in 12 s");
        assert!(parse("garbage").is_none());
    }

    #[test]
    fn tail_keeps_the_newest() {
        let dir = std::env::temp_dir().join(format!("mahdar-log-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("mahdar.log");
        let body: String = (0..50).map(|i| format!("2026-10-12 10:00:{:02}.000 WARN  app       event {i}\n", i % 60)).collect();
        std::fs::write(&p, body).unwrap();
        let t = tail_file(&p, 5);
        assert_eq!(t.len(), 5);
        assert_eq!(t[0].msg, "event 45");
        assert_eq!(t[4].msg, "event 49");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
