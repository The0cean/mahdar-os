//! On-disk storage: one folder per meeting (meeting.json + audio.wav), plus settings.json.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};

use crate::asr::Backend;
use crate::summarize::{LocalOptions, Provider, Template};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Chunk {
    pub start_s: f64,
    pub end_s: f64,
    pub text: String,
    /// "" when clean; "repetition"/"truncated" when the text is partial.
    #[serde(default)]
    pub flag: String,
    /// Speaker number (1-based, arrival order) when diarization ran.
    #[serde(default)]
    pub speaker: Option<u32>,
    /// Text was corrected by hand.
    #[serde(default)]
    pub edited: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMsg {
    /// "user" or "assistant"
    pub role: String,
    pub text: String,
    pub created: DateTime<Local>,
    #[serde(default)]
    pub provider: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Output {
    pub id: String,
    pub template_id: String,
    pub template_name: String,
    pub provider: String,
    pub created: DateTime<Local>,
    pub text: String,
}

/// A stretch of a recording where no audio was captured (laptop asleep, microphone lost).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Gap {
    /// Position in the audio where it happened.
    pub at_s: f64,
    /// Real time that went unrecorded.
    pub missing_s: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// The microphone is recording into `recording.wav`.
    Recording,
    Transcribing,
    Ready,
    Failed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Meeting {
    pub id: String,
    pub title: String,
    pub created: DateTime<Local>,
    pub source: String,
    pub duration_s: f64,
    pub language: String,
    pub status: Status,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub transcribe_secs: Option<f64>,
    #[serde(default)]
    pub chunks: Vec<Chunk>,
    #[serde(default)]
    pub outputs: Vec<Output>,
    /// Names given to speaker numbers.
    #[serde(default)]
    pub speakers: BTreeMap<u32, String>,
    #[serde(default)]
    pub chat: Vec<ChatMsg>,
    /// The user's own notes, typed in Mahdar or imported. "[mm:ss]" marks a time in the recording.
    #[serde(default)]
    pub notes: String,
    /// Parts of a recording that were not captured.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gaps: Vec<Gap>,
}

impl Meeting {
    pub fn speaker_name(&self, n: u32) -> String {
        self.speakers.get(&n).cloned().unwrap_or_else(|| format!("Speaker {n}"))
    }

    /// Transcript for the minutes writer: "[mm:ss] Speaker: text" per line.
    pub fn transcript_text(&self) -> String {
        self.transcript_between(0.0, f64::INFINITY)
    }

    /// Transcript lines that overlap `from_s..to_s`, in the same format. A gap in the
    /// recording is marked before the line it falls in.
    pub fn transcript_between(&self, from_s: f64, to_s: f64) -> String {
        let mut lines = Vec::new();
        for c in self.chunks.iter().filter(|c| c.end_s > from_s && c.start_s < to_s) {
            for g in self.gaps.iter().filter(|g| g.at_s >= c.start_s && g.at_s < c.end_s) {
                lines.push(format!("[{}] (recording gap: {} not recorded)", fmt_ts(g.at_s), fmt_ts(g.missing_s)));
            }
            let t = crate::asr::clean_text(&c.text);
            if t.is_empty() {
                continue;
            }
            lines.push(match c.speaker {
                Some(n) => format!("[{}] {}: {}", fmt_ts(c.start_s), self.speaker_name(n), t),
                None => format!("[{}] {}", fmt_ts(c.start_s), t),
            });
        }
        lines.join("\n")
    }
}

pub fn fmt_ts(s: f64) -> String {
    let s = s as u64;
    if s >= 3600 {
        format!("{}:{:02}:{:02}", s / 3600, (s % 3600) / 60, s % 60)
    } else {
        format!("{:02}:{:02}", s / 60, s % 60)
    }
}

/// Short summary for the sidebar.
#[derive(Clone, Debug, Serialize)]
pub struct MeetingSummary {
    pub id: String,
    pub title: String,
    pub created: DateTime<Local>,
    pub duration_s: f64,
    pub status: Status,
    pub n_outputs: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    pub model_path: String,
    pub backend: Backend,
    pub language: String,
    pub output_language: String,
    #[serde(deserialize_with = "crate::summarize::provider_or_local")]
    pub provider: Provider,
    #[serde(default)]
    pub custom_templates: Vec<Template>,
    /// Folder to auto-import new recordings from (e.g. the phone's iCloud folder).
    #[serde(default)]
    pub watch_folder: Option<String>,
    /// Names and terms, one per line, given to the minutes writer for correct spelling.
    #[serde(default)]
    pub glossary: String,
    /// Detect speakers when transcribing (needs the diarizer model).
    #[serde(default = "yes")]
    pub speakers: bool,
    #[serde(default)]
    pub diarizer_path: String,
    /// Mahdar Remote: serve the phone UI/API on the network (token-protected).
    #[serde(default)]
    pub remote_enabled: bool,
    #[serde(default = "default_port")]
    pub remote_port: u16,
    #[serde(default)]
    pub remote_token: String,
    /// Watchdog: restart Mahdar after a crash and start it at Windows login.
    #[serde(default)]
    pub watchdog: bool,
    /// Engine for the Brain (tagging and questions across meetings); None = same as `provider`.
    #[serde(default, deserialize_with = "crate::summarize::provider_or_none")]
    pub brain_provider: Option<Provider>,
    /// Output preselected in a meeting's output panel.
    #[serde(default = "default_template")]
    pub default_template: String,
    /// Output written automatically when a transcription finishes; empty = none.
    #[serde(default)]
    pub auto_output: String,
    /// Add each newly transcribed meeting to the Brain.
    #[serde(default)]
    pub auto_index: bool,
    #[serde(default)]
    pub local: LocalOptions,
    /// Download new versions in the background and offer to install them.
    #[serde(default = "yes")]
    pub auto_update: bool,
}

impl Settings {
    /// The engine the Brain uses.
    pub fn brain_engine(&self) -> Provider {
        self.brain_provider.clone().unwrap_or_else(|| self.provider.clone())
    }
}

fn default_port() -> u16 {
    8787
}

fn default_template() -> String {
    "mom".into()
}

fn yes() -> bool {
    true
}

impl Settings {
    pub fn defaults(data_dir: &Path) -> Self {
        Settings {
            model_path: data_dir.join("models").join(crate::MODEL_FILE).to_string_lossy().into_owned(),
            backend: Backend::Vulkan,
            language: "ar".into(),
            output_language: "en".into(),
            provider: Provider::local_default(),
            custom_templates: Vec::new(),
            watch_folder: None,
            glossary: String::new(),
            speakers: true,
            diarizer_path: String::new(),
            remote_enabled: false,
            remote_port: default_port(),
            remote_token: String::new(),
            watchdog: false,
            brain_provider: None,
            default_template: default_template(),
            auto_output: String::new(),
            auto_index: false,
            local: LocalOptions::default(),
            auto_update: true,
        }
    }
}

/// Claude API spend in USD, as priced by Mahdar from each response's token counts.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Spend {
    pub total: f64,
    pub days: BTreeMap<String, f64>,
}

impl Spend {
    pub fn today(&self) -> f64 {
        self.days.get(&chrono::Local::now().format("%Y-%m-%d").to_string()).copied().unwrap_or(0.0)
    }
    pub fn this_month(&self) -> f64 {
        let month = chrono::Local::now().format("%Y-%m").to_string();
        self.days.iter().filter(|(d, _)| d.starts_with(&month)).map(|(_, v)| v).sum()
    }
}

pub struct Store {
    root: PathBuf,
}

impl Store {
    pub fn new(root: PathBuf) -> Result<Self> {
        fs::create_dir_all(root.join("meetings"))?;
        Ok(Store { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn meeting_dir(&self, id: &str) -> PathBuf {
        self.root.join("meetings").join(id)
    }

    pub fn audio_path(&self, id: &str) -> PathBuf {
        self.meeting_dir(id).join("audio.wav")
    }

    /// The microphone recording as it is written (see `audio::read_spool`); removed once
    /// `audio.wav` is saved.
    pub fn recording_path(&self, id: &str) -> PathBuf {
        self.meeting_dir(id).join("recording.wav")
    }

    pub fn save_meeting(&self, m: &Meeting) -> Result<()> {
        let dir = self.meeting_dir(&m.id);
        fs::create_dir_all(&dir)?;
        let tmp = dir.join("meeting.json.tmp");
        fs::write(&tmp, serde_json::to_vec_pretty(m)?)?;
        fs::rename(&tmp, dir.join("meeting.json"))?;
        Ok(())
    }

    pub fn load_meeting(&self, id: &str) -> Result<Meeting> {
        let p = self.meeting_dir(id).join("meeting.json");
        let bytes = fs::read(&p).with_context(|| format!("meeting {id} not found"))?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    pub fn list_meetings(&self) -> Result<Vec<MeetingSummary>> {
        let mut out = Vec::new();
        for entry in fs::read_dir(self.root.join("meetings"))? {
            let id = entry?.file_name().to_string_lossy().into_owned();
            if let Ok(m) = self.load_meeting(&id) {
                out.push(MeetingSummary {
                    id: m.id,
                    title: m.title,
                    created: m.created,
                    duration_s: m.duration_s,
                    status: m.status,
                    n_outputs: m.outputs.len(),
                });
            }
        }
        out.sort_by(|a, b| b.created.cmp(&a.created));
        Ok(out)
    }

    pub fn delete_meeting(&self, id: &str) -> Result<()> {
        let dir = self.meeting_dir(id);
        if dir.starts_with(self.root.join("meetings")) && dir.exists() {
            fs::remove_dir_all(dir)?;
        }
        Ok(())
    }

    pub fn load_settings(&self) -> Settings {
        let mut s: Settings = fs::read(self.root.join("settings.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_else(|| Settings::defaults(&self.root));
        if s.diarizer_path.is_empty() {
            s.diarizer_path = self.root.join("models").join(crate::DIARIZER_FILE).to_string_lossy().into_owned();
        }
        s
    }

    /// Files the folder watcher has already seen, keyed by "path|size|mtime".
    pub fn load_seen(&self) -> std::collections::HashSet<String> {
        fs::read(self.root.join("watched.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default()
    }

    pub fn save_seen(&self, seen: &std::collections::HashSet<String>) -> Result<()> {
        fs::write(self.root.join("watched.json"), serde_json::to_vec(seen)?)?;
        Ok(())
    }

    /// Add a Claude API charge to `usage.json` (per day and total).
    pub fn add_spend(&self, usd: f64) -> Result<()> {
        let mut s = self.spend();
        let day = chrono::Local::now().format("%Y-%m-%d").to_string();
        *s.days.entry(day).or_default() += usd;
        s.total += usd;
        fs::write(self.root.join("usage.json"), serde_json::to_vec_pretty(&s)?)?;
        Ok(())
    }

    pub fn spend(&self) -> Spend {
        fs::read(self.root.join("usage.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
    }

    pub fn save_settings(&self, s: &Settings) -> Result<()> {
        fs::write(self.root.join("settings.json"), serde_json::to_vec_pretty(s)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_settings_get_new_defaults() {
        let old = r#"{"model_path":"m.gguf","backend":"vulkan","language":"ar","output_language":"en",
            "provider":{"kind":"local","base_url":"http://127.0.0.1:11434/v1","model":"gemma4-it:12b"}}"#;
        let s: Settings = serde_json::from_str(old).unwrap();
        assert_eq!(s.default_template, "mom");
        assert!(s.auto_output.is_empty() && !s.auto_index);
        assert_eq!((s.local.temperature, s.local.context, s.local.autostart, s.local.timeout_min), (0.2, 65_536, true, 30));
        let partial: Settings = serde_json::from_str(&old.replace("}}", r#"},"local":{"temperature":0.5}}"#)).unwrap();
        assert_eq!((partial.local.temperature, partial.local.timeout_min), (0.5, 30));
    }

    #[test]
    fn old_meetings_have_empty_notes_and_windows_work() {
        let old = r#"{"id":"x","title":"T","created":"2026-10-01T10:00:00+03:00","source":"a.m4a","duration_s":300,
            "language":"ar","status":"ready","chunks":[
            {"start_s":0,"end_s":100,"text":"hello"},{"start_s":100,"end_s":200,"text":"price is 40","speaker":1},
            {"start_s":200,"end_s":300,"text":"bye"}]}"#;
        let m: Meeting = serde_json::from_str(old).unwrap();
        assert!(m.notes.is_empty());
        assert_eq!(m.transcript_between(120.0, 150.0), "[01:40] Speaker 1: price is 40");
        assert_eq!(m.transcript_between(90.0, 210.0).lines().count(), 3);
        assert_eq!(m.transcript_text().lines().count(), 3);
        let mut m = m;
        m.gaps.push(Gap { at_s: 150.0, missing_s: 185.0 });
        assert_eq!(m.transcript_between(120.0, 150.0), "[02:30] (recording gap: 03:05 not recorded)\n[01:40] Speaker 1: price is 40");
    }
}
