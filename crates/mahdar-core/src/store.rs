//! On-disk storage: one folder per meeting (meeting.json + audio.wav), plus settings.json.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};

use crate::asr::Backend;
use crate::summarize::{Provider, Template};

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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Status {
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
}

impl Meeting {
    pub fn speaker_name(&self, n: u32) -> String {
        self.speakers.get(&n).cloned().unwrap_or_else(|| format!("Speaker {n}"))
    }

    /// Transcript for the minutes writer: "[mm:ss] Speaker: text" per line.
    pub fn transcript_text(&self) -> String {
        self.chunks
            .iter()
            .map(|c| (c, crate::asr::clean_text(&c.text)))
            .filter(|(_, t)| !t.is_empty())
            .map(|(c, t)| match c.speaker {
                Some(n) => format!("[{}] {}: {}", fmt_ts(c.start_s), self.speaker_name(n), t),
                None => format!("[{}] {}", fmt_ts(c.start_s), t),
            })
            .collect::<Vec<_>>()
            .join("\n")
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
        }
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

    pub fn save_settings(&self, s: &Settings) -> Result<()> {
        fs::write(self.root.join("settings.json"), serde_json::to_vec_pretty(s)?)?;
        Ok(())
    }
}
