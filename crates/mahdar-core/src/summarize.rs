//! Turning transcripts into minutes, notes and action items.
//!
//! Providers:
//! - `CursorCli`: Cursor's CLI (`agent -p`) with the user's own login. Leaves the laptop.
//! - `Local`: an OpenAI-compatible server on this machine, e.g. FastFlowLM
//!   (`flm serve gemma4-it:12b`) or Ollama. Nothing leaves the laptop.

use std::process::{Child, Command, Stdio};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Provider {
    /// Cursor's CLI (`agent -p`). Empty model = the account's default.
    CursorCli { model: String },
    Local { base_url: String, model: String },
}

/// The local writer's defaults: FastFlowLM on its usual port.
pub const LOCAL_URL: &str = "http://127.0.0.1:11434/v1";
pub const LOCAL_MODEL: &str = "gemma4-it:12b";

impl Provider {
    pub fn local_default() -> Self {
        Provider::Local { base_url: LOCAL_URL.into(), model: LOCAL_MODEL.into() }
    }

    pub fn label(&self) -> String {
        match self {
            Provider::CursorCli { model } if model.is_empty() => "Cursor (cloud)".into(),
            Provider::CursorCli { model } => format!("Cursor {model} (cloud)"),
            Provider::Local { model, .. } => format!("{model} (local)"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Template {
    pub id: String,
    pub name: String,
    pub prompt: String,
    #[serde(default)]
    pub builtin: bool,
}

const SYSTEM: &str = "You turn meeting transcripts into clear written outputs. \
The transcript was produced by speech recognition from a business meeting that mixes Arabic \
(Egyptian, Gulf, Levantine dialects and MSA) and English; English terms are often written in \
Arabic script. Recognition errors exist: infer the \
intended meaning from context, but never invent facts, names, numbers or decisions that are not \
supported by the transcript. If something is unclear, say so briefly. Output Markdown only.";

pub fn builtin_templates() -> Vec<Template> {
    let t = |id: &str, name: &str, prompt: &str| Template {
        id: id.into(),
        name: name.into(),
        prompt: prompt.into(),
        builtin: true,
    };
    vec![
        t("mom", "Minutes of Meeting", "Write formal Minutes of Meeting with these sections: \
**Meeting purpose**, **Participants** (names/roles mentioned; mark unknowns), **Key discussion points** \
(grouped by topic, concise bullets), **Decisions**, **Action items** (table: action | owner | due date, \
use \"TBD\" when not stated), **Open questions / risks**. Keep amounts, limits, dates and product names exact."),
        t("notes", "Notes", "Write short, informal notes for someone who missed the meeting: a 3-5 sentence \
summary of what it was about and where it landed, then at most 10 bullets with the most important points \
(numbers, decisions, next steps). Keep it under 350 words. Do not use the Minutes of Meeting layout: no \
participants list, no section headings, no action-item table. Keep numbers and names exact."),
        t("actions", "Action items", "List only the action items and follow-ups as a checklist: \
\"- [ ] action — owner — due date\". Use \"TBD\" where owner or date is not stated. No other text."),
    ]
}

fn glossary_block(glossary: &str) -> String {
    let g = glossary.trim();
    if g.is_empty() {
        String::new()
    } else {
        format!("\n\nNames and terms that appear in this meeting (use these spellings; the transcript may misspell them):\n{g}")
    }
}

/// Settings saved by older versions can name a writer that no longer exists; those fall
/// back to the local writer instead of resetting every other setting.
pub fn provider_or_local<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Provider, D::Error> {
    let v = serde_json::Value::deserialize(d)?;
    Ok(serde_json::from_value(v).unwrap_or_else(|_| Provider::local_default()))
}

/// Like [`provider_or_local`], for the optional Brain engine: an unknown writer means "same as minutes".
pub fn provider_or_none<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Option<Provider>, D::Error> {
    let v = Option::<serde_json::Value>::deserialize(d)?;
    Ok(v.and_then(|v| serde_json::from_value(v).ok()))
}

/// Build the full user prompt for a template.
pub fn build_prompt(template: &Template, output_language: &str, title: &str, transcript: &str, glossary: &str) -> String {
    let lang = match output_language {
        "ar" => "Write the output in Arabic (Modern Standard Arabic, business register).",
        "auto" => "Write the output in the main language of the meeting.",
        _ => "Write the output in English.",
    };
    format!(
        "{}\n{}{}\n\nMeeting title: {}\n\n<transcript>\n{}\n</transcript>",
        template.prompt,
        lang,
        glossary_block(glossary),
        title,
        transcript
    )
}

/// Prompt for a question about one meeting, with recent Q&A for follow-ups.
pub fn build_question(title: &str, transcript: &str, glossary: &str, history: &[(String, String)], question: &str) -> String {
    let mut prior = String::new();
    for (role, text) in history {
        prior.push_str(&format!("{}: {}\n\n", if role == "user" { "Q" } else { "A" }, text));
    }
    format!(
        "Answer the question using only this meeting's transcript. Be concise. Cite the timestamps you rely on, \
         like [12:34]. If the transcript doesn't contain the answer, say so plainly. Answer in the language of the question.{}\n\n\
         Meeting title: {}\n\n<transcript>\n{}\n</transcript>\n\n{}{}Q: {}",
        glossary_block(glossary),
        title,
        transcript,
        if prior.is_empty() { "" } else { "Earlier in this conversation:\n\n" },
        prior,
        question
    )
}

/// Send a prompt to the writer.
pub fn generate(provider: &Provider, prompt: &str) -> Result<String> {
    match provider {
        Provider::CursorCli { model } => cursor_cli(model, prompt),
        Provider::Local { base_url, model } => openai_compatible(base_url, model, prompt),
    }
}

/// How to launch Cursor's CLI. The Windows installer puts a bundled node.exe +
/// index.js under %LOCALAPPDATA%\cursor-agent\versions\<date-commit>; running
/// those directly avoids the agent.cmd → PowerShell hop. Otherwise use PATH.
fn cursor_command() -> Command {
    if let Some(local) = std::env::var_os("LOCALAPPDATA") {
        let versions = std::path::PathBuf::from(local).join("cursor-agent").join("versions");
        let latest = std::fs::read_dir(&versions)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.join("node.exe").is_file() && p.join("index.js").is_file())
            .max(); // names are YYYY.MM.DD-commit, so the max sorts newest for same-width dates
        if let Some(dir) = latest {
            let mut c = Command::new(dir.join("node.exe"));
            c.arg(dir.join("index.js")).env("CURSOR_INVOKED_AS", "agent");
            return c;
        }
    }
    if cfg!(windows) {
        let mut c = Command::new("cmd");
        c.args(["/C", "agent"]);
        c
    } else {
        Command::new("agent")
    }
}

fn cursor_cli(model: &str, prompt: &str) -> Result<String> {
    // The request goes in a file inside a throwaway workspace: transcripts are
    // too long for a Windows command line, and ask mode keeps the agent read-only.
    let ws = std::env::temp_dir().join(format!("mahdar-cursor-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&ws)?;
    std::fs::write(ws.join("mahdar_request.md"), format!("# Instructions\n\n{SYSTEM}\n\n{prompt}\n"))?;
    let mut cmd = cursor_command();
    cmd.args(["-p", "--output-format", "json", "--mode", "ask", "--trust", "--workspace"])
        .arg(&ws)
        .current_dir(&ws);
    if !model.trim().is_empty() {
        cmd.args(["--model", model.trim()]);
    }
    cmd.arg(
        "Read the file mahdar_request.md in this workspace in full and follow its instructions exactly. \
         Reply with only the requested Markdown output: no preamble, no notes about reading files.",
    )
    .stdin(Stdio::null())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let out = cmd.output();
    let _ = std::fs::remove_dir_all(&ws);
    let out = out.context("could not start Cursor's CLI. Install it with: irm 'https://cursor.com/install?win32=true' | iex, then run `agent login`")?;
    let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let msg = if err.trim().is_empty() { &stdout } else { err.trim() };
        if msg.contains("is not recognized") {
            bail!("Cursor's CLI isn't installed. In PowerShell run: irm 'https://cursor.com/install?win32=true' | iex, then: agent login");
        }
        bail!("Cursor CLI failed: {msg}");
    }
    // JSON mode ends with a result object; fall back to raw text if the shape differs.
    let text = stdout
        .lines()
        .rev()
        .find_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .and_then(|v| v["result"].as_str().map(str::to_string))
        .unwrap_or(stdout);
    if text.trim().is_empty() {
        bail!("Cursor returned no text. Run `agent login` (or check `agent status`).");
    }
    Ok(text.trim().to_string())
}

/// Context window FastFlowLM is started with: enough for ~1.5 h of transcript.
pub const FLM_CONTEXT: u32 = 65_536;

/// Make sure the local OpenAI-compatible server is up. If nothing answers at
/// `base_url`, start FastFlowLM (`flm serve <model>`) on that port and wait for it.
/// Returns the child process when Mahdar started it (caller stops it on exit).
pub fn ensure_local_server(base_url: &str, model: &str, on_starting: impl FnOnce()) -> Result<Option<Child>> {
    let base = base_url.trim_end_matches('/');
    let client = reqwest::blocking::Client::builder().timeout(Duration::from_secs(3)).build()?;
    let models = |c: &reqwest::blocking::Client| -> Option<serde_json::Value> {
        c.get(format!("{base}/models")).send().ok()?.json().ok()
    };
    let check = |v: &serde_json::Value| -> Result<()> {
        let list = v["data"].as_array().cloned().unwrap_or_default();
        if list.iter().any(|m| m["id"] == model) {
            return Ok(());
        }
        let is_flm = list.iter().any(|m| m["owned_by"] == "FastFlowLM");
        if is_flm {
            bail!("FastFlowLM doesn't have {model}. Download it with: flm pull {model}");
        }
        bail!("another server (Ollama?) is using {base} and doesn't have {model}. Stop it, or change the server URL in Settings.");
    };
    if let Some(v) = models(&client) {
        check(&v)?;
        return Ok(None);
    }

    let port = base.rsplit(':').next().and_then(|p| p.split('/').next()).and_then(|p| p.parse::<u16>().ok()).unwrap_or(11434);
    on_starting();
    let mut cmd = Command::new("flm");
    cmd.args(["serve", model, "--quiet", "-c", &FLM_CONTEXT.to_string(), "-p", &port.to_string()])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut child = cmd.spawn().context("could not start FastFlowLM (`flm`). Is it installed and on PATH?")?;
    let deadline = std::time::Instant::now() + Duration::from_secs(240);
    while std::time::Instant::now() < deadline {
        if let Some(status) = child.try_wait()? {
            bail!("FastFlowLM exited while starting ({status}). Try `flm serve {model}` in a terminal to see why.");
        }
        if let Some(v) = models(&client) {
            check(&v)?;
            return Ok(Some(child));
        }
        std::thread::sleep(Duration::from_millis(1500));
    }
    let _ = child.kill();
    bail!("FastFlowLM didn't start within 4 minutes")
}

fn openai_compatible(base_url: &str, model: &str, prompt: &str) -> Result<String> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "stream": false
    });
    let resp = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(60 * 30))
        .build()?
        .post(&url)
        .json(&body)
        .send()
        .with_context(|| format!("could not reach the local model server at {base_url}. Start it with: flm serve {model}"))?;
    let status = resp.status();
    let v: serde_json::Value = resp.json()?;
    if !status.is_success() {
        bail!("local model server error {status}: {v}");
    }
    let text = v["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| anyhow!("unexpected response: {v}"))?;
    // Some reasoning models wrap thoughts in <think>…</think>; drop them.
    let text = match text.rfind("</think>") {
        Some(i) => &text[i + "</think>".len()..],
        None => text,
    };
    Ok(text.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Deserialize)]
    struct Saved {
        #[serde(deserialize_with = "provider_or_local")]
        provider: Provider,
        #[serde(default, deserialize_with = "provider_or_none")]
        brain: Option<Provider>,
    }

    #[test]
    fn removed_writers_fall_back() {
        let s: Saved = serde_json::from_str(r#"{"provider":{"kind":"removed_writer","model":"x"},"brain":{"kind":"removed_writer","model":"x"}}"#).unwrap();
        assert_eq!(s.provider.label(), "gemma4-it:12b (local)");
        assert!(s.brain.is_none());
        let s: Saved = serde_json::from_str(r#"{"provider":{"kind":"cursor_cli","model":""},"brain":null}"#).unwrap();
        assert_eq!(s.provider.label(), "Cursor (cloud)");
        let s: Saved = serde_json::from_str(r#"{"provider":{"kind":"local","base_url":"u","model":"m"}}"#).unwrap();
        assert_eq!(s.provider.label(), "m (local)");
        assert!(s.brain.is_none());
    }
}
