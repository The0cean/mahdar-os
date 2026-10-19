//! Turning transcripts into minutes, notes and action items.
//!
//! Providers:
//! - `ClaudeCli`: runs the local `claude -p` (Claude Code) with the user's own
//!   login. Leaves the laptop (Anthropic). Personal use only.
//! - `ClaudeApi`: the Claude API (Messages API) with the key saved in Settings
//!   (Windows Credential Manager). Leaves the laptop (Anthropic); billed per token.
//! - `Local`: an OpenAI-compatible server on this machine, e.g. FastFlowLM
//!   (`flm serve gemma4-it:12b`) or Ollama. Nothing leaves the laptop.

use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Provider {
    ClaudeCli { model: String },
    /// Claude API with the saved key; `model` is an API id like "claude-sonnet-5-5".
    /// `effort`: "low" | "medium" | "high", or empty for the model's default.
    ClaudeApi {
        model: String,
        #[serde(default)]
        effort: String,
    },
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
            Provider::ClaudeCli { model } => format!("Claude {model} (cloud)"),
            Provider::ClaudeApi { model, effort } if effort.is_empty() => format!("Claude {} (API)", api_model_name(model)),
            Provider::ClaudeApi { model, effort } => format!("Claude {} · {effort} (API)", api_model_name(model)),
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
supported by the transcript. If something is unclear, say so briefly. Output Markdown only. \
Begin directly with the requested output: no preamble, and no remarks about reading files or what you are about to do.";

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
        t("enhanced", "Enhanced notes", "Turn my notes (in <my_notes>) into complete meeting notes. Keep my \
structure, order and headings, and my wording where it is clear. Under each of my points add what the transcript \
says about it: numbers, names, who said what, decisions, with timestamps like [12:34]. Answer the points I marked \
as missed or unclear. Then add a short **Also discussed** section with anything important my notes leave out. \
If there are no notes, write concise notes from the transcript instead."),
    ]
}

/// The user's notes, with how to use them. Goes after the transcript (which API engines cache).
fn notes_block(notes: &str) -> String {
    let n = notes.trim();
    if n.is_empty() {
        return String::new();
    }
    format!(
        "\n\nMy own notes from this meeting are below. They show what mattered to me: cover every point in them, \
         filled out with detail from the transcript. A time in a note (like [08:12], @8, 8:30 or min 8) points to that \
         moment in the recording; use the transcript around it. Where a note says I missed something or marks it \
         unclear (?), fill it in from the transcript at that time. If a note disagrees with the transcript, follow the \
         transcript and point out the difference briefly. The notes may be terse, misspelled or in another language.\n\
         <my_notes>\n{n}\n</my_notes>"
    )
}

fn glossary_block(glossary: &str) -> String {
    let g = glossary.trim();
    if g.is_empty() {
        String::new()
    } else {
        format!("\n\nNames and terms that appear in this meeting (use these spellings; the transcript may misspell them):\n{g}")
    }
}

/// Tuning for the local writer, from Settings → Writing.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct LocalOptions {
    /// Sampling temperature: low keeps minutes close to the transcript.
    pub temperature: f32,
    /// Context window FastFlowLM is started with, in tokens.
    pub context: u32,
    /// Start FastFlowLM when nothing answers at the server URL.
    pub autostart: bool,
    /// Give up on one answer after this many minutes.
    pub timeout_min: u32,
}

impl Default for LocalOptions {
    fn default() -> Self {
        LocalOptions { temperature: 0.2, context: FLM_CONTEXT, autostart: true, timeout_min: 30 }
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
pub fn build_prompt(template: &Template, output_language: &str, title: &str, transcript: &str, glossary: &str, notes: &str) -> String {
    let lang = match output_language {
        "ar" => "Write the output in Arabic (Modern Standard Arabic, business register).",
        "auto" => "Write the output in the main language of the meeting.",
        _ => "Write the output in English.",
    };
    format!(
        "{}\n{}{}\n\nMeeting title: {}\n\n<transcript>\n{}\n</transcript>{}",
        template.prompt,
        lang,
        glossary_block(glossary),
        title,
        transcript,
        notes_block(notes)
    )
}

/// Prompt for a question about one meeting, with recent Q&A for follow-ups.
pub fn build_question(
    title: &str,
    transcript: &str,
    glossary: &str,
    notes: &str,
    history: &[(String, String)],
    question: &str,
) -> String {
    let mut prior = String::new();
    for (role, text) in history {
        prior.push_str(&format!("{}: {}\n\n", if role == "user" { "Q" } else { "A" }, text));
    }
    format!(
        "Answer the question from this meeting's transcript. Be concise unless asked for detail. Cite the timestamps you rely on, \
         like [12:34]. If the transcript doesn't contain what the question needs, say so plainly. When asked what you think, or for \
         advice, suggestions, risks or next steps, give your own view: direct and practical, based on the transcript, and kept \
         clearly apart from what was actually said. Answer in the language of the question.{}\n\n\
         Meeting title: {}\n\n<transcript>\n{}\n</transcript>{}\n\n{}{}Q: {}",
        glossary_block(glossary),
        title,
        transcript,
        notes_block(notes),
        if prior.is_empty() { "" } else { "Earlier in this conversation:\n\n" },
        prior,
        question
    )
}

/// Prompt to fill in one note from the transcript around the time it points to.
/// `excerpt` is the transcript from a few minutes either side of `at_s`.
pub fn build_fill(title: &str, excerpt: &str, glossary: &str, at_s: f64, note: &str) -> String {
    format!(
        "During this meeting I wrote the note below at about [{}] in the recording, but I missed part of what was said. \
         From the transcript excerpt, say what was said there that my note is about: the facts, numbers, names and who \
         said them, with timestamps like [12:34]. Answer in one to three plain-text sentences (no Markdown), in the language of my note, with no \
         preamble. If the excerpt doesn't cover it, say so in one sentence.{}\n\n\
         Meeting title: {}\n\n<transcript>\n{}\n</transcript>\n\nMy note: {}",
        crate::store::fmt_ts(at_s),
        glossary_block(glossary),
        title,
        excerpt,
        note.trim()
    )
}

/// A writer's answer. API engines also report tokens and cost.
#[derive(Clone, Debug, Default)]
pub struct Reply {
    pub text: String,
    pub cost_usd: Option<f64>,
    /// e.g. "21,400 in (18,000 cached) / 1,250 out"
    pub usage: String,
}

impl From<String> for Reply {
    fn from(text: String) -> Self {
        Reply { text, ..Default::default() }
    }
}

/// Send a prompt to the writer. `reuse`: the transcript part of this prompt will be sent
/// again soon (questions), so the API engine caches it.
pub fn generate(provider: &Provider, prompt: &str, reuse: bool, local: &LocalOptions) -> Result<Reply> {
    match provider {
        Provider::ClaudeCli { model } => claude_cli(model, prompt).map(Reply::from),
        Provider::ClaudeApi { model, effort } => claude_api(model, effort, prompt, reuse),
        Provider::CursorCli { model } => cursor_cli(model, prompt).map(Reply::from),
        Provider::Local { base_url, model } => openai_compatible(base_url, model, prompt, local).map(Reply::from),
    }
}

// ---------- Claude API ----------

pub const API_URL: &str = "https://api.anthropic.com/v1";
const API_VERSION: &str = "2023-06-01";

/// Claude models offered in the engine menus: (API id, short name).
pub const API_MODELS: &[(&str, &str)] =
    &[("claude-haiku-5-5", "Haiku 5.5"), ("claude-sonnet-5-5", "Sonnet 5.5"), ("claude-opus-5-5", "Opus 5.5")];

pub fn api_model_name(model: &str) -> String {
    API_MODELS.iter().find(|(id, _)| *id == model).map(|(_, n)| n.to_string()).unwrap_or_else(|| model.to_string())
}

/// USD per million tokens: (input, output, cache write, cache read). Same table as Mizan
/// (platform.claude.com pricing, read 2026-10-12). Unknown models are priced high so the
/// spend shown errs on the safe side.
pub fn api_prices(model: &str, prompt_tokens: u64) -> (f64, f64, f64, f64) {
    match model {
        m if m.starts_with("claude-haiku-5-5") && prompt_tokens > 100_000 => (0.50, 2.50, 0.625, 0.05),
        m if m.starts_with("claude-haiku-5-5") => (0.10, 0.50, 0.125, 0.01),
        m if m.starts_with("claude-sonnet-5") => (2.0, 10.0, 2.50, 0.10),
        m if m.starts_with("claude-opus-5-5") => (4.0, 20.0, 5.0, 0.20),
        _ => (10.0, 50.0, 12.50, 1.0),
    }
}

fn thousands(v: u64) -> String {
    let s = v.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Cost in USD of one response's `usage` block, and a short description of it.
pub fn price_usage(model: &str, usage: &serde_json::Value) -> (f64, String) {
    let n = |k: &str| usage.get(k).and_then(serde_json::Value::as_u64).unwrap_or(0);
    let (input, output, cw, cr) = (n("input_tokens"), n("output_tokens"), n("cache_creation_input_tokens"), n("cache_read_input_tokens"));
    let (pi, po, pw, pr) = api_prices(model, input + cw + cr);
    let usd = (input as f64 * pi + output as f64 * po + cw as f64 * pw + cr as f64 * pr) / 1e6;
    let cached = if cr + cw > 0 { format!(" ({} cached)", thousands(cr + cw)) } else { String::new() };
    (usd, format!("{} in{cached} / {} out", thousands(input + cw + cr), thousands(output)))
}

/// The user message, with the transcript part marked for caching when it will be reused.
fn api_content(prompt: &str, reuse: bool) -> serde_json::Value {
    const END: &str = "</transcript>";
    if reuse {
        if let Some(i) = prompt.find(END) {
            let (head, rest) = prompt.split_at(i + END.len());
            let mut blocks = vec![serde_json::json!({"type": "text", "text": head, "cache_control": {"type": "ephemeral"}})];
            if !rest.trim().is_empty() {
                blocks.push(serde_json::json!({"type": "text", "text": rest}));
            }
            return serde_json::Value::Array(blocks);
        }
    }
    serde_json::json!([{"type": "text", "text": prompt}])
}

fn api_client(timeout: Duration) -> Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder().timeout(timeout).build()?)
}

fn api_error(status: reqwest::StatusCode, body: &str) -> String {
    let msg = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
        .unwrap_or_else(|| body.chars().take(300).collect());
    if status.as_u16() == 401 {
        return "the Claude API key was rejected. Check it in Settings → Claude API.".into();
    }
    format!("Claude API error {}: {msg}", status.as_u16())
}

fn claude_api(model: &str, effort: &str, prompt: &str, reuse: bool) -> Result<Reply> {
    let (key, _) = crate::secrets::anthropic_key().ok_or_else(|| anyhow!("no Claude API key yet. Add it in Settings → Claude API."))?;
    let mut body = serde_json::json!({
        "model": model,
        // Room for thinking at high effort plus long minutes.
        "max_tokens": 32000,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": api_content(prompt, reuse)}],
    });
    if !effort.is_empty() {
        body["output_config"] = serde_json::json!({"effort": effort});
    }
    let client = api_client(Duration::from_secs(15 * 60))?;
    let mut last = String::new();
    for attempt in 0..4u32 {
        if attempt > 0 {
            std::thread::sleep(Duration::from_secs(2u64.pow(attempt)));
        }
        let resp = client
            .post(format!("{API_URL}/messages"))
            .header("x-api-key", &key)
            .header("anthropic-version", API_VERSION)
            .json(&body)
            .send();
        let resp = match resp {
            Ok(r) => r,
            Err(e) => {
                last = format!("could not reach the Claude API: {e}");
                continue;
            }
        };
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        if !status.is_success() {
            last = api_error(status, &text);
            // Overloaded / rate limited / server error: try again; anything else is final.
            if status.as_u16() == 429 || status.as_u16() == 529 || status.is_server_error() {
                continue;
            }
            bail!(last);
        }
        let v: serde_json::Value = serde_json::from_str(&text).context("the Claude API returned invalid JSON")?;
        let out: String = v["content"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|b| b["type"] == "text")
            .filter_map(|b| b["text"].as_str())
            .collect();
        let (usd, usage) = price_usage(model, &v["usage"]);
        if out.trim().is_empty() {
            bail!("Claude returned no text (stop reason: {})", v["stop_reason"].as_str().unwrap_or("?"));
        }
        let mut out = out.trim().to_string();
        if v["stop_reason"] == "max_tokens" {
            out.push_str("\n\n_(cut off at the length limit)_");
        }
        return Ok(Reply { text: out, cost_usd: Some(usd), usage });
    }
    bail!(last)
}

/// Check a key without spending anything (lists models). Says which Mahdar models it can use.
pub fn test_api_key(key: &str) -> Result<String> {
    let resp = api_client(Duration::from_secs(20))?
        .get(format!("{API_URL}/models"))
        .header("x-api-key", key.trim())
        .header("anthropic-version", API_VERSION)
        .send()
        .context("could not reach the Claude API")?;
    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    if !status.is_success() {
        bail!(api_error(status, &text));
    }
    let v: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
    let ids: Vec<&str> = v["data"].as_array().into_iter().flatten().filter_map(|m| m["id"].as_str()).collect();
    let ours: Vec<String> = API_MODELS.iter().filter(|(id, _)| ids.contains(id)).map(|(_, n)| n.to_string()).collect();
    Ok(if ours.is_empty() { format!("key works ({} models)", ids.len()) } else { format!("key works · {}", ours.join(", ")) })
}

fn claude_cli(model: &str, prompt: &str) -> Result<String> {
    let mut cmd = Command::new("claude");
    cmd.args(["-p", "--model", model, "--output-format", "text", "--no-session-persistence"])
        .args(["--system-prompt", SYSTEM])
        .args(["--tools", ""])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut child = cmd.spawn().context("could not start `claude`. Is Claude Code installed and on PATH?")?;
    child.stdin.take().unwrap().write_all(prompt.as_bytes())?;
    let out = child.wait_with_output()?;
    if !out.status.success() {
        bail!("claude failed: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if text.is_empty() {
        bail!("claude returned no text");
    }
    Ok(text)
}

/// How to launch Cursor's CLI. The Windows installer puts a bundled node.exe +
/// index.js under %LOCALAPPDATA%\cursor-agent\versions\<date-commit>; running
/// those directly avoids the agent.cmd → PowerShell hop. Otherwise (or with
/// `direct` false) use the `agent` command, which picks its own version.
fn cursor_command(direct: bool) -> Command {
    let versions = std::env::var_os("LOCALAPPDATA")
        .map(|l| std::path::PathBuf::from(l).join("cursor-agent").join("versions"));
    if let Some(dir) = versions.filter(|_| direct).and_then(|v| newest_cursor_version(&v)) {
        let mut c = Command::new(dir.join("node.exe"));
        c.arg(dir.join("index.js")).env("CURSOR_INVOKED_AS", "agent");
        return c;
    }
    if cfg!(windows) {
        let mut c = Command::new("cmd");
        c.args(["/C", "agent"]);
        c
    } else {
        Command::new("agent")
    }
}

/// The newest version folder. Only dated ones (YYYY.MM.DD-commit) count: others, like
/// `dist-package`, can be incomplete, and a plain name sort puts them after the dates.
fn newest_cursor_version(versions: &std::path::Path) -> Option<std::path::PathBuf> {
    std::fs::read_dir(versions)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with(|c: char| c.is_ascii_digit()))
                && p.join("node.exe").is_file()
                && p.join("index.js").is_file()
        })
        .max()
}

fn cursor_cli(model: &str, prompt: &str) -> Result<String> {
    match cursor_cli_with(model, prompt, true) {
        // A broken version folder: let Cursor's own launcher choose.
        Err(e) if format!("{e:#}").contains("MODULE_NOT_FOUND") => cursor_cli_with(model, prompt, false),
        r => r,
    }
}

fn cursor_cli_with(model: &str, prompt: &str, direct: bool) -> Result<String> {
    // The request goes in a file inside a throwaway workspace: transcripts are
    // too long for a Windows command line, and ask mode keeps the agent read-only.
    let ws = std::env::temp_dir().join(format!("mahdar-cursor-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&ws)?;
    std::fs::write(ws.join("mahdar_request.md"), format!("# Instructions\n\n{SYSTEM}\n\n{prompt}\n"))?;
    let mut cmd = cursor_command(direct);
    cmd.args(["-p", "--output-format", "stream-json", "--mode", "ask", "--trust", "--workspace"])
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
    let text = cursor_answer(&stdout);
    if text.trim().is_empty() {
        bail!("Cursor returned no text. Run `agent login` (or check `agent status`).");
    }
    Ok(text.trim().to_string())
}

/// The answer from Cursor's stream-json output: only the assistant text after its last
/// tool call. The agent narrates before reading the request file ("I'll read the file
/// first."), and the final `result` field joins that narration onto the answer.
fn cursor_answer(stdout: &str) -> String {
    let mut after_tools = String::new();
    let mut result = None;
    let mut events = 0;
    for v in stdout.lines().filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok()) {
        events += 1;
        match v["type"].as_str() {
            Some("tool_call") => after_tools.clear(),
            Some("assistant") => {
                for b in v["message"]["content"].as_array().into_iter().flatten() {
                    if let Some(t) = b["text"].as_str() {
                        after_tools.push_str(t);
                    }
                }
            }
            Some("result") => result = v["result"].as_str().map(str::to_string),
            _ => {}
        }
    }
    let text = if !after_tools.trim().is_empty() {
        after_tools
    } else if let Some(r) = result {
        r
    } else if events == 0 {
        stdout.to_string()
    } else {
        String::new()
    };
    strip_preamble(text.trim()).to_string()
}

/// Drop a leading "I'll read the file first." style sentence, in case it still gets through.
fn strip_preamble(text: &str) -> &str {
    let end = text.find(['\n', '.', ':']).map(|i| i + 1).unwrap_or(text.len());
    let first = text[..end].to_lowercase();
    let narrates = ["i'll", "i will", "let me", "i'm going to", "first,"].iter().any(|w| first.starts_with(w));
    let about_file = ["read", "file", "request", "instructions", "open", "look at"].iter().any(|w| first.contains(w));
    if end < 160 && narrates && about_file {
        text[end..].trim_start()
    } else {
        text
    }
}

/// Default context window for FastFlowLM: enough for ~1.5 h of transcript.
pub const FLM_CONTEXT: u32 = 65_536;

/// Make sure the local OpenAI-compatible server is up. If nothing answers at
/// `base_url`, start FastFlowLM (`flm serve <model>`) on that port and wait for it,
/// unless auto-start is off. Returns the child process when Mahdar started it
/// (caller stops it on exit).
pub fn ensure_local_server(base_url: &str, model: &str, local: &LocalOptions, on_starting: impl FnOnce()) -> Result<Option<Child>> {
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

    if !local.autostart {
        bail!("nothing answers at {base}. Start the local server (flm serve {model}), or turn on auto-start in Settings → Writing.");
    }
    let port = base.rsplit(':').next().and_then(|p| p.split('/').next()).and_then(|p| p.parse::<u16>().ok()).unwrap_or(11434);
    on_starting();
    let mut cmd = Command::new("flm");
    cmd.args(["serve", model, "--quiet", "-c", &local.context.to_string(), "-p", &port.to_string()])
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

fn openai_compatible(base_url: &str, model: &str, prompt: &str, local: &LocalOptions) -> Result<String> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt}
        ],
        "temperature": local.temperature,
        "stream": false
    });
    let resp = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(60 * u64::from(local.timeout_min.max(1))))
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

    #[test]
    fn prices_api_usage() {
        let u = serde_json::json!({"input_tokens": 1000, "output_tokens": 2000, "cache_read_input_tokens": 20000});
        let (usd, desc) = price_usage("claude-sonnet-5-5", &u);
        // 1k × $2 + 2k × $10 + 20k × $0.10 per million
        assert!((usd - 0.024).abs() < 1e-9, "{usd}");
        assert_eq!(desc, "21,000 in (20,000 cached) / 2,000 out");
    }

    #[test]
    fn caches_the_transcript_only_for_questions() {
        let q = build_question("T", "line", "", "@8 price?", &[], "who?");
        let c = api_content(&q, true);
        assert_eq!(c.as_array().unwrap().len(), 2);
        assert!(c[0]["text"].as_str().unwrap().ends_with("</transcript>"));
        assert!(c[0]["cache_control"].is_object());
        assert!(c[1]["text"].as_str().unwrap().contains("Q: who?"));
        assert!(c[1]["text"].as_str().unwrap().contains("<my_notes>\n@8 price?\n</my_notes>"));
        let p = build_prompt(&builtin_templates()[0], "en", "T", "line", "", "");
        assert!(api_content(&p, false)[0]["cache_control"].is_null());
        assert!(!p.contains("my_notes"));
        let p = build_prompt(&builtin_templates()[0], "en", "T", "line", "", " \n ");
        assert!(!p.contains("my_notes"));
    }

    /// Hits the real API with a made-up key: checks the URL, headers and 401 handling.
    /// `cargo test -p mahdar-core -- --ignored live_`
    #[test]
    #[ignore]
    fn live_bad_key_is_rejected() {
        let e = test_api_key("sk-ant-api03-not-a-real-key").unwrap_err().to_string();
        assert!(e.contains("rejected"), "{e}");
    }

    /// The key saved in Settings (or Mizan) works. Free: lists models only.
    #[test]
    #[ignore]
    fn live_saved_key_works() {
        let (key, src) = crate::secrets::anthropic_key().expect("no key saved");
        println!("key from {src}: {}", test_api_key(&key).unwrap());
    }

    /// One real question on a made-up transcript with Haiku (well under a cent).
    #[test]
    #[ignore]
    fn live_generate_question() {
        let t = "[00:00] Speaker 1: We approve the plan.
[00:12] Speaker 2: I'll send the report by Thursday.";
        let q = build_question("Demo", t, "", "", &[], "Who sends the report, and when?");
        let r = generate(&Provider::ClaudeApi { model: "claude-haiku-5-5".into(), effort: "high".into() }, &q, true, &LocalOptions::default()).unwrap();
        println!("answer: {}
cost: ${:.5} · {}", r.text, r.cost_usd.unwrap(), r.usage);
        assert!(r.text.contains("Speaker 2"));
    }

    #[test]
    fn cursor_keeps_only_the_answer() {
        let out = [
            r#"{"type":"system","subtype":"init","model":"x"}"#,
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"I'll read the file first."}]}}"#,
            r#"{"type":"tool_call","subtype":"started","call_id":"1","tool_call":{}}"#,
            r#"{"type":"tool_call","subtype":"completed","call_id":"1","tool_call":{}}"#,
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"- [ ] Re-check the scope"}]}}"#,
            r#"{"type":"result","subtype":"success","result":"I'll read the file first.- [ ] Re-check the scope"}"#,
        ]
        .join("\n");
        assert_eq!(cursor_answer(&out), "- [ ] Re-check the scope");
        // Only a result line: the preamble is cut from it.
        let only = r#"{"type":"result","subtype":"success","result":"I'll read the file first.- [ ] Re-check the scope"}"#;
        assert_eq!(cursor_answer(only), "- [ ] Re-check the scope");
        assert_eq!(cursor_answer("plain text answer"), "plain text answer");
        assert_eq!(strip_preamble("I will send the report: Thursday"), "I will send the report: Thursday");
    }

    #[test]
    fn api_provider_round_trips() {
        let p: Provider = serde_json::from_str(r#"{"kind":"claude_api","model":"claude-opus-5-5"}"#).unwrap();
        assert_eq!(p.label(), "Claude Opus 5.5 (API)");
        let p: Provider = serde_json::from_str(r#"{"kind":"claude_api","model":"claude-haiku-5-5","effort":"high"}"#).unwrap();
        assert_eq!(p.label(), "Claude Haiku 5.5 · high (API)");
    }
}
