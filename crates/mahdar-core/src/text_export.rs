//! A meeting as text: the transcript as plain text, Markdown or subtitles (.srt), and the whole
//! meeting (notes, outputs, transcript) as one Markdown file.

use serde::Deserialize;

use crate::store::{fmt_ts, Meeting};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextFormat {
    /// "[mm:ss] Speaker: text" per line, as the writers get it.
    Text,
    /// The transcript as Markdown, with a title.
    Markdown,
    /// Subtitles for a video player.
    Srt,
    /// The whole meeting as Markdown: notes, every output, the transcript.
    Meeting,
}

pub fn render(m: &Meeting, format: TextFormat) -> String {
    match format {
        TextFormat::Text => m.transcript_text(),
        TextFormat::Markdown => format!("# {}\n\nTranscript · {}\n\n{}\n", m.title, about(m), transcript_md(m)),
        TextFormat::Srt => srt(m),
        TextFormat::Meeting => meeting_md(m),
    }
}

/// "04 November 2026 · 20:23 · 1:02:03"
fn about(m: &Meeting) -> String {
    let mut s = m.created.format("%d %B %Y · %H:%M").to_string();
    if m.duration_s > 0.0 {
        s.push_str(&format!(" · {}", fmt_ts(m.duration_s)));
    }
    s
}

enum Line {
    /// A break in the recording, at a time, as `Gap::label` says it.
    Break(f64, String),
    Said { start: f64, end: f64, who: Option<String>, text: String },
}

/// The transcript's lines in order, with each break before the line it falls in, or the next
/// one (as in `Meeting::transcript_between`).
fn lines(m: &Meeting) -> Vec<Line> {
    let mut out = Vec::new();
    let mut prev_end = f64::NEG_INFINITY;
    for c in &m.chunks {
        let after = std::mem::replace(&mut prev_end, c.end_s);
        for g in m.gaps.iter().filter(|g| g.at_s >= after && g.at_s < c.end_s) {
            out.push(Line::Break(g.at_s, g.label()));
        }
        let text = crate::asr::clean_text(&c.text);
        if !text.is_empty() {
            out.push(Line::Said { start: c.start_s, end: c.end_s, who: c.speaker.map(|n| m.speaker_name(n)), text });
        }
    }
    out
}

fn transcript_md(m: &Meeting) -> String {
    lines(m)
        .into_iter()
        .map(|l| match l {
            Line::Break(at, label) => format!("*[{}] {label}*", fmt_ts(at)),
            Line::Said { start, who: Some(who), text, .. } => format!("**[{}] {who}:** {text}", fmt_ts(start)),
            Line::Said { start, text, .. } => format!("**[{}]** {text}", fmt_ts(start)),
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn srt(m: &Meeting) -> String {
    let time = |s: f64| {
        let ms = (s.max(0.0) * 1000.0).round() as u64;
        format!("{:02}:{:02}:{:02},{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
    };
    lines(m)
        .into_iter()
        .filter_map(|l| match l {
            Line::Said { start, end, who, text } => Some((start, end, who.map(|w| format!("{w}: {text}")).unwrap_or(text))),
            Line::Break(..) => None,
        })
        .enumerate()
        .map(|(i, (start, end, text))| format!("{}\n{} --> {}\n{text}\n", i + 1, time(start), time(end)))
        .collect::<Vec<_>>()
        .join("\n")
}

fn meeting_md(m: &Meeting) -> String {
    let mut parts = vec![format!("# {}\n\n{}", m.title, about(m))];
    if !m.notes.trim().is_empty() {
        parts.push(format!("## Notes\n\n{}", m.notes.trim()));
    }
    for o in &m.outputs {
        parts.push(format!("## {}\n\n*{} · {}*\n\n{}", o.template_name, o.provider, o.created.format("%d %B %Y %H:%M"), demote(o.text.trim())));
    }
    let t = transcript_md(m);
    if !t.is_empty() {
        parts.push(format!("## Transcript\n\n{t}"));
    }
    parts.join("\n\n") + "\n"
}

/// Headings two levels down, so an output's own "# Title" sits under its "## Output" heading.
/// Lines inside ``` code blocks are left alone.
fn demote(md: &str) -> String {
    let mut code = false;
    md.lines()
        .map(|l| {
            if l.trim_start().starts_with("```") {
                code = !code;
            }
            let hashes = l.len() - l.trim_start_matches('#').len();
            if !code && (1..=4).contains(&hashes) && l[hashes..].starts_with(' ') {
                format!("##{l}")
            } else {
                l.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{Chunk, Gap, GapKind, Output};
    use chrono::{Local, TimeZone};

    fn meeting() -> Meeting {
        let at = Local.with_ymd_and_hms(2026, 11, 4, 20, 23, 0).unwrap();
        let chunk = |a: f64, b: f64, text: &str, speaker: Option<u32>| Chunk { start_s: a, end_s: b, text: text.into(), flag: String::new(), speaker, edited: false };
        let mut m: Meeting = serde_json::from_value(serde_json::json!({
            "id": "m1", "title": "Board", "created": at, "source": "microphone", "duration_s": 75.5,
            "language": "ar", "status": "ready"
        }))
        .unwrap();
        m.speakers.insert(1, "Amjad".into());
        m.chunks = vec![chunk(0.0, 12.5, "Welcome", Some(1)), chunk(12.5, 30.0, "  ", None), chunk(30.0, 75.5, "Budget is 40k", Some(2))];
        m.gaps = vec![Gap { at_s: 30.0, missing_s: 300.0, kind: GapKind::Paused }];
        m.notes = "[00:10] agenda\n".into();
        m.outputs = vec![Output {
            id: "o1".into(),
            template_id: "mom".into(),
            template_name: "Minutes of Meeting".into(),
            provider: "Cursor (cloud)".into(),
            created: at,
            text: "# Minutes\n\n## Decisions\n- go\n\n```\n# not a heading\n```".into(),
        }];
        m
    }

    #[test]
    fn transcript_as_markdown_and_text() {
        let m = meeting();
        assert_eq!(
            render(&m, TextFormat::Markdown),
            "# Board\n\nTranscript · 04 November 2026 · 20:23 · 01:15\n\n**[00:00] Amjad:** Welcome\n\n*[00:30] (recording paused for 05:00)*\n\n**[00:30] Speaker 2:** Budget is 40k\n"
        );
        assert_eq!(render(&m, TextFormat::Text), m.transcript_text());
    }

    #[test]
    fn subtitles() {
        assert_eq!(
            render(&meeting(), TextFormat::Srt),
            "1\n00:00:00,000 --> 00:00:12,500\nAmjad: Welcome\n\n2\n00:00:30,000 --> 00:01:15,500\nSpeaker 2: Budget is 40k\n"
        );
    }

    #[test]
    fn whole_meeting() {
        let md = render(&meeting(), TextFormat::Meeting);
        assert!(md.starts_with("# Board\n\n04 November 2026 · 20:23 · 01:15\n\n## Notes\n\n[00:10] agenda\n\n## Minutes of Meeting\n\n*Cursor (cloud) · 04 November 2026 20:23*\n\n### Minutes\n\n#### Decisions\n- go\n\n```\n# not a heading\n```\n\n## Transcript\n\n**[00:00] Amjad:**"), "{md}");
        assert!(md.ends_with("Budget is 40k\n"));
    }
}
