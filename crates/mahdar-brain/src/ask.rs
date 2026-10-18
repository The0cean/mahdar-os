//! Questions across all meetings: fuse keyword and meaning search, send only the best
//! excerpts to the writer, and turn its [#n] citations into links.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::Hit;

/// Reciprocal rank fusion of ranked id lists (best first): score = Σ 1 / (k + rank).
pub fn rrf(lists: &[Vec<i64>], k: f64) -> Vec<(i64, f64)> {
    let mut score: HashMap<i64, f64> = HashMap::new();
    for list in lists {
        for (rank, id) in list.iter().enumerate() {
            *score.entry(*id).or_default() += 1.0 / (k + rank as f64 + 1.0);
        }
    }
    let mut out: Vec<(i64, f64)> = score.into_iter().collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    out
}

/// The question with numbered excerpts and the conversation so far (question, answer), oldest
/// first. Only these excerpts are sent, never whole transcripts.
pub fn prompt(question: &str, hits: &[Hit], history: &[(String, String)]) -> String {
    let mut ex = String::new();
    for (i, h) in hits.iter().enumerate() {
        let date = h.created.get(..10).unwrap_or("");
        ex.push_str(&format!("[#{}] {} ({}) at {}: {}\n\n", i + 1, h.title, date, crate::text::fmt_ts(h.start_s), h.text));
    }
    let mut prior = String::new();
    for (q, a) in history {
        // Earlier answers cited other excerpt numbers; drop them so they can't be confused with these.
        prior.push_str(&format!("User: {q}\n\nYou: {}\n\n", strip_cites(a)));
    }
    let prior = if prior.is_empty() { String::new() } else { format!("<conversation>\n{prior}</conversation>\n\n") };
    format!(
        "You are helping the user think through their meetings. Your facts come from these excerpts from the user's meeting \
         transcripts (speech recognition, Arabic and English; English terms are often written in Arabic script). Cite every \
         statement about what was said with the excerpt numbers in square brackets, like [#3] or [#3][#7]. When excerpts from \
         different meetings differ, say which meeting said what, oldest first. When the user asks what you think, or for advice, \
         suggestions, risks or next steps, give your own view: direct and practical, based on the excerpts and the conversation, \
         and kept clearly apart from what was actually said. Never invent facts, names, numbers or decisions; if the excerpts \
         don't contain something the question needs, say so plainly. This may be a follow-up: read it in the light of the \
         conversation so far. Answer in the language of the question. Be concise unless asked for detail.\n\n\
         <excerpts>\n{ex}</excerpts>\n\n{prior}Question: {question}"
    )
}

/// Remove "[#n]" citation markers.
pub fn strip_cites(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(i) = rest.find("[#") {
        out.push_str(&rest[..i]);
        let after = &rest[i + 2..];
        let digits = after.len() - after.trim_start_matches(|c: char| c.is_ascii_digit()).len();
        if digits > 0 && after[digits..].starts_with(']') {
            rest = &after[digits + 1..];
        } else {
            out.push_str("[#");
            rest = after;
        }
    }
    out.push_str(rest);
    out.replace("  ", " ").replace(" .", ".")
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Citation {
    pub n: usize,
    pub meeting_id: String,
    pub title: String,
    pub start_s: f64,
    pub speaker: Option<String>,
}

/// Normalize citations in `answer` to "[#n]" (also accepting "[#3, #5]", "[#3,5]", "[3]"),
/// drop numbers that aren't excerpts, and list the excerpts used in order of first use.
pub fn cite(answer: &str, hits: &[Hit]) -> (String, Vec<Citation>) {
    let mut out = String::new();
    let mut used: Vec<usize> = Vec::new();
    let mut rest = answer;
    while let Some(open) = rest.find('[') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let Some(close) = after.find(']') else {
            out.push_str(&rest[open..]);
            rest = "";
            break;
        };
        let inner = &after[..close];
        let looks_like = !inner.is_empty() && inner.chars().all(|c| c.is_ascii_digit() || matches!(c, '#' | ',' | ' ' | ';' | '،'));
        let nums: Vec<usize> = if looks_like {
            inner.split(|c: char| !c.is_ascii_digit()).filter_map(|s| s.parse().ok()).collect()
        } else {
            Vec::new()
        };
        if looks_like && inner.contains(|c: char| c.is_ascii_digit()) {
            for n in nums.into_iter().filter(|n| *n >= 1 && *n <= hits.len()) {
                out.push_str(&format!("[#{n}]"));
                if !used.contains(&n) {
                    used.push(n);
                }
            }
        } else {
            out.push('[');
            out.push_str(inner);
            out.push(']');
        }
        rest = &after[close + 1..];
    }
    out.push_str(rest);
    let cites = used
        .into_iter()
        .map(|n| {
            let h = &hits[n - 1];
            Citation { n, meeting_id: h.doc_id.clone(), title: h.title.clone(), start_s: h.start_s, speaker: h.speaker.clone() }
        })
        .collect();
    (out, cites)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(id: &str, t: f64) -> Hit {
        Hit { passage_id: 0, doc_id: id.into(), title: format!("Meeting {id}"), created: "2026-10-01T10:00:00+03:00".into(), start_s: t, end_s: t + 30.0, speaker: Some("Speaker 2".into()), text: "x".into(), score: 0.0 }
    }

    #[test]
    fn fuses_rankings() {
        let r = rrf(&[vec![1, 2, 3], vec![3, 1, 9]], 60.0);
        let ids: Vec<i64> = r.iter().map(|x| x.0).collect();
        assert_eq!(ids, vec![1, 3, 2, 9]);
    }

    #[test]
    fn citations_become_links() {
        let hits = vec![hit("a", 10.0), hit("b", 754.0), hit("c", 0.0)];
        let (text, c) = cite("It goes out Thursday [#2, #1]. Total is 5 [3][#9]. See [note].", &hits);
        assert_eq!(text, "It goes out Thursday [#2][#1]. Total is 5 [#3]. See [note].");
        assert_eq!(c.iter().map(|x| x.n).collect::<Vec<_>>(), vec![2, 1, 3]);
        assert_eq!(c[0].meeting_id, "b");
        assert_eq!(c[0].start_s, 754.0);
        let (t2, c2) = cite("nothing cited [", &hits);
        assert_eq!((t2.as_str(), c2.len()), ("nothing cited [", 0));
    }

    #[test]
    fn prompt_numbers_excerpts() {
        let p = prompt("who?", &[hit("a", 75.0)], &[]);
        assert!(p.contains("[#1] Meeting a (2026-10-01) at 01:15: x"));
        assert!(p.ends_with("Question: who?"));
        assert!(!p.contains("<conversation>"));
        let f = prompt("what do you think?", &[hit("a", 75.0)], &[("who?".into(), "Speaker 2 [#4][#1]. Next [#x].".into())]);
        assert!(f.contains("User: who?\n\nYou: Speaker 2. Next [#x].\n\n</conversation>"), "{f}");
        assert!(f.ends_with("Question: what do you think?"));
    }
}
