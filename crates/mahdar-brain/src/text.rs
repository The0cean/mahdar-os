//! Text helpers: Arabic-aware normalization for keyword search, passage grouping, timestamps.

/// One transcript line (or any timestamped piece of a document).
#[derive(Clone, Debug)]
pub struct Line {
    pub start_s: f64,
    pub end_s: f64,
    pub speaker: Option<String>,
    pub text: String,
}

/// A searchable window of a document: ~30–60 s of speech.
#[derive(Clone, Debug, PartialEq)]
pub struct Passage {
    pub start_s: f64,
    pub end_s: f64,
    /// Speaker, or "A, B" when a short exchange was kept together.
    pub speaker: Option<String>,
    /// "Speaker: text" for each line, joined.
    pub text: String,
}

/// Longest passage, in seconds.
pub const MAX_SPAN: f64 = 60.0;
/// A passage shorter than this may take in the next speaker's turn, so a short
/// question stays with its answer.
pub const MIN_SPAN: f64 = 15.0;

/// Group lines into passages: a new passage starts when the speaker changes (unless the
/// current one is still under [`MIN_SPAN`]) or when it would grow past [`MAX_SPAN`].
pub fn passages(lines: &[Line]) -> Vec<Passage> {
    let mut out: Vec<Passage> = Vec::new();
    let mut cur: Option<(Passage, Vec<String>)> = None;
    for l in lines {
        let text = l.text.trim();
        if text.is_empty() {
            continue;
        }
        let piece = match &l.speaker {
            Some(s) => format!("{s}: {text}"),
            None => text.to_string(),
        };
        if let Some((p, speakers)) = cur.as_mut() {
            let same = speakers.last() == l.speaker.as_ref() || l.speaker.is_none();
            let fits = l.end_s - p.start_s <= MAX_SPAN;
            let short = p.end_s - p.start_s < MIN_SPAN;
            if fits && (same || short) {
                p.text.push(' ');
                p.text.push_str(&piece);
                p.end_s = p.end_s.max(l.end_s);
                if let Some(s) = &l.speaker {
                    if !speakers.contains(s) {
                        speakers.push(s.clone());
                    }
                }
                continue;
            }
            let (mut done, sp) = cur.take().unwrap();
            done.speaker = if sp.is_empty() { None } else { Some(sp.join(", ")) };
            out.push(done);
        }
        cur = Some((
            Passage { start_s: l.start_s, end_s: l.end_s, speaker: None, text: piece },
            l.speaker.iter().cloned().collect(),
        ));
    }
    if let Some((mut done, sp)) = cur {
        done.speaker = if sp.is_empty() { None } else { Some(sp.join(", ")) };
        out.push(done);
    }
    out
}

fn is_arabic_mark(c: char) -> bool {
    // Harakat, Quranic marks, superscript alef, tatweel.
    matches!(c, '\u{0610}'..='\u{061A}' | '\u{064B}'..='\u{065F}' | '\u{0670}' | '\u{06D6}'..='\u{06ED}' | '\u{0640}')
}

/// Fold Arabic spelling variants so the same word matches however the ASR wrote it:
/// no diacritics or tatweel, أ/إ/آ/ٱ → ا, ى → ي, ة → ه, ؤ → و, ئ → ي, Arabic-Indic digits → 0-9.
/// Latin text is lowercased.
pub fn fold(s: &str) -> String {
    s.chars()
        .filter(|c| !is_arabic_mark(*c))
        .map(|c| match c {
            'أ' | 'إ' | 'آ' | 'ٱ' => 'ا',
            'ى' => 'ي',
            'ة' => 'ه',
            'ؤ' => 'و',
            'ئ' => 'ي',
            '٠'..='٩' => char::from(b'0' + (c as u32 - '٠' as u32) as u8),
            '۰'..='۹' => char::from(b'0' + (c as u32 - '۰' as u32) as u8),
            _ => c,
        })
        .flat_map(char::to_lowercase)
        .collect()
}

fn is_arabic(c: char) -> bool {
    ('\u{0600}'..='\u{06FF}').contains(&c)
}

/// Light Arabic stemming for keyword search: drop the article and attached prepositions
/// (ال، وال، بال، فال، كال، لل) when at least 3 letters remain, so "البنك", "والبنك" and
/// "بنك" all match.
fn stem(word: &str) -> &str {
    if !word.chars().next().map(is_arabic).unwrap_or(false) {
        return word;
    }
    for pre in ["وال", "بال", "فال", "كال", "لل", "ال"] {
        if let Some(rest) = word.strip_prefix(pre) {
            if rest.chars().count() >= 3 {
                return rest;
            }
        }
    }
    word
}

/// Words of a text, folded and stemmed, for the keyword index and queries.
pub fn search_words(s: &str) -> Vec<String> {
    fold(s)
        .split(|c: char| !(c.is_alphanumeric()))
        .filter(|w| !w.is_empty())
        .map(|w| stem(w).to_string())
        .collect()
}

/// Text as stored in the keyword index.
pub fn index_text(s: &str) -> String {
    search_words(s).join(" ")
}

const STOP: &[&str] = &[
    "the", "and", "for", "with", "what", "who", "when", "where", "which", "how", "did", "does", "was", "were", "are", "is",
    "about", "from", "that", "this", "have", "has", "had", "any", "all", "our", "we", "they", "you", "said", "say", "tell",
    "me", "of", "to", "in", "on", "at", "a", "an", "or", "be", "it", "do", "us", "there", "their",
    "في", "من", "علي", "الي", "عن", "ما", "ماذا", "هل", "مين", "ايه", "اللي", "هو", "هي", "كان", "قال", "مع", "او", "و",
];

/// An FTS5 query: every meaningful word, quoted, OR-ed. None when nothing is left.
pub fn fts_query(q: &str, extra: &[String]) -> Option<String> {
    let mut words: Vec<String> = Vec::new();
    for w in search_words(q).into_iter().chain(extra.iter().flat_map(|e| search_words(e))) {
        if w.chars().count() < 2 || STOP.contains(&w.as_str()) || words.contains(&w) {
            continue;
        }
        words.push(w);
    }
    if words.is_empty() {
        return None;
    }
    // Prefix match from 3 letters: "ايتمان" also finds "ايتماني", "limit" finds "limits".
    Some(
        words
            .iter()
            .map(|w| if w.chars().count() >= 3 { format!("\"{}\"*", w.replace('"', "")) } else { format!("\"{}\"", w.replace('"', "")) })
            .collect::<Vec<_>>()
            .join(" OR "),
    )
}

/// "12:34" or "1:02:03" (optionally in brackets) → seconds.
pub fn parse_ts(s: &str) -> Option<f64> {
    let s = s.trim().trim_start_matches('[').trim_end_matches(']').trim();
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() < 2 || parts.len() > 3 {
        return None;
    }
    let mut total = 0f64;
    for p in &parts {
        total = total * 60.0 + p.trim().parse::<f64>().ok()?;
    }
    Some(total)
}

pub fn fmt_ts(s: f64) -> String {
    let s = s.max(0.0) as u64;
    if s >= 3600 {
        format!("{}:{:02}:{:02}", s / 3600, (s % 3600) / 60, s % 60)
    } else {
        format!("{:02}:{:02}", s / 60, s % 60)
    }
}

/// FNV-1a, to notice when a transcript changed after it was indexed.
pub fn hash(s: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(a: f64, b: f64, sp: &str, t: &str) -> Line {
        Line { start_s: a, end_s: b, speaker: if sp.is_empty() { None } else { Some(sp.into()) }, text: t.into() }
    }

    #[test]
    fn groups_by_speaker_and_length() {
        let lines = vec![
            line(0.0, 20.0, "Speaker 1", "we need the plan approved"),
            line(20.0, 45.0, "Speaker 1", "before the review"),
            line(45.0, 70.0, "Speaker 1", "and the timeline too"), // would pass 60 s
            line(70.0, 90.0, "Speaker 2", "agreed"),
            line(90.0, 95.0, "Speaker 1", "ok?"),             // short: takes the answer
            line(95.0, 110.0, "Speaker 2", "yes, by Thursday"),
            line(110.0, 112.0, "Speaker 2", "   "),           // empty lines are skipped
        ];
        let p = passages(&lines);
        assert_eq!(p.len(), 4, "{p:#?}");
        assert_eq!(p[0].text, "Speaker 1: we need the plan approved Speaker 1: before the review");
        assert_eq!((p[0].start_s, p[0].end_s), (0.0, 45.0));
        assert_eq!(p[1].speaker.as_deref(), Some("Speaker 1"));
        assert_eq!(p[2].speaker.as_deref(), Some("Speaker 2"));
        // "agreed" (20 s) ended the 70–90 passage; then speaker 1's 5 s question keeps speaker 2's answer.
        assert_eq!(p[3].speaker.as_deref(), Some("Speaker 1, Speaker 2"));
        assert_eq!(p[3].text, "Speaker 1: ok? Speaker 2: yes, by Thursday");
    }

    #[test]
    fn arabic_folding_and_stems_match() {
        assert_eq!(fold("إِدارة الْمَخاطِر"), "اداره المخاطر");
        assert_eq!(index_text("والفريق المركزي"), "فريق مركزي");
        assert_eq!(index_text("فريق"), "فريق");
        assert_eq!(fold("٢٠٢٤"), "2024");
        assert_eq!(fts_query("ما هو حد الائتمان of project alpha?", &[]).unwrap(), "\"حد\" OR \"ايتمان\"* OR \"project\"* OR \"alpha\"*");
        assert!(fts_query("what was it?", &[]).is_none());
    }

    #[test]
    fn timestamps() {
        assert_eq!(parse_ts("[12:34]"), Some(754.0));
        assert_eq!(parse_ts("1:02:03"), Some(3723.0));
        assert_eq!(parse_ts("soon"), None);
        assert_eq!(fmt_ts(3723.0), "1:02:03");
    }
}
