//! Joining several recordings into one meeting before it is transcribed.

use chrono::{DateTime, Local};

use crate::store::{fmt_ts, Gap, GapKind};

/// One recording going into a join, in order.
#[derive(Clone, Debug, Default)]
pub struct Part {
    pub title: String,
    /// When it started recording, if known (files imported from elsewhere: unknown).
    pub created: Option<DateTime<Local>>,
    /// Length of its audio.
    pub duration_s: f64,
    /// Its own gaps and pauses, at positions within the part.
    pub gaps: Vec<Gap>,
    pub notes: String,
}

/// Gaps and notes for the joined meeting: each part's gaps moved to where the part starts, a
/// "next part" marker at each seam (with the real time between the parts, when known), and the
/// parts' notes one after another with their `[mm:ss]` stamps moved the same way.
pub fn merge(parts: &[Part]) -> (Vec<Gap>, String) {
    let mut gaps = Vec::new();
    let mut notes = Vec::new();
    let mut start = 0.0;
    for (i, p) in parts.iter().enumerate() {
        if i > 0 {
            let prev = &parts[i - 1];
            // The previous part ended after its audio plus the time its own gaps and pauses took.
            let between = match (prev.created, p.created) {
                (Some(a), Some(b)) => {
                    let ended = prev.duration_s + prev.gaps.iter().map(|g| g.missing_s).sum::<f64>();
                    ((b - a).num_milliseconds() as f64 / 1000.0 - ended).max(0.0)
                }
                _ => 0.0,
            };
            gaps.push(Gap { at_s: start, missing_s: between.round(), kind: GapKind::Joined });
        }
        gaps.extend(p.gaps.iter().map(|g| Gap { at_s: g.at_s + start, ..g.clone() }));
        let text = p.notes.trim();
        if !text.is_empty() {
            let shifted = text.lines().map(|l| shift_stamp(l, start)).collect::<Vec<_>>().join("\n");
            notes.push(if i == 0 { shifted } else { format!("── Part {} · {} · from [{}] ──\n{shifted}", i + 1, p.title, fmt_ts(start)) });
        }
        start += p.duration_s;
    }
    (gaps, notes.join("\n\n"))
}

/// Move a leading `[mm:ss]` / `[h:mm:ss]` stamp (as typed while recording) by `by_s` seconds.
fn shift_stamp(line: &str, by_s: f64) -> String {
    let body = line.trim_start();
    let indent = &line[..line.len() - body.len()];
    let Some(close) = body.strip_prefix('[').and_then(|r| r.find(']')) else { return line.to_string() };
    let inner = &body[1..=close];
    let nums: Option<Vec<u64>> = inner.split(':').map(|n| n.parse().ok()).collect();
    let secs = match nums.as_deref() {
        Some([m, s]) if *s < 60 => m * 60 + s,
        Some([h, m, s]) if *m < 60 && *s < 60 => h * 3600 + m * 60 + s,
        _ => return line.to_string(),
    };
    format!("{indent}[{}]{}", fmt_ts(secs as f64 + by_s), &body[close + 2..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn parts_join_with_markers_and_shifted_notes() {
        let at = |h, m| Some(Local.with_ymd_and_hms(2026, 11, 2, h, m, 0).unwrap());
        let parts = vec![
            Part {
                title: "Board".into(),
                created: at(10, 0),
                duration_s: 1500.0, // 25 min of audio + a 5 min pause: ended 10:30
                gaps: vec![Gap { at_s: 600.0, missing_s: 300.0, kind: GapKind::Paused }],
                notes: "[00:10] budget\n    ↳ 40k".into(),
            },
            Part { title: "Board, after lunch".into(), created: at(11, 15), duration_s: 600.0, gaps: vec![], notes: "".into() },
            Part {
                title: "Recording 3".into(),
                created: None,
                duration_s: 60.0,
                gaps: vec![Gap::lost(30.0, 4.0)],
                notes: "[05:00] vote\n@2 who seconded?\n[1:02:03] late".into(),
            },
        ];
        let (gaps, notes) = merge(&parts);
        assert_eq!(
            gaps,
            vec![
                Gap { at_s: 600.0, missing_s: 300.0, kind: GapKind::Paused },
                Gap { at_s: 1500.0, missing_s: 2700.0, kind: GapKind::Joined }, // 10:30 → 11:15
                Gap { at_s: 2100.0, missing_s: 0.0, kind: GapKind::Joined },    // time unknown
                Gap::lost(2130.0, 4.0),
            ]
        );
        assert_eq!(
            notes,
            "[00:10] budget\n    ↳ 40k\n\n── Part 3 · Recording 3 · from [35:00] ──\n[40:00] vote\n@2 who seconded?\n[1:37:03] late"
        );
    }

    #[test]
    fn only_real_stamps_move() {
        assert_eq!(shift_stamp("  [01:05] hi", 60.0), "  [02:05] hi");
        assert_eq!(shift_stamp("[note] hi", 60.0), "[note] hi");
        assert_eq!(shift_stamp("[01:75] hi", 60.0), "[01:75] hi");
        assert_eq!(shift_stamp("no stamp [01:00]", 60.0), "no stamp [01:00]");
    }
}
