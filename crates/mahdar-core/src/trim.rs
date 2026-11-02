//! Trimming a recording: cutting off its start or end, or parts in the middle. The transcript,
//! gaps and note stamps follow, so a trimmed meeting doesn't need transcribing again.

use serde::{Deserialize, Serialize};

use crate::store::{Chunk, Gap, GapKind, Meeting};

/// Parts to remove as (from_s, to_s): clamped to the audio, sorted, overlaps merged, and
/// anything shorter than a tenth of a second dropped.
pub fn normalize(cuts: &[(f64, f64)], duration_s: f64) -> Vec<(f64, f64)> {
    let mut v: Vec<(f64, f64)> = cuts
        .iter()
        .map(|&(a, b)| (a.min(b).clamp(0.0, duration_s), a.max(b).clamp(0.0, duration_s)))
        .filter(|(a, b)| b - a >= 0.1)
        .collect();
    v.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut out: Vec<(f64, f64)> = Vec::new();
    for (a, b) in v {
        match out.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => out.push((a, b)),
        }
    }
    out
}

/// Where time `t` of the original lands after the cuts. A time inside a cut lands where the
/// cut was.
pub fn map(cuts: &[(f64, f64)], t: f64) -> f64 {
    t - cuts.iter().map(|&(a, b)| (t.min(b) - a).max(0.0)).sum::<f64>()
}

fn inside(cuts: &[(f64, f64)], t: f64) -> bool {
    cuts.iter().any(|&(a, b)| t > a && t < b)
}

/// The audio without the cut parts.
pub fn pcm(pcm: &[f32], rate: usize, cuts: &[(f64, f64)]) -> Vec<f32> {
    let at = |s: f64| ((s * rate as f64).round() as usize).min(pcm.len());
    let mut out = Vec::with_capacity(pcm.len());
    let mut from = 0;
    for &(a, b) in cuts {
        out.extend_from_slice(&pcm[from.min(at(a))..at(a)]);
        from = at(b);
    }
    out.extend_from_slice(&pcm[from.min(pcm.len())..]);
    out
}

/// What a meeting had before its first trim, to put back with RESTORE ORIGINAL.
#[derive(Serialize, Deserialize)]
pub struct Backup {
    pub duration_s: f64,
    pub chunks: Vec<Chunk>,
    pub gaps: Vec<Gap>,
    pub notes: String,
    /// The notes right after the latest trim: if they're unchanged at restore time, the
    /// original notes come back too (otherwise the user's later edits are kept).
    pub notes_after: String,
}

impl Backup {
    pub fn of(m: &Meeting) -> Self {
        Backup { duration_s: m.duration_s, chunks: m.chunks.clone(), gaps: m.gaps.clone(), notes: m.notes.clone(), notes_after: String::new() }
    }

    pub fn restore(self, m: &mut Meeting) {
        m.duration_s = self.duration_s;
        m.chunks = self.chunks;
        m.gaps = self.gaps;
        if m.notes == self.notes_after {
            m.notes = self.notes;
        }
        m.trimmed_s = 0.0;
    }
}

/// Cut `cuts` (already normalized) out of the meeting's transcript, gaps and notes. Lines
/// mostly in a cut part are dropped; the rest move to their new times. A part cut from the
/// middle leaves a marker in the transcript.
pub fn apply(m: &mut Meeting, cuts: &[(f64, f64)]) {
    let duration = m.duration_s;
    let removed: f64 = cuts.iter().map(|(a, b)| b - a).sum();
    m.chunks = std::mem::take(&mut m.chunks)
        .into_iter()
        .filter(|c| !inside(cuts, (c.start_s + c.end_s) / 2.0))
        .map(|c| Chunk { start_s: map(cuts, c.start_s), end_s: map(cuts, c.end_s), ..c })
        .filter(|c| c.end_s - c.start_s >= 0.3)
        .collect();
    let mut gaps: Vec<Gap> = std::mem::take(&mut m.gaps)
        .into_iter()
        .filter(|g| !inside(cuts, g.at_s))
        .map(|g| Gap { at_s: map(cuts, g.at_s), ..g })
        .collect();
    gaps.extend(
        cuts.iter()
            .filter(|&&(a, b)| a > 0.0 && b < duration)
            .map(|&(a, b)| Gap { at_s: map(cuts, a), missing_s: b - a, kind: GapKind::Cut }),
    );
    gaps.sort_by(|x, y| x.at_s.total_cmp(&y.at_s));
    m.gaps = gaps;
    m.notes = m.notes.lines().map(|l| crate::join::restamp(l, |t| map(cuts, t))).collect::<Vec<_>>().join("\n");
    m.duration_s = (duration - removed).max(0.0);
    m.trimmed_s += removed;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meeting() -> Meeting {
        serde_json::from_str(
            r#"{"id":"x","title":"T","created":"2026-11-02T10:00:00+03:00","source":"mic","duration_s":600,
            "language":"ar","status":"ready","notes":"[00:30] small talk\n[03:20] price 40k\n  [09:50] bye\nplain line",
            "gaps":[{"at_s":100,"missing_s":20},{"at_s":250,"missing_s":30,"kind":"paused"}],
            "chunks":[{"start_s":0,"end_s":60,"text":"hello"},{"start_s":60,"end_s":100,"text":"so"},
            {"start_s":100,"end_s":200,"text":"price"},{"start_s":200,"end_s":260,"text":"off record"},
            {"start_s":260,"end_s":500,"text":"plan"},{"start_s":500,"end_s":600,"text":"bye"}]}"#,
        )
        .unwrap()
    }

    #[test]
    fn cuts_are_tidied() {
        assert_eq!(normalize(&[(500.0, 700.0), (90.0, 70.0), (80.0, 120.0), (5.0, 5.05)], 600.0), vec![(70.0, 120.0), (500.0, 600.0)]);
        let cuts = [(0.0, 60.0), (200.0, 260.0)];
        assert_eq!(map(&cuts, 30.0), 0.0);
        assert_eq!(map(&cuts, 100.0), 40.0);
        assert_eq!(map(&cuts, 230.0), 140.0);
        assert_eq!(map(&cuts, 300.0), 180.0);
    }

    #[test]
    fn trimming_moves_the_transcript_and_notes() {
        let mut m = meeting();
        let cuts = normalize(&[(0.0, 60.0), (200.0, 260.0), (540.0, 600.0)], 600.0);
        apply(&mut m, &cuts);
        let lines: Vec<(f64, f64, &str)> = m.chunks.iter().map(|c| (c.start_s, c.end_s, c.text.as_str())).collect();
        // "hello" and "off record" were in cut parts; "bye" (500–600) mostly in the cut tail.
        assert_eq!(lines, vec![(0.0, 40.0, "so"), (40.0, 140.0, "price"), (140.0, 380.0, "plan")]);
        assert_eq!(m.duration_s, 420.0);
        assert_eq!(m.trimmed_s, 180.0);
        // The pause at 250 s was in the cut part; the cut itself leaves a marker.
        assert_eq!(m.gaps, vec![Gap::lost(40.0, 20.0), Gap { at_s: 140.0, missing_s: 60.0, kind: GapKind::Cut }]);
        assert_eq!(
            m.transcript_between(130.0, 150.0),
            "[00:40] (recording gap: 00:20 not recorded)\n[00:40] price\n[02:20] (part cut out: 01:00)\n[02:20] plan"
        );
        assert_eq!(m.notes, "[00:00] small talk\n[02:20] price 40k\n  [07:00] bye\nplain line");
    }

    #[test]
    fn audio_loses_the_cut_samples() {
        let pcm: Vec<f32> = (0..10).map(|i| i as f32).collect();
        // 1 sample per second: cut 0..2 and 5..7.
        assert_eq!(super::pcm(&pcm, 1, &[(0.0, 2.0), (5.0, 7.0)]), vec![2.0, 3.0, 4.0, 7.0, 8.0, 9.0]);
        assert_eq!(super::pcm(&pcm, 1, &[(8.0, 10.0)]), (0..8).map(|i| i as f32).collect::<Vec<_>>());
    }

    #[test]
    fn restore_puts_everything_back() {
        let mut m = meeting();
        let mut backup = Backup::of(&m);
        apply(&mut m, &[(0.0, 60.0)]);
        backup.notes_after = m.notes.clone();
        backup.restore(&mut m);
        assert_eq!((m.duration_s, m.chunks.len(), m.trimmed_s), (600.0, 6, 0.0));
        assert!(m.notes.starts_with("[00:30] small talk"));
    }
}
