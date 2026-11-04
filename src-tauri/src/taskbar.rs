//! Mahdar's state on the Windows taskbar button, so it shows while another window is in front:
//! - a red dot while recording, an amber pause badge while paused;
//! - a progress bar while a job runs: filling as a transcription goes, moving while writing
//!   or preparing;
//! - the window title says the same ("● REC 12:34 · Mahdar"), for the taskbar label and Alt+Tab;
//! - the button flashes when a transcription ends while Mahdar isn't the active window.
//!
//! A recording in stealth shows none of it: no badge, no REC in the title.

use std::time::Duration;

use tauri::window::{ProgressBarState, ProgressBarStatus};
use tauri::{AppHandle, Manager};

use crate::AppState;
use mahdar_core::store::fmt_ts;

const TITLE: &str = "Mahdar";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Badge {
    None,
    Recording,
    Paused,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Bar {
    None,
    /// Percent done, 1..=100.
    Percent(u64),
    /// Working, progress unknown.
    Busy,
}

#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Shown {
    pub badge: Badge,
    pub bar: Bar,
    pub title: String,
}

/// A job as the progress map has it: (stage, done, total).
type Job<'a> = (&'a str, usize, usize);

/// What the taskbar should show. `recording`: (seconds, paused) when the microphone is on;
/// `stealth`: the recording isn't shown.
pub(crate) fn decide(recording: Option<(f64, bool)>, stealth: bool, jobs: &[Job]) -> Shown {
    let recording = recording.filter(|_| !stealth);
    // Failed jobs linger in the map for a while; they are not work in progress.
    let jobs: Vec<&Job> = jobs.iter().filter(|(stage, _, _)| !stage.starts_with("failed")).collect();
    let percent = jobs
        .iter()
        .filter(|(stage, _, total)| *total > 0 && matches!(*stage, "transcribing" | "retrying"))
        .map(|(_, done, total)| ((100 * done / total) as u64).clamp(1, 100))
        .min();
    let bar = match (percent, jobs.is_empty()) {
        (Some(p), _) => Bar::Percent(p),
        (None, false) => Bar::Busy,
        (None, true) => Bar::None,
    };
    let work = match (percent, jobs.first()) {
        (Some(p), _) => format!("Transcribing {p}%"),
        (None, Some((stage, _, _))) => match *stage {
            "writing" | "starting local model" => "Writing…".to_string(),
            "joining" => "Joining recordings…".to_string(),
            "saving audio" => "Saving audio…".to_string(),
            _ => "Transcribing…".to_string(),
        },
        (None, None) => String::new(),
    };
    let (badge, rec) = match recording {
        Some((s, false)) => (Badge::Recording, format!("● REC {}", fmt_ts(s))),
        Some((s, true)) => (Badge::Paused, format!("❚❚ PAUSED {}", fmt_ts(s))),
        None => (Badge::None, String::new()),
    };
    let title = [rec, work, TITLE.to_string()].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" · ");
    Shown { badge, bar, title }
}

/// Keep the taskbar up to date, once a second.
pub(crate) fn spawn(app: AppHandle) {
    std::thread::spawn(move || {
        let mut last: Option<Shown> = None;
        loop {
            std::thread::sleep(Duration::from_secs(1));
            let state = app.state::<AppState>();
            let recording = state.recorder.lock().unwrap().as_ref().map(|(r, _)| (r.seconds(), r.is_paused()));
            let stealth = state.stealth.load(std::sync::atomic::Ordering::Relaxed);
            let shown = {
                let map = state.progress.lock().unwrap();
                let jobs: Vec<Job> = map.values().map(|p| (p.stage.as_str(), p.done, p.total)).collect();
                decide(recording, stealth, &jobs)
            };
            if last.as_ref() != Some(&shown) {
                apply(&app, &shown, last.as_ref());
                last = Some(shown);
            }
        }
    });
}

fn apply(app: &AppHandle, now: &Shown, before: Option<&Shown>) {
    let Some(w) = app.get_webview_window("main") else { return };
    if before.map(|b| &b.title) != Some(&now.title) {
        let _ = w.set_title(&now.title);
    }
    if before.map(|b| b.bar) != Some(now.bar) {
        let (status, progress) = match now.bar {
            Bar::None => (ProgressBarStatus::None, None),
            Bar::Percent(p) => (ProgressBarStatus::Normal, Some(p)),
            Bar::Busy => (ProgressBarStatus::Indeterminate, None),
        };
        let _ = w.set_progress_bar(ProgressBarState { status: Some(status), progress });
    }
    #[cfg(windows)]
    if before.map(|b| b.badge) != Some(now.badge) {
        let icon = match now.badge {
            Badge::None => None,
            Badge::Recording => Some(badge_icon([229, 50, 45], false)),
            Badge::Paused => Some(badge_icon([240, 160, 0], true)),
        };
        let _ = w.set_overlay_icon(icon);
    }
}

/// Flash the taskbar button (until Mahdar is brought to the front) when it isn't the active window.
pub(crate) fn flash(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        if !w.is_focused().unwrap_or(true) {
            let _ = w.request_user_attention(Some(tauri::UserAttentionType::Informational));
        }
    }
}

/// A 32×32 round badge in `rgb` with a white rim, and two white bars for "paused".
#[cfg_attr(not(windows), allow(dead_code))]
fn badge_icon(rgb: [u8; 3], bars: bool) -> tauri::image::Image<'static> {
    const N: usize = 32;
    let mut px = vec![0u8; N * N * 4];
    let c = (N as f32 - 1.0) / 2.0;
    for y in 0..N {
        for x in 0..N {
            let d = ((x as f32 - c).powi(2) + (y as f32 - c).powi(2)).sqrt();
            // Coverage of the disc (radius 15.5) and of the colored inside (radius 13), antialiased.
            let outer = (15.5 - d + 0.5).clamp(0.0, 1.0);
            let inner = (13.0 - d + 0.5).clamp(0.0, 1.0);
            let bar = bars && (9..=22).contains(&y) && ((10..=13).contains(&x) || (18..=21).contains(&x));
            let (r, g, b) = if bar {
                (255.0, 255.0, 255.0)
            } else {
                let mix = |col: u8| col as f32 * inner + 255.0 * (1.0 - inner);
                (mix(rgb[0]), mix(rgb[1]), mix(rgb[2]))
            };
            let i = (y * N + x) * 4;
            px[i..i + 4].copy_from_slice(&[r as u8, g as u8, b as u8, (outer * 255.0) as u8]);
        }
    }
    tauri::image::Image::new_owned(px, N as u32, N as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shows_recording_and_work() {
        let idle = decide(None, false, &[]);
        assert_eq!(idle, Shown { badge: Badge::None, bar: Bar::None, title: "Mahdar".into() });

        let rec = decide(Some((754.0, false)), false, &[]);
        assert_eq!((rec.badge, rec.bar, rec.title.as_str()), (Badge::Recording, Bar::None, "● REC 12:34 · Mahdar"));

        // Paused while another meeting transcribes: badge for the recording, bar for the job.
        let both = decide(Some((60.0, true)), false, &[("transcribing", 21, 50)]);
        assert_eq!((both.badge, both.bar, both.title.as_str()), (Badge::Paused, Bar::Percent(42), "❚❚ PAUSED 01:00 · Transcribing 42% · Mahdar"));

        let prep = decide(None, false, &[("loading model", 0, 0)]);
        assert_eq!((prep.bar, prep.title.as_str()), (Bar::Busy, "Transcribing… · Mahdar"));
        assert_eq!(decide(None, false, &[("writing", 0, 0)]).title, "Writing… · Mahdar");
        assert_eq!(decide(None, false, &[("joining", 1, 3)]).bar, Bar::Busy);
        // Two transcriptions: the one furthest behind; just started still shows on the bar.
        assert_eq!(decide(None, false, &[("transcribing", 9, 10), ("transcribing", 0, 40)]).bar, Bar::Percent(1));
        // A failed job hanging around is not work.
        assert_eq!(decide(None, false, &[("failed: no key", 0, 0)]), idle);

        // Stealth: the recording leaves no trace; other work still shows.
        assert_eq!(decide(Some((754.0, false)), true, &[]), idle);
        assert_eq!(decide(Some((60.0, true)), true, &[]), idle);
        assert_eq!(decide(Some((60.0, false)), true, &[("transcribing", 21, 50)]).title, "Transcribing 42% · Mahdar");
    }

    #[test]
    fn badges_are_round() {
        let img = badge_icon([229, 50, 45], true);
        let px = img.rgba();
        let at = |x: usize, y: usize| &px[(y * 32 + x) * 4..(y * 32 + x) * 4 + 4];
        assert_eq!(at(0, 0)[3], 0); // corner: transparent
        assert_eq!(at(16, 4), &[229, 50, 45, 255]); // inside: the color
        assert_eq!(at(11, 15), &[255, 255, 255, 255]); // a pause bar
    }
}
