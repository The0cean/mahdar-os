//! Audio decoding (via ffmpeg), quiet-point chunking, and microphone recording.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Context, Result};

pub use crate::store::Gap;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

pub const SAMPLE_RATE: usize = 16_000;

/// Chunk bounds for the ASR model. Short chunks avoid decode loops
/// (measured on Cohere Arabic: 6-minute chunks looped, 15-30 s chunks rarely do).
pub const MIN_CHUNK_S: usize = 15;
pub const MAX_CHUNK_S: usize = 30;

#[cfg(windows)]
fn hide_window(cmd: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    cmd.creation_flags(CREATE_NO_WINDOW);
}
#[cfg(not(windows))]
fn hide_window(_: &mut Command) {}

/// Decode any audio/video file ffmpeg understands to 16 kHz mono f32 PCM.
pub fn decode_to_pcm(path: &Path) -> Result<Vec<f32>> {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-v", "error", "-nostdin", "-i"])
        .arg(path)
        .args(["-f", "f32le", "-ac", "1", "-ar", &SAMPLE_RATE.to_string(), "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    hide_window(&mut cmd);
    let mut child = cmd
        .spawn()
        .context("ffmpeg not found. Install it (winget install Gyan.FFmpeg) and restart Mahdar")?;
    let mut bytes = Vec::new();
    child.stdout.take().unwrap().read_to_end(&mut bytes)?;
    let mut err = String::new();
    child.stderr.take().unwrap().read_to_string(&mut err)?;
    if !child.wait()?.success() {
        bail!("ffmpeg could not decode {}: {}", path.display(), err.trim());
    }
    Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

/// Sample indices where to cut: the quietest 0.5 s inside each MIN..MAX window.
pub fn split_points(pcm: &[f32]) -> Vec<usize> {
    let frame = SAMPLE_RATE / 20; // 50 ms
    let n_frames = pcm.len() / frame;
    let rms: Vec<f32> = (0..n_frames)
        .map(|i| {
            let f = &pcm[i * frame..(i + 1) * frame];
            (f.iter().map(|s| s * s).sum::<f32>() / frame as f32).sqrt()
        })
        .collect();
    // 0.5 s moving average
    let win = 10usize;
    let mut smooth = vec![0f32; n_frames];
    let mut acc = 0f32;
    for i in 0..n_frames {
        acc += rms[i];
        if i >= win {
            acc -= rms[i - win];
        }
        smooth[i.saturating_sub(win / 2)] = acc / win.min(i + 1) as f32;
    }

    let mut cuts = vec![0usize];
    let mut start = 0usize;
    while (pcm.len() - start) / SAMPLE_RATE > MAX_CHUNK_S {
        let lo = (start + MIN_CHUNK_S * SAMPLE_RATE) / frame;
        let hi = ((start + MAX_CHUNK_S * SAMPLE_RATE) / frame).min(n_frames);
        let best = (lo..hi)
            .min_by(|&a, &b| smooth[a].partial_cmp(&smooth[b]).unwrap())
            .unwrap_or(lo);
        start = best * frame;
        cuts.push(start);
    }
    cuts.push(pcm.len());
    cuts
}

/// Sample index of the quietest 0.5 s window whose centre lies in `lo..hi`.
pub fn quiet_point(pcm: &[f32], lo: usize, hi: usize) -> usize {
    let frame = SAMPLE_RATE / 20; // 50 ms
    let win = SAMPLE_RATE / 2;
    let energy = |c: usize| -> f32 {
        let a = c.saturating_sub(win / 2);
        let b = (c + win / 2).min(pcm.len());
        pcm[a..b].iter().map(|s| s * s).sum::<f32>()
    };
    let mut best = (lo + hi) / 2;
    let mut best_e = f32::MAX;
    let mut c = lo;
    while c < hi.min(pcm.len()) {
        let e = energy(c);
        if e < best_e {
            best_e = e;
            best = c;
        }
        c += frame;
    }
    best
}

/// Read a WAV written by `write_wav` back into f32 PCM.
pub fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let mut r = hound::WavReader::open(path).with_context(|| format!("could not open {}", path.display()))?;
    let spec = r.spec();
    if spec.sample_rate as usize != SAMPLE_RATE || spec.channels != 1 || spec.bits_per_sample != 16 {
        bail!("unexpected WAV format in {}", path.display());
    }
    Ok(r.samples::<i16>().map(|s| s.map(|v| v as f32 / i16::MAX as f32)).collect::<Result<_, _>>()?)
}

/// Compressed copy of a meeting's WAV for phones (AAC 32 kbps mono, ~14 MB/hour),
/// next to it as `audio-phone.m4a`. Re-made only when the WAV is newer.
pub fn phone_copy(wav: &Path) -> Result<std::path::PathBuf> {
    let m4a = wav.with_file_name("audio-phone.m4a");
    let fresh = match (std::fs::metadata(wav).and_then(|m| m.modified()), std::fs::metadata(&m4a).and_then(|m| m.modified())) {
        (Ok(w), Ok(m)) => m >= w,
        _ => false,
    };
    if fresh {
        return Ok(m4a);
    }
    let tmp = m4a.with_extension("part.m4a");
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-v", "error", "-y", "-i"])
        .arg(wav)
        .args(["-ac", "1", "-c:a", "aac", "-b:a", "32k", "-movflags", "+faststart"])
        .arg(&tmp)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    hide_window(&mut cmd);
    if !cmd.status().context("ffmpeg not found")?.success() {
        let _ = std::fs::remove_file(&tmp);
        bail!("ffmpeg could not compress {}", wav.display());
    }
    std::fs::rename(&tmp, &m4a)?;
    Ok(m4a)
}

/// Write 16 kHz mono PCM to a 16-bit WAV.
pub fn write_wav(path: &Path, pcm: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for &s in pcm {
        w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    w.finalize()?;
    Ok(())
}

/// The recording as it is written: mono 16-bit WAV at the microphone's own rate. The header is
/// rewritten every few seconds, so after a crash or power loss the file is valid up to that point.
struct Spool {
    w: hound::WavWriter<std::io::BufWriter<std::fs::File>>,
    unflushed: usize,
    rate: usize,
}

/// Seconds of audio between header rewrites: at most this much is lost if the laptop dies.
const SPOOL_CHECKPOINT_S: usize = 3;

impl Spool {
    fn create(path: &Path, rate: u32) -> Result<Self> {
        let spec = hound::WavSpec { channels: 1, sample_rate: rate, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let w = hound::WavWriter::create(path, spec).with_context(|| format!("could not create {}", path.display()))?;
        Ok(Spool { w, unflushed: 0, rate: rate as usize })
    }

    fn push(&mut self, samples: &[f32]) -> Result<()> {
        for &s in samples {
            self.w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
        }
        self.unflushed += samples.len();
        if self.unflushed >= self.rate * SPOOL_CHECKPOINT_S {
            self.w.flush()?;
            self.unflushed = 0;
        }
        Ok(())
    }

    fn finish(self) -> Result<()> {
        self.w.finalize()?;
        Ok(())
    }
}

/// Read a recording spool (complete, or cut short by a crash) as 16 kHz mono PCM.
pub fn read_spool(path: &Path) -> Result<Vec<f32>> {
    let mut r = hound::WavReader::open(path).with_context(|| format!("could not open {}", path.display()))?;
    let rate = r.spec().sample_rate as usize;
    // A crash can leave a partly written last sample: keep everything before it.
    let pcm: Vec<f32> = r.samples::<i16>().map_while(|s| s.ok()).map(|v| v as f32 / i16::MAX as f32).collect();
    Ok(resample_linear(&pcm, rate, SAMPLE_RATE))
}

/// Seconds of audio in a recording spool, without reading it.
pub fn spool_seconds(path: &Path) -> f64 {
    hound::WavReader::open(path).map(|r| r.duration() as f64 / r.spec().sample_rate.max(1) as f64).unwrap_or(0.0)
}

/// A pause in capture shorter than this is jitter, not a gap.
const GAP_MIN_S: f64 = 2.0;
/// Reopen the microphone after this long without audio (or at once when it reports an error).
const REOPEN_AFTER_S: f64 = 3.0;

/// Finds gaps in a recording: stretches of wall-clock time with no audio from the microphone,
/// e.g. while the laptop slept or after the microphone was lost.
#[derive(Default)]
struct GapTracker {
    /// (position in the recording where audio stopped, seconds missing so far)
    stall: Option<(f64, f64)>,
}

#[derive(Debug, PartialEq)]
enum Tick {
    Ok,
    /// Audio flowing again after a real gap.
    Gap(Gap),
    /// No audio for a while: the microphone should be reopened.
    Stalled,
}

impl GapTracker {
    /// One check: `wall_s` of real time passed and `got_s` of audio arrived; the recording is
    /// now `position_s` long. `failed`: the microphone reported an error.
    fn tick(&mut self, wall_s: f64, got_s: f64, position_s: f64, failed: bool) -> Tick {
        if !failed && got_s >= wall_s * 0.5 {
            return match self.stall.take() {
                Some((at_s, missing_s)) if missing_s >= GAP_MIN_S => Tick::Gap(Gap { at_s, missing_s }),
                _ => Tick::Ok,
            };
        }
        let stall = self.stall.get_or_insert((position_s, 0.0));
        stall.1 += (wall_s - got_s).max(0.0);
        if failed || stall.1 >= REOPEN_AFTER_S {
            Tick::Stalled
        } else {
            Tick::Ok
        }
    }
}

/// Keeps Windows from sleeping on its own while held (closing the lid still follows the power settings).
struct StayAwake;

impl StayAwake {
    fn new() -> Self {
        #[cfg(windows)]
        unsafe {
            use windows_sys::Win32::System::Power::{SetThreadExecutionState, ES_CONTINUOUS, ES_SYSTEM_REQUIRED};
            SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED);
        }
        StayAwake
    }
}

impl Drop for StayAwake {
    fn drop(&mut self) {
        #[cfg(windows)]
        unsafe {
            use windows_sys::Win32::System::Power::{SetThreadExecutionState, ES_CONTINUOUS};
            SetThreadExecutionState(ES_CONTINUOUS);
        }
    }
}

/// What the microphone callbacks share with the recorder thread.
#[derive(Clone)]
struct Capture {
    buf: Arc<Mutex<Vec<f32>>>,
    level: Arc<Mutex<f32>>,
    failed: Arc<std::sync::atomic::AtomicBool>,
}

/// Open the default microphone, feeding mono samples into `cap.buf`. Returns the stream and its rate.
fn open_mic(cap: &Capture) -> Result<(cpal::Stream, u32)> {
    let device = cpal::default_host()
        .default_input_device()
        .ok_or_else(|| anyhow!("no microphone found"))?;
    let config = device.default_input_config()?;
    let rate = config.sample_rate().0;
    let channels = config.channels() as usize;
    let (buf, level, failed) = (cap.buf.clone(), cap.level.clone(), cap.failed.clone());
    let push = move |mono: &mut dyn Iterator<Item = f32>| {
        let mut b = buf.lock().unwrap();
        let before = b.len();
        b.extend(mono);
        let new = &b[before..];
        if !new.is_empty() {
            *level.lock().unwrap() = new.iter().fold(0f32, |m, s| m.max(s.abs()));
        }
    };
    let err_fn = move |e| {
        crate::log::warn("record", format!("microphone error: {e}"));
        failed.store(true, std::sync::atomic::Ordering::Relaxed);
    };
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |d: &[f32], _: &_| push(&mut d.chunks(channels).map(|f| f.iter().sum::<f32>() / channels as f32)),
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config.into(),
            move |d: &[i16], _: &_| {
                push(&mut d.chunks(channels).map(|f| f.iter().map(|&s| s as f32 / i16::MAX as f32).sum::<f32>() / channels as f32))
            },
            err_fn,
            None,
        )?,
        other => bail!("unsupported microphone sample format {other:?}"),
    };
    stream.play()?;
    Ok((stream, rate))
}

/// Microphone recorder. Runs the cpal stream on its own thread (cpal streams are not Send on
/// Windows) and writes the audio to a spool file as it arrives. While recording it keeps Windows
/// awake, reopens the microphone if it stops delivering audio (e.g. after sleep), and reports
/// the gaps that leaves.
pub struct Recorder {
    stop_tx: mpsc::Sender<()>,
    thread: std::thread::JoinHandle<Result<()>>,
    level: Arc<Mutex<f32>>,
    /// Seconds of audio in the spool, as f64 bits.
    seconds: Arc<std::sync::atomic::AtomicU64>,
    gaps: Arc<Mutex<Vec<Gap>>>,
}

impl Recorder {
    /// Start recording into `spool` (see [`read_spool`]). `on_gap` is called from the recorder
    /// thread when audio resumes after a gap.
    pub fn start(spool: &Path, on_gap: impl Fn(&Gap) + Send + 'static) -> Result<Self> {
        use std::sync::atomic::Ordering::Relaxed;
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<()>>();
        let cap = Capture { buf: Default::default(), level: Default::default(), failed: Default::default() };
        let level = cap.level.clone();
        let seconds = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let seconds_w = seconds.clone();
        let gaps = Arc::new(Mutex::new(Vec::new()));
        let gaps_w = gaps.clone();
        let spool = spool.to_path_buf();

        let thread = std::thread::spawn(move || -> Result<()> {
            let opened = open_mic(&cap).and_then(|(stream, rate)| Ok((stream, rate, Spool::create(&spool, rate)?)));
            let (stream, mut mic_rate, mut file) = match opened {
                Ok(x) => {
                    let _ = ready_tx.send(Ok(()));
                    x
                }
                Err(e) => {
                    let msg = e.to_string();
                    let _ = ready_tx.send(Err(e));
                    return Err(anyhow!(msg));
                }
            };
            let _awake = StayAwake::new();
            let mut stream = Some(stream);
            let mut written = 0usize;
            let mut tracker = GapTracker::default();
            // Wall-clock time, so time spent asleep counts.
            let mut last = std::time::SystemTime::now();
            let mut next_reopen = 0.0f64;
            // Move the captured audio to disk twice a second until told to stop.
            loop {
                let stop = !matches!(
                    stop_rx.recv_timeout(std::time::Duration::from_millis(500)),
                    Err(mpsc::RecvTimeoutError::Timeout)
                );
                if stop {
                    drop(stream.take()); // ends the callbacks, so the buffer is final
                }
                let samples = std::mem::take(&mut *cap.buf.lock().unwrap());
                let got_s = samples.len() as f64 / mic_rate as f64;
                let samples =
                    if mic_rate as usize == file.rate { samples } else { resample_linear(&samples, mic_rate as usize, file.rate) };
                file.push(&samples)?;
                written += samples.len();
                let position_s = written as f64 / file.rate as f64;
                seconds_w.store(position_s.to_bits(), Relaxed);
                if stop {
                    return file.finish();
                }

                let now = std::time::SystemTime::now();
                let wall_s = now.duration_since(last).map(|d| d.as_secs_f64()).unwrap_or(0.0);
                last = now;
                match tracker.tick(wall_s, got_s, position_s, cap.failed.swap(false, Relaxed)) {
                    Tick::Ok => next_reopen = 0.0,
                    Tick::Gap(gap) => {
                        crate::log::warn("record", format!("{} not recorded at {}", crate::store::fmt_ts(gap.missing_s), crate::store::fmt_ts(gap.at_s)));
                        on_gap(&gap);
                        gaps_w.lock().unwrap().push(gap);
                        next_reopen = 0.0;
                    }
                    Tick::Stalled => {
                        // Retry every 2 s until a microphone answers; audio resumes into the same file.
                        let stalled_s = tracker.stall.map(|s| s.1).unwrap_or(0.0);
                        if stalled_s >= next_reopen {
                            next_reopen = stalled_s + 2.0;
                            drop(stream.take());
                            cap.buf.lock().unwrap().clear();
                            match open_mic(&cap) {
                                Ok((s, rate)) => {
                                    crate::log::info("record", format!("microphone reopened after {stalled_s:.0} s without audio"));
                                    stream = Some(s);
                                    mic_rate = rate;
                                }
                                Err(e) => crate::log::warn("record", format!("microphone not back yet: {e:#}")),
                            }
                        }
                    }
                }
            }
        });

        ready_rx.recv().map_err(|_| anyhow!("recorder thread died"))??;
        Ok(Recorder { stop_tx, thread, level, seconds, gaps })
    }

    /// Latest peak level 0..1, for a live meter.
    pub fn level(&self) -> f32 {
        *self.level.lock().unwrap()
    }

    /// Length of the recording so far, in seconds of audio (gaps not included).
    pub fn seconds(&self) -> f64 {
        f64::from_bits(self.seconds.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// Gaps so far: where audio stopped, and for how long.
    pub fn gaps(&self) -> Vec<Gap> {
        self.gaps.lock().unwrap().clone()
    }

    /// Stop and close the spool file.
    pub fn stop(self) -> Result<()> {
        let _ = self.stop_tx.send(());
        self.thread.join().map_err(|_| anyhow!("recorder thread panicked"))?
    }
}

/// Simple linear resampler; adequate for speech going into ASR.
fn resample_linear(input: &[f32], from: usize, to: usize) -> Vec<f32> {
    if from == to || input.is_empty() {
        return input.to_vec();
    }
    let ratio = from as f64 / to as f64;
    let n_out = (input.len() as f64 / ratio) as usize;
    (0..n_out)
        .map(|i| {
            let pos = i as f64 * ratio;
            let idx = pos as usize;
            let frac = (pos - idx as f64) as f32;
            let a = input[idx];
            let b = *input.get(idx + 1).unwrap_or(&a);
            a + (b - a) * frac
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("mahdar-audio-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("recording.wav")
    }

    #[test]
    fn spool_survives_a_crash_up_to_the_last_checkpoint() {
        let path = tmp("crash");
        let mut s = Spool::create(&path, 48_000).unwrap();
        let second = vec![0.25f32; 48_000];
        for _ in 0..4 {
            s.push(&second).unwrap(); // checkpoint after 3 s
        }
        // Power cut: nothing after the last checkpoint reaches the disk.
        std::mem::forget(s);
        assert_eq!(spool_seconds(&path), 3.0);
        let pcm = read_spool(&path).unwrap();
        assert_eq!(pcm.len(), 3 * SAMPLE_RATE);
        assert!((pcm[100] - 0.25).abs() < 1e-3);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn gaps_are_found_after_sleep_and_mic_loss() {
        let mut t = GapTracker::default();
        // Normal capture, with jitter.
        assert_eq!(t.tick(0.5, 0.5, 10.0, false), Tick::Ok);
        assert_eq!(t.tick(0.5, 0.2, 10.2, false), Tick::Ok);
        assert_eq!(t.tick(0.5, 0.8, 11.0, false), Tick::Ok);
        // Laptop asleep for 3 minutes: one long tick with no audio, then audio again.
        assert_eq!(t.tick(180.0, 0.1, 11.1, false), Tick::Stalled);
        assert_eq!(t.tick(0.5, 0.5, 11.6, false), Tick::Gap(Gap { at_s: 11.1, missing_s: 179.9 }));
        // Microphone goes quiet: reopen after 3 s, gap reported once audio is back.
        for i in 0..5 {
            assert_eq!(t.tick(0.5, 0.0, 20.0, false), Tick::Ok, "tick {i}");
        }
        assert_eq!(t.tick(0.5, 0.0, 20.0, false), Tick::Stalled);
        assert_eq!(t.tick(0.5, 0.5, 20.5, false), Tick::Gap(Gap { at_s: 20.0, missing_s: 3.0 }));
        // An error reopens at once; a short hiccup is not a gap.
        assert_eq!(t.tick(0.5, 0.4, 30.0, true), Tick::Stalled);
        assert_eq!(t.tick(0.5, 0.5, 30.5, false), Tick::Ok);
    }

    #[test]
    fn finished_spool_has_everything() {
        let path = tmp("done");
        let mut s = Spool::create(&path, 44_100).unwrap();
        s.push(&vec![0.0f32; 44_100 / 2]).unwrap();
        s.finish().unwrap();
        assert_eq!(spool_seconds(&path), 0.5);
        assert_eq!(read_spool(&path).unwrap().len(), SAMPLE_RATE / 2);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }
}
