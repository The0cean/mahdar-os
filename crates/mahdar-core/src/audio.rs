//! Audio decoding (via ffmpeg), quiet-point chunking, and microphone recording.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Context, Result};
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

/// Microphone recorder. Runs the cpal stream on its own thread (cpal streams
/// are not Send on Windows) and resamples to 16 kHz mono when stopped.
pub struct Recorder {
    stop_tx: mpsc::Sender<()>,
    thread: std::thread::JoinHandle<Result<(Vec<f32>, u32)>>,
    level: Arc<Mutex<f32>>,
}

impl Recorder {
    pub fn start() -> Result<Self> {
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<()>>();
        let level = Arc::new(Mutex::new(0f32));
        let level_cb = level.clone();

        let thread = std::thread::spawn(move || -> Result<(Vec<f32>, u32)> {
            let setup = || -> Result<(cpal::Stream, Arc<Mutex<Vec<f32>>>, u32)> {
                let device = cpal::default_host()
                    .default_input_device()
                    .ok_or_else(|| anyhow!("no microphone found"))?;
                let config = device.default_input_config()?;
                let rate = config.sample_rate().0;
                let channels = config.channels() as usize;
                let buf = Arc::new(Mutex::new(Vec::<f32>::new()));
                let buf_cb = buf.clone();
                let err_fn = |e| eprintln!("mic stream error: {e}");
                let push = move |mono: &mut dyn Iterator<Item = f32>| {
                    let mut b = buf_cb.lock().unwrap();
                    let before = b.len();
                    b.extend(mono);
                    let new = &b[before..];
                    if !new.is_empty() {
                        let peak = new.iter().fold(0f32, |m, s| m.max(s.abs()));
                        *level_cb.lock().unwrap() = peak;
                    }
                };
                let stream = match config.sample_format() {
                    cpal::SampleFormat::F32 => device.build_input_stream(
                        &config.into(),
                        move |d: &[f32], _: &_| {
                            push(&mut d.chunks(channels).map(|f| f.iter().sum::<f32>() / channels as f32))
                        },
                        err_fn,
                        None,
                    )?,
                    cpal::SampleFormat::I16 => device.build_input_stream(
                        &config.into(),
                        move |d: &[i16], _: &_| {
                            push(&mut d.chunks(channels).map(|f| {
                                f.iter().map(|&s| s as f32 / i16::MAX as f32).sum::<f32>() / channels as f32
                            }))
                        },
                        err_fn,
                        None,
                    )?,
                    other => bail!("unsupported microphone sample format {other:?}"),
                };
                stream.play()?;
                Ok((stream, buf, rate))
            };
            match setup() {
                Ok((stream, buf, rate)) => {
                    let _ = ready_tx.send(Ok(()));
                    let _ = stop_rx.recv();
                    drop(stream);
                    let samples = std::mem::take(&mut *buf.lock().unwrap());
                    Ok((samples, rate))
                }
                Err(e) => {
                    let msg = e.to_string();
                    let _ = ready_tx.send(Err(e));
                    Err(anyhow!(msg))
                }
            }
        });

        ready_rx.recv().map_err(|_| anyhow!("recorder thread died"))??;
        Ok(Recorder { stop_tx, thread, level })
    }

    /// Latest peak level 0..1, for a live meter.
    pub fn level(&self) -> f32 {
        *self.level.lock().unwrap()
    }

    /// Stop and return 16 kHz mono PCM.
    pub fn stop(self) -> Result<Vec<f32>> {
        let _ = self.stop_tx.send(());
        let (samples, rate) = self.thread.join().map_err(|_| anyhow!("recorder thread panicked"))??;
        Ok(resample_linear(&samples, rate as usize, SAMPLE_RATE))
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
