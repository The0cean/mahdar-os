//! Mahdar core: shared by the desktop app (and later the mobile app and MCP server).

pub mod asr;
pub mod audio;
pub mod export;
pub mod log;
pub mod notes;
pub mod secrets;
pub mod store;
pub mod summarize;

use std::path::Path;
use std::time::Instant;

use anyhow::Result;

pub use asr::{Backend, Engine};
pub use store::{Chunk, Meeting, Status, Store};

/// Default ASR model: Cohere Transcribe Arabic (Apache 2.0), Q8_0 GGUF for transcribe.cpp.
pub const MODEL_FILE: &str = "cohere-transcribe-arabic-07-2026-Q8_0.gguf";
pub const MODEL_URL: &str = "https://huggingface.co/handy-computer/cohere-transcribe-arabic-07-2026-gguf/resolve/main/cohere-transcribe-arabic-07-2026-Q8_0.gguf";
pub const MODEL_BYTES: u64 = 2_410_655_136;

/// Speaker diarizer: NVIDIA Streaming Sortformer 4spk v2.1 (NVIDIA Open Model License), F16 GGUF.
/// Up to 4 speakers. F16 rather than Q8_0: quantization can permute speaker labels.
pub const DIARIZER_FILE: &str = "diar_streaming_sortformer_4spk-v2.1-F16.gguf";
pub const DIARIZER_URL: &str = "https://huggingface.co/handy-computer/diar_streaming_sortformer_4spk-v2.1-gguf/resolve/main/diar_streaming_sortformer_4spk-v2.1-F16.gguf";

/// A span of audio to transcribe: sample range plus the speaker, if known.
pub type Span = (usize, usize, Option<u32>);

/// Spans cut at quiet points only (no speaker information).
pub fn plain_spans(pcm: &[f32]) -> Vec<Span> {
    audio::split_points(pcm).windows(2).map(|w| (w[0], w[1], None)).collect()
}

/// Spans that follow speaker turns: the timeline is partitioned at each turn
/// start (so no audio is dropped), same-speaker turns separated by short
/// pauses are merged, slivers fold into the previous span, and long spans
/// are split further at quiet points to stay within the ASR chunk limit.
pub fn speaker_spans(pcm: &[f32], turns: &[asr::SpeakerTurn]) -> Vec<Span> {
    let sr = audio::SAMPLE_RATE as f64;
    let total_s = pcm.len() as f64 / sr;
    let mut merged: Vec<asr::SpeakerTurn> = Vec::new();
    for t in turns {
        if let Some(last) = merged.last_mut() {
            if last.speaker == t.speaker && t.start_s - last.end_s < 1.5 {
                last.end_s = last.end_s.max(t.end_s);
                continue;
            }
        }
        merged.push(*t);
    }
    // (start_s, end_s, speaker), contiguous from 0 to the end of the audio.
    let mut parts: Vec<(f64, f64, u32)> = Vec::new();
    for (i, t) in merged.iter().enumerate() {
        let start = parts.last().map(|p| p.1).unwrap_or(0.0);
        let end = merged.get(i + 1).map(|n| n.start_s).unwrap_or(total_s).max(start);
        if end - start < 0.8 {
            if let Some(last) = parts.last_mut() {
                last.1 = end;
                continue;
            }
        }
        parts.push((start, end, t.speaker));
    }
    if parts.is_empty() {
        return plain_spans(pcm);
    }
    if let Some(last) = parts.last_mut() {
        last.1 = total_s;
    }
    let mut spans = Vec::new();
    for (start, end, speaker) in parts {
        let a = ((start * sr) as usize).min(pcm.len());
        let b = ((end * sr) as usize).min(pcm.len());
        if b <= a {
            continue;
        }
        for w in audio::split_points(&pcm[a..b]).windows(2) {
            spans.push((a + w[0], a + w[1], Some(speaker)));
        }
    }
    spans
}

/// Transcribe 16 kHz mono PCM span by span, reporting progress after each.
pub fn transcribe_spans(
    engine: &mut Engine,
    pcm: &[f32],
    spans: &[Span],
    language: &str,
    mut on_chunk: impl FnMut(usize, usize, &Chunk),
) -> Result<(Vec<Chunk>, f64)> {
    let sr = audio::SAMPLE_RATE as f64;
    let t0 = Instant::now();
    let mut chunks = Vec::with_capacity(spans.len());
    for (i, &(a, b, speaker)) in spans.iter().enumerate() {
        let r = engine.transcribe(&pcm[a..b], language)?;
        let chunk = Chunk { start_s: a as f64 / sr, end_s: b as f64 / sr, text: r.text, flag: r.flag, speaker, edited: false };
        on_chunk(i + 1, spans.len(), &chunk);
        chunks.push(chunk);
    }
    Ok((chunks, t0.elapsed().as_secs_f64()))
}

/// Transcribe 16 kHz mono PCM chunk by chunk (quiet-point chunks, no speakers).
pub fn transcribe_pcm(
    engine: &mut Engine,
    pcm: &[f32],
    language: &str,
    on_chunk: impl FnMut(usize, usize, &Chunk),
) -> Result<(Vec<Chunk>, f64)> {
    transcribe_spans(engine, pcm, &plain_spans(pcm), language, on_chunk)
}

/// Re-run chunks whose decode got stuck (or any chosen chunks) on smaller pieces.
///
/// The model loops less on shorter audio, so each chunk is split at quiet points
/// into 2, then 3, then 4 pieces until every piece decodes cleanly. If no split is
/// fully clean, the longest result is kept and the chunk stays flagged.
/// Returns how many chunks came out clean.
pub fn retry_chunks(
    engine: &mut Engine,
    pcm: &[f32],
    chunks: &mut [Chunk],
    indices: &[usize],
    language: &str,
    mut on_progress: impl FnMut(usize, usize),
) -> Result<usize> {
    let sr = audio::SAMPLE_RATE as f64;
    let mut fixed = 0;
    for (n, &i) in indices.iter().enumerate() {
        let Some(chunk) = chunks.get_mut(i) else { continue };
        let a = ((chunk.start_s * sr) as usize).min(pcm.len());
        let b = ((chunk.end_s * sr) as usize).min(pcm.len());
        let seg = &pcm[a..b];
        let mut best = (chunk.text.clone(), chunk.flag.clone());
        for parts in 2..=4usize {
            let len = seg.len();
            let mut cuts = vec![0];
            for k in 1..parts {
                let centre = len * k / parts;
                let slack = len / (parts * 4);
                cuts.push(audio::quiet_point(seg, centre.saturating_sub(slack), centre + slack));
            }
            cuts.push(len);
            let mut texts = Vec::new();
            let mut clean = true;
            for w in cuts.windows(2) {
                if w[1] <= w[0] + audio::SAMPLE_RATE / 2 {
                    continue; // skip slivers under 0.5 s
                }
                let r = engine.transcribe(&seg[w[0]..w[1]], language)?;
                clean &= r.flag.is_empty();
                if !r.text.is_empty() {
                    texts.push(r.text);
                }
            }
            let text = texts.join(" ");
            if clean {
                best = (text, String::new());
                break;
            }
            if !best.1.is_empty() && text.chars().count() > best.0.chars().count() {
                best = (text, best.1.clone());
            }
        }
        if best.1.is_empty() {
            fixed += 1;
        }
        chunk.text = best.0;
        chunk.flag = best.1;
        on_progress(n + 1, indices.len());
    }
    Ok(fixed)
}

/// Download the default model with progress callbacks (bytes done, bytes total).
pub fn download_model(dest: &Path, progress: impl FnMut(u64, u64)) -> Result<()> {
    download(MODEL_URL, dest, MODEL_BYTES, progress)
}

/// Download a file with progress callbacks (bytes done, bytes total), via a .part file.
pub fn download(url: &str, dest: &Path, expected: u64, mut progress: impl FnMut(u64, u64)) -> Result<()> {
    use std::io::{Read, Write};
    std::fs::create_dir_all(dest.parent().unwrap())?;
    let part = dest.with_extension("gguf.part");
    let mut resp = reqwest::blocking::Client::builder().timeout(None).build()?.get(url).send()?.error_for_status()?;
    let total = resp.content_length().unwrap_or(expected);
    let mut file = std::fs::File::create(&part)?;
    let mut buf = vec![0u8; 1 << 20];
    let mut done = 0u64;
    loop {
        let n = resp.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        done += n as u64;
        progress(done, total);
    }
    file.flush()?;
    drop(file);
    if done != total {
        anyhow::bail!("download incomplete ({done} of {total} bytes)");
    }
    std::fs::rename(part, dest)?;
    Ok(())
}
