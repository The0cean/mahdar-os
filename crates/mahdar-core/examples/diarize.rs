//! Smoke test: cargo run -p mahdar-core --example diarize -- <native_dir> <diarizer.gguf> <asr.gguf> <audio> [seconds]
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let mut pcm = mahdar_core::audio::decode_to_pcm(Path::new(&a[4]))?;
    if let Some(secs) = a.get(5) {
        pcm.truncate(secs.parse::<usize>()? * mahdar_core::audio::SAMPLE_RATE);
    }
    let native = Path::new(&a[1]);
    let mut diar = mahdar_core::Engine::load(native, Path::new(&a[2]), mahdar_core::Backend::Vulkan)?;
    let t = std::time::Instant::now();
    let turns = diar.diarize(&pcm)?;
    println!("diarized {:.0} s of audio in {:.1} s on {}: {} turns", pcm.len() as f64 / 16000.0, t.elapsed().as_secs_f64(), diar.backend_name(), turns.len());
    let mut per = std::collections::BTreeMap::new();
    for t in &turns {
        *per.entry(t.speaker).or_insert(0.0) += t.end_s - t.start_s;
    }
    println!("talk time per speaker (s): {per:?}");
    let spans = mahdar_core::speaker_spans(&pcm, &turns);
    println!("{} spans", spans.len());
    drop(diar);
    let mut asr = mahdar_core::Engine::load(native, Path::new(&a[3]), mahdar_core::Backend::Vulkan)?;
    let (chunks, secs) = mahdar_core::transcribe_spans(&mut asr, &pcm, &spans, "ar", |_, _, _| {})?;
    for c in chunks.iter().take(25) {
        println!("[{:>6.1}] S{} {}", c.start_s, c.speaker.unwrap_or(0), c.text.chars().take(70).collect::<String>());
    }
    println!("asr {secs:.1}s");
    Ok(())
}
