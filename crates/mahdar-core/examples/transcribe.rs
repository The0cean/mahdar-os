//! Smoke test: cargo run -p mahdar-core --example transcribe -- <native_dir> <model.gguf> <audio> [seconds]
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let mut pcm = mahdar_core::audio::decode_to_pcm(Path::new(&a[3]))?;
    if let Some(secs) = a.get(4) {
        pcm.truncate(secs.parse::<usize>()? * mahdar_core::audio::SAMPLE_RATE);
    }
    let mut engine = mahdar_core::Engine::load(Path::new(&a[1]), Path::new(&a[2]), mahdar_core::Backend::Vulkan)?;
    println!("backend: {}", engine.backend_name());
    let (chunks, secs) = mahdar_core::transcribe_pcm(&mut engine, &pcm, "ar", |d, t, c| {
        println!("[{d}/{t}] {:.0}s {} {}", c.start_s, c.flag, c.text.chars().take(80).collect::<String>())
    })?;
    println!("{} chunks in {secs:.1}s", chunks.len());
    Ok(())
}
