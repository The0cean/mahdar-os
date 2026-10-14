# Mahdar (محضر)

Mahdar is a desktop app that turns Arabic and English meetings into transcripts, minutes, notes and action items. Transcription runs on this laptop.

- **Speech to text**: [Cohere Transcribe Arabic](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026) (Apache 2.0), Q8_0 GGUF, run through [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) (MIT) on the Radeon GPU (Vulkan) or the CPU. On the Ryzen AI 9 HX 370 / Radeon 890M, 30 minutes of meeting audio takes about 1.5 minutes.
- **Minutes and notes**: written by a local OpenAI-compatible server such as FastFlowLM (`flm serve gemma4-it:12b`), which keeps everything on the laptop. Optional cloud writers: the Claude API with your own key, Claude Code (`claude -p`) or Cursor (`agent -p`) with your own login.
- **Audio**: record from the microphone, or import m4a/mp3/wav/etc. Importing needs `ffmpeg` on PATH.

## Install

Download `Mahdar_<version>_x64-setup.exe` from the [Releases](https://github.com/The0cean/mahdar-os/releases) page and run it. It installs for the current user and doesn't need admin rights. On first run, click DOWNLOAD to fetch the speech model from Hugging Face.

After that Mahdar updates itself: new versions download in the background from this repo's Releases and install when you click **UPDATE READY · RESTART** or close Mahdar (never while recording, transcribing or writing). Turn it off in Settings › System.

## Features

- **Speakers**: [NVIDIA Streaming Sortformer 4spk v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) (NVIDIA Open Model License) runs first, up to 4 speakers, and transcription chunks follow the speaker turns. Click a label to rename the speaker or reassign the line. The names are used in the minutes.
- **Watch folder**: new recordings in a chosen folder are imported automatically. Files already there when the folder is chosen are skipped.
- **Edit**: double-click a line to correct it. A names & terms glossary is passed to the minutes writer and to Ask.
- **Ask**: Q&A about a meeting. Answers cite clickable `[mm:ss]` timestamps.
- **Export**: Word (.docx) for outputs and the transcript, PDF through the print dialog, and Markdown.
- **Unclear parts**: FIX UNCLEAR, or ↻ on a single line, re-runs that audio in smaller pieces.
- **Writers**: a local OpenAI-compatible server (the default; FastFlowLM is started automatically when it isn't already running), the Claude API (bring your own key, kept in Windows Credential Manager; spend is shown in Monitor), Claude Code (`claude -p`) or Cursor (`agent -p`).

## Layout

| Path | What |
|---|---|
| `crates/mahdar-core` | Shared Rust core: audio decode and chunking, transcribe.cpp FFI (`asr.rs`), summaries, storage. The mobile app and MCP server will reuse it. |
| `src-tauri` | Tauri 2 desktop shell and commands. `native/` holds `transcribe.dll` and the ggml backends (v0.3.1, CPU + Vulkan). |
| `src`, `index.html` | UI (TypeScript + Vite). |

Data is stored in `%APPDATA%\com.mahdar.app\`: `settings.json` and `meetings\<id>\{meeting.json,audio.wav}`.

## Develop

```bash
npm install
npx tauri dev           # run with hot reload
npx tauri build         # installer in target/release/bundle/nsis/
cargo run -p mahdar-core --example transcribe -- <native_dir> <model.gguf> <audio> [seconds]
```

## Releasing

```bash
node tools/bump.mjs 0.2.1          # package.json, Cargo.toml, tauri.conf.json, lock files
git commit -am "v0.2.1" && git tag v0.2.1
git push && git push origin v0.2.1
```

The tag runs `.github/workflows/windows-installer.yml`. It builds the installer, signs it with the update key (repository secret `TAURI_SIGNING_PRIVATE_KEY`), and publishes a release with the installer, its `.sig`, and `latest.json`. The release notes are the commit subjects since the previous tag. Installed copies read `releases/latest/download/latest.json` and only install files that verify against the public key in `src-tauri/tauri.conf.json`. Keep the private key safe. Without it, no update can reach installed copies, and replacing it means everyone reinstalls once by hand.

## Notes

- Audio is cut into 15–30 s chunks at quiet points. Long chunks make the model loop. Chunks where it still loops are kept, flagged ⚠, and contain partial text.
- The model needs a language hint and has no auto-detect. "Arabic" is the default and handles mixed meetings best, but writes English terms in Arabic script.

## License

MIT. See [LICENSE](LICENSE). The speech and speaker models are downloaded separately and keep their own licenses (Apache 2.0 and the NVIDIA Open Model License).
