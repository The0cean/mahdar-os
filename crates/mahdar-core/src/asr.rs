//! Local speech-to-text through transcribe.cpp's C API (loaded at runtime).
//!
//! The library ships as `transcribe.dll` plus ggml backend modules (CPU,
//! Vulkan) in one directory. Struct layouts mirror `include/transcribe.h`
//! v0.3.1 and are checked against the library's own ABI sizes at load.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use libloading::Library;

const ABI_MODEL_LOAD_PARAMS: c_int = 0;
const ABI_SESSION_PARAMS: c_int = 1;
const ABI_RUN_PARAMS: c_int = 2;

const STATUS_OK: c_int = 0;
const STATUS_OUTPUT_TRUNCATED: c_int = 18;
const STATUS_OUTPUT_REPETITION: c_int = 19;

const TIMESTAMPS_NONE: c_int = 0;

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Auto,
    Cpu,
    Vulkan,
}

impl Backend {
    fn code(self) -> c_int {
        match self {
            Backend::Auto => 0,
            Backend::Cpu => 1,
            Backend::Vulkan => 3,
        }
    }
}

#[repr(C)]
struct ModelLoadParams {
    struct_size: u64,
    backend: c_int,
    device: *mut c_void,
}

#[repr(C)]
struct SessionParams {
    struct_size: u64,
    n_threads: c_int,
    kv_type: c_int,
    n_ctx: i32,
}

#[repr(C)]
struct RunParams {
    struct_size: u64,
    task: c_int,
    timestamps: c_int,
    pnc: c_int,
    itn: c_int,
    diarize: c_int,
    language: *const c_char,
    target_language: *const c_char,
    keep_special_tags: bool,
    family: *const c_void,
    spec_k_drafts: i32,
    vocabulary: *const *const c_char,
    n_vocabulary: i32,
    prompt: *const c_char,
    prefix: *const c_char,
}

type FnInitBackends = unsafe extern "C" fn(*const c_char) -> c_int;
type FnAbiSize = unsafe extern "C" fn(c_int) -> usize;
type FnStatusString = unsafe extern "C" fn(c_int) -> *const c_char;
type FnLoadParamsInit = unsafe extern "C" fn(*mut ModelLoadParams);
type FnModelLoad = unsafe extern "C" fn(*const c_char, *const ModelLoadParams, *mut *mut c_void) -> c_int;
type FnModelFree = unsafe extern "C" fn(*mut c_void);
type FnModelBackend = unsafe extern "C" fn(*const c_void) -> *const c_char;
type FnSessionParamsInit = unsafe extern "C" fn(*mut SessionParams);
type FnSessionInit = unsafe extern "C" fn(*mut c_void, *const SessionParams, *mut *mut c_void) -> c_int;
type FnSessionFree = unsafe extern "C" fn(*mut c_void);
type FnRunParamsInit = unsafe extern "C" fn(*mut RunParams);
type FnRun = unsafe extern "C" fn(*mut c_void, *const f32, c_int, *const RunParams) -> c_int;
type FnFullText = unsafe extern "C" fn(*const c_void) -> *const c_char;
type FnSortformerExtInit = unsafe extern "C" fn(*mut SortformerExt);
type FnNSpeakerSegments = unsafe extern "C" fn(*const c_void) -> c_int;
type FnGetSpeakerSegment = unsafe extern "C" fn(*const c_void, c_int, *mut SpeakerSegment) -> c_int;
type FnSpeakerSegmentInit = unsafe extern "C" fn(*mut SpeakerSegment);

/// transcribe/sortformer.h: `struct transcribe_ext` header + preset.
#[repr(C)]
struct SortformerExt {
    size: u64,
    kind: u32,
    preset: c_int,
}
const SORTFORMER_PRESET_VERY_HIGH_LATENCY: c_int = 1; // most accurate; the offline operating point

#[repr(C)]
struct SpeakerSegment {
    struct_size: u64,
    t0_ms: i64,
    t1_ms: i64,
    speaker_id: i32,
    p: f32,
}

/// One "who spoke when" row: seconds, 1-based speaker in order of first appearance.
#[derive(Clone, Copy, Debug)]
pub struct SpeakerTurn {
    pub start_s: f64,
    pub end_s: f64,
    pub speaker: u32,
}

/// Outcome of one chunk.
pub struct ChunkText {
    pub text: String,
    /// Empty when clean; otherwise why decoding stopped early (text is partial).
    pub flag: String,
}

/// A loaded model plus one session. Not thread-safe: wrap in a Mutex.
pub struct Engine {
    model: *mut c_void,
    session: *mut c_void,
    backend_name: String,
    status_string: FnStatusString,
    run_params_init: FnRunParamsInit,
    run: FnRun,
    full_text: FnFullText,
    session_free: FnSessionFree,
    model_free: FnModelFree,
    sortformer_ext_init: Option<FnSortformerExtInit>,
    n_speaker_segments: FnNSpeakerSegments,
    get_speaker_segment: FnGetSpeakerSegment,
    speaker_segment_init: FnSpeakerSegmentInit,
    _lib: Library, // must outlive every fn pointer above
}

// The handles are only touched through &mut self, behind a Mutex.
unsafe impl Send for Engine {}

#[cfg(windows)]
fn open_library(path: &Path) -> Result<Library> {
    use libloading::os::windows::{Library as WinLib, LOAD_WITH_ALTERED_SEARCH_PATH};
    // Altered search path lets transcribe.dll find ggml*.dll next to it.
    let lib = unsafe { WinLib::load_with_flags(path, LOAD_WITH_ALTERED_SEARCH_PATH) }?;
    Ok(lib.into())
}

#[cfg(not(windows))]
fn open_library(path: &Path) -> Result<Library> {
    Ok(unsafe { Library::new(path) }?)
}

fn library_file(dir: &Path) -> std::path::PathBuf {
    if cfg!(windows) {
        dir.join("transcribe.dll")
    } else if cfg!(target_os = "macos") {
        dir.join("libtranscribe.dylib")
    } else {
        dir.join("libtranscribe.so")
    }
}

unsafe fn sym<T: Copy>(lib: &Library, name: &[u8]) -> Result<T> {
    let s: libloading::Symbol<T> = lib
        .get(name)
        .with_context(|| format!("missing symbol {}", String::from_utf8_lossy(name)))?;
    Ok(*s)
}

impl Engine {
    /// Load the native library from `native_dir` and the GGUF model at `model_path`.
    pub fn load(native_dir: &Path, model_path: &Path, backend: Backend) -> Result<Self> {
        let lib = open_library(&library_file(native_dir))
            .with_context(|| format!("could not load transcription library from {}", native_dir.display()))?;
        unsafe {
            let abi_size: FnAbiSize = sym(&lib, b"transcribe_abi_struct_size\0")?;
            for (id, ours, name) in [
                (ABI_MODEL_LOAD_PARAMS, std::mem::size_of::<ModelLoadParams>(), "model_load_params"),
                (ABI_SESSION_PARAMS, std::mem::size_of::<SessionParams>(), "session_params"),
                (ABI_RUN_PARAMS, std::mem::size_of::<RunParams>(), "run_params"),
            ] {
                let theirs = abi_size(id);
                if theirs != ours {
                    bail!("transcribe.dll ABI mismatch for {name}: library {theirs} bytes, app {ours} bytes");
                }
            }

            let status_string: FnStatusString = sym(&lib, b"transcribe_status_string\0")?;
            let check = |st: c_int, what: &str| -> Result<()> {
                if st == STATUS_OK {
                    return Ok(());
                }
                let msg = CStr::from_ptr(status_string(st)).to_string_lossy();
                Err(anyhow!("{what}: {msg} (status {st})"))
            };

            let init_backends: FnInitBackends = sym(&lib, b"transcribe_init_backends\0")?;
            let dir = CString::new(native_dir.to_string_lossy().as_bytes())?;
            check(init_backends(dir.as_ptr()), "init backends")?;

            let load_params_init: FnLoadParamsInit = sym(&lib, b"transcribe_model_load_params_init\0")?;
            let model_load: FnModelLoad = sym(&lib, b"transcribe_model_load_file\0")?;
            let mut lp = std::mem::zeroed::<ModelLoadParams>();
            load_params_init(&mut lp);
            lp.backend = backend.code();
            let path = CString::new(model_path.to_string_lossy().as_bytes())?;
            let mut model = std::ptr::null_mut();
            check(model_load(path.as_ptr(), &lp, &mut model), "load model")?;

            let model_backend: FnModelBackend = sym(&lib, b"transcribe_model_backend\0")?;
            let backend_name = CStr::from_ptr(model_backend(model)).to_string_lossy().into_owned();

            let model_free: FnModelFree = sym(&lib, b"transcribe_model_free\0")?;
            let session_params_init: FnSessionParamsInit = sym(&lib, b"transcribe_session_params_init\0")?;
            let session_init: FnSessionInit = sym(&lib, b"transcribe_session_init\0")?;
            let mut sp = std::mem::zeroed::<SessionParams>();
            session_params_init(&mut sp);
            let mut session = std::ptr::null_mut();
            if let Err(e) = check(session_init(model, &sp, &mut session), "open session") {
                model_free(model);
                return Err(e);
            }

            Ok(Engine {
                model,
                session,
                backend_name,
                status_string,
                run_params_init: sym(&lib, b"transcribe_run_params_init\0")?,
                run: sym(&lib, b"transcribe_run\0")?,
                full_text: sym(&lib, b"transcribe_full_text\0")?,
                session_free: sym(&lib, b"transcribe_session_free\0")?,
                model_free,
                sortformer_ext_init: sym(&lib, b"transcribe_sortformer_stream_ext_init\0").ok(),
                n_speaker_segments: sym(&lib, b"transcribe_n_speaker_segments\0")?,
                get_speaker_segment: sym(&lib, b"transcribe_get_speaker_segment\0")?,
                speaker_segment_init: sym(&lib, b"transcribe_speaker_segment_init\0")?,
                _lib: lib,
            })
        }
    }

    /// Which compute backend the model landed on ("vulkan", "cpu", ...).
    pub fn backend_name(&self) -> &str {
        &self.backend_name
    }

    /// Transcribe one chunk of 16 kHz mono f32 PCM (the model accepts up to ~400 s).
    pub fn transcribe(&mut self, pcm: &[f32], language: &str) -> Result<ChunkText> {
        if pcm.is_empty() {
            return Ok(ChunkText { text: String::new(), flag: String::new() });
        }
        let lang = CString::new(language)?;
        unsafe {
            let mut rp = std::mem::zeroed::<RunParams>();
            (self.run_params_init)(&mut rp);
            rp.language = lang.as_ptr();
            rp.timestamps = TIMESTAMPS_NONE;
            let st = (self.run)(self.session, pcm.as_ptr(), pcm.len() as c_int, &rp);
            let flag = match st {
                STATUS_OK => "",
                STATUS_OUTPUT_REPETITION => "repetition",
                STATUS_OUTPUT_TRUNCATED => "truncated",
                _ => {
                    let msg = CStr::from_ptr((self.status_string)(st)).to_string_lossy();
                    bail!("transcription failed: {msg} (status {st})");
                }
            };
            // On truncation/repetition the session keeps the partial transcript.
            let p = (self.full_text)(self.session);
            let text = if p.is_null() { String::new() } else { clean_text(&CStr::from_ptr(p).to_string_lossy()) };
            Ok(ChunkText { text, flag: flag.to_string() })
        }
    }
}

impl Engine {
    /// Run a diarization model (Sortformer) over a whole recording.
    pub fn diarize(&mut self, pcm: &[f32]) -> Result<Vec<SpeakerTurn>> {
        if pcm.is_empty() {
            return Ok(Vec::new());
        }
        unsafe {
            let mut rp = std::mem::zeroed::<RunParams>();
            (self.run_params_init)(&mut rp);
            rp.timestamps = TIMESTAMPS_NONE;
            let mut ext = std::mem::zeroed::<SortformerExt>();
            if let Some(init) = self.sortformer_ext_init {
                init(&mut ext);
                ext.preset = SORTFORMER_PRESET_VERY_HIGH_LATENCY;
                rp.family = &ext as *const SortformerExt as *const c_void;
            }
            let st = (self.run)(self.session, pcm.as_ptr(), pcm.len() as c_int, &rp);
            if st != STATUS_OK {
                let msg = CStr::from_ptr((self.status_string)(st)).to_string_lossy();
                bail!("speaker detection failed: {msg} (status {st})");
            }
            let n = (self.n_speaker_segments)(self.session);
            let mut turns = Vec::with_capacity(n.max(0) as usize);
            for i in 0..n {
                let mut seg = std::mem::zeroed::<SpeakerSegment>();
                (self.speaker_segment_init)(&mut seg);
                if (self.get_speaker_segment)(self.session, i, &mut seg) == STATUS_OK && seg.speaker_id > 0 {
                    turns.push(SpeakerTurn {
                        start_s: seg.t0_ms as f64 / 1000.0,
                        end_s: seg.t1_ms as f64 / 1000.0,
                        speaker: seg.speaker_id as u32,
                    });
                }
            }
            turns.sort_by(|a, b| a.start_s.partial_cmp(&b.start_s).unwrap());
            Ok(turns)
        }
    }
}

/// Drop non-speech tags the model emits on silence/noise, e.g. "@@@ضوضاء" ("noise").
/// Also drops a chunk that is only "فراغ" ("empty"), which the model emits on near-silent slivers.
pub fn clean_text(text: &str) -> String {
    let t = if text.contains('@') {
        text.replace("ضوضاء", "").replace('@', "").split_whitespace().collect::<Vec<_>>().join(" ")
    } else {
        text.trim().to_string()
    };
    if t.trim_matches(|c: char| c.is_ascii_punctuation() || c == '.' || c.is_whitespace()) == "فراغ" {
        return String::new();
    }
    t
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            (self.session_free)(self.session);
            (self.model_free)(self.model);
        }
    }
}
