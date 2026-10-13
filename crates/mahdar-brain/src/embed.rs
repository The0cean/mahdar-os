//! Local sentence embeddings (meaning search). Runs on the CPU with ONNX Runtime; nothing
//! leaves the machine.
//!
//! Model: multilingual-e5-base, int8 (Xenova's ONNX export, 278 MB). Arabic (MSA and dialects
//! as written by the ASR) and English in one vector space, 768 dimensions. Chosen over e5-small
//! (118 MB) after a bake-off on made-up Arabic/English meeting lines: 5/5 vs 4/5 top-1 on
//! cross-language questions, at ~50 passages/s on the HX 370 CPU (~1.5 s per meeting hour).
//! e5 expects "query: " / "passage: " prefixes.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use fastembed::{InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel};

pub const MODEL_NAME: &str = "multilingual-e5-base (int8)";
pub const MODEL_DIR: &str = "multilingual-e5-base-int8";
pub const DIM: usize = 768;
const REPO: &str = "https://huggingface.co/Xenova/multilingual-e5-base/resolve/main";

/// Files to download: (URL, file name in the model folder, expected bytes).
pub fn model_files() -> Vec<(String, &'static str, u64)> {
    vec![
        (format!("{REPO}/onnx/model_int8.onnx"), "model.onnx", 278_184_162),
        (format!("{REPO}/tokenizer.json"), "tokenizer.json", 17_082_660),
        (format!("{REPO}/config.json"), "config.json", 686),
        (format!("{REPO}/special_tokens_map.json"), "special_tokens_map.json", 280),
        (format!("{REPO}/tokenizer_config.json"), "tokenizer_config.json", 418),
    ]
}

pub fn model_dir(models_root: &Path) -> PathBuf {
    models_root.join(MODEL_DIR)
}

/// Total download size in bytes.
pub fn model_bytes() -> u64 {
    model_files().iter().map(|f| f.2).sum()
}

pub fn model_present(models_root: &Path) -> bool {
    let dir = model_dir(models_root);
    model_files().iter().all(|(_, f, _)| dir.join(f).is_file())
}

pub struct Embedder {
    model: TextEmbedding,
}

impl Embedder {
    pub fn load(models_root: &Path) -> Result<Self> {
        let dir = model_dir(models_root);
        let read = |f: &str| std::fs::read(dir.join(f)).with_context(|| format!("embedding model file {f} missing in {}", dir.display()));
        let tokenizer = TokenizerFiles {
            tokenizer_file: read("tokenizer.json")?,
            config_file: read("config.json")?,
            special_tokens_map_file: read("special_tokens_map.json")?,
            tokenizer_config_file: read("tokenizer_config.json")?,
        };
        let model = UserDefinedEmbeddingModel::new(read("model.onnx")?, tokenizer).with_pooling(Pooling::Mean);
        // Passages are ≤ 60 s of speech (~150–250 tokens); 512 is the model's limit.
        let opts = InitOptionsUserDefined::new().with_max_length(512);
        let model = TextEmbedding::try_new_from_user_defined(model, opts).context("could not load the embedding model")?;
        Ok(Embedder { model })
    }

    fn run(&mut self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let mut out = self.model.embed(texts, Some(32)).context("embedding failed")?;
        for v in &mut out {
            normalize(v);
        }
        Ok(out)
    }

    pub fn passages(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.run(texts.iter().map(|t| format!("passage: {t}")).collect())
    }

    pub fn query(&mut self, q: &str) -> Result<Vec<f32>> {
        Ok(self.run(vec![format!("query: {q}")])?.remove(0))
    }
}

pub fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn from_blob(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
