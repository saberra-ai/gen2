//! Models from the Hugging Face Hub, by reference.
//!
//! Anywhere the API takes a model path it also takes a reference:
//!
//! ```text
//! hf:owner/repo                 the repo's Q4_K_M GGUF (see below)
//! hf:owner/repo:Q8_0            the file tagged with that quantization
//! hf:owner/repo:Model-Q5_K_M.gguf   that exact file
//! ```
//!
//! This is the grammar `llama-cli -hf` and `ollama run hf.co/…` use, so a
//! string copied from a model card works as written. `hf://` and
//! `https://huggingface.co/owner/repo` are accepted as a courtesy.
//!
//! ```no_run
//! let model = gen2::load("hf:unsloth/Qwen3-0.6B-GGUF")?;
//! let text = model.generate("Reply with one word: hello").max_tokens(8).text()?;
//! # Ok::<(), gen2::Error>(())
//! ```
//!
//! # Which file
//!
//! With no suffix the repo's GGUF files are listed and one is chosen, in
//! this order: the file tagged `Q4_K_M`; else the smallest file tagged
//! `Q4`-anything; else the smallest GGUF. A `:QUANT` suffix picks the file
//! tagged with it (case-insensitive) and fails, naming the tags the repo
//! does have, when there is none. Shards past the first and `mmproj` files
//! are never candidates. When the repo also carries an `mmproj*.gguf`
//! projector it is downloaded and wired in as the vision projector.
//!
//! # Where it goes
//!
//! Files land in a cache in the Hub's own layout
//! (`models--owner--repo/snapshots/<sha>/file`), so a cache shared with
//! `huggingface-cli` or `llama-cli` is a hit. The directory is the first of:
//! `GEN2_MODELS_DIR`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`,
//! `$HF_HOME/hub`, then the platform cache directory (`~/.cache/gen2/hf`
//! on Linux, `~/Library/Caches/gen2/hf` on macOS) — see [`cache_dir`].
//! A second load of the same reference reads the cache and makes no network
//! request, so a warm cache works offline. `HF_TOKEN` is sent when set,
//! which gated repos need and which lifts the anonymous rate limit.
//!
//! # The typed form
//!
//! [`HfModel`] is the reference as a value, for programs that build the
//! choice rather than type it, with [`HfModel::on_progress`] for a download
//! bar. `Engine::builder().hf(model)` and [`Runtime::load_hf`] take it;
//! [`HfModel::download`] returns the local paths for anything else.
//!
//! [`Runtime::load_hf`]: crate::Runtime::load_hf
//!
//! Everything here except the download itself compiles without the `hf`
//! feature: the parser, the selection rules and the cache lookup are plain
//! code, and without the feature a reference that is not already cached
//! fails with a message that says which feature to turn on.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use regex::Regex;

use super::error::Error;

/// The quantization chosen when a reference names none.
pub const DEFAULT_QUANT: &str = "Q4_K_M";

/// The environment variable that overrides the cache directory.
pub const MODELS_DIR_VAR: &str = "GEN2_MODELS_DIR";

// ── The reference ───────────────────────────────────────────────────────────

/// A model on the Hugging Face Hub: a repo, and optionally which file in it.
///
/// Built by [`HfModel::parse`] from the `hf:` grammar or by hand:
///
/// ```
/// use gen2::hf::HfModel;
///
/// let a = HfModel::parse("hf:unsloth/Qwen3-0.6B-GGUF:Q8_0")?;
/// let b = HfModel::new("unsloth/Qwen3-0.6B-GGUF").quant("Q8_0");
/// assert_eq!(a.repo, b.repo);
/// assert_eq!(a.quant, b.quant);
/// # Ok::<(), gen2::Error>(())
/// ```
#[derive(Clone)]
pub struct HfModel {
    /// `owner/name`.
    pub repo: String,
    /// A quantization tag to pick the file by — `Q8_0`, `UD-Q4_K_XL`.
    /// `None` means [`DEFAULT_QUANT`] with the fallbacks described in the
    /// module docs. Ignored when `file` is set.
    pub quant: Option<String>,
    /// An exact file in the repo. Set, nothing is listed or chosen.
    pub file: Option<String>,
    /// The projector file, when the repo's own `mmproj*.gguf` is not wanted
    /// or the caller knows which one.
    mmproj: Mmproj,
    /// Where files go; `None` means [`cache_dir`].
    cache_dir: Option<PathBuf>,
    /// A token for this download; `None` means `HF_TOKEN` and the Hub
    /// client's own fallbacks.
    token: Option<String>,
    progress: Option<ProgressHook>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Mmproj {
    /// Use the repo's projector when it has one.
    Auto,
    /// Never download a projector.
    None,
    /// This file.
    File(String),
}

type ProgressHook = Arc<Mutex<dyn FnMut(Progress) + Send + 'static>>;

impl HfModel {
    /// A model by repo id, resolving to the default quantization.
    pub fn new(repo: impl Into<String>) -> Self {
        Self {
            repo: repo.into(),
            quant: None,
            file: None,
            mmproj: Mmproj::Auto,
            cache_dir: None,
            token: None,
            progress: None,
        }
    }

    /// Parse the `hf:` grammar (and the URL forms the module docs list).
    ///
    /// Everything about the string is checked here, so a malformed
    /// reference fails before any network is touched.
    pub fn parse(reference: &str) -> Result<Self, HfError> {
        let parsed = parse_reference(reference)?;
        let mut model = Self::new(parsed.repo);
        model.quant = parsed.quant;
        model.file = parsed.file;
        Ok(model)
    }

    /// Pick the file by quantization tag.
    pub fn quant(mut self, quant: impl Into<String>) -> Self {
        self.quant = Some(quant.into());
        self
    }

    /// Pick this exact file.
    pub fn file(mut self, file: impl Into<String>) -> Self {
        self.file = Some(file.into());
        self
    }

    /// The projector to download alongside the weights.
    ///
    /// By default the repo's `mmproj*.gguf` is taken when there is one;
    /// this names a specific file instead.
    pub fn mmproj(mut self, file: impl Into<String>) -> Self {
        self.mmproj = Mmproj::File(file.into());
        self
    }

    /// Download no projector, even when the repo has one.
    pub fn without_mmproj(mut self) -> Self {
        self.mmproj = Mmproj::None;
        self
    }

    /// Where to cache the files, instead of [`cache_dir`].
    pub fn cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }

    /// The token to send, instead of `HF_TOKEN`.
    pub fn token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Called as bytes arrive, with the file name and the running total.
    ///
    /// Fires only for a real download: a cache hit calls it never. The
    /// callback runs on the download thread, so it should hand the numbers
    /// to whatever is drawing rather than draw itself.
    pub fn on_progress(mut self, f: impl FnMut(Progress) + Send + 'static) -> Self {
        self.progress = Some(Arc::new(Mutex::new(f)));
        self
    }

    /// The cache directory this model reads and writes.
    pub fn effective_cache_dir(&self) -> PathBuf {
        self.cache_dir.clone().unwrap_or_else(cache_dir)
    }

    /// The reference in `hf:` form.
    pub fn reference(&self) -> String {
        match (&self.file, &self.quant) {
            (Some(file), _) => format!("hf:{}:{file}", self.repo),
            (None, Some(quant)) => format!("hf:{}:{quant}", self.repo),
            (None, None) => format!("hf:{}", self.repo),
        }
    }

    /// Decide which files this reference means, without downloading them.
    ///
    /// Answered from the cache when the cache already holds the exact
    /// choice; otherwise the repo is listed (which needs the `hf` feature
    /// and the network).
    pub fn resolve(&self) -> Result<HfResolved, HfError> {
        if let Some(resolved) = self.resolve_from_cache() {
            return Ok(resolved);
        }
        self.resolve_remote()
    }

    /// Resolve and fetch, returning the local paths.
    ///
    /// A warm cache makes no network request at all — not to list, not to
    /// check freshness. That is deliberate: a model file on the Hub is
    /// immutable in practice, and a load that works on the plane is worth
    /// more than one that notices a re-upload.
    pub fn download(&self) -> Result<HfDownload, HfError> {
        // Fully cached: the names are known and the files are here.
        if let Some(resolved) = self.resolve_from_cache()
            && let Some(download) = self.fetch_cached(&resolved)
        {
            return Ok(download);
        }
        self.download_remote()
    }

    /// What the cache alone can answer: the exact file, or the exact tag
    /// asked for (never a fallback tier — the Hub may hold the better file).
    fn resolve_from_cache(&self) -> Option<HfResolved> {
        let cache = self.effective_cache_dir();
        let files = cached_gguf_files(&cache, &self.repo);
        let file = match &self.file {
            Some(file) => files
                .iter()
                .any(|f| f.name == *file)
                .then(|| file.clone())?,
            None => match choose_gguf(&files, self.quant.as_deref()) {
                Choice::Exact(name) => name,
                _ => return None,
            },
        };
        let mmproj = match &self.mmproj {
            Mmproj::None => None,
            // A projector named but not cached means the cache cannot answer.
            Mmproj::File(name) => Some(
                files
                    .iter()
                    .any(|f| f.name == *name)
                    .then(|| name.clone())?,
            ),
            Mmproj::Auto => choose_mmproj(&files),
        };
        Some(HfResolved {
            repo: self.repo.clone(),
            file,
            mmproj,
            from_cache: true,
        })
    }
}

/// The files a reference resolved to, by name.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct HfResolved {
    /// `owner/name`.
    pub repo: String,
    /// The model file within the repo.
    pub file: String,
    /// The projector within the repo, when one was chosen.
    pub mmproj: Option<String>,
    /// Whether the cache answered without listing the repo.
    pub from_cache: bool,
}

/// Local paths for a downloaded reference.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct HfDownload {
    /// What was chosen.
    pub resolved: HfResolved,
    /// The model file on disk.
    pub model: PathBuf,
    /// The projector on disk, when one was chosen.
    pub mmproj: Option<PathBuf>,
    /// Whether every file was already cached — no bytes moved.
    pub cache_hit: bool,
}

/// One download's running total, for [`HfModel::on_progress`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Progress {
    /// The file within the repo.
    pub file: String,
    /// Bytes received so far.
    pub bytes: u64,
    /// Bytes expected. Zero when the size is not known yet.
    pub total: u64,
}

impl Progress {
    /// `bytes / total`, when the total is known.
    pub fn fraction(&self) -> Option<f64> {
        (self.total > 0).then(|| self.bytes as f64 / self.total as f64)
    }
}

impl fmt::Debug for HfModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HfModel")
            .field("repo", &self.repo)
            .field("quant", &self.quant)
            .field("file", &self.file)
            .field("mmproj", &self.mmproj)
            .field("cache_dir", &self.cache_dir)
            .field("token", &self.token.as_ref().map(|_| "…"))
            .field("progress", &self.progress.is_some())
            .finish()
    }
}

impl std::str::FromStr for HfModel {
    type Err = HfError;

    fn from_str(s: &str) -> Result<Self, HfError> {
        Self::parse(s)
    }
}

impl TryFrom<&str> for HfModel {
    type Error = HfError;

    fn try_from(s: &str) -> Result<Self, HfError> {
        Self::parse(s)
    }
}

// ── Errors ──────────────────────────────────────────────────────────────────

/// What can go wrong between a reference and a file on disk.
///
/// Each case says what to do about it. Converts into [`Error::Load`], so
/// through `gen2::load` these are load failures with this text.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum HfError {
    /// The string is not `hf:owner/repo[:QUANT|:file.gguf]`.
    InvalidReference {
        /// As given.
        reference: String,
        /// What is wrong with it.
        reason: String,
    },
    /// The Hub has no such repo.
    RepoNotFound {
        /// `owner/name`.
        repo: String,
    },
    /// The repo has no file tagged with the requested quantization.
    QuantNotFound {
        /// `owner/name`.
        repo: String,
        /// As requested.
        quant: String,
        /// The tags the repo does have, sorted.
        available: Vec<String>,
    },
    /// The repo has no file by that name.
    FileNotFound {
        /// `owner/name`.
        repo: String,
        /// As requested.
        file: String,
    },
    /// The repo holds no GGUF file at all.
    NoGguf {
        /// `owner/name`.
        repo: String,
    },
    /// The Hub wants a token, or refused the one it got.
    Gated {
        /// `owner/name`.
        repo: String,
        /// Whether a token was sent.
        had_token: bool,
    },
    /// The Hub is rate-limiting this address.
    RateLimited {
        /// When to try again, when the Hub said.
        retry_after: Option<Duration>,
    },
    /// The Hub could not be reached, and the cache could not answer.
    Network {
        /// `owner/name`.
        repo: String,
        /// The transport's own words.
        message: String,
        /// Where the cache was looked in.
        cache_dir: PathBuf,
    },
    /// The cache directory could not be read or written.
    Cache {
        /// The path involved.
        path: PathBuf,
        /// The I/O error.
        message: String,
    },
    /// The `hf` feature is off and the reference is not in the cache.
    FeatureDisabled {
        /// The reference.
        reference: String,
    },
    /// Anything the Hub client reported that fits none of the above.
    Other {
        /// `owner/name`.
        repo: String,
        /// The client's own words.
        message: String,
    },
}

impl fmt::Display for HfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidReference { reference, reason } => write!(
                f,
                "`{reference}` is not a Hugging Face reference: {reason}. The form is \
                 hf:owner/repo, hf:owner/repo:QUANT (e.g. Q8_0) or hf:owner/repo:file.gguf"
            ),
            Self::RepoNotFound { repo } => write!(
                f,
                "no repo `{repo}` on the Hugging Face Hub — check the spelling at \
                 https://huggingface.co/{repo} (a private repo needs HF_TOKEN)"
            ),
            Self::QuantNotFound {
                repo,
                quant,
                available,
            } => {
                write!(f, "`{repo}` has no GGUF tagged `{quant}`")?;
                if available.is_empty() {
                    write!(f, "; it has no tagged GGUF files at all")
                } else {
                    write!(f, "; it has: {}", available.join(", "))
                }
            }
            Self::FileNotFound { repo, file } => {
                write!(
                    f,
                    "`{repo}` has no file `{file}` — see https://huggingface.co/{repo}/tree/main"
                )
            }
            Self::NoGguf { repo } => write!(
                f,
                "`{repo}` holds no GGUF file; `hf:` loads GGUF repos (try the model's `-GGUF` \
                 repo, e.g. from unsloth, bartowski or ggml-org)"
            ),
            Self::Gated { repo, had_token } => {
                write!(f, "`{repo}` needs a token")?;
                if *had_token {
                    write!(
                        f,
                        ": the Hub refused the one sent. Accept the model's terms at \
                         https://huggingface.co/{repo}, and check the token has read access"
                    )
                } else {
                    write!(
                        f,
                        " (gated or private): set HF_TOKEN to a token from \
                         https://huggingface.co/settings/tokens, after accepting the model's \
                         terms at https://huggingface.co/{repo}"
                    )
                }
            }
            Self::RateLimited { retry_after } => {
                write!(f, "the Hugging Face Hub is rate-limiting this address")?;
                if let Some(d) = retry_after {
                    write!(f, "; retry in {}s", d.as_secs())?;
                }
                write!(f, ". Setting HF_TOKEN lifts the anonymous limit")
            }
            Self::Network {
                repo,
                message,
                cache_dir,
            } => write!(
                f,
                "could not reach the Hugging Face Hub for `{repo}` ({message}), and nothing \
                 usable for it is cached under {} — connect, or point {MODELS_DIR_VAR} at a \
                 cache that holds it",
                cache_dir.display()
            ),
            Self::Cache { path, message } => {
                write!(f, "model cache at {}: {message}", path.display())
            }
            Self::FeatureDisabled { reference } => write!(
                f,
                "`{reference}` is a Hugging Face reference, but this build of gen2 has the `hf` \
                 feature off and the file is not cached; enable the feature, or download the \
                 GGUF and load it by path"
            ),
            Self::Other { repo, message } => {
                write!(f, "Hugging Face Hub, `{repo}`: {message}")
            }
        }
    }
}

impl std::error::Error for HfError {}

impl From<HfError> for Error {
    fn from(e: HfError) -> Self {
        Error::Load(e.to_string())
    }
}

// ── The grammar ─────────────────────────────────────────────────────────────

/// Whether a path string is a reference rather than a file.
///
/// The only trigger for the resolver: a plain path never gets past this
/// prefix check, so loading a file costs nothing extra.
pub fn is_reference(s: &str) -> bool {
    s.starts_with("hf:")
        || s.starts_with("hf.co/")
        || s.starts_with("https://huggingface.co/")
        || s.starts_with("https://hf.co/")
        || s.starts_with("http://huggingface.co/")
}

/// The reference in a builder's model path, when it is one.
pub(crate) fn reference_from_path(path: &Path) -> Option<&str> {
    path.to_str().filter(|s| is_reference(s))
}

#[derive(Debug, PartialEq, Eq)]
struct Parsed {
    repo: String,
    quant: Option<String>,
    file: Option<String>,
}

fn invalid(reference: &str, reason: impl Into<String>) -> HfError {
    HfError::InvalidReference {
        reference: reference.to_string(),
        reason: reason.into(),
    }
}

fn parse_reference(reference: &str) -> Result<Parsed, HfError> {
    let trimmed = reference.trim();
    if trimmed != reference {
        return Err(invalid(reference, "leading or trailing whitespace"));
    }
    if let Some(rest) = reference
        .strip_prefix("https://huggingface.co/")
        .or_else(|| reference.strip_prefix("http://huggingface.co/"))
        .or_else(|| reference.strip_prefix("https://hf.co/"))
    {
        return parse_url_path(reference, rest);
    }
    let body = reference
        .strip_prefix("hf://")
        .or_else(|| reference.strip_prefix("hf:"))
        .or_else(|| reference.strip_prefix("hf.co/"))
        .ok_or_else(|| invalid(reference, "it does not start with `hf:`"))?;
    if body.is_empty() {
        return Err(invalid(reference, "nothing after `hf:`"));
    }
    let (repo, suffix) = match body.split_once(':') {
        Some((repo, suffix)) => (repo, Some(suffix)),
        None => (body, None),
    };
    check_repo(reference, repo)?;
    let (quant, file) = match suffix {
        None => (None, None),
        Some("") => {
            return Err(invalid(
                reference,
                "nothing after the colon; drop it, or name a quantization or a .gguf file",
            ));
        }
        Some(s) if s.to_ascii_lowercase().ends_with(".gguf") => {
            check_file(reference, s)?;
            (None, Some(s.to_string()))
        }
        Some(s) => {
            check_quant(reference, s)?;
            (Some(s.to_string()), None)
        }
    };
    Ok(Parsed {
        repo: repo.to_string(),
        quant,
        file,
    })
}

/// `owner/repo[/resolve|blob/<rev>/<file>][?query]`, as copied from a browser.
fn parse_url_path(reference: &str, rest: &str) -> Result<Parsed, HfError> {
    let rest = rest.split(['?', '#']).next().unwrap_or_default();
    let mut parts = rest.trim_end_matches('/').splitn(3, '/');
    let owner = parts.next().unwrap_or_default();
    let name = parts.next().unwrap_or_default();
    let repo = format!("{owner}/{name}");
    check_repo(reference, &repo)?;
    let file = match parts.next() {
        None | Some("") => None,
        Some(tail) => {
            // `resolve/main/file.gguf`, `blob/main/file.gguf`, `tree/main`.
            let mut segs = tail.splitn(3, '/');
            match (segs.next(), segs.next(), segs.next()) {
                (Some("resolve" | "blob"), Some(_rev), Some(file)) => {
                    if !file.to_ascii_lowercase().ends_with(".gguf") {
                        return Err(invalid(
                            reference,
                            "the URL names a file that is not a .gguf",
                        ));
                    }
                    check_file(reference, file)?;
                    Some(file.to_string())
                }
                (Some("tree"), _, _) => None,
                _ => {
                    return Err(invalid(
                        reference,
                        "a URL is accepted only as the repo page or a `resolve`/`blob` file link",
                    ));
                }
            }
        }
    };
    Ok(Parsed {
        repo,
        quant: None,
        file,
    })
}

fn check_repo(reference: &str, repo: &str) -> Result<(), HfError> {
    let Some((owner, name)) = repo.split_once('/') else {
        return Err(invalid(reference, "the repo must be `owner/name`"));
    };
    if owner.is_empty() || name.is_empty() {
        return Err(invalid(reference, "the repo must be `owner/name`"));
    }
    if name.contains('/') {
        return Err(invalid(
            reference,
            "the repo must be `owner/name` — a file in it goes after a colon",
        ));
    }
    let ok = |s: &str| {
        s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    };
    if !ok(owner) || !ok(name) {
        return Err(invalid(
            reference,
            "repo names use letters, digits, `-`, `_` and `.` only",
        ));
    }
    Ok(())
}

fn check_quant(reference: &str, quant: &str) -> Result<(), HfError> {
    if !quant
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    {
        return Err(invalid(
            reference,
            "a quantization tag uses letters, digits, `-`, `_` and `.` only",
        ));
    }
    Ok(())
}

fn check_file(reference: &str, file: &str) -> Result<(), HfError> {
    if file.starts_with('/') || file.split('/').any(|seg| seg.is_empty() || seg == "..") {
        return Err(invalid(
            reference,
            "a file is a path within the repo, without `..` or a leading `/`",
        ));
    }
    if file.chars().any(char::is_whitespace) {
        return Err(invalid(reference, "a file name cannot contain whitespace"));
    }
    Ok(())
}

// ── Choosing a file ─────────────────────────────────────────────────────────

/// A GGUF in a repo, as listed by the Hub or found in the cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufFile {
    /// The path within the repo.
    pub name: String,
    /// The size in bytes, when known.
    pub size: Option<u64>,
}

/// What [`choose_gguf`] decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Choice {
    /// A file tagged exactly as asked (or as [`DEFAULT_QUANT`]).
    Exact(String),
    /// No file carries the default tag, so this is the next best: the
    /// smallest `Q4`-anything, else the smallest GGUF.
    Fallback(String),
    /// A tag was asked for and nothing carries it; these tags exist.
    QuantMissing(Vec<String>),
    /// Nothing in the list is a loadable GGUF.
    NoGguf,
}

/// Shards past the first are loaded by llama.cpp from the first's name.
static SHARD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)-(\d{5})-of-\d{5}\.gguf$").expect("shard regex"));

/// Whether a listed file is a model candidate: a `.gguf` that is not a
/// projector and not a later shard.
fn is_candidate(name: &str) -> bool {
    let base = name.rsplit('/').next().unwrap_or(name);
    let lower = base.to_ascii_lowercase();
    if !lower.ends_with(".gguf") || lower.starts_with("mmproj") {
        return false;
    }
    match SHARD.captures(base) {
        Some(c) => &c[1] == "00001",
        None => true,
    }
}

/// The file's name as tag segments: the stem split on `-` and `.`,
/// lower-cased. `Qwen3-0.6B-UD-Q4_K_XL.gguf` → `qwen3 0 6b ud q4_k_xl`.
fn segments(name: &str) -> Vec<String> {
    let base = name.rsplit('/').next().unwrap_or(name);
    let stem = base
        .strip_suffix(".gguf")
        .or_else(|| base.strip_suffix(".GGUF"))
        .unwrap_or(base);
    let stem = SHARD.find(base).map(|m| &base[..m.start()]).unwrap_or(stem);
    stem.split(['-', '.'])
        .filter(|s| !s.is_empty())
        .map(|s| s.to_ascii_lowercase())
        .collect()
}

/// Whether the file's segments contain the tag's segments contiguously.
fn has_tag(name: &str, tag: &str) -> bool {
    let want: Vec<String> = tag
        .split(['-', '.'])
        .filter(|s| !s.is_empty())
        .map(|s| s.to_ascii_lowercase())
        .collect();
    if want.is_empty() {
        return false;
    }
    segments(name)
        .windows(want.len())
        .any(|w| w == want.as_slice())
}

/// Whether a segment reads as a quantization: `q4_k_m`, `iq3_xxs`, `f16`.
fn looks_like_quant(seg: &str) -> bool {
    let rest = seg.strip_prefix("iq").or_else(|| seg.strip_prefix("q"));
    match rest {
        Some(r) => r.starts_with(|c: char| c.is_ascii_digit()),
        None => matches!(seg, "f16" | "f32" | "bf16" | "fp16" | "fp32"),
    }
}

/// The tag a file carries, in the file's own casing: the last segment that
/// reads as a quantization, with an `UD-` prefix kept when one precedes it.
pub fn quant_of(name: &str) -> Option<String> {
    let base = name.rsplit('/').next().unwrap_or(name);
    let stem = base
        .strip_suffix(".gguf")
        .or_else(|| base.strip_suffix(".GGUF"))
        .unwrap_or(base);
    let stem = SHARD.find(base).map(|m| &base[..m.start()]).unwrap_or(stem);
    let raw: Vec<&str> = stem.split(['-', '.']).filter(|s| !s.is_empty()).collect();
    let at = raw
        .iter()
        .rposition(|s| looks_like_quant(&s.to_ascii_lowercase()))?;
    let mut tag = raw[at].to_string();
    if at > 0 && raw[at - 1].eq_ignore_ascii_case("ud") {
        tag = format!("{}-{tag}", raw[at - 1]);
    }
    Some(tag)
}

/// Pick the model file for a reference from what the repo holds.
///
/// `quant` is the requested tag; `None` asks for [`DEFAULT_QUANT`] with the
/// fallbacks. Order, given no tag: the file tagged `Q4_K_M`; else the
/// smallest tagged `Q4*`; else the smallest GGUF. Given a tag: the file
/// tagged with it, else [`Choice::QuantMissing`] with the tags present.
/// Ties are broken by name so the answer is stable across listings.
pub fn choose_gguf(files: &[GgufFile], quant: Option<&str>) -> Choice {
    let mut candidates: Vec<&GgufFile> = files.iter().filter(|f| is_candidate(&f.name)).collect();
    candidates.sort_by(|a, b| a.name.cmp(&b.name));
    if candidates.is_empty() {
        return Choice::NoGguf;
    }
    let wanted = quant.unwrap_or(DEFAULT_QUANT);
    let by_size = |a: &&GgufFile, b: &&GgufFile| {
        a.size
            .unwrap_or(u64::MAX)
            .cmp(&b.size.unwrap_or(u64::MAX))
            .then_with(|| a.name.cmp(&b.name))
    };
    let mut tagged: Vec<&GgufFile> = candidates
        .iter()
        .copied()
        .filter(|f| has_tag(&f.name, wanted))
        .collect();
    if !tagged.is_empty() {
        tagged.sort_by(by_size);
        return Choice::Exact(tagged[0].name.clone());
    }
    if quant.is_some() {
        let mut available: Vec<String> = candidates
            .iter()
            .filter_map(|f| quant_of(&f.name))
            .collect();
        available.sort_by_key(|t| t.to_ascii_lowercase());
        available.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
        return Choice::QuantMissing(available);
    }
    let mut q4: Vec<&GgufFile> = candidates
        .iter()
        .copied()
        .filter(|f| {
            quant_of(&f.name).is_some_and(|t| {
                t.to_ascii_lowercase()
                    .trim_start_matches("ud-")
                    .starts_with("q4")
            })
        })
        .collect();
    if !q4.is_empty() {
        q4.sort_by(by_size);
        return Choice::Fallback(q4[0].name.clone());
    }
    let mut all = candidates;
    all.sort_by(by_size);
    Choice::Fallback(all[0].name.clone())
}

/// The projector to pair with the weights, when the repo has one: an
/// `mmproj*.gguf`, preferring an F16 one, then the first by name.
pub fn choose_mmproj(files: &[GgufFile]) -> Option<String> {
    let mut projectors: Vec<&str> = files
        .iter()
        .map(|f| f.name.as_str())
        .filter(|n| {
            let base = n.rsplit('/').next().unwrap_or(n).to_ascii_lowercase();
            base.starts_with("mmproj") && base.ends_with(".gguf")
        })
        .collect();
    projectors.sort_unstable();
    // "F16" as a whole segment: `mmproj-BF16.gguf` is not it.
    projectors
        .iter()
        .find(|n| segments(n).iter().any(|s| s == "f16"))
        .or_else(|| projectors.first())
        .map(|n| n.to_string())
}

// ── The cache ───────────────────────────────────────────────────────────────

/// The directory `hf:` references download into.
///
/// The first set of `GEN2_MODELS_DIR`, `HF_HUB_CACHE`,
/// `HUGGINGFACE_HUB_CACHE`, `$HF_HOME/hub`; else the platform cache
/// directory plus `gen2/hf`; else `gen2/hf` under the system temp dir.
/// Always absolute, and the same from any working directory.
pub fn cache_dir() -> PathBuf {
    cache_dir_from(
        |k| std::env::var(k).ok().filter(|v| !v.is_empty()),
        dirs::cache_dir(),
    )
}

/// [`cache_dir`] over an explicit environment, so the order is testable.
fn cache_dir_from(env: impl Fn(&str) -> Option<String>, platform: Option<PathBuf>) -> PathBuf {
    let dir = if let Some(d) = env(MODELS_DIR_VAR) {
        PathBuf::from(d)
    } else if let Some(d) = env("HF_HUB_CACHE") {
        PathBuf::from(d)
    } else if let Some(d) = env("HUGGINGFACE_HUB_CACHE") {
        PathBuf::from(d)
    } else if let Some(home) = env("HF_HOME") {
        PathBuf::from(home).join("hub")
    } else {
        platform
            .unwrap_or_else(std::env::temp_dir)
            .join("gen2")
            .join("hf")
    };
    if dir.is_absolute() {
        dir
    } else {
        std::env::current_dir().map(|c| c.join(&dir)).unwrap_or(dir)
    }
}

/// The Hub layout's directory for a repo: `models--owner--name`.
fn repo_dir(cache: &Path, repo: &str) -> PathBuf {
    cache.join(format!("models--{}", repo.replace('/', "--")))
}

/// The GGUF files the cache holds for a repo, with their sizes.
///
/// Reads the snapshot `refs/main` points at, or every snapshot when there
/// is no ref. A file the Hub client is still writing is not listed: it
/// completes into the snapshot only when done.
pub fn cached_gguf_files(cache: &Path, repo: &str) -> Vec<GgufFile> {
    let root = repo_dir(cache, repo);
    let snapshots = root.join("snapshots");
    let mut dirs: Vec<PathBuf> = match std::fs::read_to_string(root.join("refs").join("main")) {
        Ok(sha) if !sha.trim().is_empty() => vec![snapshots.join(sha.trim())],
        _ => std::fs::read_dir(&snapshots)
            .map(|rd| rd.flatten().map(|e| e.path()).collect())
            .unwrap_or_default(),
    };
    dirs.sort();
    let mut files = Vec::new();
    for dir in dirs {
        collect_gguf(&dir, &dir, &mut files);
    }
    files.sort_by(|a, b| a.name.cmp(&b.name));
    files.dedup_by(|a, b| a.name == b.name);
    files
}

fn collect_gguf(root: &Path, dir: &Path, out: &mut Vec<GgufFile>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        // `metadata` follows the symlink into `blobs/`, which is where the
        // bytes — and the size — are.
        let Ok(meta) = std::fs::metadata(&path) else {
            continue;
        };
        if meta.is_dir() {
            collect_gguf(root, &path, out);
        } else if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
            && let Ok(rel) = path.strip_prefix(root)
        {
            out.push(GgufFile {
                name: rel.to_string_lossy().replace('\\', "/"),
                size: Some(meta.len()),
            });
        }
    }
}

/// The cached path of a repo file, when the cache holds it.
fn cached_path(cache: &Path, repo: &str, file: &str) -> Option<PathBuf> {
    let root = repo_dir(cache, repo);
    let snapshots = root.join("snapshots");
    let mut dirs: Vec<PathBuf> = match std::fs::read_to_string(root.join("refs").join("main")) {
        Ok(sha) if !sha.trim().is_empty() => vec![snapshots.join(sha.trim())],
        _ => std::fs::read_dir(&snapshots)
            .map(|rd| rd.flatten().map(|e| e.path()).collect())
            .unwrap_or_default(),
    };
    dirs.sort();
    dirs.into_iter()
        .map(|d| d.join(file))
        .find(|p| std::fs::metadata(p).is_ok_and(|m| m.is_file()))
}

impl HfModel {
    /// The paths for a cache-answered resolution, when every file is here.
    fn fetch_cached(&self, resolved: &HfResolved) -> Option<HfDownload> {
        let cache = self.effective_cache_dir();
        let model = cached_path(&cache, &self.repo, &resolved.file)?;
        let mmproj = match &resolved.mmproj {
            Some(name) => Some(cached_path(&cache, &self.repo, name)?),
            None => None,
        };
        Some(HfDownload {
            resolved: resolved.clone(),
            model,
            mmproj,
            cache_hit: true,
        })
    }
}

// ── The network half, behind the feature ────────────────────────────────────

#[cfg(feature = "hf")]
mod remote {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
    use hf_hub::repository::files::RepoTreeEntry;
    use hf_hub::{HFClientSync, HFError, HFRepositorySync, RepoTypeModel};

    use super::{
        Choice, GgufFile, HfDownload, HfError as E, HfModel, HfResolved, Mmproj, Progress,
        ProgressHook, choose_gguf, choose_mmproj,
    };

    /// Forwards the Hub client's events to the caller's hook.
    struct Forward {
        file: String,
        hook: ProgressHook,
    }

    impl Forward {
        fn emit(&self, file: &str, bytes: u64, total: u64) {
            if bytes == 0 && total == 0 {
                return;
            }
            if let Ok(mut f) = self.hook.lock() {
                f(Progress {
                    file: file.to_string(),
                    bytes,
                    total,
                });
            }
        }
    }

    impl ProgressHandler for Forward {
        fn on_progress(&self, event: &ProgressEvent) {
            let ProgressEvent::Download(event) = event else {
                return;
            };
            match event {
                DownloadEvent::Progress { files } => {
                    for f in files {
                        self.emit(&f.filename, f.bytes_completed, f.total_bytes);
                    }
                }
                DownloadEvent::AggregateProgress {
                    bytes_completed,
                    total_bytes,
                    ..
                } => self.emit(&self.file, *bytes_completed, *total_bytes),
                DownloadEvent::Start { .. } | DownloadEvent::Complete => {}
            }
        }
    }

    impl HfModel {
        fn client(&self, cache: &Path) -> Result<HFClientSync, E> {
            let mut builder = hf_hub::HFClient::builder()
                .cache_dir(cache)
                .user_agent(format!("gen2/{} hf-hub/{}", env!("CARGO_PKG_VERSION"), "1"));
            // Explicit so the choice is this crate's and not the client's
            // default of the day: the caller's token, else `HF_TOKEN`; the
            // client's own file-based fallbacks apply when neither is set.
            if let Some(token) = self
                .token
                .clone()
                .or_else(|| std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty()))
            {
                builder = builder.token(token);
            }
            builder.build_sync().map_err(|e| self.map_error(e, cache))
        }

        fn had_token(&self) -> bool {
            self.token.is_some() || std::env::var("HF_TOKEN").is_ok_and(|t| !t.is_empty())
        }

        fn map_error(&self, e: HFError, cache: &Path) -> E {
            let repo = self.repo.clone();
            match e {
                HFError::RepoNotFound { .. } => E::RepoNotFound { repo },
                // The listing endpoint's failures come back unmapped, with
                // the Hub's own code in a header (`x-error-code`) and the
                // status; read those the way the client does for downloads.
                HFError::Http { context } => {
                    match (context.error_code.as_deref(), context.status.as_u16()) {
                        (Some("RepoNotFound"), _) | (None, 404) => E::RepoNotFound { repo },
                        (Some("GatedRepo"), _) | (_, 401 | 403) => E::Gated {
                            repo,
                            had_token: self.had_token(),
                        },
                        (_, 429) => E::RateLimited { retry_after: None },
                        (_, status) => E::Other {
                            repo,
                            message: format!(
                                "HTTP {status}{}",
                                context
                                    .server_message
                                    .as_deref()
                                    .map(|m| format!(": {m}"))
                                    .unwrap_or_default()
                            ),
                        },
                    }
                }
                HFError::AuthRequired { .. } | HFError::Forbidden { .. } => E::Gated {
                    repo,
                    had_token: self.had_token(),
                },
                HFError::RateLimited { retry_after, .. } => E::RateLimited { retry_after },
                HFError::EntryNotFound { path, .. } => E::FileNotFound { repo, file: path },
                HFError::Request { source, .. } => E::Network {
                    repo,
                    message: source.to_string(),
                    cache_dir: cache.to_path_buf(),
                },
                HFError::Io(io) => E::Cache {
                    path: cache.to_path_buf(),
                    message: io.to_string(),
                },
                HFError::CacheLockTimeout { path } => E::Cache {
                    path,
                    message: "another process holds the download lock".into(),
                },
                other => E::Other {
                    repo,
                    message: other.to_string(),
                },
            }
        }

        fn repository(&self, client: &HFClientSync) -> HFRepositorySync<RepoTypeModel> {
            let (owner, name) = self.repo.split_once('/').unwrap_or(("", &self.repo));
            client.model(owner, name)
        }

        /// List the repo and choose.
        fn list(
            &self,
            repo: &HFRepositorySync<RepoTypeModel>,
            cache: &Path,
        ) -> Result<Vec<GgufFile>, E> {
            let entries = repo
                .list_tree()
                .recursive(true)
                .send()
                .map_err(|e| self.map_error(e, cache))?;
            Ok(entries
                .into_iter()
                .filter_map(|e| match e {
                    RepoTreeEntry::File { path, size, .. } => Some(GgufFile {
                        name: path,
                        size: Some(size),
                    }),
                    RepoTreeEntry::Directory { .. } => None,
                })
                .collect())
        }

        fn choose(&self, files: &[GgufFile]) -> Result<HfResolved, E> {
            let file = match &self.file {
                Some(file) => {
                    if !files.iter().any(|f| f.name == *file) {
                        return Err(E::FileNotFound {
                            repo: self.repo.clone(),
                            file: file.clone(),
                        });
                    }
                    file.clone()
                }
                None => match choose_gguf(files, self.quant.as_deref()) {
                    Choice::Exact(name) | Choice::Fallback(name) => name,
                    Choice::QuantMissing(available) => {
                        return Err(E::QuantNotFound {
                            repo: self.repo.clone(),
                            quant: self.quant.clone().unwrap_or_default(),
                            available,
                        });
                    }
                    Choice::NoGguf => {
                        return Err(E::NoGguf {
                            repo: self.repo.clone(),
                        });
                    }
                },
            };
            let mmproj = match &self.mmproj {
                Mmproj::None => None,
                Mmproj::File(name) => Some(name.clone()),
                Mmproj::Auto => choose_mmproj(files),
            };
            Ok(HfResolved {
                repo: self.repo.clone(),
                file,
                mmproj,
                from_cache: false,
            })
        }

        pub(super) fn resolve_remote(&self) -> Result<HfResolved, E> {
            let cache = self.effective_cache_dir();
            let client = self.client(&cache)?;
            let repo = self.repository(&client);
            // An exact file with no projector to discover needs no listing.
            if let (Some(file), Mmproj::None | Mmproj::File(_)) = (&self.file, &self.mmproj) {
                return Ok(HfResolved {
                    repo: self.repo.clone(),
                    file: file.clone(),
                    mmproj: match &self.mmproj {
                        Mmproj::File(name) => Some(name.clone()),
                        _ => None,
                    },
                    from_cache: false,
                });
            }
            let files = self.list(&repo, &cache)?;
            self.choose(&files)
        }

        /// The cached path, else a download. Only the download gets the
        /// progress hook, so a cache hit never calls it.
        fn fetch(
            &self,
            repo: &HFRepositorySync<RepoTypeModel>,
            file: &str,
            cache: &Path,
            hit: &mut bool,
        ) -> Result<PathBuf, E> {
            match repo
                .download_file()
                .filename(file)
                .local_files_only(true)
                .send()
            {
                Ok(path) => return Ok(path),
                Err(HFError::LocalEntryNotFound { .. }) => {}
                Err(e) => return Err(self.map_error(e, cache)),
            }
            *hit = false;
            let progress = self.progress.as_ref().map(|hook| {
                hf_hub::progress::Progress::new(Forward {
                    file: file.to_string(),
                    hook: Arc::clone(hook),
                })
            });
            repo.download_file()
                .filename(file)
                .maybe_progress(progress)
                .send()
                .map_err(|e| self.map_error(e, cache))
        }

        pub(super) fn download_remote(&self) -> Result<HfDownload, E> {
            let cache = self.effective_cache_dir();
            std::fs::create_dir_all(&cache).map_err(|e| E::Cache {
                path: cache.clone(),
                message: e.to_string(),
            })?;
            let client = self.client(&cache)?;
            let repo = self.repository(&client);
            let resolved = match self.resolve_from_cache() {
                Some(r) => r,
                None => {
                    if let (Some(file), Mmproj::None | Mmproj::File(_)) = (&self.file, &self.mmproj)
                    {
                        HfResolved {
                            repo: self.repo.clone(),
                            file: file.clone(),
                            mmproj: match &self.mmproj {
                                Mmproj::File(name) => Some(name.clone()),
                                _ => None,
                            },
                            from_cache: false,
                        }
                    } else {
                        let files = self.list(&repo, &cache)?;
                        self.choose(&files)?
                    }
                }
            };
            let mut hit = true;
            let model = self.fetch(&repo, &resolved.file, &cache, &mut hit)?;
            let mmproj = match &resolved.mmproj {
                Some(name) => Some(self.fetch(&repo, name, &cache, &mut hit)?),
                None => None,
            };
            Ok(HfDownload {
                resolved,
                model,
                mmproj,
                cache_hit: hit,
            })
        }
    }
}

#[cfg(not(feature = "hf"))]
impl HfModel {
    fn resolve_remote(&self) -> Result<HfResolved, HfError> {
        Err(HfError::FeatureDisabled {
            reference: self.reference(),
        })
    }

    fn download_remote(&self) -> Result<HfDownload, HfError> {
        Err(HfError::FeatureDisabled {
            reference: self.reference(),
        })
    }
}

// ── Facade glue ─────────────────────────────────────────────────────────────

/// Resolve a builder's model path when it is a reference; `None` when it is
/// a plain path. The one place the facade asks.
pub(crate) fn resolve_model_path(path: &Path) -> Result<Option<HfDownload>, Error> {
    let Some(reference) = reference_from_path(path) else {
        return Ok(None);
    };
    Ok(Some(HfModel::parse(reference)?.download()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> Parsed {
        parse_reference(s).unwrap_or_else(|e| panic!("{s}: {e}"))
    }

    fn reject(s: &str) -> String {
        match parse_reference(s) {
            Ok(p) => panic!("{s} parsed as {p:?}"),
            Err(HfError::InvalidReference { reference, reason }) => {
                assert_eq!(reference, s);
                reason
            }
            Err(e) => panic!("{s}: wrong error {e}"),
        }
    }

    // ── grammar ────────────────────────────────────────────────────────

    #[test]
    fn bare_repo_means_default_quant() {
        let p = parse("hf:unsloth/Qwen3-0.6B-GGUF");
        assert_eq!(p.repo, "unsloth/Qwen3-0.6B-GGUF");
        assert_eq!(p.quant, None);
        assert_eq!(p.file, None);
    }

    #[test]
    fn a_tag_after_the_colon_is_a_quant() {
        let p = parse("hf:unsloth/Qwen3-0.6B-GGUF:Q8_0");
        assert_eq!(p.quant.as_deref(), Some("Q8_0"));
        assert_eq!(p.file, None);
        let p = parse("hf:unsloth/Qwen3-0.6B-GGUF:UD-Q4_K_XL");
        assert_eq!(p.quant.as_deref(), Some("UD-Q4_K_XL"));
        // Case is kept as typed; matching is case-insensitive later.
        assert_eq!(parse("hf:a/b:q4_k_m").quant.as_deref(), Some("q4_k_m"));
    }

    #[test]
    fn a_gguf_name_after_the_colon_is_a_file() {
        let p = parse("hf:unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q5_K_M.gguf");
        assert_eq!(p.file.as_deref(), Some("Qwen3-0.6B-Q5_K_M.gguf"));
        assert_eq!(p.quant, None);
        // A file in a subdirectory, and upper-case extension.
        let p = parse("hf:a/b:Q4_K_M/model-00001-of-00002.GGUF");
        assert_eq!(p.file.as_deref(), Some("Q4_K_M/model-00001-of-00002.GGUF"));
    }

    #[test]
    fn courtesy_forms_are_accepted() {
        assert_eq!(parse("hf://a/b").repo, "a/b");
        assert_eq!(parse("hf.co/a/b:Q8_0").quant.as_deref(), Some("Q8_0"));
        assert_eq!(
            parse("https://huggingface.co/unsloth/Qwen3-0.6B-GGUF").repo,
            "unsloth/Qwen3-0.6B-GGUF"
        );
        assert_eq!(parse("https://huggingface.co/a/b/").repo, "a/b");
        assert_eq!(parse("https://huggingface.co/a/b/tree/main").repo, "a/b");
        assert_eq!(parse("https://hf.co/a/b").repo, "a/b");
        let p = parse("https://huggingface.co/a/b/resolve/main/m-Q4_K_M.gguf?download=true");
        assert_eq!(p.file.as_deref(), Some("m-Q4_K_M.gguf"));
        let p = parse("https://huggingface.co/a/b/blob/main/sub/m.gguf");
        assert_eq!(p.file.as_deref(), Some("sub/m.gguf"));
    }

    #[test]
    fn bad_forms_are_rejected_with_a_reason() {
        assert!(reject("hf:").contains("nothing after"));
        assert!(reject("hf:owner").contains("owner/name"));
        assert!(reject("hf:owner/").contains("owner/name"));
        assert!(reject("hf:/repo").contains("owner/name"));
        assert!(reject("hf:owner/repo/extra").contains("after a colon"));
        assert!(reject("hf:owner/repo:").contains("nothing after the colon"));
        assert!(reject("hf:owner/repo:Q4:Q5").contains("quantization tag"));
        assert!(reject("hf:owner/repo:../x.gguf").contains("`..`"));
        assert!(reject("hf:owner/repo:/abs.gguf").contains("leading `/`"));
        assert!(reject("hf:owner/repo:a b.gguf").contains("whitespace"));
        assert!(reject("hf:ow ner/repo").contains("letters, digits"));
        assert!(reject(" hf:a/b").contains("whitespace"));
        assert!(reject("/models/model.gguf").contains("`hf:`"));
        assert!(reject("https://huggingface.co/a").contains("owner/name"));
        assert!(
            reject("https://huggingface.co/a/b/resolve/main/README.md").contains("not a .gguf")
        );
        assert!(reject("https://huggingface.co/a/b/discussions/1").contains("resolve"));
        // The message a user of `gen2::load` sees carries the form.
        let e = Error::from(parse_reference("hf:x").unwrap_err());
        assert!(e.to_string().contains("hf:owner/repo:QUANT"), "{e}");
    }

    #[test]
    fn only_references_trigger_the_resolver() {
        assert!(is_reference("hf:a/b"));
        assert!(is_reference("hf://a/b"));
        assert!(is_reference("https://huggingface.co/a/b"));
        assert!(!is_reference("/models/hf:a/b.gguf"));
        assert!(!is_reference("./hf/model.gguf"));
        assert!(!is_reference("C:\\models\\model.gguf"));
        assert!(!is_reference("http://localhost:11434/v1"));
        assert_eq!(reference_from_path(Path::new("/models/model.gguf")), None);
        assert_eq!(reference_from_path(Path::new("hf:a/b")), Some("hf:a/b"));
    }

    #[test]
    fn typed_form_round_trips_the_reference() {
        let m = HfModel::parse("hf:a/b:Q8_0").unwrap();
        assert_eq!(m.reference(), "hf:a/b:Q8_0");
        assert_eq!(HfModel::new("a/b").reference(), "hf:a/b");
        assert_eq!(
            HfModel::new("a/b").file("x.gguf").reference(),
            "hf:a/b:x.gguf"
        );
        let m: HfModel = "hf:a/b".parse().unwrap();
        assert_eq!(m.repo, "a/b");
        assert!(format!("{m:?}").contains("a/b"));
    }

    // ── choosing ───────────────────────────────────────────────────────

    fn files(list: &[(&str, u64)]) -> Vec<GgufFile> {
        list.iter()
            .map(|(n, s)| GgufFile {
                name: n.to_string(),
                size: Some(*s),
            })
            .collect()
    }

    /// unsloth/Qwen3-0.6B-GGUF's listing, roughly.
    fn qwen() -> Vec<GgufFile> {
        files(&[
            ("README.md", 1_000),
            (".gitattributes", 100),
            ("Qwen3-0.6B-BF16.gguf", 1_200_000_000),
            ("Qwen3-0.6B-Q8_0.gguf", 639_000_000),
            ("Qwen3-0.6B-Q4_K_M.gguf", 397_000_000),
            ("Qwen3-0.6B-UD-Q4_K_XL.gguf", 405_000_000),
            ("Qwen3-0.6B-Q5_K_M.gguf", 444_000_000),
            ("Qwen3-0.6B-Q4_0.gguf", 382_000_000),
        ])
    }

    #[test]
    fn no_tag_picks_q4_k_m() {
        assert_eq!(
            choose_gguf(&qwen(), None),
            Choice::Exact("Qwen3-0.6B-Q4_K_M.gguf".into())
        );
    }

    #[test]
    fn a_tag_matches_case_insensitively_and_whole() {
        assert_eq!(
            choose_gguf(&qwen(), Some("q8_0")),
            Choice::Exact("Qwen3-0.6B-Q8_0.gguf".into())
        );
        // `Q4_K` is not a prefix match for Q4_K_M: it is a tag of its own.
        assert!(matches!(
            choose_gguf(&qwen(), Some("Q4_K")),
            Choice::QuantMissing(_)
        ));
        assert_eq!(
            choose_gguf(&qwen(), Some("UD-Q4_K_XL")),
            Choice::Exact("Qwen3-0.6B-UD-Q4_K_XL.gguf".into())
        );
        // A dotted, TheBloke-style name.
        let f = files(&[("llama-2-7b.Q4_K_M.gguf", 4), ("llama-2-7b.Q2_K.gguf", 2)]);
        assert_eq!(
            choose_gguf(&f, Some("Q2_K")),
            Choice::Exact("llama-2-7b.Q2_K.gguf".into())
        );
    }

    #[test]
    fn a_missing_tag_lists_what_the_repo_has() {
        let Choice::QuantMissing(available) = choose_gguf(&qwen(), Some("Q6_K")) else {
            panic!("expected QuantMissing");
        };
        assert_eq!(
            available,
            vec!["BF16", "Q4_0", "Q4_K_M", "Q5_K_M", "Q8_0", "UD-Q4_K_XL"]
        );
        let e = HfError::QuantNotFound {
            repo: "a/b".into(),
            quant: "Q6_K".into(),
            available,
        };
        assert!(
            e.to_string()
                .contains("no GGUF tagged `Q6_K`; it has: BF16, Q4_0"),
            "{e}"
        );
    }

    #[test]
    fn fallback_order_is_q4_then_smallest() {
        // No Q4_K_M: the smallest Q4-anything.
        let f = files(&[
            ("m-Q8_0.gguf", 8),
            ("m-Q4_K_S.gguf", 4),
            ("m-Q4_0.gguf", 3),
            ("m-Q2_K.gguf", 2),
        ]);
        assert_eq!(
            choose_gguf(&f, None),
            Choice::Fallback("m-Q4_0.gguf".into())
        );
        // No Q4 at all: the smallest GGUF.
        let f = files(&[("m-Q8_0.gguf", 8), ("m-Q5_K_M.gguf", 5), ("m-F16.gguf", 16)]);
        assert_eq!(
            choose_gguf(&f, None),
            Choice::Fallback("m-Q5_K_M.gguf".into())
        );
        // Unknown sizes sort last; ties by name.
        let f = vec![
            GgufFile {
                name: "b.gguf".into(),
                size: None,
            },
            GgufFile {
                name: "a.gguf".into(),
                size: None,
            },
        ];
        assert_eq!(choose_gguf(&f, None), Choice::Fallback("a.gguf".into()));
    }

    #[test]
    fn projectors_and_later_shards_are_never_the_model() {
        let f = files(&[
            ("mmproj-F16.gguf", 200),
            ("mmproj-BF16.gguf", 200),
            ("m-Q4_K_M-00002-of-00002.gguf", 100),
            ("m-Q4_K_M-00001-of-00002.gguf", 300),
            ("README.md", 1),
        ]);
        assert_eq!(
            choose_gguf(&f, None),
            Choice::Exact("m-Q4_K_M-00001-of-00002.gguf".into())
        );
        assert_eq!(choose_mmproj(&f).as_deref(), Some("mmproj-F16.gguf"));
        assert_eq!(
            choose_gguf(&files(&[("README.md", 1), ("mmproj-F16.gguf", 2)]), None),
            Choice::NoGguf
        );
        assert_eq!(choose_mmproj(&qwen()), None);
        // Without an F16 projector, the first by name.
        let f = files(&[("mmproj-Q8_0.gguf", 1), ("mmproj-BF16.gguf", 1)]);
        assert_eq!(choose_mmproj(&f).as_deref(), Some("mmproj-BF16.gguf"));
        // Files in subdirectories count by their own name.
        let f = files(&[("Q4_K_M/m-Q4_K_M.gguf", 1), ("mmproj/mmproj-F16.gguf", 1)]);
        assert_eq!(
            choose_gguf(&f, None),
            Choice::Exact("Q4_K_M/m-Q4_K_M.gguf".into())
        );
        assert_eq!(choose_mmproj(&f).as_deref(), Some("mmproj/mmproj-F16.gguf"));
    }

    #[test]
    fn the_tag_of_a_file_is_its_last_quant_segment() {
        assert_eq!(
            quant_of("Qwen3-0.6B-Q4_K_M.gguf").as_deref(),
            Some("Q4_K_M")
        );
        assert_eq!(
            quant_of("Qwen3-0.6B-UD-Q4_K_XL.gguf").as_deref(),
            Some("UD-Q4_K_XL")
        );
        assert_eq!(quant_of("llama-2-7b.Q2_K.gguf").as_deref(), Some("Q2_K"));
        assert_eq!(quant_of("m-IQ3_XXS.gguf").as_deref(), Some("IQ3_XXS"));
        assert_eq!(quant_of("m-bf16.gguf").as_deref(), Some("bf16"));
        assert_eq!(
            quant_of("m-Q4_K_M-00001-of-00002.gguf").as_deref(),
            Some("Q4_K_M")
        );
        assert_eq!(quant_of("model.gguf"), None);
        assert_eq!(quant_of("Llama-3.2-1B-Instruct.gguf"), None);
    }

    // ── cache ──────────────────────────────────────────────────────────

    fn env_of<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| {
            pairs
                .iter()
                .find(|(n, _)| *n == k)
                .map(|(_, v)| v.to_string())
        }
    }

    #[test]
    fn cache_dir_order_is_gen2_then_hf_then_platform() {
        let platform = Some(PathBuf::from("/plat"));
        let all = [
            (MODELS_DIR_VAR, "/g"),
            ("HF_HUB_CACHE", "/hub"),
            ("HUGGINGFACE_HUB_CACHE", "/old"),
            ("HF_HOME", "/home"),
        ];
        assert_eq!(
            cache_dir_from(env_of(&all), platform.clone()),
            PathBuf::from("/g")
        );
        assert_eq!(
            cache_dir_from(env_of(&all[1..]), platform.clone()),
            PathBuf::from("/hub")
        );
        assert_eq!(
            cache_dir_from(env_of(&all[2..]), platform.clone()),
            PathBuf::from("/old")
        );
        assert_eq!(
            cache_dir_from(env_of(&all[3..]), platform.clone()),
            PathBuf::from("/home/hub")
        );
        assert_eq!(
            cache_dir_from(env_of(&[]), platform),
            PathBuf::from("/plat/gen2/hf")
        );
        let fallback = cache_dir_from(env_of(&[]), None);
        assert!(fallback.ends_with("gen2/hf"), "{}", fallback.display());
        assert!(fallback.is_absolute());
    }

    #[test]
    fn a_relative_models_dir_is_anchored_to_the_cwd() {
        let dir = cache_dir_from(env_of(&[(MODELS_DIR_VAR, "models")]), None);
        assert!(dir.is_absolute(), "{}", dir.display());
        assert!(dir.ends_with("models"));
        // Reads the real environment without mutating it: whatever it is,
        // it is absolute.
        assert!(cache_dir().is_absolute());
    }

    /// A cache in the Hub's layout: a blob, a snapshot symlink, a ref.
    fn fake_cache(files: &[(&str, usize)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path().join("models--unsloth--Qwen3-0.6B-GGUF");
        let blobs = repo.join("blobs");
        let snap = repo.join("snapshots").join("abc123");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::write(repo.join("refs").join("main"), "abc123\n").unwrap();
        for (i, (name, size)) in files.iter().enumerate() {
            let blob = blobs.join(format!("blob{i}"));
            std::fs::write(&blob, vec![0u8; *size]).unwrap();
            let link = snap.join(name);
            if let Some(parent) = link.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            #[cfg(unix)]
            std::os::unix::fs::symlink(&blob, &link).unwrap();
            #[cfg(not(unix))]
            std::fs::copy(&blob, &link).unwrap();
        }
        dir
    }

    #[test]
    fn the_cache_lists_gguf_files_with_sizes_through_symlinks() {
        let cache = fake_cache(&[
            ("Qwen3-0.6B-Q4_K_M.gguf", 40),
            ("Qwen3-0.6B-Q8_0.gguf", 80),
            ("README.md", 1),
            ("sub/mmproj-F16.gguf", 20),
        ]);
        let got = cached_gguf_files(cache.path(), "unsloth/Qwen3-0.6B-GGUF");
        assert_eq!(
            got,
            files(&[
                ("Qwen3-0.6B-Q4_K_M.gguf", 40),
                ("Qwen3-0.6B-Q8_0.gguf", 80),
                ("sub/mmproj-F16.gguf", 20),
            ])
        );
        assert!(cached_gguf_files(cache.path(), "nobody/nothing").is_empty());
        assert!(
            cached_path(
                cache.path(),
                "unsloth/Qwen3-0.6B-GGUF",
                "Qwen3-0.6B-Q8_0.gguf"
            )
            .is_some()
        );
        assert!(cached_path(cache.path(), "unsloth/Qwen3-0.6B-GGUF", "missing.gguf").is_none());
    }

    #[test]
    fn a_warm_cache_answers_without_the_network() {
        let cache = fake_cache(&[("Qwen3-0.6B-Q4_K_M.gguf", 40), ("Qwen3-0.6B-Q8_0.gguf", 80)]);
        let model = HfModel::parse("hf:unsloth/Qwen3-0.6B-GGUF")
            .unwrap()
            .cache_dir(cache.path());
        let resolved = model.resolve().unwrap();
        assert_eq!(resolved.file, "Qwen3-0.6B-Q4_K_M.gguf");
        assert!(resolved.from_cache);
        let download = model.download().unwrap();
        assert!(download.cache_hit);
        assert!(
            download
                .model
                .ends_with("snapshots/abc123/Qwen3-0.6B-Q4_K_M.gguf")
        );
        assert_eq!(download.mmproj, None);

        // The typed and the string forms agree.
        let by_tag = HfModel::new("unsloth/Qwen3-0.6B-GGUF")
            .quant("q8_0")
            .cache_dir(cache.path())
            .download()
            .unwrap();
        assert!(by_tag.model.ends_with("Qwen3-0.6B-Q8_0.gguf"));
        let by_file = HfModel::new("unsloth/Qwen3-0.6B-GGUF")
            .file("Qwen3-0.6B-Q8_0.gguf")
            .cache_dir(cache.path())
            .download()
            .unwrap();
        assert_eq!(by_file.model, by_tag.model);
    }

    #[test]
    fn the_cache_never_answers_with_a_fallback_tier() {
        // Only Q8_0 cached: the default must go to the Hub for Q4_K_M
        // rather than quietly load the bigger file.
        let cache = fake_cache(&[("Qwen3-0.6B-Q8_0.gguf", 80)]);
        let model = HfModel::parse("hf:unsloth/Qwen3-0.6B-GGUF")
            .unwrap()
            .cache_dir(cache.path());
        assert_eq!(model.resolve_from_cache(), None);
        // Likewise a tag the cache lacks, and a file it lacks.
        let model = HfModel::new("unsloth/Qwen3-0.6B-GGUF")
            .quant("Q5_K_M")
            .cache_dir(cache.path());
        assert_eq!(model.resolve_from_cache(), None);
        let model = HfModel::new("unsloth/Qwen3-0.6B-GGUF")
            .file("other.gguf")
            .cache_dir(cache.path());
        assert_eq!(model.resolve_from_cache(), None);
    }

    #[test]
    fn a_cached_projector_rides_along() {
        let cache = fake_cache(&[("m-Q4_K_M.gguf", 40), ("mmproj-F16.gguf", 20)]);
        let base = HfModel::new("unsloth/Qwen3-0.6B-GGUF").cache_dir(cache.path());
        let d = base.clone().download().unwrap();
        assert!(
            d.mmproj
                .as_ref()
                .is_some_and(|p| p.ends_with("mmproj-F16.gguf"))
        );
        assert_eq!(
            base.clone().without_mmproj().download().unwrap().mmproj,
            None
        );
        assert!(
            base.clone()
                .mmproj("mmproj-F16.gguf")
                .download()
                .unwrap()
                .mmproj
                .is_some()
        );
        // A named projector that is not cached means the cache cannot answer.
        assert_eq!(base.mmproj("mmproj-BF16.gguf").resolve_from_cache(), None);
    }

    #[cfg(not(feature = "hf"))]
    #[test]
    fn without_the_feature_a_cold_reference_says_so() {
        let cache = tempfile::tempdir().unwrap();
        let err = HfModel::parse("hf:a/b")
            .unwrap()
            .cache_dir(cache.path())
            .download()
            .unwrap_err();
        assert!(matches!(err, HfError::FeatureDisabled { .. }));
        assert!(err.to_string().contains("`hf` feature"));
    }

    // ── errors ─────────────────────────────────────────────────────────

    #[test]
    fn every_error_says_what_to_do_and_maps_to_load() {
        let cases: Vec<(HfError, &str)> = vec![
            (
                HfError::RepoNotFound { repo: "a/b".into() },
                "no repo `a/b`",
            ),
            (
                HfError::FileNotFound {
                    repo: "a/b".into(),
                    file: "x.gguf".into(),
                },
                "no file `x.gguf`",
            ),
            (HfError::NoGguf { repo: "a/b".into() }, "holds no GGUF"),
            (
                HfError::Gated {
                    repo: "a/b".into(),
                    had_token: false,
                },
                "set HF_TOKEN",
            ),
            (
                HfError::Gated {
                    repo: "a/b".into(),
                    had_token: true,
                },
                "refused the one sent",
            ),
            (
                HfError::RateLimited {
                    retry_after: Some(Duration::from_secs(30)),
                },
                "retry in 30s",
            ),
            (HfError::RateLimited { retry_after: None }, "HF_TOKEN lifts"),
            (
                HfError::Network {
                    repo: "a/b".into(),
                    message: "dns".into(),
                    cache_dir: "/c".into(),
                },
                "could not reach the Hugging Face Hub for `a/b` (dns), and nothing usable for it is cached under /c",
            ),
            (
                HfError::Cache {
                    path: "/c".into(),
                    message: "read-only".into(),
                },
                "model cache at /c: read-only",
            ),
            (
                HfError::FeatureDisabled {
                    reference: "hf:a/b".into(),
                },
                "`hf` feature off",
            ),
            (
                HfError::Other {
                    repo: "a/b".into(),
                    message: "odd".into(),
                },
                "`a/b`: odd",
            ),
        ];
        for (err, needle) in cases {
            let text = err.to_string();
            assert!(text.contains(needle), "{text:?} lacks {needle:?}");
            let mapped: Error = err.clone().into();
            match mapped {
                Error::Load(msg) => assert_eq!(msg, text),
                other => panic!("{err:?} mapped to {other:?}"),
            }
        }
    }

    #[test]
    fn a_plain_path_never_touches_the_resolver() {
        assert!(
            resolve_model_path(Path::new("/models/model.gguf"))
                .unwrap()
                .is_none()
        );
        assert!(
            resolve_model_path(Path::new("relative.gguf"))
                .unwrap()
                .is_none()
        );
        // And a malformed reference fails before any I/O.
        let err = resolve_model_path(Path::new("hf:nope")).unwrap_err();
        assert!(
            matches!(err, Error::Load(ref m) if m.contains("owner/name")),
            "{err}"
        );
    }

    #[test]
    fn progress_fraction_needs_a_total() {
        let p = Progress {
            file: "f".into(),
            bytes: 50,
            total: 200,
        };
        assert_eq!(p.fraction(), Some(0.25));
        let p = Progress {
            file: "f".into(),
            bytes: 50,
            total: 0,
        };
        assert_eq!(p.fraction(), None);
    }
}
