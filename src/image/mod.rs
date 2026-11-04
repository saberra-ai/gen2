//! Image models owned by [`Runtime`](crate::Runtime).
//!
//! Enable `backend-stable-diffusion` and build the host-supplied Metal bridge
//! with `bash tools/build-image-backend.sh`. No model or native library is
//! downloaded during a Cargo build. See `docs/image-generation.md`.
//!
//! ```no_run
//! use gen2::{Runtime, image::{ImageBundle, ImageLoadOptions}};
//! let runtime = Runtime::new()?;
//! let bundle = ImageBundle::qwen_image_2_1_turbo("denoiser.gguf", "encoder.gguf", "vae.safetensors");
//! let model = runtime.load_image_model(bundle, ImageLoadOptions::new("libgen2_image.dylib"))?;
//! let response = model.generate("A red teapot in morning sunlight").seed(42).run()?;
//! response.image.save("teapot.png")?;
//! # Ok::<(), gen2::Error>(())
//! ```

#[cfg(feature = "backend-stable-diffusion")]
mod ffi;
mod job;
#[cfg(test)]
mod tests;

use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use crate::{Runtime, api::model::ModelId};
pub use job::{ImageCanceller, ImageEvent, ImageEventStream, ImageGeneration};

/// Image-specific errors have stable codes through [`crate::Error::code`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ImageError {
    #[error("image backend unavailable: {0}")]
    BackendUnavailable(String),
    #[error("invalid image bundle: {0}")]
    InvalidBundle(String),
    #[error("invalid image request: {0}")]
    InvalidRequest(String),
    #[error("image memory admission denied: {0}")]
    ResourceLimit(String),
    #[error("this image model already has an active request")]
    Busy,
    #[error("image generation cancelled")]
    Cancelled,
    #[error("image execution failed: {0}")]
    Execution(String),
}

impl ImageError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::BackendUnavailable(_) => "image_backend_unavailable",
            Self::InvalidBundle(_) => "image_invalid_bundle",
            Self::InvalidRequest(_) => "image_invalid_request",
            Self::ResourceLimit(_) => "image_resource_limit",
            Self::Busy => "image_busy",
            Self::Cancelled => "image_cancelled",
            Self::Execution(_) => "image_execution_failed",
        }
    }
}

pub(crate) type NativeResult<T> = std::result::Result<T, ImageError>;

/// Explicit local component paths. The GGUF denoiser alone is insufficient.
#[derive(Debug, Clone)]
pub struct ImageBundle {
    pub denoiser: PathBuf,
    pub text_encoder: PathBuf,
    pub vae: PathBuf,
}

impl ImageBundle {
    /// The eight-step Qwen-Image-2.1-Turbo pipeline, with its matching 2.1 VAE.
    pub fn qwen_image_2_1_turbo(
        denoiser: impl Into<PathBuf>,
        text_encoder: impl Into<PathBuf>,
        vae: impl Into<PathBuf>,
    ) -> Self {
        Self {
            denoiser: denoiser.into(),
            text_encoder: text_encoder.into(),
            vae: vae.into(),
        }
    }

    pub(crate) fn verify(mut self) -> NativeResult<(Self, Vec<FileIdentity>)> {
        let mut identities = Vec::new();
        for path in [&mut self.denoiser, &mut self.text_encoder, &mut self.vae] {
            *path = path
                .canonicalize()
                .map_err(|e| ImageError::InvalidBundle(format!("{}: {e}", path.display())))?;
            let identity = FileIdentity::read(path)?;
            identities.push(identity);
        }
        Ok((self, identities))
    }
}

/// Host-supplied native bridge and the largest request this model reserves for.
/// Lower limits reduce the reservation; unsupported larger limits are refused.
#[derive(Debug, Clone)]
pub struct ImageLoadOptions {
    pub native_library: PathBuf,
    /// Maximum width × height, at most 1,048,576 in this backend release.
    pub max_pixels: u64,
    /// UTF-8 prompt byte limit, at most 4096. Counted before native tokenization.
    pub max_prompt_bytes: usize,
}

impl ImageLoadOptions {
    pub fn new(native_library: impl Into<PathBuf>) -> Self {
        Self {
            native_library: native_library.into(),
            max_pixels: 1024 * 1024,
            max_prompt_bytes: 4096,
        }
    }

    pub(crate) fn plan(&self, files: &[FileIdentity]) -> NativeResult<ImageMemoryPlan> {
        if !(1024..=1024 * 1024).contains(&self.max_pixels)
            || !(1..=4096).contains(&self.max_prompt_bytes)
        {
            return Err(ImageError::InvalidRequest(
                "max_pixels must be 1024..=1048576 and max_prompt_bytes 1..=4096".into(),
            ));
        }
        let bytes = files
            .iter()
            .try_fold(0u64, |sum, file| sum.checked_add(file.bytes))
            .ok_or_else(|| ImageError::ResourceLimit("component sizes overflow".into()))?;
        // Reserve format conversion/alignment overhead plus the measured native
        // pipeline's workspace, including longer conditioning and retained caches.
        let weights_mb = bytes
            .div_ceil(1024 * 1024)
            .checked_mul(5)
            .map(|n| n.div_ceil(4))
            .and_then(|n| n.checked_add(512))
            .ok_or_else(|| ImageError::ResourceLimit("component reservation overflows".into()))?;
        let working_mb = 2048
            + (self.max_pixels * 2048).div_ceil(1024 * 1024)
            + (self.max_prompt_bytes as u64).div_ceil(2);
        let total_mb = weights_mb
            .checked_add(working_mb)
            .ok_or_else(|| ImageError::ResourceLimit("image reservation overflows".into()))?;
        Ok(ImageMemoryPlan {
            weights_mb,
            working_mb,
            total_mb,
            max_pixels: self.max_pixels,
            max_prompt_bytes: self.max_prompt_bytes,
        })
    }
}

/// Conservative peak reservation in MiB, held until the native context is freed.
/// An estimate, not measured RSS or a hard limit on driver allocations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageMemoryPlan {
    pub weights_mb: u64,
    pub working_mb: u64,
    pub total_mb: u64,
    pub max_pixels: u64,
    pub max_prompt_bytes: usize,
}

/// Owned pixels. Clones share immutable storage; native allocations are freed
/// before the result reaches the caller and are never borrowed by this value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedImage {
    width: u32,
    height: u32,
    channels: u32,
    pixels: Arc<[u8]>,
}

impl GeneratedImage {
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn channels(&self) -> u32 {
        self.channels
    }
    pub fn bytes(&self) -> &[u8] {
        &self.pixels
    }
    /// Save RGB/RGBA pixels using the file extension (for example, PNG).
    pub fn save(&self, path: impl AsRef<Path>) -> crate::Result<()> {
        let color = if self.channels == 4 {
            ::image::ColorType::Rgba8
        } else {
            ::image::ColorType::Rgb8
        };
        ::image::save_buffer(path, &self.pixels, self.width, self.height, color)
            .map_err(|e| ImageError::Execution(format!("saving image: {e}")))?;
        Ok(())
    }

    #[cfg(any(feature = "backend-stable-diffusion", test))]
    pub(crate) fn from_pixels(
        width: u32,
        height: u32,
        channels: u32,
        pixels: Vec<u8>,
    ) -> NativeResult<Self> {
        let length = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|n| n.checked_mul(u64::from(channels)));
        if width == 0
            || height == 0
            || !matches!(channels, 3 | 4)
            || length != Some(pixels.len() as u64)
        {
            return Err(ImageError::Execution(
                "invalid native image dimensions or storage".into(),
            ));
        }
        Ok(Self {
            width,
            height,
            channels,
            pixels: pixels.into(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct ImageResponse {
    pub image: GeneratedImage,
    pub seed: i64,
    pub elapsed: Duration,
}

/// A completed denoising step; step 0 reports the start of sampling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageProgress {
    pub step: u32,
    pub total_steps: u32,
    pub elapsed: Duration,
}

/// Cheap, thread-safe handle. Eviction releases weights; the next request
/// restores them under the same runtime admission policy.
#[derive(Clone)]
pub struct ImageModel {
    pub(crate) runtime: Runtime,
    pub(crate) id: ModelId,
    pub(crate) entry: Arc<Entry>,
}

impl std::fmt::Debug for ImageModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageModel")
            .field("id", &self.id)
            .field("resident", &self.is_resident())
            .finish()
    }
}

impl ImageModel {
    pub fn id(&self) -> ModelId {
        self.id
    }
    pub fn runtime(&self) -> Runtime {
        self.runtime.clone()
    }
    pub fn memory_plan(&self) -> &ImageMemoryPlan {
        &self.entry.plan
    }
    pub fn is_resident(&self) -> bool {
        self.entry.resident.load(Ordering::SeqCst)
    }
    pub fn is_busy(&self) -> bool {
        self.entry.busy.load(Ordering::SeqCst)
    }
    pub fn generate(&self, prompt: impl Into<String>) -> ImageGeneration {
        ImageGeneration::new(self.clone(), prompt.into())
    }
}

pub(crate) trait Driver: Send {
    fn generate(
        &mut self,
        request: &job::Request,
        cancel: &ImageCanceller,
        progress: &mut dyn FnMut(u32, u32) -> bool,
    ) -> NativeResult<GeneratedImage>;
}

pub(crate) type Factory = Arc<dyn Fn() -> NativeResult<Box<dyn Driver>> + Send + Sync>;

pub(crate) struct Entry {
    pub name: String,
    pub plan: ImageMemoryPlan,
    pub factory: Factory,
    pub state: Mutex<Option<Box<dyn Driver>>>,
    pub resident: AtomicBool,
    pub busy: AtomicBool,
    pub last_used: Mutex<Option<Instant>>,
}

impl Entry {
    pub fn last_used(&self) -> Option<Instant> {
        *self.last_used.lock().unwrap_or_else(|e| e.into_inner())
    }
    pub fn touch(&self) {
        *self.last_used.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
    }
    pub fn unload(&self) -> NativeResult<()> {
        if self.busy.load(Ordering::SeqCst) {
            return Err(ImageError::Busy);
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| ImageError::Execution("image context lock poisoned".into()))?;
        drop(state.take()); // Keep the reservation until native teardown completes.
        self.resident.store(false, Ordering::SeqCst);
        Ok(())
    }
    pub fn restore(&self) -> NativeResult<()> {
        if !self.resident.load(Ordering::SeqCst) {
            let driver = (self.factory)()?;
            *self
                .state
                .lock()
                .map_err(|_| ImageError::Execution("image context lock poisoned".into()))? =
                Some(driver);
            self.resident.store(true, Ordering::SeqCst);
        }
        self.touch();
        Ok(())
    }
}

#[derive(Clone)]
pub(crate) struct FileIdentity {
    #[cfg(any(feature = "backend-stable-diffusion", test))]
    pub path: PathBuf,
    pub bytes: u64,
    #[cfg(any(feature = "backend-stable-diffusion", test))]
    pub modified: std::time::SystemTime,
}
impl FileIdentity {
    fn read(path: &Path) -> NativeResult<Self> {
        let metadata = path
            .metadata()
            .map_err(|e| ImageError::InvalidBundle(format!("{}: {e}", path.display())))?;
        if !metadata.is_file() || metadata.len() == 0 {
            return Err(ImageError::InvalidBundle(format!(
                "{} is not a nonempty file",
                path.display()
            )));
        }
        Ok(Self {
            #[cfg(any(feature = "backend-stable-diffusion", test))]
            path: path.to_owned(),
            bytes: metadata.len(),
            #[cfg(any(feature = "backend-stable-diffusion", test))]
            modified: metadata
                .modified()
                .map_err(|e| ImageError::InvalidBundle(e.to_string()))?,
        })
    }
    #[cfg(any(feature = "backend-stable-diffusion", test))]
    pub fn check(&self) -> NativeResult<()> {
        let now = Self::read(&self.path)?;
        if now.bytes != self.bytes || now.modified != self.modified {
            return Err(ImageError::InvalidBundle(format!(
                "{} changed since admission; load a new bundle",
                self.path.display()
            )));
        }
        Ok(())
    }
}

pub(crate) fn factory(
    bundle: ImageBundle,
    options: ImageLoadOptions,
    files: Vec<FileIdentity>,
    plan: ImageMemoryPlan,
) -> NativeResult<Factory> {
    #[cfg(not(feature = "backend-stable-diffusion"))]
    {
        let _ = (bundle, options, files, plan);
        Err(ImageError::BackendUnavailable(
            "enable backend-stable-diffusion".into(),
        ))
    }
    #[cfg(feature = "backend-stable-diffusion")]
    {
        let api = ffi::Api::open(&options.native_library)?;
        Ok(Arc::new(move || {
            for file in &files {
                file.check()?;
            }
            Ok(Box::new(ffi::Context::load(Arc::clone(&api), &bundle, &plan)?) as Box<dyn Driver>)
        }))
    }
}
