use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
    },
    time::Instant,
};

use super::{GeneratedImage, ImageError, ImageModel, ImageProgress, ImageResponse, NativeResult};

/// Cooperative cancellation. Sampling stops at a native progress boundary;
/// an already submitted GPU graph finishes first. Memory stays reserved until
/// the native call returns, even if the event stream has been dropped.
#[derive(Debug, Clone, Default)]
pub struct ImageCanceller(Arc<AtomicBool>);
impl ImageCanceller {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

pub(crate) struct Request {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub seed: i64,
}

impl Request {
    fn validate(&self, model: &ImageModel) -> NativeResult<()> {
        let plan = &model.entry.plan;
        if self.seed < 0 {
            return Err(ImageError::InvalidRequest(
                "seed must be nonnegative".into(),
            ));
        }
        if self.prompt.trim().is_empty()
            || self.prompt.len() > plan.max_prompt_bytes
            || self.prompt.contains('\0')
        {
            return Err(ImageError::InvalidRequest(format!(
                "prompt must be nonempty UTF-8 without NUL, at most {} bytes",
                plan.max_prompt_bytes
            )));
        }
        if self.width == 0
            || self.height == 0
            || !self.width.is_multiple_of(32)
            || !self.height.is_multiple_of(32)
            || self.width > 2048
            || self.height > 2048
            || u64::from(self.width) * u64::from(self.height) > plan.max_pixels
        {
            return Err(ImageError::InvalidRequest(format!(
                "dimensions must be multiples of 32, each at most 2048, with at most {} pixels",
                plan.max_pixels
            )));
        }
        Ok(())
    }
}

/// Configure one image. Turbo always uses eight Euler steps, CFG 1, and the
/// checkpoint's sigma schedule. Defaults to 1024 × 1024 and seed 42.
#[must_use = "call run() or stream() to start generation"]
pub struct ImageGeneration {
    model: ImageModel,
    request: Request,
    cancel: ImageCanceller,
}
impl ImageGeneration {
    pub(crate) fn new(model: ImageModel, prompt: String) -> Self {
        Self {
            model,
            request: Request {
                prompt,
                width: 1024,
                height: 1024,
                seed: 42,
            },
            cancel: ImageCanceller::default(),
        }
    }
    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.request.width = width;
        self.request.height = height;
        self
    }
    pub fn seed(mut self, seed: i64) -> Self {
        self.request.seed = seed;
        self
    }
    pub fn canceller(&self) -> ImageCanceller {
        self.cancel.clone()
    }
    pub fn cancellation(mut self, cancel: ImageCanceller) -> Self {
        self.cancel = cancel;
        self
    }
    pub fn run(self) -> crate::Result<ImageResponse> {
        self.stream()?.finish()
    }

    /// Start a bounded worker and receive progress plus one terminal result.
    /// One job per model; another request on the same handle returns `image_busy`.
    /// Dropping the stream requests cancellation without blocking the caller.
    pub fn stream(self) -> crate::Result<ImageEventStream> {
        self.request.validate(&self.model)?;
        if self.cancel.is_cancelled() {
            return Err(ImageError::Cancelled.into());
        }
        self.model.runtime.begin_image_job(&self.model)?;
        let lease = JobLease(self.model.clone());
        let (sender, receiver) = mpsc::sync_channel(16);
        let cancel = self.cancel.clone();
        std::thread::Builder::new()
            .name("gen2-image".into())
            .spawn(move || {
                let started = Instant::now();
                let result = (|| -> NativeResult<GeneratedImage> {
                    if self.cancel.is_cancelled() {
                        return Err(ImageError::Cancelled);
                    }
                    let mut state =
                        self.model.entry.state.lock().map_err(|_| {
                            ImageError::Execution("image context lock poisoned".into())
                        })?;
                    let driver = state.as_mut().ok_or_else(|| {
                        ImageError::Execution("admitted image context is missing".into())
                    })?;
                    let mut progress = |step, total_steps| {
                        let _ = sender.try_send(Message::Progress(ImageProgress {
                            step,
                            total_steps,
                            elapsed: started.elapsed(),
                        }));
                        !self.cancel.is_cancelled()
                    };
                    let image = driver.generate(&self.request, &self.cancel, &mut progress)?;
                    if self.cancel.is_cancelled() {
                        return Err(ImageError::Cancelled);
                    }
                    if image.width() != self.request.width || image.height() != self.request.height
                    {
                        return Err(ImageError::Execution(
                            "native output dimensions differ from request".into(),
                        ));
                    }
                    Ok(image)
                })()
                .map(|image| ImageResponse {
                    image,
                    seed: self.request.seed,
                    elapsed: started.elapsed(),
                });
                // Releasing busy before publishing terminal makes finish() a reliable
                // boundary for retry, eviction and inspecting released job state.
                drop(lease);
                let _ = sender.send(Message::Completed(result));
            })
            .map_err(|e| ImageError::Execution(format!("starting image worker: {e}")))?;
        Ok(ImageEventStream {
            receiver,
            cancel,
            terminal: None,
            done: false,
        })
    }
}

struct JobLease(ImageModel);
impl Drop for JobLease {
    fn drop(&mut self) {
        self.0.runtime.finish_image_job(&self.0);
    }
}

/// Progress while sampling, followed by exactly one completed response or error.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ImageEvent {
    Progress(ImageProgress),
    Completed(ImageResponse),
}

enum Message {
    Progress(ImageProgress),
    Completed(NativeResult<ImageResponse>),
}

/// A blocking event iterator. Use from a worker thread in a GUI or async host.
/// Returned image storage is shared across the completion event and finish().
pub struct ImageEventStream {
    receiver: Receiver<Message>,
    cancel: ImageCanceller,
    terminal: Option<NativeResult<ImageResponse>>,
    done: bool,
}
impl ImageEventStream {
    pub fn canceller(&self) -> ImageCanceller {
        self.cancel.clone()
    }
    pub fn finish(mut self) -> crate::Result<ImageResponse> {
        while self.next().is_some() {}
        self.terminal
            .take()
            .unwrap_or_else(|| {
                Err(ImageError::Execution(
                    "image worker ended without a result".into(),
                ))
            })
            .map_err(Into::into)
    }
}
impl Iterator for ImageEventStream {
    type Item = crate::Result<ImageEvent>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match self.receiver.recv() {
            Ok(Message::Progress(progress)) => Some(Ok(ImageEvent::Progress(progress))),
            terminal => {
                let result = match terminal {
                    Ok(Message::Completed(result)) => result,
                    _ => Err(ImageError::Execution(
                        "image worker ended without a result".into(),
                    )),
                };
                self.done = true;
                self.terminal = Some(result.clone());
                Some(result.map(ImageEvent::Completed).map_err(Into::into))
            }
        }
    }
}
impl Drop for ImageEventStream {
    fn drop(&mut self) {
        if !self.done {
            self.cancel.cancel();
        }
    }
}
impl std::iter::FusedIterator for ImageEventStream {}
