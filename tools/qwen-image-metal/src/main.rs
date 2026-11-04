//! Live, in-process Metal qualification; see ../README.md.
use std::{
    ffi::{CStr, CString, c_char, c_int, c_void},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, ensure};
use gen2::{Runtime, Session, model::ThinkingMode};
use libloading::Library;
use serde_json::json;
use sha2::{Digest, Sha256};

const PROMPT: &str = "A red ceramic teapot on a wooden table beside a small green plant, warm morning sunlight through a window, beautiful editorial product photograph, detailed, natural colors.";
type ProgressFn = unsafe extern "C" fn(c_int, c_int, f32, *mut c_void);
type LoadFn = unsafe extern "C" fn(
    *const c_char,
    *const c_char,
    *const c_char,
    *mut c_char,
    usize,
) -> *mut c_void;
type GenerateFn = unsafe extern "C" fn(
    *mut c_void,
    *const c_char,
    i64,
    ProgressFn,
    *mut c_void,
    *mut NativeImage,
    *mut c_char,
    usize,
) -> bool;

#[repr(C)]
#[derive(Default)]
struct NativeImage {
    width: u32,
    height: u32,
    channels: u32,
    pixels: *const u8,
    owner: *mut c_void,
    count: c_int,
}

// Every function pointer is kept alive by _library. The bridge owns SD's
// changing C structs and its private GGML; no upstream layouts enter Rust.
struct Api {
    _library: Library,
    load: LoadFn,
    generate: GenerateFn,
    free_image: unsafe extern "C" fn(*mut NativeImage),
    unload: unsafe extern "C" fn(*mut c_void),
    devices: unsafe extern "C" fn(*mut c_char, usize) -> usize,
}

impl Api {
    fn open(path: &Path) -> Result<Self> {
        // SAFETY: only the bridge built by build.sh is supported; ABI is
        // checked before resolving typed functions. Library outlives pointers.
        unsafe {
            let library = Library::new(path)?;
            let abi = library.get::<unsafe extern "C" fn() -> u32>(b"gen2_qwen_probe_abi\0")?;
            ensure!(abi() == 1, "unsupported probe bridge ABI");
            Ok(Self {
                load: *library.get(b"gen2_qwen_probe_load\0")?,
                generate: *library.get(b"gen2_qwen_probe_generate\0")?,
                free_image: *library.get(b"gen2_qwen_probe_free_image\0")?,
                unload: *library.get(b"gen2_qwen_probe_unload\0")?,
                devices: *library.get(b"gen2_qwen_probe_devices\0")?,
                _library: library,
            })
        }
    }

    fn devices(&self) -> Result<String> {
        // SAFETY: capacity is queried first, buffer has room for a terminator.
        unsafe {
            let size = (self.devices)(std::ptr::null_mut(), 0);
            ensure!(size < 1024 * 1024, "invalid native device list size");
            let mut bytes = vec![0u8; size + 1];
            (self.devices)(bytes.as_mut_ptr().cast(), bytes.len());
            Ok(CStr::from_ptr(bytes.as_ptr().cast())
                .to_string_lossy()
                .into_owned())
        }
    }
}

struct ImageModel<'a> {
    api: &'a Api,
    ctx: *mut c_void,
}

impl<'a> ImageModel<'a> {
    fn load(api: &'a Api, weights: &Path) -> Result<Self> {
        let paths = [
            weights.join("qwen-image-2.1-turbo-Q4_K_M.gguf"),
            weights.join("vae/qwen_image_2.1_vae_bf16.safetensors"),
            weights.join("Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"),
        ];
        for path in &paths {
            ensure!(
                path.is_file(),
                "missing model component: {}",
                path.display()
            );
        }
        let paths: Vec<_> = paths
            .iter()
            .map(|p| CString::new(p.as_os_str().as_encoded_bytes()))
            .collect::<std::result::Result<_, _>>()?;
        let mut error = [0 as c_char; 2048];
        // SAFETY: all strings/buffers live through the synchronous load call.
        let ctx = unsafe {
            (api.load)(
                paths[0].as_ptr(),
                paths[1].as_ptr(),
                paths[2].as_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ensure!(!ctx.is_null(), "image load failed: {}", error_text(&error));
        Ok(Self { api, ctx })
    }

    fn generate(&self, prompt: &str, progress: &Progress) -> Result<(Vec<u8>, u32)> {
        let prompt = CString::new(prompt)?;
        let mut error = [0 as c_char; 2048];
        let mut image = OwnedImage {
            api: self.api,
            raw: NativeImage::default(),
        };
        // SAFETY: this synchronous call completes all callback and image work
        // before returning. Progress stays alive, and the guard frees output
        // with the allocating library on both success and validation failure.
        let ok = unsafe {
            (self.api.generate)(
                self.ctx,
                prompt.as_ptr(),
                42,
                on_progress,
                (progress as *const Progress).cast_mut().cast(),
                &mut image.raw,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        ensure!(ok, "image generation failed: {}", error_text(&error));
        let raw = &image.raw;
        ensure!(
            raw.width == 1024 && raw.height == 1024,
            "unexpected image dimensions"
        );
        ensure!(
            matches!(raw.channels, 3 | 4),
            "unexpected channel count: {}",
            raw.channels
        );
        ensure!(
            !raw.pixels.is_null() && !raw.owner.is_null() && raw.count == 1,
            "invalid native image ownership"
        );
        let len = 1024 * 1024 * raw.channels as usize;
        // SAFETY: ABI-1 promises width*height*channels initialized bytes owned
        // by raw.owner until free_image, and the dimensions above bound length.
        let pixels = unsafe { std::slice::from_raw_parts(raw.pixels, len) }.to_vec();
        ensure!(
            pixels
                .chunks_exact(raw.channels as usize)
                .any(|pixel| pixel[..3] != pixels[..3]),
            "image is a constant color"
        );
        Ok((pixels, raw.channels))
    }
}

impl Drop for ImageModel<'_> {
    fn drop(&mut self) {
        // SAFETY: ctx is exclusively owned, no generation is in flight.
        unsafe { (self.api.unload)(self.ctx) };
    }
}

struct OwnedImage<'a> {
    api: &'a Api,
    raw: NativeImage,
}
impl Drop for OwnedImage<'_> {
    fn drop(&mut self) {
        // SAFETY: output is null or allocated by this exact bridge library.
        unsafe { (self.api.free_image)(&mut self.raw) };
    }
}

fn error_text(buffer: &[c_char]) -> String {
    // SAFETY: zero-initialized buffers; bridge always terminates its errors.
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

struct Progress {
    started: Instant,
    steps: Arc<AtomicUsize>,
    sender: mpsc::SyncSender<()>,
}

unsafe extern "C" fn on_progress(step: c_int, total: c_int, seconds: f32, user: *mut c_void) {
    // SAFETY: generate passes a live Progress and clears the callback before
    // returning. Only atomics/shared channel operations occur in this callback.
    let progress = unsafe { &*user.cast::<Progress>() };
    if step > 0 {
        progress.steps.fetch_max(step as usize, Ordering::Relaxed);
    }
    if step == 1 {
        let _ = progress.sender.try_send(());
    }
    // Avoid unwinding across C if stderr has been closed.
    use std::io::Write;
    let _ = writeln!(
        std::io::stderr(),
        "image step {step}/{total}: {seconds:.2}s, elapsed {:.2}s",
        progress.started.elapsed().as_secs_f64()
    );
}

fn ask(model: &gen2::Model, session: &mut Session, prompt: &str) -> Result<serde_json::Value> {
    let started = Instant::now();
    let response = model
        .turn(session)
        .user(prompt)
        .reasoning(ThinkingMode::Off)
        .temperature(0.0)
        .max_tokens(64)
        .run()?;
    let text = response.text();
    ensure!(
        text.to_uppercase().contains("MAPLE"),
        "llama session recall failed: {text:?}"
    );
    let resident = model.runtime().residency().is_resident(model.id());
    ensure!(resident, "llama model was evicted");
    Ok(
        json!({"text": text, "seconds": started.elapsed().as_secs_f64(), "resident": resident,
        "completion_tokens": response.usage().completion_tokens}),
    )
}

fn main() -> Result<()> {
    ensure!(
        cfg!(all(target_os = "macos", target_arch = "aarch64")),
        "requires Apple silicon macOS"
    );
    let mut args = Vec::new();
    let mut prompt = PROMPT.to_owned();
    let mut output_override = None;
    let mut cli = std::env::args_os().skip(1);
    while let Some(arg) = cli.next() {
        match arg.to_str() {
            Some("--help" | "-h") => {
                println!(
                    "Usage: gen2-qwen-image-metal-probe [--prompt TEXT] [--output DIR] [BRIDGE_DYLIB IMAGE_WEIGHTS_DIR LLAMA_GGUF [OUTPUT_DIR]]\n\nWith no paths, uses the bridge built in this checkout and models under $HOME/models.\nGenerates a 1024x1024 image.png and report.json in a fresh output directory."
                );
                return Ok(());
            }
            Some("--prompt") => {
                prompt = cli
                    .next()
                    .context("--prompt needs text")?
                    .into_string()
                    .map_err(|_| anyhow::anyhow!("prompt must be UTF-8"))?;
                ensure!(!prompt.trim().is_empty(), "prompt must not be empty");
                CString::new(prompt.as_str()).context("prompt contains a NUL byte")?;
            }
            Some("--output") => {
                output_override = Some(PathBuf::from(
                    cli.next().context("--output needs a directory")?,
                ));
            }
            Some(flag) if flag.starts_with('-') => anyhow::bail!("unknown option: {flag}"),
            _ => args.push(arg),
        }
    }
    ensure!(
        args.is_empty() || (3..=4).contains(&args.len()),
        "provide all three model/bridge paths or none; see --help"
    );
    ensure!(
        args.len() != 4 || output_override.is_none(),
        "specify the output directory only once"
    );
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    if args.is_empty() {
        let user_home = PathBuf::from(
            std::env::var_os("HOME").context("HOME is not set; provide explicit model paths")?,
        );
        args = vec![
            root.join("target/qwen-image-metal/build/bin/libgen2_qwen_probe.dylib")
                .into_os_string(),
            std::env::var_os("GEN2_QWEN_MODELS_DIR").unwrap_or_else(|| {
                user_home
                    .join("models/qwen-image-2.1-turbo")
                    .into_os_string()
            }),
            user_home
                .join("models/qwen3-0.6b-gguf/Qwen3-0.6B-Q4_K_M.gguf")
                .into_os_string(),
        ];
    }
    let bridge = PathBuf::from(&args[0]).canonicalize()?;
    let weights = PathBuf::from(&args[1]).canonicalize()?;
    let llama = PathBuf::from(&args[2]).canonicalize()?;
    let revision = std::process::Command::new("git")
        .arg("-C")
        .arg(&root)
        .args(["describe", "--always", "--dirty", "--abbrev=40"])
        .output()?;
    ensure!(
        revision.status.success(),
        "could not identify gen2 revision"
    );
    let revision = String::from_utf8(revision.stdout)?.trim().to_owned();
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let output = output_override
        .or_else(|| args.get(3).map(PathBuf::from))
        .unwrap_or_else(|| root.join(format!("target/qwen-image-metal/run-{timestamp}")));
    std::fs::create_dir_all(&output)?;
    ensure!(
        !output.join("report.json").exists() && !output.join("image.png").exists(),
        "use a fresh output directory"
    );

    let runtime = Runtime::new()?;
    let model = runtime.load(&llama)?;
    let info = model.info();
    let offload = info.offload.context("llama placement not reported")?;
    ensure!(
        offload.on_gpu()
            && offload
                .backend
                .as_deref()
                .is_some_and(|b| b == "MTL" || b.to_lowercase().contains("metal")),
        "llama did not load on Metal: {offload}"
    );
    eprintln!("llama placement: {offload}");
    let mut session = Session::new();
    let before = ask(
        &model,
        &mut session,
        "The secret word is MAPLE. Remember it. Reply with only the secret word.",
    )?;
    eprintln!("llama before SD load: {before}");

    let api = Api::open(&bridge)?;
    let devices = api.devices()?;
    ensure!(
        devices.to_lowercase().contains("metal") || devices.to_lowercase().contains("mtl"),
        "SD has no Metal device: {devices}"
    );
    eprintln!("SD devices: {devices}");
    let load_start = Instant::now();
    let image_model = ImageModel::load(&api, &weights)?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    ensure!(
        runtime.residency().is_resident(model.id()),
        "llama evicted while loading image weights"
    );
    eprintln!("image bundle loaded in {load_seconds:.2}s; llama still resident");

    let started = Instant::now();
    let (tx, rx) = mpsc::sync_channel(1);
    let steps = Arc::new(AtomicUsize::new(0));
    let progress = Progress {
        started,
        steps: Arc::clone(&steps),
        sender: tx,
    };
    let (generated, concurrent, generation_seconds) = std::thread::scope(|scope| {
        let model = &model;
        let observed_steps = Arc::clone(&steps);
        let worker = scope.spawn(move || -> Result<_> {
            rx.recv_timeout(Duration::from_secs(900))
                .context("no first denoising step received")?;
            let at_start = observed_steps.load(Ordering::Relaxed);
            let mut result = ask(
                model,
                &mut session,
                "What is the secret word? Reply with only that word.",
            )?;
            let at_end = observed_steps.load(Ordering::Relaxed);
            ensure!(
                (1..8).contains(&at_start) && at_end < 8,
                "llama turn did not finish during denoising: steps {at_start}..{at_end}"
            );
            result["denoising_step_at_start"] = json!(at_start);
            result["denoising_step_at_end"] = json!(at_end);
            eprintln!("llama during denoising: {result}");
            Ok((session, result))
        });
        let generated = image_model.generate(&prompt, &progress);
        let generation_seconds = started.elapsed().as_secs_f64();
        let steps = progress.steps.load(Ordering::Relaxed);
        drop(progress); // Wake the receiver on failures before the first step.
        (
            generated.map(|image| (image, steps)),
            worker.join(),
            generation_seconds,
        )
    });
    let ((pixels, channels), steps) = generated?;
    ensure!(steps == 8, "expected 8 denoising steps, observed {steps}");
    let (mut session, during) =
        concurrent.map_err(|_| anyhow::anyhow!("llama worker panicked"))??;
    let after = ask(
        &model,
        &mut session,
        "Repeat the secret word, and nothing else.",
    )?;
    eprintln!("llama with SD still loaded: {after}");
    drop(image_model);
    drop(api); // Exercise native context and dylib teardown, then llama again.
    let after_unload = ask(
        &model,
        &mut session,
        "What secret word did I give you? Reply with that word only.",
    )?;
    eprintln!("llama after SD unload: {after_unload}");
    ensure!(
        runtime.stats().evictions == 0,
        "runtime evicted a model during qualification"
    );

    let color = if channels == 4 {
        image::ColorType::Rgba8
    } else {
        image::ColorType::Rgb8
    };
    let png = output.join("image.png");
    image::save_buffer_with_format(&png, &pixels, 1024, 1024, color, image::ImageFormat::Png)?;
    let decoded = image::open(&png)?;
    ensure!(
        decoded.width() == 1024 && decoded.height() == 1024,
        "saved PNG dimensions differ"
    );
    let png_hash: String = Sha256::digest(std::fs::read(&png)?)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let report = json!({
        "passed": true, "pid": std::process::id(), "unix_timestamp": timestamp,
        "mode": "in-process, private GGML dylib, concurrent llama turn at first denoising step",
        "gen2_revision": revision, "sd_revision": "68285df319457e2d79310b48513e4ba1bf8ebe9a",
        "hardware": runtime.hardware(), "runtime_evictions": runtime.stats().evictions,
        "bridge": bridge, "weights": weights, "llama_model": llama,
        "sd_devices": devices, "llama_placement": offload.to_string(),
        "prompt": prompt, "seed": 42, "width": 1024, "height": 1024, "channels": channels,
        "steps": steps, "cfg": 1.0, "sampler": "euler",
        "sigmas": [1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0],
        "load_seconds": load_seconds, "generation_seconds": generation_seconds,
        "llama_before": before, "llama_during": during, "llama_after": after,
        "llama_after_sd_unload": after_unload,
        "image": png.canonicalize()?, "image_sha256": png_hash,
        "limits": "Standalone qualification harness; image allocations are outside gen2 runtime admission. No automatic bundle downloads or public image API."
    });
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
