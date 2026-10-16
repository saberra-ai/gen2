//! N0 diagnostic, not the Gen2 diarization API. Run in a fresh process per order.
use libloading::Library;
use std::{
    ffi::{CStr, CString, c_char, c_void},
    path::Path,
    sync::{Arc, Barrier, mpsc},
};

struct Diarizer {
    model: *mut c_void,
    run: unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char) -> i32,
    close: unsafe extern "C" fn(*mut c_void),
    error: unsafe extern "C" fn() -> *const c_char,
    _library: Library,
}
impl Diarizer {
    fn open(library: &str, model: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let model = CString::new(model)?;
        // SAFETY: this N0 shim is compiled from the checked-in source against
        // the locked header. Library lifetime covers all handles and symbols.
        unsafe {
            let library = Library::new(library)?;
            let open = library
                .get::<unsafe extern "C" fn(*const c_char, i32) -> *mut c_void>(b"n0_open\0")?;
            let run = *library.get(b"n0_run\0")?;
            let close = *library.get(b"n0_close\0")?;
            let error: unsafe extern "C" fn() -> *const c_char =
                *library.get(b"n0_last_error\0")?;
            let pointer = open(model.as_ptr(), -1);
            if pointer.is_null() {
                return Err(CStr::from_ptr(error())
                    .to_string_lossy()
                    .into_owned()
                    .into());
            }
            Ok(Self {
                model: pointer,
                run,
                close,
                error,
                _library: library,
            })
        }
    }
    fn run(
        &self,
        wav: &str,
        output: &Path,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let wav = CString::new(wav)?;
        let out = CString::new(output.to_str().ok_or("non-UTF8 output path")?)?;
        // SAFETY: calls stay on this thread; the model, library and strings are live.
        unsafe {
            if (self.run)(self.model, wav.as_ptr(), out.as_ptr()) != 0 {
                return Err(CStr::from_ptr((self.error)())
                    .to_string_lossy()
                    .into_owned()
                    .into());
            }
        }
        Ok(serde_json::from_slice(&std::fs::read(output)?)?)
    }
}
impl Drop for Diarizer {
    fn drop(&mut self) {
        // SAFETY: the stream was closed synchronously by the shim; library is live.
        unsafe { (self.close)(self.model) }
    }
}

fn chat(model: &gen2::Model) -> gen2::Result<String> {
    Ok(model
        .generate("Reply with the single word hello.")
        .max_tokens(8)
        .temperature(0.)
        .run()?
        .text())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 || !["chat-first", "diar-first"].contains(&args[1].as_str()) {
        return Err(
            "usage: probe chat-first|diar-first SHIM MODEL WAV CHAT_GGUF OUTPUT_DIR".into(),
        );
    }
    let output = Path::new(&args[6]);
    std::fs::create_dir_all(output)?;
    let runtime = gen2::Runtime::builder()
        .resident_memory_budget_mb(8192)
        .build()?;
    let (model, diarizer, isolated_chat) = if args[1] == "chat-first" {
        let model = runtime.load(&args[5])?;
        let baseline = chat(&model)?;
        (model, Diarizer::open(&args[2], &args[3])?, Some(baseline))
    } else {
        let diarizer = Diarizer::open(&args[2], &args[3])?;
        (runtime.load(&args[5])?, diarizer, None)
    };
    let chat_before = chat(&model)?;
    let first = diarizer.run(&args[4], &output.join("first.json"))?;
    let chat_after = chat(&model)?;
    let second = diarizer.run(&args[4], &output.join("second.json"))?;
    drop(diarizer);
    let reloaded = Diarizer::open(&args[2], &args[3])?;
    let third = reloaded.run(&args[4], &output.join("reloaded.json"))?;
    let concurrent = std::thread::scope(|scope| -> Result<_, Box<dyn std::error::Error>> {
        let barrier = Arc::new(Barrier::new(2));
        let worker_barrier = Arc::clone(&barrier);
        let (ready_tx, ready_rx) = mpsc::channel();
        let args = &args;
        let worker = scope.spawn(move || -> Result<_, String> {
            let other = Diarizer::open(&args[2], &args[3]).map_err(|e| e.to_string())?;
            ready_tx.send(()).map_err(|e| e.to_string())?;
            worker_barrier.wait();
            other
                .run(&args[4], &output.join("concurrent.json"))
                .map_err(|e| e.to_string())
        });
        ready_rx.recv()?;
        barrier.wait();
        let text = chat(&model)?;
        let result = worker.join().map_err(|_| "native worker panicked")??;
        if text != chat_before {
            return Err("concurrent native execution changed the chat output".into());
        }
        Ok(result)
    })?;
    drop(reloaded);
    let chat_after_unload = chat(&model)?;
    if isolated_chat
        .as_ref()
        .is_some_and(|text| text != &chat_before)
        || chat_before.is_empty()
        || chat_before != chat_after
        || chat_before != chat_after_unload
        || first["segments"] != second["segments"]
        || first["segments"] != third["segments"]
        || first["segments"] != concurrent["segments"]
    {
        return Err("coexistence changed chat or repeated diarization output".into());
    }
    println!(
        "{}",
        serde_json::json!({"order":args[1], "chat_before":chat_before,
        "chat_after":chat_after, "isolated_chat":isolated_chat,
        "chat_after_unload":chat_after_unload, "repeated_and_reloaded_segments_equal":true,
        "concurrent_chat_and_diarization_equal":true})
    );
    Ok(())
}
