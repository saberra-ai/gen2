use super::{worker::DecisionBackend, *};
use ort::{session::Session, value::Tensor};
pub(crate) struct OnnxBackend {
    bundle: LayaBundle,
    tokenizer: std::sync::Arc<LayaTokenizer>,
    session: Session,
    execution: ExecutionOptions,
}
fn native(e: impl std::fmt::Display) -> DecisionError {
    DecisionError::Execution(e.to_string())
}
impl OnnxBackend {
    pub(crate) fn load(bundle: LayaBundle, options: LoadOptions) -> Result<Self> {
        let _span = tracing::info_span!("laya_load", manifest = bundle.manifest_sha256(), provider = ?options.execution).entered();
        #[cfg(feature = "laya-dynamic")]
        {
            // ORT is process-global. Refuse a different library after the first
            // successful initialization, rather than silently using the old one.
            static LIBRARY: std::sync::Mutex<Option<std::path::PathBuf>> =
                std::sync::Mutex::new(None);
            let path = options
                .native_library
                .as_ref()
                .ok_or_else(|| {
                    DecisionError::BackendUnavailable(
                        "set LoadOptions.native_library to the packaged ONNX Runtime 1.24 library"
                            .into(),
                    )
                })?
                .canonicalize()
                .map_err(|e| DecisionError::BackendUnavailable(e.to_string()))?;
            let mut loaded = LIBRARY.lock().map_err(|_| {
                DecisionError::BackendUnavailable("ONNX initialization lock poisoned".into())
            })?;
            if loaded.as_ref().is_some_and(|p| p != &path) {
                return Err(DecisionError::BackendUnavailable(
                    "another ONNX library is already initialized in this process".into(),
                ));
            }
            if loaded.is_none() {
                ort::init_from(&path)
                    .map_err(|e| DecisionError::BackendUnavailable(e.to_string()))?
                    .commit();
                *loaded = Some(path);
            }
        }
        let m = bundle.manifest();
        let tokenizer =
            LayaTokenizer::from_file(bundle.path(&m.tokenizer)?, m.special_tokens.clone())?;
        let mut builder = Session::builder()
            .map_err(native)?
            .with_intra_threads(options.intra_threads)
            .map_err(native)?;
        use ort::ep::{self, ExecutionProvider as _};
        let provider = match options.execution.provider {
            ExecutionProvider::Cpu => ep::CPU::default().build(),
            ExecutionProvider::DirectMl { device_id } => {
                if !cfg!(target_os = "windows") {
                    return Err(DecisionError::BackendUnavailable(
                        "DirectML requires Windows".into(),
                    ));
                }
                builder = builder
                    .with_parallel_execution(false)
                    .map_err(native)?
                    .with_memory_pattern(false)
                    .map_err(native)?;
                ep::DirectML::default().with_device_id(device_id).build()
            }
            ExecutionProvider::Cuda { device_id } => ep::CUDA::default()
                .with_device_id(device_id)
                .with_tf32(false)
                .build(),
            ExecutionProvider::CoreMl => {
                if !ep::CoreML::default().supported_by_platform() {
                    return Err(DecisionError::BackendUnavailable(
                        "CoreML requires an Apple platform".into(),
                    ));
                }
                ep::CoreML::default()
                    .with_low_precision_accumulation_on_gpu(false)
                    .build()
            }
            ExecutionProvider::Nnapi => {
                if !ep::NNAPI::default().supported_by_platform() {
                    return Err(DecisionError::BackendUnavailable(
                        "NNAPI requires Android".into(),
                    ));
                }
                ep::NNAPI::default().with_fp16(false).build()
            }
        };
        builder = builder
            .with_execution_providers([provider.error_on_failure()])
            .map_err(|e| DecisionError::BackendUnavailable(e.to_string()))?;
        if options.execution.provider != ExecutionProvider::Cpu
            && !options.execution.allow_cpu_fallback
        {
            builder = builder.with_disable_cpu_fallback().map_err(native)?;
        }
        let session = builder
            .commit_from_file(bundle.path(&m.graph)?)
            .map_err(native)?;
        tracing::info!(runtime_build = ort::info(), "Laya native session ready");
        Ok(Self {
            bundle,
            tokenizer: std::sync::Arc::new(tokenizer),
            session,
            execution: options.execution,
        })
    }
}
impl DecisionBackend for OnnxBackend {
    fn run_long(
        &mut self,
        request: DecisionRequest,
        scan: LongStateOptions,
        options: DecisionOptions,
    ) -> Result<LongDecisionResult> {
        let tokenizer = self.tokenizer.clone();
        let envelope = self.bundle.manifest().envelope.clone();
        super::long::scan(
            &tokenizer,
            &envelope,
            request,
            scan,
            options,
            |request, options| {
                self.run_batch(&[request], &options)?
                    .pop()
                    .ok_or_else(|| DecisionError::InvalidOutput("missing window result".into()))
            },
        )
    }
    fn run_batch(
        &mut self,
        requests: &[DecisionRequest],
        options: &DecisionOptions,
    ) -> Result<Vec<DecisionResult>> {
        options.check()?;
        let envelope = &self.bundle.manifest().envelope;
        let encode = options.encode.clone().unwrap_or(EncodeOptions {
            max_len: envelope.max_len,
            head_max_len: envelope.head_max_len,
            overflow: OverflowPolicy::Reject,
        });
        if encode.max_len > envelope.max_len || encode.head_max_len > envelope.head_max_len {
            return Err(DecisionError::ResourceLimit(
                "requested sequence budget exceeds the exported envelope".into(),
            ));
        }
        // Validate and encode the entire job before issuing any native work.
        // Indices map flattened rows back to their original state and question.
        let mut rows = Vec::new();
        let mut positions = Vec::new();
        let mut results = Vec::with_capacity(requests.len());
        let encoding_span = tracing::debug_span!("laya_encode", states = requests.len()).entered();
        for (state, request) in requests.iter().enumerate() {
            options.check()?;
            let encoded = self.tokenizer.encode(request, &encode)?;
            if encoded
                .iter()
                .any(|r| r.marker_pos.len() > envelope.max_options)
            {
                return Err(DecisionError::ResourceLimit(
                    "option count exceeds the exported envelope".into(),
                ));
            }
            positions.extend((0..encoded.len()).map(|question| (state, question)));
            rows.extend(encoded);
            results.push(DecisionResult {
                answers: Vec::with_capacity(request.questions.len()),
                inputs: Vec::with_capacity(request.questions.len()),
                manifest_sha256: self.bundle.manifest_sha256().into(),
                checkpoint: self.bundle.manifest().checkpoint,
                calibration_diagnostics: self.bundle.calibration().diagnostics.clone(),
                execution: self.execution.clone(),
                queue_micros: 0,
                execution_micros: 0,
            });
        }
        drop(encoding_span);
        for (chunk_index, chunk) in rows.chunks(envelope.max_batch).enumerate() {
            options.check()?;
            let b = chunk.len();
            let s = chunk.iter().map(|r| r.input_ids.len()).max().unwrap_or(0);
            let k = chunk.iter().map(|r| r.marker_pos.len()).max().unwrap_or(0);
            if k > envelope.max_options {
                return Err(DecisionError::ResourceLimit(
                    "option count exceeds the exported envelope".into(),
                ));
            }
            // The traced action head uses topk(2). A one-option request gets
            // one masked padding slot; decoding still consumes exactly one.
            let k = k.max(2);
            let bs = b
                .checked_mul(s)
                .ok_or_else(|| DecisionError::ResourceLimit("tensor size overflow".into()))?;
            let bk = b
                .checked_mul(k)
                .ok_or_else(|| DecisionError::ResourceLimit("tensor size overflow".into()))?;
            let mut ids = vec![i64::from(self.tokenizer.special_tokens().pad); bs];
            let mut attention = vec![0i64; bs];
            let mut markers = vec![0i64; bk];
            let mut mask = vec![false; bk];
            let mut types = Vec::with_capacity(b);
            for (i, row) in chunk.iter().enumerate() {
                ids[i * s..i * s + row.input_ids.len()].copy_from_slice(&row.input_ids);
                attention[i * s..i * s + row.input_ids.len()].fill(1);
                markers[i * k..i * k + row.marker_pos.len()].copy_from_slice(&row.marker_pos);
                mask[i * k..i * k + row.marker_pos.len()].fill(true);
                types.push(row.qtype);
            }
            let forward_span =
                tracing::debug_span!("laya_forward", batch = b, sequence = s, options = k)
                    .entered();
            let output = self
                .session
                .run(ort::inputs![
                    "input_ids"=>Tensor::from_array(([b,s],ids)).map_err(native)?,
                    "attention_mask"=>Tensor::from_array(([b,s],attention)).map_err(native)?,
                    "marker_pos"=>Tensor::from_array(([b,k],markers)).map_err(native)?,
                    "marker_mask"=>Tensor::from_array(([b,k],mask)).map_err(native)?,
                    "qtype"=>Tensor::from_array(([b],types)).map_err(native)?
                ])
                .map_err(native)?;
            drop(forward_span);
            options.check()?;
            let _decode_span = tracing::debug_span!("laya_decode", batch = b).entered();
            let (shape, logits) = output["logits"]
                .try_extract_tensor::<f32>()
                .map_err(native)?;
            let (act_shape, act) = output["act_logits"]
                .try_extract_tensor::<f32>()
                .map_err(native)?;
            if shape.as_ref() != [b as i64, k as i64] || act_shape.as_ref() != [b as i64, 2] {
                return Err(DecisionError::InvalidOutput(
                    "graph returned wrong output dimensions".into(),
                ));
            }
            for (i, row) in chunk.iter().enumerate() {
                let (state, question_index) = positions[chunk_index * envelope.max_batch + i];
                let question = &requests[state].questions[question_index].1;
                let answer = self.bundle.calibration().decode(
                    question,
                    &logits[i * k..(i + 1) * k],
                    [act[i * 2], act[i * 2 + 1]],
                    options.language.as_deref(),
                )?;
                results[state].answers.push((row.id.clone(), answer));
            }
        }
        for (row, (state, _)) in rows.into_iter().zip(positions) {
            results[state].inputs.push(row);
        }
        Ok(results)
    }
}
