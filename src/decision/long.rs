use super::*;
use serde::{Deserialize, Serialize};
/// Scan policy. A scan fails if its configured limit cannot cover every token.
#[derive(Debug, Clone)]
pub struct LongStateOptions {
    pub window_tokens: Option<usize>,
    pub stride_tokens: Option<usize>,
    pub max_windows: usize,
}
impl Default for LongStateOptions {
    fn default() -> Self {
        Self {
            window_tokens: None,
            stride_tokens: None,
            max_windows: 256,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowAnswer {
    pub id: String,
    pub answer: Answer,
    pub window_index: usize,
    pub token_start: usize,
    pub token_end: usize,
}
/// Selected-window probabilities are not calibrated whole-document confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongDecisionResult {
    pub answers: Vec<WindowAnswer>,
    pub windows: Vec<DecisionResult>,
    pub state_tokens: usize,
    pub window_tokens: usize,
    pub stride_tokens: usize,
    /// Queue time for the complete scan; nested windows are not queued again.
    pub queue_micros: u64,
    /// Complete scan time, including source tokenization and all windows.
    pub execution_micros: u64,
}
pub(crate) fn scan_encode(
    envelope: &SequenceEnvelope,
    options: &DecisionOptions,
) -> Result<EncodeOptions> {
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
    Ok(encode)
}
#[cfg(feature = "backend-laya-onnx")]
pub(crate) fn scan(
    tokenizer: &LayaTokenizer,
    envelope: &SequenceEnvelope,
    request: DecisionRequest,
    scan: LongStateOptions,
    options: DecisionOptions,
    mut run: impl FnMut(DecisionRequest, DecisionOptions) -> Result<DecisionResult>,
) -> Result<LongDecisionResult> {
    options.check()?;
    if request.questions.is_empty() {
        return Ok(LongDecisionResult {
            answers: Vec::new(),
            windows: Vec::new(),
            state_tokens: 0,
            window_tokens: 0,
            stride_tokens: 0,
            queue_micros: 0,
            execution_micros: 0,
        });
    }
    let mut encode = scan_encode(envelope, &options)?;
    // Windows must fit fully after decoding and retokenizing. The caller's
    // truncation policy cannot silently make a scan lose evidence.
    encode.overflow = OverflowPolicy::Reject;
    let empty = DecisionRequest {
        state: DecisionState::Text(String::new()),
        questions: request.questions.clone(),
    };
    let heads = tokenizer.encode(&empty, &encode)?;
    let room = heads
        .iter()
        .map(|h| encode.max_len.saturating_sub(h.input_ids.len()))
        .min()
        .unwrap_or(encode.max_len);
    let requested = scan.window_tokens.unwrap_or(
        encode
            .max_len
            .saturating_sub(encode.head_max_len.saturating_add(8))
            .max(64),
    );
    let window = requested.min(room);
    let stride = scan.stride_tokens.unwrap_or((window / 2).max(1));
    if window == 0 || stride == 0 || stride > window || scan.max_windows == 0 {
        return Err(DecisionError::InvalidRequest(
            "long-state window, stride or limit is invalid".into(),
        ));
    }
    let tokens = tokenizer.state_ids(&request.state)?;
    options.check()?;
    let count = if tokens.len() <= window {
        1
    } else {
        1 + (tokens.len() - window).div_ceil(stride)
    };
    if count > scan.max_windows {
        return Err(DecisionError::ResourceLimit(format!(
            "scan requires {count} windows; limit is {}",
            scan.max_windows
        )));
    }
    let mut windows = Vec::new();
    let mut selected: Vec<WindowAnswer> = Vec::new();
    for i in 0..count {
        options.check()?;
        let start = i * stride;
        let end = (start + window).min(tokens.len());
        let state = if count == 1 {
            request.state.clone()
        } else {
            DecisionState::Text(tokenizer.decode_state(&tokens[start..end])?)
        };
        let started = std::time::Instant::now();
        let mut result = run(
            DecisionRequest {
                state,
                questions: request.questions.clone(),
            },
            DecisionOptions {
                encode: Some(encode.clone()),
                ..options.clone()
            },
        )?;
        result.queue_micros = 0;
        result.execution_micros = started.elapsed().as_micros().min(u64::MAX as u128) as u64;
        for (j, (id, answer)) in result.answers.iter().enumerate() {
            let score = |a: &Answer| match a.value {
                AnswerValue::YesNo { probability_true } => probability_true,
                _ => a.answer_confidence,
            };
            if i == 0 {
                selected.push(WindowAnswer {
                    id: id.clone(),
                    answer: answer.clone(),
                    window_index: i,
                    token_start: start,
                    token_end: end,
                });
            } else if score(answer) > score(&selected[j].answer) {
                selected[j] = WindowAnswer {
                    id: id.clone(),
                    answer: answer.clone(),
                    window_index: i,
                    token_start: start,
                    token_end: end,
                };
            }
        }
        windows.push(result);
    }
    Ok(LongDecisionResult {
        answers: selected,
        windows,
        state_tokens: tokens.len(),
        window_tokens: window,
        stride_tokens: stride,
        queue_micros: 0,
        execution_micros: 0,
    })
}
