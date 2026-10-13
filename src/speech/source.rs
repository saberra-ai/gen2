use std::path::{Path, PathBuf};

/// Explicit architecture selection; a bundle never silently switches models.
#[derive(Debug, Clone)]
pub enum SpeechSource {
    BarkOnnx(PathBuf),
    WhistleOnnx(PathBuf),
}
impl SpeechSource {
    pub fn bark(path: impl AsRef<Path>) -> Self {
        Self::BarkOnnx(path.as_ref().into())
    }
    pub fn whistle(path: impl AsRef<Path>) -> Self {
        Self::WhistleOnnx(path.as_ref().into())
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum SpeechModel {
    Bark,
    Whistle,
}
impl SpeechModel {
    pub(crate) fn minimum_reservation_mb(self) -> u64 {
        match self {
            Self::Bark => 4096,
            Self::Whistle => 2048,
        }
    }
    pub fn supported_languages(self) -> &'static [&'static str] {
        match self {
            Self::Bark => &["en"],
            Self::Whistle => &["en", "de", "fr", "es", "it", "nl", "pl"],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SpeechLanguage {
    En,
    De,
    Fr,
    Es,
    It,
    Nl,
    Pl,
}
impl SpeechLanguage {
    #[cfg(feature = "backend-asr-onnx")]
    pub(crate) fn index(self) -> usize {
        self as usize
    }
    #[cfg(feature = "backend-asr-onnx")]
    pub(crate) fn from_index(i: usize) -> Self {
        [
            Self::En,
            Self::De,
            Self::Fr,
            Self::Es,
            Self::It,
            Self::Nl,
            Self::Pl,
        ][i]
    }
}

/// Shared request controls. Query the loaded transcriber's profile for support.
#[derive(Debug, Clone, Default)]
pub struct TranscriptionSettings {
    pub language: Option<SpeechLanguage>,
    /// 1–1024 text tokens. `None` uses the upstream duration-based cap (≤320).
    pub max_tokens: Option<usize>,
    /// Detailed calls may return explicitly marked partial text. Basic calls fail.
    pub allow_partial: bool,
}
impl TranscriptionSettings {
    pub(crate) fn validate_for(&self, model: SpeechModel) -> super::Result<()> {
        self.validate()?;
        if model == SpeechModel::Bark
            && (self.max_tokens.is_some() || self.language.is_some_and(|l| l != SpeechLanguage::En))
        {
            return Err(super::SpeechError::InvalidRequest(
                "Bark supports English and does not support a text-token limit".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn validate(&self) -> super::Result<()> {
        if self.max_tokens.is_some_and(|n| !(1..=1024).contains(&n)) {
            return Err(super::SpeechError::InvalidRequest(
                "max_tokens must be 1–1024 when supported by the model".into(),
            ));
        }
        Ok(())
    }
}

/// The same request interface serves every speech adapter.
#[derive(Debug, Clone)]
pub struct TranscriptionRequest {
    pub audio: super::Audio,
    pub options: super::TranscribeOptions,
    pub settings: TranscriptionSettings,
}
impl TranscriptionRequest {
    pub fn new(audio: super::Audio) -> Self {
        Self {
            audio,
            options: Default::default(),
            settings: Default::default(),
        }
    }
    pub fn language(mut self, language: SpeechLanguage) -> Self {
        self.settings.language = Some(language);
        self
    }
    pub fn max_tokens(mut self, limit: usize) -> Self {
        self.settings.max_tokens = Some(limit);
        self
    }
    pub fn allow_partial(mut self, allow: bool) -> Self {
        self.settings.allow_partial = allow;
        self
    }
    pub fn options(mut self, options: super::TranscribeOptions) -> Self {
        self.options = options;
        self
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SpeechProfile {
    pub model: SpeechModel,
    pub languages: &'static [&'static str],
    pub automatic_language: bool,
    pub maximum_text_tokens: Option<usize>,
    pub partial_transcripts: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum Completion {
    EndOfSpeech,
    TokenLimit,
}
#[derive(Debug, Clone, serde::Serialize)]
pub struct DetailedTranscription {
    pub transcription: super::Transcription,
    /// Forced language when supplied, otherwise the model's selected language.
    pub language: Option<SpeechLanguage>,
    pub completion: Completion,
    /// Independent language-ID decision; `None` for adapters without language ID.
    pub detected_language: Option<SpeechLanguage>,
}
