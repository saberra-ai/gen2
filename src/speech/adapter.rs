use super::*;

/// Only model-specific inference varies here. The worker owns all lifecycle.
pub(crate) enum Adapter {
    Bark(Box<super::onnx::OnnxBackend>),
    Whistle(Box<super::whistle::WhistleBackend>),
}
impl Adapter {
    pub(crate) fn load(bundle: &SpeechBundle, options: &LoadOptions) -> Result<Self> {
        match bundle.model() {
            SpeechModel::Bark => {
                super::onnx::OnnxBackend::load(bundle, options).map(|b| Self::Bark(Box::new(b)))
            }
            SpeechModel::Whistle => super::whistle::WhistleBackend::load(bundle, options)
                .map(|b| Self::Whistle(Box::new(b))),
        }
    }
    pub(crate) fn run(
        &mut self,
        audio: Audio,
        options: &TranscribeOptions,
        request: Option<&TranscriptionSettings>,
    ) -> Result<DetailedTranscription> {
        match self {
            Self::Bark(backend) => {
                if let Some(request) = request {
                    request.validate_for(SpeechModel::Bark)?;
                }
                backend
                    .run(audio, options)
                    .map(|transcription| DetailedTranscription {
                        transcription,
                        language: Some(SpeechLanguage::En),
                        detected_language: None,
                        completion: Completion::EndOfSpeech,
                    })
            }
            Self::Whistle(backend) => backend.run(
                audio,
                options,
                request.unwrap_or(&TranscriptionSettings::default()),
            ),
        }
    }
}
