use std::ffi::CString;

use llama_cpp_2::mtmd::{
    MtmdBitmap, MtmdContext, MtmdInputChunks, MtmdInputText, MtmdVideoBitmap, MtmdVideoPart,
};

use crate::engine::ExecError;
use crate::input::Video;
use crate::session_rt::media_util::{decode_limits, media_parts, validate_image_path};
use crate::types::message::{Message, MessageChunk};

const PCM_SAMPLE_RATE: usize = 16_000;
const MAX_AUDIO_SAMPLES: usize = 192 * PCM_SAMPLE_RATE;
const MAX_VIDEO_FRAMES: usize = 384;
const MAX_RESIZED_VIDEO_PIXELS: u64 = 32 * 1024 * 1024;

enum Attachment {
    Bitmap(MtmdBitmap),
    Video(MtmdVideoBitmap),
}

pub(super) fn tokenize(
    ctx: &MtmdContext,
    prompt: String,
    marker: &str,
    messages: &[Message],
    add_special: bool,
) -> Result<MtmdInputChunks, ExecError> {
    let parts = media_parts(messages).collect::<Vec<_>>();
    if prompt.matches(marker).count() != parts.len() {
        return Err(ExecError::InvalidArg(
            "media marker count does not match attachments",
        ));
    }
    let attachments = parts
        .into_iter()
        .map(|part| match part {
            MessageChunk::ImageUrl { image_url } => {
                let path = validate_image_path(&image_url.url)?;
                let bitmap = MtmdBitmap::from_file(ctx, &path, false)
                    .map_err(|err| ExecError::Other(err.into()))?;
                Ok(Attachment::Bitmap(bitmap))
            }
            MessageChunk::AudioPcm { samples } => {
                validate_pcm(samples, MAX_AUDIO_SAMPLES)?;
                let bitmap = MtmdBitmap::from_audio_data(samples)
                    .map_err(|err| ExecError::Other(err.into()))?;
                Ok(Attachment::Bitmap(bitmap))
            }
            MessageChunk::Video { video } => prepare_video(ctx, video).map(Attachment::Video),
            MessageChunk::Text { .. } => unreachable!("media parts excludes text"),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let bitmaps = attachments
        .iter()
        .map(|attachment| match attachment {
            Attachment::Bitmap(bitmap) => bitmap,
            Attachment::Video(video) => video.as_bitmap(),
        })
        .collect::<Vec<_>>();
    ctx.tokenize(
        MtmdInputText {
            text: prompt,
            add_special,
            parse_special: true,
        },
        &bitmaps,
    )
    .map_err(|err| ExecError::Other(err.into()))
}

fn validate_pcm(samples: &[f32], max_samples: usize) -> Result<(), ExecError> {
    if !(320..=max_samples).contains(&samples.len())
        || samples.iter().any(|sample| !(-1.0..=1.0).contains(sample))
    {
        return Err(ExecError::InvalidArg(
            "expected finite mono 16 kHz PCM with at least 20 ms and a supported duration",
        ));
    }
    Ok(())
}

fn prepare_video(ctx: &MtmdContext, video: &Video) -> Result<MtmdVideoBitmap, ExecError> {
    const SAMPLES_PER_FRAME: usize = PCM_SAMPLE_RATE / 2;
    let frames = video.frames();
    let pcm = video.soundtrack();
    if !(1..=MAX_VIDEO_FRAMES).contains(&frames.len()) {
        return Err(ExecError::InvalidArg(
            "video requires 1 to 384 frames at 2 fps",
        ));
    }
    if !pcm.is_empty() {
        validate_pcm(pcm, frames.len() * SAMPLES_PER_FRAME)?;
    }

    let mut expected = None;
    let mut parts = Vec::new();
    for (i, group) in frames.chunks(2).enumerate() {
        // Timestamp each temporal pair at its midpoint, as in llama.cpp's
        // LFM3 video template. An odd final frame is its own midpoint.
        let seconds = i as f32 + if group.len() == 2 { 0.25 } else { 0.0 };
        let stamp = CString::new(format!("<{seconds:.1} seconds>"))
            .map_err(|err| ExecError::Other(err.into()))?;
        parts.push(MtmdVideoPart::Timestamp(stamp));

        for frame in group {
            let path = validate_image_path(frame.source())?;
            let mut reader = image::ImageReader::open(&path)
                .and_then(image::ImageReader::with_guessed_format)
                .map_err(|err| ExecError::Io(format!("open video frame {path}: {err}")))?;
            reader.limits(decode_limits());
            let rgb = reader
                .decode()
                .map_err(|err| ExecError::Other(err.into()))?
                .into_rgb8();
            let dimensions = rgb.dimensions();
            let size = match expected {
                Some((input, output)) if input == dimensions => output,
                Some(_) => {
                    return Err(ExecError::InvalidArg(
                        "video frames must have equal dimensions",
                    ));
                }
                None => {
                    let output = ctx
                        .lfm3_video_size(dimensions.0, dimensions.1, frames.len())
                        .ok_or(ExecError::InvalidArg("unsupported video frame dimensions"))?;
                    if u64::from(output.0) * u64::from(output.1) * frames.len() as u64
                        > MAX_RESIZED_VIDEO_PIXELS
                    {
                        return Err(ExecError::InvalidArg("resized video exceeds pixel limit"));
                    }
                    expected = Some((dimensions, output));
                    output
                }
            };
            let (width, height) = size;
            let rgb = if dimensions == size {
                rgb
            } else {
                image::imageops::resize(
                    &rgb,
                    width,
                    height,
                    image::imageops::FilterType::CatmullRom,
                )
            };
            parts.push(MtmdVideoPart::Frame {
                width,
                height,
                rgb: rgb.into_raw(),
            });
        }

        let start = i * 2 * SAMPLES_PER_FRAME;
        let end = (start + group.len() * SAMPLES_PER_FRAME).min(pcm.len());
        if start < end {
            parts.push(MtmdVideoPart::Soundtrack(pcm[start..end].to_vec()));
        }
    }

    MtmdVideoBitmap::new(ctx, parts).map_err(|err| ExecError::Other(err.into()))
}
