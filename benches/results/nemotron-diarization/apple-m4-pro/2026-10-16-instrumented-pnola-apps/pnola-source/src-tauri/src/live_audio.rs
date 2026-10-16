//! A bounded PCM section writer. Each finalized WAV is published atomically;
//! the session never depends on one duration-limited RIFF file.
use std::{
    fs,
    path::{Path, PathBuf},
};
pub const RATE: u32 = 16_000;
pub const SECTION_FRAMES: usize = RATE as usize * 8;

pub struct Sections {
    directory: PathBuf,
    buffer: Vec<i16>,
    sequence: u64,
    pub frames: u64,
}
impl Sections {
    pub fn new(directory: &Path) -> Result<Self, String> {
        fs::create_dir_all(directory).map_err(|e| e.to_string())?;
        Ok(Self {
            directory: directory.into(),
            buffer: Vec::with_capacity(SECTION_FRAMES),
            sequence: 0,
            frames: 0,
        })
    }
    pub fn push(&mut self, samples: &[i16]) -> Result<(), String> {
        for &sample in samples {
            self.buffer.push(sample);
            self.frames += 1;
            if self.buffer.len() == SECTION_FRAMES {
                self.flush()?;
            }
        }
        Ok(())
    }
    pub fn flush(&mut self) -> Result<(), String> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let path = self.directory.join(format!("{:020}.wav", self.sequence));
        let temporary = path.with_extension("wav.tmp");
        let mut writer = hound::WavWriter::create(
            &temporary,
            hound::WavSpec {
                channels: 1,
                sample_rate: RATE,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .map_err(|e| e.to_string())?;
        for sample in &self.buffer {
            writer.write_sample(*sample).map_err(|e| e.to_string())?;
        }
        writer.finalize().map_err(|e| e.to_string())?;
        fs::File::open(&temporary)
            .and_then(|f| f.sync_all())
            .map_err(|e| e.to_string())?;
        fs::rename(temporary, path).map_err(|e| e.to_string())?;
        self.sequence += 1;
        self.buffer.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sections_continue_past_two_hours_and_preserve_the_tail() {
        let directory = tempfile::tempdir().unwrap();
        let mut writer = Sections::new(directory.path()).unwrap();
        // The counter can cross the old cap without allocating hours of PCM.
        writer.frames = 7200 * RATE as u64;
        writer.push(&vec![123; SECTION_FRAMES + 17]).unwrap();
        writer.flush().unwrap();
        assert_eq!(
            writer.frames,
            7200 * RATE as u64 + SECTION_FRAMES as u64 + 17
        );
        let a = hound::WavReader::open(directory.path().join("00000000000000000000.wav")).unwrap();
        let b = hound::WavReader::open(directory.path().join("00000000000000000001.wav")).unwrap();
        assert_eq!(a.duration(), SECTION_FRAMES as u32);
        assert_eq!(b.duration(), 17);
        writer.flush().unwrap();
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 2);
        assert!(writer.buffer.capacity() <= SECTION_FRAMES);
    }
}
