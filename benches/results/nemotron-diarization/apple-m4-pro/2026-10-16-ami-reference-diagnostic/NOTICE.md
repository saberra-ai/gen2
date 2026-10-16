# Reference and scorer notices

AMI forced-alignment reference from Shota Horiguchi, Naohiro Tawara, Takanori
Ashihara, Atsushi Ando and Marc Delcroix, *Can We Really Repurpose Multi-Speaker
ASR Corpus for Speaker Diarization?*, IEEE ASRU 2025, December 2025.
[Source and citation](https://github.com/nttcslab-sp/diar-forced-alignment/tree/9527b7c64846fb38316a610f32e9d3466bd6d8b7).
The original ES2004a RTTM is retained unchanged. Cropped-reference JSON clips
all intersecting intervals to source [300,420) seconds and subtracts 300 seconds.
CC BY 4.0 terms are retained in LICENSE.reference. Alignment labels can contain
errors and ambiguity; their use here does not imply author endorsement.

md_eval.py is unmodified NVIDIA NeMo source at the pinned revision in protocol.json.
Its Apache-2.0 header and original BSD-2-Clause/NIST notices remain in the file;
Apache terms are retained in LICENSE.Apache-2.0. The diagnostic supplies only a
standard Python logging adapter for nemo.utils.logging. It uses exact continuous
intervals serialized with 17 significant digits, not NeMo CLI's 3-decimal RTTM
writer. NumPy/SciPy match the original oracle environment, not the full NeMo lock.

This diagnostic does not change the original frozen reference or gate status.
