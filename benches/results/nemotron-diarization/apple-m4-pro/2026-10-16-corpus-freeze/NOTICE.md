# Frozen evaluation audio provenance

AMI Meeting Corpus ES2004a and ICSI Meeting Corpus Bmr005 signals/transcripts: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/. Sources: https://groups.inf.ed.ac.uk/ami/download/ and https://groups.inf.ed.ac.uk/ami/icsi/download/. Cropped source PCM to the exact intervals in manifest.json; generated PCM16 WAV headers. AMI: Carletta et al., The AMI Meeting Corpus: A Pre-announcement (2005). ICSI: Janin et al., The ICSI Meeting Corpus (ICASSP 2003). Keep source archive credits.

VoxConverse: original author research-purpose distribution under CC BY 4.0; original video owners retain copyright. Source/license/citation: https://github.com/joonson/voxconverse/blob/24bf60be297701cd7e4ef18550c6d390c1b87365/README.md. Chung, Huh, Nagrani, Afouras, Zisserman, Spot the Conversation: Speaker Diarisation in the Wild (Interspeech 2020). Complete original WAV members, no audio crop or sample conversion.

AMI transcriber boundaries are human-reviewed, initialized automatically; word times are forced-aligned human text. ICSI references use lexical MRT utterances, not invented word times. Vox references are author-manually-checked semi-automatic annotations.

Local tuning-heldout only: ICSI Full and VoxConverse v0.3 dev/test explicitly appear in checkpoint training inventory. ES2004 is outside declared AMI train/dev under official partition; no guarantee of zero exposure through other sources. See docs/research/nemotron-heldout-corpus.md.
