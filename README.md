# voice_dataset_builder
Build Voice datasets for emotional variance with IndexTTS
# What this script actually does
## 1. Reads entire RAVDESS dataset
decodes emotion from filename
preserves actor + intensity info
## 2. Balances emotion distribution
~40 samples per emotion (adjustable)
avoids dataset bias
## 3. Runs IndexTTS2 inference
For every sample:
your voice + RAVDESS emotion + random text
→ synthesized emotional speech
## 4. Produces structured dataset
indextts_dataset/
    sample_0000.wav
    sample_0001.wav
    ...
    metadata.csv
## 5. Full dataset coverage
all RAVDESS emotions
balanced per emotion
actor diversity enforced
## 6. Safe long-run execution
resume-safe (skips completed files)
retry logic (3 attempts)
failure tracking
## 7. Data integrity
WAV validation
rejects silent/corrupt outputs
## 8. Research-grade logging
CSV metadata
JSON failure log
reproducible job list
## 9. GPU efficiency
no redundant recomputation
sequential stable inference (safe for TTS models)
