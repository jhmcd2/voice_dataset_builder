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
