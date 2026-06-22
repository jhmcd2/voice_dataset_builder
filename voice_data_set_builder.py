import os
import random
import csv
from collections import defaultdict
from indextts.infer_v2 import IndexTTS2

# =========================
# CONFIG
# =========================

RAVDESS_DIR = "ravdess/"
VOICE_WAV = "my_voice.wav"
OUTPUT_DIR = "indextts_dataset/"
METADATA_CSV = os.path.join(OUTPUT_DIR, "metadata.csv")

TEXTS = [
    "Hello, how are you today?",
    "This is a test of emotional speech synthesis.",
    "I can't believe this actually works.",
    "We are generating expressive speech samples.",
    "The weather looks nice today.",
    "I will call you later.",
]

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# =========================
# INIT MODEL
# =========================

tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    is_fp16=True,
    use_cuda_kernel=False
)

# =========================
# PARSE RAVDESS
# =========================

def parse_file(fname):
    parts = fname.replace(".wav", "").split("-")
    return {
        "emotion_id": parts[2],
        "emotion": EMOTION_MAP.get(parts[2], "unknown"),
        "intensity": parts[3],
        "actor": parts[6],
        "path": os.path.join(RAVDESS_DIR, fname)
    }

# =========================
# LOAD DATASET
# =========================

def load_ravdess():
    data = []
    for f in os.listdir(RAVDESS_DIR):
        if f.endswith(".wav"):
            data.append(parse_file(f))
    return data

# =========================
# GROUP BY EMOTION
# =========================

def group_by_emotion(data):
    groups = defaultdict(list)
    for item in data:
        groups[item["emotion"]].append(item)
    return groups

# =========================
# BALANCED SAMPLER
# =========================

def sample_balanced(groups, target_per_emotion=40):
    sampled = []

    for emotion, items in groups.items():
        random.shuffle(items)
        sampled.extend(items[:target_per_emotion])

    return sampled

# =========================
# MAIN PIPELINE
# =========================

def run_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = load_ravdess()
    groups = group_by_emotion(dataset)

    sampled = sample_balanced(groups, target_per_emotion=40)

    print(f"Total samples selected: {len(sampled)}")

    # CSV logging
    with open(METADATA_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "output_file",
            "emotion",
            "intensity",
            "actor",
            "emotion_ref",
            "text"
        ])

        for i, item in enumerate(sampled):

            text = random.choice(TEXTS)
            output_path = os.path.join(OUTPUT_DIR, f"sample_{i:04d}.wav")

            print(f"[{i}] {item['emotion']} | actor {item['actor']}")

            # =========================
            # INDEXTTS2 INFERENCE
            # =========================
            tts.infer(
                spk_audio_prompt=VOICE_WAV,
                emo_audio_prompt=item["path"],
                text=text,
                output_path=output_path,
                verbose=False
            )

            # log metadata
            writer.writerow([
                output_path,
                item["emotion"],
                item["intensity"],
                item["actor"],
                item["path"],
                text
            ])

    print("\nDONE: Dataset generated successfully.")

# =========================
# RUN
# =========================

if __name__ == "__main__":
    run_pipeline()
