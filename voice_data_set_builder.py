import os
import sys
import json
import csv
import time
import random
import soundfile as sf
from collections import defaultdict

from indextts.infer_v2 import IndexTTS2

# =========================
# CONFIG
# =========================

RAVDESS_DIR = "ravdess/"
VOICE_WAV = None

OUTPUT_DIR = "indextts_dataset"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")

MANIFEST_PATH = os.path.join(OUTPUT_DIR, "manifest.json")
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.csv")
FAILED_PATH = os.path.join(OUTPUT_DIR, "failed_jobs.json")

MAX_RETRIES = 3
TARGET_PER_EMOTION = 40

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
# UTILITIES
# =========================

def parse_ravdess(fname):
    parts = fname.replace(".wav", "").split("-")
    return {
        "emotion_id": parts[2],
        "emotion": EMOTION_MAP.get(parts[2], "unknown"),
        "intensity": parts[3],
        "actor": parts[6],
        "path": os.path.join(RAVDESS_DIR, fname)
    }

def load_dataset():
    data = []
    for f in os.listdir(RAVDESS_DIR):
        if f.endswith(".wav"):
            data.append(parse_ravdess(f))
    return data

def group_by_emotion(data):
    groups = defaultdict(list)
    for item in data:
        groups[item["emotion"]].append(item)
    return groups

def validate_wav(path):
    try:
        audio, sr = sf.read(path)
        if len(audio) == 0:
            return False
        if abs(audio).max() < 1e-5:
            return False
        return True
    except:
        return False

def already_done(path):
    return os.path.exists(path) and validate_wav(path)

# =========================
# BALANCED SAMPLING
# =========================

def build_jobs(groups):
    jobs = []

    for emotion, items in groups.items():
        random.shuffle(items)

        # actor-aware spread
        actor_map = defaultdict(list)
        for x in items:
            actor_map[x["actor"]].append(x)

        actors = list(actor_map.keys())
        random.shuffle(actors)

        selected = []

        # round-robin actor sampling
        while len(selected) < TARGET_PER_EMOTION:
            for a in actors:
                if actor_map[a]:
                    selected.append(actor_map[a].pop())
                if len(selected) >= TARGET_PER_EMOTION:
                    break

        for item in selected:
            jobs.append({
                "emotion": item["emotion"],
                "intensity": item["intensity"],
                "actor": item["actor"],
                "emotion_ref": item["path"],
                "text": random.choice(TEXTS)
            })

    return jobs

# =========================
# PIPELINE
# =========================

def run(VOICE_WAV):
    os.makedirs(AUDIO_DIR, exist_ok=True)

    data = load_dataset()
    groups = group_by_emotion(data)
    jobs = build_jobs(groups)

    failed = []

    # CSV metadata
    with open(METADATA_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "output",
            "emotion",
            "intensity",
            "actor",
            "emotion_ref",
            "text",
            "status",
            "retries"
        ])

        start_time = time.time()

        for i, job in enumerate(jobs):

            output_path = os.path.join(AUDIO_DIR, f"sample_{i:05d}.wav")

            if already_done(output_path):
                print(f"[SKIP] {output_path}")
                continue

            print(f"[{i}/{len(jobs)}] {job['emotion']} | actor {job['actor']}")

            success = False
            retries = 0

            for attempt in range(MAX_RETRIES):

                try:
                    tts.infer(
                        spk_audio_prompt=VOICE_WAV,
                        emo_audio_prompt=job["emotion_ref"],
                        text=job["text"],
                        output_path=output_path,
                        verbose=False
                    )

                    if validate_wav(output_path):
                        success = True
                        break

                except Exception as e:
                    print(f"Retry {attempt+1} failed: {e}")

                retries += 1

            if not success:
                failed.append(job)

            writer.writerow([
                output_path,
                job["emotion"],
                job["intensity"],
                job["actor"],
                job["emotion_ref"],
                job["text"],
                "ok" if success else "failed",
                retries
            ])

            # ETA
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = len(jobs) - (i + 1)
            eta = remaining / rate if rate > 0 else 0

            print(f"ETA: {eta/60:.1f} min")

    # Save failures
    with open(FAILED_PATH, "w") as f:
        json.dump(failed, f, indent=2)

    print("\nDONE")
    print(f"Total jobs: {len(jobs)}")
    print(f"Failed: {len(failed)}")

# =========================
# RUN
# =========================

if __name__ == "__main__":
    # =========================
    # Injest File
    # =========================
    
    # Ensure the user provided the file argument
    if len(sys.argv) < 2:
        print("Error: Please provide a voice to modify.")
        print("Usage: voice_data_set_builder.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # Read and process the file
    try:
        with open(filename, 'r') as file:
            VOICE_WAV = file.read()
            print(f"Successfully ingested {filename}!")
            run(VOICE_WAV)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    
