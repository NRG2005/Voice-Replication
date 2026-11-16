"""
kNN-VC voice conversion, local version (no Colab).

What it does:
  1. Converts your teacher recording(s) to 16 kHz mono wav clips (the "matching set").
  2. Makes ordinary speech for the 20 test sentences with a generic voice (edge-tts, needs internet).
  3. Converts that speech into the teacher's voice with kNN-VC.
  4. Saves 00.wav ... 19.wav and a knnvc_results.zip (same layout as the Colab notebooks produced).

Nothing is trained. Run it from a terminal:  python knn_vc_local.py
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

# ============================ CONFIG: edit this part ============================

# Your teacher recording(s): mp4, wav, mp3, m4a... Only the teacher's voice should be in the part used.
TEACHER_FILES = [
    "/Users/nihalraviganesh/Documents/VoiceReplication/teacher_source3.wav"
]

START_SEC = 0        # skip this many seconds at the start of EACH file (e.g. to skip an intro with other voices)
MAX_MINUTES = 5      # use at most this many minutes of teacher audio in total (more than ~5 does not help much)
OUTPUT_DIR = "knnvc_run"      # everything is created inside this folder, next to where you run the script

# The generic voice that gets converted into the teacher's voice. Pick the one matching your teacher:
#   en-IN-PrabhatNeural (Indian English, male)   en-IN-NeerjaNeural (Indian English, female)
VOICE = "en-IN-NeerjaNeural"

QUICK_TEST = False   # True = only the first 2 sentences, to check that everything works before the full run

# Keep these 20 sentences identical to the ones you used in the XTTS and F5-TTS notebooks.
SENTENCES = [
    "Good morning everyone, let us begin with a quick recap of the previous lecture.",
    "Please open your notebooks and follow along with the example on the screen.",
    "Today we will look at how the method works and why it matters in practice.",
    "If you have any doubts, please ask them at the end of this section.",
    "The assignment is due next week, so please start working on it early.",
    "Let us go through the main steps one by one.",
    "This is a common mistake that students make during the exam.",
    "Remember that the goal is to understand the idea, not just memorize the formula.",
    "We will now move on to the next topic.",
    "To summarize, there are three key points you should take away from this lecture.",
    "A convolution slides a small kernel across the image to produce a feature map.",
    "Histogram equalization improves the contrast of an image.",
    "The Canny edge detector uses two thresholds for hysteresis.",
    "Backpropagation computes gradients using the chain rule.",
    "Gradient descent updates the weights in the direction that reduces the loss.",
    "A convolutional neural network applies pooling to reduce the spatial resolution.",
    "The ResNet architecture uses skip connections to ease the training of deep networks.",
    "Overfitting happens when the model memorizes the training data.",
    "Precision and recall describe different kinds of errors in a classifier.",
    "SIFT detects keypoints that are invariant to scale and rotation.",
]
# ================================================================================

# On macOS, Python sometimes cannot verify download certificates; this uses certifi's list instead.
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

import torch
import soundfile as sf


def run(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:])
        raise RuntimeError("Command failed: " + " ".join(str(c) for c in cmd))


def load16k(path):
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    assert sr == 16000, f"{path} is {sr} Hz, expected 16000"
    return torch.from_numpy(data)


def main():
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg was not found. Install it first (on a Mac: brew install ffmpeg).")

    files = [Path(f).expanduser() for f in TEACHER_FILES]
    for f in files:
        if not f.exists():
            sys.exit(f"Teacher file not found: {f}\nFix TEACHER_FILES at the top of this script.")

    sentences = SENTENCES[:2] if QUICK_TEST else SENTENCES

    root = Path(OUTPUT_DIR).resolve()
    clip_dir = root / "teacher_16k"      # the 16 kHz wav clips of the teacher
    src_dir = root / "source_speech"     # generic-voice speech that gets converted
    out_dir = root / "outputs"           # final converted speech: 00.wav ... 19.wav
    for d in (clip_dir, src_dir, out_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)

    # ---- Step 1: teacher recordings -> 16 kHz mono wav clips of 10 seconds ----
    print("Step 1/4: converting teacher audio to 16 kHz clips...")
    for k, f in enumerate(files):
        full = root / f"full_{k:02d}.wav"
        run(["ffmpeg", "-y", "-loglevel", "error", "-ss", START_SEC, "-t", MAX_MINUTES * 60,
             "-i", f, "-vn", "-ac", 1, "-ar", 16000, full])
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", full, "-f", "segment",
             "-segment_time", 10, "-c", "copy", clip_dir / f"f{k:02d}_%03d.wav"])
        full.unlink()

    total = 0.0
    for p in sorted(clip_dir.glob("*.wav")):
        d = sf.info(str(p)).duration
        if d < 2.0 or total + d > MAX_MINUTES * 60:
            p.unlink()
        else:
            total += d
    clips = sorted(clip_dir.glob("*.wav"))
    print(f"  {len(clips)} clips, {total / 60:.1f} minutes of teacher audio")
    if not clips:
        sys.exit("No usable teacher audio was found. Check TEACHER_FILES and START_SEC.")
    if total < 120:
        print("  WARNING: under 2 minutes. kNN-VC works but sounds weaker. More clean teacher audio helps.")

    # ---- Step 2: ordinary speech for each sentence in a generic voice (needs internet) ----
    print(f"Step 2/4: making source speech with the voice {VOICE}...")
    for i, s in enumerate(sentences):
        mp3, wav = src_dir / f"{i:02d}.mp3", src_dir / f"{i:02d}.wav"
        run([sys.executable, "-m", "edge_tts", "--voice", VOICE, "--text", s, "--write-media", mp3])
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp3, "-ac", 1, "-ar", 16000, wav])

    # ---- Step 3: convert to the teacher's voice ----
    print("Step 3/4: loading kNN-VC (the first run downloads large model files, this can take a while)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    knn_vc = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True,
                            pretrained=True, device=device)
    matching_set = knn_vc.get_matching_set([load16k(p) for p in clips])
    for i in range(len(sentences)):
        query = knn_vc.get_features(load16k(src_dir / f"{i:02d}.wav"))
        w = knn_vc.match(query, matching_set, topk=4)
        sf.write(str(out_dir / f"{i:02d}.wav"), w.detach().cpu().numpy().reshape(-1), 16000)
        print(f"  {i + 1}/{len(sentences)}")

    # ---- Step 4: results zip, same layout the compare notebook expects ----
    print("Step 4/4: making knnvc_results.zip...")
    res = root / "knnvc_results"
    shutil.rmtree(res, ignore_errors=True)
    (res / "teacher_audio").mkdir(parents=True)
    shutil.copytree(out_dir, res / "outputs")
    for p in clips[:20]:
        shutil.copy(p, res / "teacher_audio")
    (res / "sentences.txt").write_text("\n".join(sentences) + "\n")
    zip_path = shutil.make_archive(str(root / "knnvc_results"), "zip", res)

    print("\nDone.")
    print("  Converted audio :", out_dir)
    print("  Results zip     :", zip_path, " <- upload this to the compare notebook")


if __name__ == "__main__":
    main()
