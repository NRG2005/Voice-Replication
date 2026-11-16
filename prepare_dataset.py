"""
Build a fine-tuning dataset from your lecture videos (runs on your Mac, no GPU needed).

Two stages, run one after the other:

  python prepare_dataset.py prepare
      lecture mp4 -> audio -> cut at the pauses into clips of 3-11 s -> Whisper transcript for each clip
      Writes  dataset/clips.csv   <- OPEN THIS AND CHECK IT (see below)

  python prepare_dataset.py build
      reads your checked clips.csv -> keeps the good clips -> reserves a held-out test set (never trained on)
      Writes  dataset/dataset_for_training.zip   <- this is what you upload to the GPU machine
              dataset/held_out/                  <- test audio for scoring later (16 kHz)

What to check in clips.csv between the two stages:
  - Read the `text` column against the audio (the clip files are in dataset/all_clips/). Fix wrong words.
  - Rows with a `flag` are pre-marked `exclude = x` (probably a bad transcript or a bad clip).
    Fix the text and clear the x to keep one, or leave the x to drop it.
  - Put an `x` in the `exclude` column of any clip that has a student's voice, noise, or music.
  Transcript accuracy matters a lot: the model learns to match words to sound.
"""
import csv
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

# ============================ CONFIG: edit this part ============================

# (video or audio file, start second, end second). Use only stretches where the TEACHER ALONE speaks.
# end = None means "to the end of the file". You can list several files or several ranges of one file.
LECTURES = [
    ("/Users/nihalraviganesh/Documents/VoiceReplication/cleaned_test.mp4", 0, None),
]

OUTPUT_DIR = "dataset"
WHISPER_MODEL = "small.en"   # "medium.en" is more accurate but slower on a Mac
MIN_CLIP = 3.0               # seconds
MAX_CLIP = 11.0              # seconds (the XTTS recipe limit is about 11.6 s)
JOIN_GAP = 0.8               # pauses shorter than this stay inside a clip; longer pauses cut the clip
PAD = 0.10                   # seconds of padding kept around each clip
TEST_MINUTES = 2.0           # this much speech is held out as a test set (taken from the END of your clips)

# ================================================================================

SR = 24000        # F5-TTS works at 24 kHz
VAD_SR = 16000


def run(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:])
        raise RuntimeError("Command failed: " + " ".join(str(c) for c in cmd))


def split_at_quiet_points(a16, s, e, max_len):
    """Split a too-long speech stretch at the quietest moment instead of in the middle of a word."""
    parts = []
    win, hop = int(0.05 * VAD_SR), int(0.025 * VAD_SR)
    while e - s > max_len:
        lo, hi = s + max_len * 0.5, s + max_len
        seg = a16[int(lo * VAD_SR):int(hi * VAD_SR)]
        n = max(1, (len(seg) - win) // hop + 1)
        energy = [np.sqrt(np.mean(seg[i * hop:i * hop + win] ** 2) + 1e-12) for i in range(n)]
        cut = lo + (int(np.argmin(energy)) * hop + win / 2) / VAD_SR
        parts.append((s, cut))
        s = cut
    parts.append((s, e))
    return parts


def merge_segments(segs, min_len, max_len, gap):
    clips, cur = [], None
    for s, e in segs:
        if cur is not None and s - cur[1] <= gap and e - cur[0] <= max_len:
            cur[1] = e
        else:
            if cur is not None:
                clips.append(tuple(cur))
            cur = [s, e]
    if cur is not None:
        clips.append(tuple(cur))
    return [c for c in clips if c[1] - c[0] >= min_len]


def clean_text(t):
    return " ".join(t.replace("|", " ").replace("\n", " ").split())


# --------------------------------------------------------------------------------
def stage_prepare():
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg was not found. Install it first (on a Mac: brew install ffmpeg).")
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps
    from faster_whisper import WhisperModel

    root = Path(OUTPUT_DIR).resolve()
    clip_dir, work = root / "all_clips", root / "work"
    for d in (clip_dir, work):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)

    vad = load_silero_vad()
    print(f"Loading Whisper ({WHISPER_MODEL})...")
    whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    rows = []
    for k, (path, start, end) in enumerate(LECTURES):
        path = Path(path).expanduser()
        if not path.exists():
            sys.exit(f"File not found: {path}\nFix LECTURES at the top of this script.")
        print(f"\n[{k + 1}/{len(LECTURES)}] {path.name}: converting audio...")
        span = [] if end is None else ["-t", end - start]
        f24, f16 = work / f"l{k}_24k.wav", work / f"l{k}_16k.wav"
        run(["ffmpeg", "-y", "-loglevel", "error", "-ss", start, *span, "-i", path, "-vn", "-ac", 1, "-ar", SR, f24])
        run(["ffmpeg", "-y", "-loglevel", "error", "-ss", start, *span, "-i", path, "-vn", "-ac", 1, "-ar", VAD_SR, f16])
        a24, _ = sf.read(str(f24), dtype="float32")
        a16, _ = sf.read(str(f16), dtype="float32")
        total = len(a16) / VAD_SR

        print("  finding speech and pauses...")
        ts = get_speech_timestamps(torch.from_numpy(a16), vad, sampling_rate=VAD_SR, return_seconds=True)
        segs = []
        for t in ts:
            segs += split_at_quiet_points(a16, t["start"], t["end"], MAX_CLIP)
        clips = merge_segments(segs, MIN_CLIP, MAX_CLIP, JOIN_GAP)
        print(f"  {len(clips)} clips")

        for i, (s, e) in enumerate(clips):
            s, e = max(0.0, s - PAD), min(total, e + PAD)
            name = f"L{k}_{i:04d}.wav"
            sf.write(str(clip_dir / name), a24[int(s * SR):int(e * SR)], SR, subtype="PCM_16")
            segs_w, _ = whisper.transcribe(str(clip_dir / name), language="en", beam_size=5,
                                           condition_on_previous_text=False)
            segs_w = list(segs_w)
            text = clean_text(" ".join(x.text for x in segs_w))
            avg_lp = float(np.mean([x.avg_logprob for x in segs_w])) if segs_w else -9.0
            dur = e - s
            flags = []
            if not text:
                flags.append("empty transcript")
            if avg_lp < -1.0:
                flags.append("low Whisper confidence")
            cps = len(text) / dur if dur else 0
            if text and (cps < 5 or cps > 22):
                flags.append("odd speaking rate")
            rows.append({"clip": name, "lecture": k, "start_sec": round(start + s, 2), "duration": round(dur, 2),
                         "text": text, "avg_logprob": round(avg_lp, 2),
                         "flag": "; ".join(flags), "exclude": "x" if flags else ""})
            if (i + 1) % 10 == 0 or i + 1 == len(clips):
                print(f"  transcribed {i + 1}/{len(clips)}")

    if not rows:
        sys.exit("No speech clips were found. Check LECTURES (file, start, end).")
    with open(root / "clips.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    shutil.rmtree(work, ignore_errors=True)

    mins = sum(r["duration"] for r in rows) / 60
    kept = sum(r["duration"] for r in rows if not r["exclude"]) / 60
    print(f"\nDone. {len(rows)} clips, {mins:.1f} min total, {kept:.1f} min unflagged.")
    print(f"NEXT: open {root / 'clips.csv'}, check the text against the audio in {clip_dir},")
    print("fix wrong words, mark bad clips with x in `exclude`, save it, then run:  python prepare_dataset.py build")


# --------------------------------------------------------------------------------
def stage_build():
    root = Path(OUTPUT_DIR).resolve()
    csv_path, clip_dir = root / "clips.csv", root / "all_clips"
    if not csv_path.exists():
        sys.exit("clips.csv not found. Run:  python prepare_dataset.py prepare   first.")
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    good = [r for r in rows if clean_text(r["text"]) and r["exclude"].strip().lower() != "x"]
    if not good:
        sys.exit("No usable clips. Check the text and exclude columns in clips.csv.")
    good.sort(key=lambda r: (int(r["lecture"]), float(r["start_sec"])))

    # held-out test set: the LAST clips, so no training clip is a neighbour of a test clip in time
    test, acc = [], 0.0
    for r in reversed(good):
        if acc >= TEST_MINUTES * 60:
            break
        test.append(r)
        acc += float(r["duration"])
    test.reverse()
    test_names = {r["clip"] for r in test}
    train = [r for r in good if r["clip"] not in test_names]

    out = root / "train_package"
    shutil.rmtree(out, ignore_errors=True)
    (out / "wavs").mkdir(parents=True)
    held = root / "held_out"
    shutil.rmtree(held, ignore_errors=True)
    held.mkdir(parents=True)

    for r in train:
        shutil.copy(clip_dir / r["clip"], out / "wavs" / r["clip"])
    for r in test:
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", clip_dir / r["clip"], "-ac", 1, "-ar", 16000, held / r["clip"]])

    # train.csv with RELATIVE paths: the training notebook turns them into absolute paths where it runs
    with open(out / "train.csv", "w", encoding="utf-8", newline="") as f:
        f.write("audio_file|text\n")
        for r in train:
            f.write(f"wavs/{r['clip']}|{clean_text(r['text'])}\n")
    with open(held / "held_out.csv", "w", encoding="utf-8", newline="") as f:
        f.write("clip|text\n")
        for r in test:
            f.write(f"{r['clip']}|{clean_text(r['text'])}\n")

    zpath = root / "dataset_for_training.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out / "train.csv", "train.csv")
        for p in sorted((out / "wavs").glob("*.wav")):
            z.write(p, f"wavs/{p.name}")

    tmin = sum(float(r["duration"]) for r in train) / 60
    hmin = sum(float(r["duration"]) for r in test) / 60
    summary = (f"Training clips : {len(train)}  ({tmin:.1f} min)\n"
               f"Held-out clips : {len(test)}  ({hmin:.1f} min)  <- never train on these\n"
               f"Dropped        : {len(rows) - len(good)} clips (empty text or marked x)\n")
    (root / "summary.txt").write_text(summary)
    print(summary)
    print(f"Upload this to the GPU machine : {zpath}")
    print(f"Keep for scoring later         : {held}")
    if tmin < 30:
        print(f"NOTE: {tmin:.0f} minutes of training audio is on the low side. 30-60+ minutes gives better results.")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "build"):
        sys.exit("Usage:  python prepare_dataset.py prepare    then    python prepare_dataset.py build")
    {"prepare": stage_prepare, "build": stage_build}[sys.argv[1]]()
