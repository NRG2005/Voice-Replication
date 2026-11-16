"""
Compare the voice-cloning methods, local version of 4_compare.ipynb (no Colab, no GPU needed).

It does not generate any voice. It scores the audio found in the results zips and writes:
  compare_out/results.csv   one row per generated clip
  compare_out/summary.csv   one row per method
  compare_out/report.txt    the summary, the winner per metric, and which clips to listen to

Metrics (computed for every clip, then averaged per method):
  speaker_sim  closeness to the teacher's real voice (ECAPA speaker embeddings, cosine). Higher is better.
  wer          word error rate: Whisper listens to the clip and it is compared with the intended sentence. Lower is better.
  utmos        predicted naturalness, roughly 1 to 5. Higher is better.

Run:  python compare_local.py
"""
import os
import re
import sys
import glob
import shutil
import warnings
import zipfile
from pathlib import Path

# ============================ CONFIG: edit this part ============================

# The results zips from your three methods (paths can be anywhere on your Mac).
ZIP_FILES = [
    "xtts_results.zip",
    "f5_results.zip",
    "knnvc_results.zip",
]

OUTPUT_DIR = "compare_out"
WHISPER_MODEL = "small.en"     # used to check the words. "tiny.en" is faster but less accurate.

# ================================================================================

try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
import jiwer
import soundfile as sf


def label_for(fname, taken):
    low = Path(fname).name.lower()
    lab = None
    for key, name in (("xtts", "xtts"), ("f5", "f5"), ("knn", "knnvc")):
        if key in low:
            lab = name
            break
    if lab is None:
        lab = re.sub(r"\W+", "_", Path(fname).stem)
    base, k = lab, 2
    while lab in taken:                  # two zips of the same method: f5, f5_2, ...
        lab = f"{base}_{k}"
        k += 1
    return lab


def norm(t):
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", t.lower()).split())


def main():
    out_root = Path(OUTPUT_DIR).resolve()
    work = out_root / "unzipped"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    # ---- Step 1: unzip ----
    methods = {}
    for z in ZIP_FILES:
        zp = Path(z).expanduser()
        if not zp.exists():
            sys.exit(f"Zip not found: {zp}\nFix ZIP_FILES at the top of this script.")
        lab = label_for(zp.name, methods)
        dest = work / lab
        dest.mkdir(parents=True)
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(dest)
        methods[lab] = dest
        print(f"{lab}: {len(list((dest / 'outputs').glob('*.wav')))} generated clips")

    sentence_sets = {m: (d / "sentences.txt").read_text() for m, d in methods.items()}
    if len(set(sentence_sets.values())) > 1:
        print("\nWARNING: the sentences differ between your zips. The comparison is only fair if every method spoke the same sentences.\n")

    # ---- Step 2: speaker similarity model (ECAPA-TDNN) + the teacher's real audio ----
    embed, t_mean = None, None
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                             savedir=str(out_root / "ecapa_model"))

        def embed(path):
            data, fs = sf.read(str(path), dtype="float32", always_2d=True)
            sig = torch.from_numpy(data.T)
            if fs != 16000:
                sig = torchaudio.functional.resample(sig, fs, 16000)
            if sig.shape[0] > 1:
                sig = sig.mean(dim=0, keepdim=True)
            with torch.no_grad():
                return clf.encode_batch(sig).squeeze().cpu()

        teacher_files = sorted(glob.glob(str(work / "*" / "teacher_audio" / "*.wav")))[:20]
        print(f"Speaker reference: {len(teacher_files)} real teacher clips")
        t_mean = torch.stack([embed(p) for p in teacher_files]).mean(0)
    except Exception as ex:
        print("Speaker similarity unavailable:", ex)

    # ---- Step 3: Whisper (checks the words) and UTMOS (naturalness) ----
    from faster_whisper import WhisperModel
    use_cuda = torch.cuda.is_available()
    whisper = WhisperModel(WHISPER_MODEL, device="cuda" if use_cuda else "cpu",
                           compute_type="float16" if use_cuda else "int8")

    def transcribe(path):
        segs, _ = whisper.transcribe(str(path), language="en")
        return " ".join(s.text.strip() for s in segs)

    utmos = None
    try:
        utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    except Exception as ex:
        print("UTMOS unavailable:", ex)

    # ---- Step 4: score every clip ----
    rows = []
    for m, d in methods.items():
        sents = [s.strip() for s in (d / "sentences.txt").read_text().splitlines() if s.strip()]
        clips = sorted((d / "outputs").glob("*.wav"))
        for n, p in enumerate(clips, 1):
            i = int(p.stem)
            row = {"method": m, "idx": i, "sentence": sents[i],
                   "speaker_sim": np.nan, "wer": np.nan, "utmos": np.nan}
            try:
                if embed is not None:
                    row["speaker_sim"] = torch.nn.functional.cosine_similarity(embed(p), t_mean, dim=0).item()
            except Exception as ex:
                print("speaker similarity failed:", m, i, ex)
            try:
                row["wer"] = jiwer.wer(norm(sents[i]), norm(transcribe(p)))
            except Exception as ex:
                print("wer failed:", m, i, ex)
            try:
                if utmos is not None:
                    wave, sr = librosa.load(str(p), sr=None, mono=True)
                    row["utmos"] = float(utmos(torch.from_numpy(wave).unsqueeze(0), sr).item())
            except Exception as ex:
                print("utmos failed:", m, i, ex)
            rows.append(row)
            print(f"  scored {m} {n}/{len(clips)}")

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "results.csv", index=False)
    summary = df.groupby("method")[["speaker_sim", "wer", "utmos"]].mean().round(3)
    summary.to_csv(out_root / "summary.csv")

    # ---- Step 5: report ----
    def winners(col, best):
        v = summary[col].dropna()
        if v.empty:
            return "n/a"
        top = v.max() if best == "high" else v.min()
        names = list(v.index[v == top])
        return names[0] if len(names) == 1 else "tie (" + ", ".join(names) + ")"

    lines = [
        "RESULTS (average over all clips per method)",
        summary.to_string(),
        "",
        "Winner per metric:",
        f"  closest to the teacher's voice (speaker_sim, higher): {winners('speaker_sim', 'high')}",
        f"  clearest words (wer, lower):                          {winners('wer', 'low')}",
        f"  most natural sounding (utmos, higher):                {winners('utmos', 'high')}",
        "",
        "Now LISTEN. The same sentence from every method (sentence 0 is general, 13 is technical):",
    ]
    for idx in (0, 13):
        for m, d in methods.items():
            p = d / "outputs" / f"{idx:02d}.wav"
            if p.exists():
                lines.append(f"  {m:8s} sentence {idx:2d}: {p}")
    lines += [
        "",
        "How to read this: the numbers are a guide, not a verdict. If one method wins all three it is your best.",
        "More often they disagree (e.g. closest voice but mispronounced technical words): report that trade-off.",
    ]
    report = "\n".join(lines)
    (out_root / "report.txt").write_text(report + "\n")
    print("\n" + report)
    print(f"\nSaved in {out_root}: results.csv, summary.csv, report.txt")


if __name__ == "__main__":
    main()
