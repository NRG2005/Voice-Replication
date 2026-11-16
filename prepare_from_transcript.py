"""
Build the fine-tuning clips from your lecture video + its transcript JSON (runs on your Mac, no GPU needed).

Use this INSTEAD of `python prepare_dataset.py prepare` when you already have a transcript with word timestamps.
It reads the JSON, cuts the video's audio into clips of 3-8 s at real pauses between words, and takes each clip's
text from your transcript. It writes dataset/clips.csv in the same format that `prepare_dataset.py build` reads.

  python prepare_from_transcript.py          (this script)
  ... open dataset/clips.csv, review it ...
  python prepare_dataset.py build            (the script you already have)

Clips are pre-marked  exclude = x  (and given a `flag`) when the text is unlikely to match the audio well:
  - digits or hex ("0x66887799", "26248")      the spoken form is unknown, so the text would not match the sound
  - symbol talk ("percentile X", "%n")          the transcript's spelling is not what is spoken
  - Hindi words in English letters
  - the cut could not land on a clear pause
  - a different Whisper pass heard something quite different from the transcript (text and audio may not match)
Fix the text to EXACTLY what is said and clear the x to keep a clip; leave the x to drop it.
"""
import csv
import difflib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# ============================ CONFIG: edit this part ============================

VIDEO = "/Users/nihalraviganesh/Documents/VoiceReplication/cleaned_test.mp4"      # the 35-minute teacher-only video
TRANSCRIPT_JSON = "/Users/nihalraviganesh/Documents/VoiceReplication/transcript.json"     # the JSON with word timestamps

OUTPUT_DIR = "dataset"
MIN_CLIP = 3.0          # seconds
MAX_CLIP = 8.0          # seconds. Shorter clips lose less audio when a number or symbol forces a clip out (6 keeps even more)
VERIFY_WITH_WHISPER = True     # re-listen to every clip with Whisper and flag clips whose text looks wrong
WHISPER_MODEL = "small.en"
ALIGN_WHISPER_MODEL = "base.en"   # used once, only if the transcript and video clocks drift apart irregularly

# ================================================================================

SR = 24000          # F5-TTS works at 24 kHz
AN_SR = 16000       # sample rate used only to find pauses
HOP = 0.01          # 10 ms frames for the energy curve
WIN = 0.02
HINDI = re.compile(r"\b(iska|nikalu|nikal|gaya|gayi|hai|hain|nahi|nahin|kya|karo|karna|hota|hoga|matlab|dekho|"
                   r"haan|accha|achha|toh|yeh|woh|aur|bhi|kyun|kyu|thoda|samajh)\b", re.I)
SYMBOLS = re.compile(r"percentile|percentage|0x|\bx\d{2}\b|%|\\x|@|#|\$", re.I)
HEX_LETTERS = re.compile(r"\b[A-F]{2,}\b")       # spelled-out hex such as FF or AAAA (case-sensitive on purpose)


def run(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:])
        raise RuntimeError("Command failed: " + " ".join(str(c) for c in cmd))


def clean_text(t):
    return " ".join(t.replace("|", " ").replace("\n", " ").split())


def norm_words(t):
    return re.sub(r"[^a-z0-9' ]", " ", t.lower()).split()


def load_words(json_path):
    """Flatten the transcript JSON into one list of words with start/end seconds."""
    data = json.load(open(json_path, encoding="utf-8"))
    segs = data["segments"] if isinstance(data, dict) else data
    words = []
    for s in segs:
        for w in s.get("words", []):
            t = w["word"].strip()
            if t and w["end"] >= w["start"]:
                words.append({"word": t, "start": float(w["start"]), "end": float(w["end"])})
    words.sort(key=lambda w: w["start"])
    return words


def frame_db(a, sr=AN_SR):
    """Loudness (dB) of every 10 ms frame, computed with a running sum so 35 minutes stays cheap."""
    w, h = int(WIN * sr), int(HOP * sr)
    cs = np.concatenate([[0.0], np.cumsum(a.astype(np.float64) ** 2)])
    n = 1 + (len(a) - w) // h
    starts = np.arange(n) * h
    rms = np.sqrt((cs[starts + w] - cs[starts]) / w)
    return 20 * np.log10(rms + 1e-8)


def quiet_center(db, t0, t1, thr):
    """Where to cut between t0 and t1 (seconds): the CENTER of the pause, so both clips keep a margin of silence.
    Returns (time, loudness of the quietest frame). If there is no real pause, it is just the quietest frame."""
    i0 = max(0, int(t0 / HOP))
    i1 = min(len(db), max(i0 + 1, int(t1 / HOP)))
    k = i0 + int(np.argmin(db[i0:i1]))
    v = float(db[k])
    if v >= thr:
        return k * HOP + WIN / 2, v
    left, right = k, k
    while left > 0 and db[left - 1] < thr:
        left -= 1
    while right < len(db) - 1 and db[right + 1] < thr:
        right += 1
    return (left + right) / 2 * HOP + WIN / 2, v


def boundaries(words, db, thr, slack=0.0):
    """For every gap between two words: the quietest moment around it (this is where we are allowed to cut)."""
    T, V = [], []
    for i in range(len(words) - 1):
        a, b = words[i], words[i + 1]
        lo = a["start"] + 0.4 * (a["end"] - a["start"]) - slack   # timestamps can stretch a word over the pause after it
        hi = b["start"] + 0.15 + slack
        if hi <= lo:
            lo, hi = a["end"] - 0.1, b["start"] + 0.1
        t, v = quiet_center(db, lo, hi, thr)
        T.append(t)
        V.append(v)
    return np.array(T), np.array(V)



def _bins(x, k=10):
    n = len(x) // k
    return x[:n * k].reshape(n, k).mean(1)


def _word_activity(words, n_frames):
    act = np.zeros(n_frames, np.float32)
    for w in words:
        a = int(w["start"] / HOP)
        b = min(n_frames, int((w["start"] + 0.6 * (w["end"] - w["start"])) / HOP))
        if b > a >= 0:
            act[a:b] = 1.0
    return act


def estimate_alignment(words, db, thr, max_shift=120.0):
    """Find how the transcript's clock maps onto the video's clock by matching WHEN words are spoken to WHEN the audio
    is loud. Returns a function offset(t) in seconds (audio_time = transcript_time + offset(t)) and a description."""
    audio = (db > thr).astype(np.float32)
    n_t = int(words[-1]["end"] / HOP) + 1
    windows = int(np.clip(n_t * HOP / 70, 3, 30))          # about 70 s per window
    need = min(5, windows)
    spoken = _word_activity(words, n_t)
    L = int(max_shift / 0.1)
    edges = np.linspace(0, n_t, windows + 1).astype(int)
    centers, offsets = [], []
    for k in range(windows):
        f0, f1 = edges[k], edges[k + 1]
        x = spoken[f0:f1]
        if x.sum() < 200:
            continue
        xb = _bins(x)
        b0 = f0 // 10
        lo = b0 - L
        yb_full = _bins(audio)
        y = np.zeros(len(xb) + 2 * L, np.float32)                  # audio around this window, zero-padded at the ends
        src0, src1 = max(lo, 0), min(lo + len(y), len(yb_full))
        if src1 > src0:
            y[src0 - lo:src1 - lo] = yb_full[src0:src1]
        c = np.correlate(y - y.mean(), xb - xb.mean(), "valid")
        peak = int(np.argmax(c))
        z = (c[peak] - c.mean()) / (c.std() + 1e-9)
        if z < 5:
            continue
        coarse = (peak - L) * 10                                    # in 10 ms frames
        best, best_lag = -1.0, coarse
        for lag in range(coarse - 50, coarse + 51):                 # refine to 10 ms
            a0, a1 = f0 + lag, f0 + lag + len(x)
            if a0 < 0 or a1 > len(audio):
                continue
            v = float(np.dot(x, audio[a0:a1]))
            if v > best:
                best, best_lag = v, lag
        centers.append((f0 + f1) / 2 * HOP)
        offsets.append(best_lag * HOP)
    if len(offsets) < need:
        return None, "Could not line the transcript up with this audio (the speech pattern does not match)."
    c, o = np.array(centers), np.array(offsets)
    print(f"  offset found in each stretch of the video (seconds): {', '.join(f'{v:.1f}' for v in o[:40])}"
          + (" ..." if len(o) > 40 else ""))
    keep = np.abs(o - np.median(o)) < 0.5                       # 1) constant offset: most windows agree
    if keep.sum() >= max(need, 0.6 * len(o)) and (o[keep].max() - o[keep].min()) < 0.3:
        b = float(np.median(o[keep]))
        return (lambda t: b), f"constant offset of {b:+.2f} s (transcript clock + {b:.2f} s = video clock)"
    keep = np.ones(len(o), bool)                                # 2) steady drift: fit a line, ignoring outliers
    for _ in range(3):
        p = np.polyfit(c[keep], o[keep], 1)
        keep = np.abs(np.polyval(p, c) - o) < 0.5
        if keep.sum() < need:
            break
    if keep.sum() >= max(need, 0.6 * len(o)):
        return (lambda t: float(np.polyval(p, t))), (
            f"drift: the video clock runs {p[0] * 100:+.2f}% against the transcript "
            f"(offset {np.polyval(p, 0):+.1f} s at the start, {np.polyval(p, c.max()):+.1f} s at the end)")
    # 3) the offset varies irregularly (e.g. silences were trimmed or added): follow it piece by piece
    return None, "The offset changes irregularly through the video."


def _median_filter(v, k=5):
    h = k // 2
    return np.array([np.median(v[max(0, i - h):i + h + 1]) for i in range(len(v))])


def anchor_alignment(words, a16, total, model_name):
    """Let Whisper listen to THIS video (word timestamps), then match its words to the transcript's words.
    Every matching word is an anchor: it tells how far the transcript clock is from the video clock at that moment."""
    from faster_whisper import WhisperModel
    import difflib
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segs, _ = model.transcribe(a16, language="en", word_timestamps=True, condition_on_previous_text=False)
    heard, last = [], 0
    for sg in segs:
        for w in (sg.words or []):
            heard.append({"word": w.word.strip(), "start": float(w.start), "end": float(w.end)})
        pct = int(100 * sg.end / total)
        if pct >= last + 10:
            last = pct
            print(f"    listening... {pct}%")
    if len(heard) < 50:
        return None, "Whisper found almost no speech in this video."
    A = [" ".join(norm_words(w["word"])) for w in words]
    B = [" ".join(norm_words(w["word"])) for w in heard]
    sm = difflib.SequenceMatcher(None, A, B, autojunk=False)
    t, off = [], []
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            if A[blk.a + k]:
                t.append(words[blk.a + k]["start"])
                off.append(heard[blk.b + k]["start"] - words[blk.a + k]["start"])
    share = len(t) / len(words)
    if share < 0.4 or len(t) < 100:
        return None, f"Only {share:.0%} of the transcript's words were found in what Whisper heard in this video."
    t, off = np.array(t), np.array(off)
    sane = np.abs(off) < 120.0                           # an offset of minutes is a false match, not a real shift
    t, off = t[sane], off[sane]
    smooth = _median_filter(off, 15)
    ok = np.abs(off - smooth) < 1.0
    t, off = t[ok], smooth[ok]
    desc = (f"the video clock is {off[0]:+.1f} s from the transcript clock at the start and {off[-1]:+.1f} s at the end "
            f"(lined up with {share:.0%} of the words)")
    return (lambda x: float(np.interp(x, t, off))), desc


def build_clips(words, T, V, thr, total):
    """Walk through the words and cut at pauses so clips are MIN_CLIP..MAX_CLIP long. Returns (a, j, start, end, clear_pause)."""
    n = len(words)
    clips, a, s = [], 0, max(0.0, words[0]["start"] - 0.25)
    while a < n:
        end_all = min(total, words[n - 1]["end"] + 0.25)
        if end_all - s <= MAX_CLIP:                          # the rest fits in one clip
            if end_all - s >= MIN_CLIP:
                clips.append((a, n - 1, s, end_all, True))
            break
        cands = [j for j in range(a, n - 1) if MIN_CLIP <= T[j] - s <= MAX_CLIP]
        if cands:
            good = [j for j in cands if V[j] < thr]
            sent = [j for j in good if words[j]["word"][-1:] in ".?!" and T[j] - s >= 5.0]
            if sent:
                j, clear = max(sent), True
            elif good:
                j, clear = max(good), True
            else:
                j, clear = min(cands, key=lambda k: V[k]), False
        else:                                                # no boundary in range: take the first one past MIN_CLIP
            later = [j for j in range(a, n - 1) if T[j] - s >= MIN_CLIP]
            if not later:
                break
            j, clear = later[0], False
        clips.append((a, j, s, float(T[j]), clear))
        a, s = j + 1, float(T[j])
    return clips


def main():
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg was not found. Install it first (on a Mac: brew install ffmpeg).")
    video, tj = Path(VIDEO).expanduser(), Path(TRANSCRIPT_JSON).expanduser()
    for p in (video, tj):
        if not p.exists():
            sys.exit(f"File not found: {p}\nFix VIDEO / TRANSCRIPT_JSON at the top of this script.")

    root = Path(OUTPUT_DIR).resolve()
    clip_dir, work = root / "all_clips", root / "work"
    shutil.rmtree(work, ignore_errors=True)        # only the scratch folder is reset here; existing clips are kept
    work.mkdir(parents=True)                       # until the transcript has been lined up with the audio

    print("Reading the transcript...")
    words = load_words(tj)
    if len(words) < 50:
        sys.exit("The transcript JSON has too few words with timestamps. Is this the right file?")
    print(f"  {len(words)} words, last timestamp {words[-1]['end'] / 60:.1f} min")

    print("Converting the video's audio...")
    f24, f16 = work / "a24.wav", work / "a16.wav"
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vn", "-ac", 1, "-ar", SR, f24])
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vn", "-ac", 1, "-ar", AN_SR, f16])
    a24, _ = sf.read(str(f24), dtype="float32")
    a16, _ = sf.read(str(f16), dtype="float32")
    total = len(a16) / AN_SR
    print(f"  video audio is {total / 60:.1f} min")
    db = frame_db(a16)
    speech = np.percentile(db, 90)
    noise = np.percentile(db[db > speech - 80], 5)      # ignore digital silence (exact zeros) when judging the background level
    thr = noise + 0.35 * (speech - noise)            # quieter than this counts as a pause
    print(f"  loudness: speech about {speech:.0f} dB, background about {noise:.0f} dB, pause threshold {thr:.0f} dB")

    print("Lining the transcript up with the audio...")
    offset_at, desc = estimate_alignment(words, db, thr)
    slack = 0.0
    if offset_at is None and abs(total - words[-1]["end"]) > 5:
        print("  The clocks drift apart irregularly. Letting Whisper listen to the video to line it up exactly")
        print("  (one extra pass over the whole video; this takes several minutes)...")
        offset_at, desc = anchor_alignment(words, a16, total, ALIGN_WHISPER_MODEL)
        slack = 0.3
        if offset_at is None:
            sys.exit(desc + "\nCheck that this video is the one the transcript was made from.")
    elif offset_at is None:
        print("  WARNING: " + desc + " The lengths agree, so the transcript is used as it is (no shift).")
        offset_at, desc = (lambda t: 0.0), "no shift"
    print("  " + desc)
    if abs(total - words[-1]["end"]) > 15:
        print(f"  (the video is {total - words[-1]['end']:+.0f} s longer than the transcript; that is fine once the offset above is applied)")
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        o = offset_at(mid)
        w["start"], w["end"] = w["start"] + o, w["end"] + o
    words = [w for w in words if w["end"] > 0 and w["start"] < total]
    if len(words) < 50:
        sys.exit("Almost no transcript words fall inside this video after lining them up.")

    shutil.rmtree(clip_dir, ignore_errors=True)    # alignment worked: now it is safe to replace any earlier clips
    clip_dir.mkdir(parents=True)

    T, V = boundaries(words, db, thr, slack)
    clips = build_clips(words, T, V, thr, total)
    print(f"Cut {len(clips)} clips.")

    whisper = None
    if VERIFY_WITH_WHISPER:
        from faster_whisper import WhisperModel
        print(f"Loading Whisper ({WHISPER_MODEL}) to double-check every clip...")
        whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    rows = []
    for i, (a, j, s, e, clear) in enumerate(clips):
        s0 = s
        loud = np.where(db[int(s0 / HOP):int(e / HOP)] > thr)[0]
        if len(loud):                                           # keep at most 0.25 s of silence at each end
            first, last = s0 + loud[0] * HOP, s0 + (loud[-1] + 1) * HOP
            s, e = max(s0, first - 0.25), min(e, last + 0.25)
        name = f"L0_{i:04d}.wav"
        sf.write(str(clip_dir / name), a24[int(s * SR):int(e * SR)], SR, subtype="PCM_16")
        dur = len(a24[int(s * SR):int(e * SR)]) / SR
        text = clean_text(" ".join(w["word"] for w in words[a:j + 1]))

        flags = []
        if re.search(r"\d", text):
            flags.append("digits (spoken form unknown)")
        if SYMBOLS.search(text) or HEX_LETTERS.search(text):
            flags.append("symbol/hex talk")
        if HINDI.search(text):
            flags.append("Hindi words")
        cps = len(text) / dur if dur else 0
        if cps < 5 or cps > 22:
            flags.append("odd speaking rate")
        if not clear:
            flags.append("no clear pause at the cut")
        match = ""
        if whisper is not None:
            segs_w, _ = whisper.transcribe(str(clip_dir / name), language="en", beam_size=5,
                                           condition_on_previous_text=False)
            heard = norm_words(" ".join(x.text for x in segs_w))
            ratio = difflib.SequenceMatcher(None, norm_words(text), heard).ratio()
            match = round(ratio, 2)
            if ratio < 0.75:
                flags.append(f"Whisper hears something different ({ratio:.0%} match)")
        rows.append({"clip": name, "lecture": 0, "start_sec": round(s, 2), "duration": round(dur, 2),
                     "text": text, "match_with_whisper": match,
                     "flag": "; ".join(flags), "exclude": "x" if flags else ""})
        if (i + 1) % 25 == 0 or i + 1 == len(clips):
            print(f"  prepared {i + 1}/{len(clips)}")

    with open(root / "clips.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    shutil.rmtree(work, ignore_errors=True)

    nopause = sum(1 for r in rows if "no clear pause" in r["flag"])
    if nopause > 0.25 * len(rows):
        print(f"\nWARNING: {nopause} of {len(rows)} cuts did not land on a clear pause. The transcript may not line up well "
              "with this audio; check a few clips by ear before trusting the text.")
    mins = sum(r["duration"] for r in rows) / 60
    kept = sum(r["duration"] for r in rows if not r["exclude"]) / 60
    print(f"\nDone. {len(rows)} clips, {mins:.1f} min total, {kept:.1f} min unflagged.")
    reasons = {}
    for r in rows:
        for fl in [x.split(" (")[0] for x in r["flag"].split("; ") if x]:
            reasons[fl] = reasons.get(fl, 0) + 1
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  flagged for '{k}': {v} clips")
    print(f"\nNEXT: open {root / 'clips.csv'}. Listen to clips in {clip_dir} and fix the text to exactly what is said.")
    print("Then run:  python prepare_dataset.py build")


if __name__ == "__main__":
    main()
