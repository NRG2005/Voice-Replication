"""
Review your dataset clips by ear, fast (macOS). Run it after `prepare` and before `build`:

  python review_clips.py

For every clip it plays the audio and shows the text. You decide:
  Enter               the text matches what you hear -> accept
  r                   replay the clip
  x                   drop this clip (bad audio, a student's voice, wrong text you cannot fix)
  b                   go back one clip
  q                   save and quit (come back later: reviewed clips are skipped)
  old=>new            fix words in THIS clip's text, e.g.  county=>Canny
  !old=>new           fix words in ALL clips' text at once, e.g.  !little Indian=>little endian
  anything else       typed as the whole corrected text for this clip

Everything is saved after every action, so you can stop at any time.
By default it only goes through clips that are not pre-marked `x`. Set INCLUDE_FLAGGED = True to also go through the
flagged ones: accepting a flagged clip (Enter) means "the text is right now" and removes its x.

The text must be EXACTLY what is said (the model learns word-to-sound from it). Write what is spoken, e.g. "percent S",
not "%s", and spell out numbers the way the teacher says them.
"""
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ============================ CONFIG ============================
OUTPUT_DIR = "/Users/nihalraviganesh/Documents/VoiceReplication/dataset"        # the same folder the other scripts use
INCLUDE_FLAGGED = True
# ================================================================


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows, fields = list(r), list(r.fieldnames)
    if "reviewed" not in fields:
        fields.append("reviewed")
        for row in rows:
            row["reviewed"] = ""
    return rows, fields


def save(path, rows, fields):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def play(wav):
    if os.environ.get("NO_PLAY"):
        return
    if shutil.which("afplay"):
        subprocess.run(["afplay", str(wav)])
    else:
        print(f"   (no audio player found; open {wav})")


def queue_of(rows):
    return [i for i, r in enumerate(rows)
            if r["reviewed"] != "y" and (INCLUDE_FLAGGED or r["exclude"].strip().lower() != "x")]


def main():
    root = Path(OUTPUT_DIR).resolve()
    csv_path, clip_dir = root / "clips.csv", root / "all_clips"
    if not csv_path.exists():
        sys.exit(f"{csv_path} not found. Run the prepare step first.")
    rows, fields = load(csv_path)
    todo = queue_of(rows)
    if not todo:
        sys.exit("Nothing left to review.")
    print(__doc__.split("Everything is saved")[0])
    print(f"{len(todo)} clips to review.\n")

    pos = 0
    while 0 <= pos < len(todo):
        i = todo[pos]
        r = rows[i]
        flag = f"   [flag: {r['flag']}]" if r.get("flag") else ""
        print(f"[{pos + 1}/{len(todo)}] {r['clip']}  {r['duration']}s{flag}")
        print(f"   {r['text']}")
        play(clip_dir / r["clip"])
        while True:
            cmd = input("   > ").strip()
            if cmd == "":
                r["reviewed"] = "y"
                if INCLUDE_FLAGGED:
                    r["exclude"] = ""
                save(csv_path, rows, fields)
                pos += 1
                break
            if cmd == "r":
                play(clip_dir / r["clip"])
            elif cmd == "x":
                r["exclude"], r["reviewed"] = "x", "y"
                save(csv_path, rows, fields)
                pos += 1
                break
            elif cmd == "b":
                pos = max(0, pos - 1)
                break
            elif cmd == "q":
                save(csv_path, rows, fields)
                print(f"Saved. {len(queue_of(rows))} clips still to review.")
                return
            elif cmd.startswith("!") and "=>" in cmd:
                old, new = cmd[1:].split("=>", 1)
                n = 0
                for row in rows:
                    if old and old in row["text"]:
                        row["text"] = row["text"].replace(old, new)
                        n += 1
                save(csv_path, rows, fields)
                print(f"   replaced in {n} clips. This clip now reads:\n   {r['text']}")
            elif "=>" in cmd:
                old, new = cmd.split("=>", 1)
                if old and old in r["text"]:
                    r["text"] = r["text"].replace(old, new)
                    save(csv_path, rows, fields)
                    print(f"   {r['text']}")
                else:
                    print(f"   '{old}' is not in this clip's text.")
            else:
                r["text"] = " ".join(cmd.replace("|", " ").split())
                save(csv_path, rows, fields)
                print(f"   {r['text']}")

    kept = [r for r in rows if r["text"].strip() and r["exclude"].strip().lower() != "x"]
    mins = sum(float(r["duration"]) for r in kept) / 60
    print(f"\nDone reviewing. {len(kept)} clips kept, {mins:.1f} minutes.")
    print("Next:  python prepare_dataset.py build")


if __name__ == "__main__":
    main()
