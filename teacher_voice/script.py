"""Type a sentence, get a .wav of it spoken in the teacher's voice.

    python script.py                      interactive: asks for a sentence, again and again (q to quit)
    python script.py "One sentence."      one sentence, then exit
    python script.py --no-play            do not play the audio automatically (macOS)

Every wav is saved in the outputs/ folder next to this file.
"""
import argparse, contextlib, io, os, re, subprocess, sys, time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "final_model"
OUT_DIR = HERE / "outputs"
sys.path.insert(0, str(HERE / "F5-TTS" / "src"))       # the bundled F5-TTS code (works even without `pip install -e`)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

import torch
import soundfile as sf
from num2words import num2words
from omegaconf import OmegaConf

NFE_STEPS = 32           # denoising steps; 32 is the F5-TTS default (more = slower, no real gain)
SEED = 1234


def spoken_form(text):
    """Digits and symbols -> words, the way the teacher says them: 0x41 -> 'zero X four one', %x -> 'percent X'."""
    def hex_(m):
        return " zero X " + " ".join(num2words(int(c)) if c.isdigit() else c.upper() for c in m.group(1)) + " "
    t = re.sub(r"\b0[xX]([0-9a-fA-F]+)\b", hex_, text)
    t = re.sub(r"%\s*([a-zA-Z])\b", lambda m: " percent " + m.group(1).upper() + " ", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*%", lambda m: m.group(1) + " percent", t)
    t = re.sub(r"\d+\.\d+|\d+", lambda m: num2words(float(m.group()) if "." in m.group() else int(m.group())).replace("-", " "), t)
    for sym, word in {"&": " and ", "+": " plus ", "=": " equals ", "@": " at ", "#": " hash ", "\\": " backslash ", "*": " star "}.items():
        t = t.replace(sym, word)
    t = " ".join(t.split())
    return re.sub(r"\s+([,.;:?!])", r"\1", t)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TeacherVoice:
    def __init__(self, device=None):
        from f5_tts.infer.utils_infer import load_model, load_vocoder, preprocess_ref_audio_text
        from f5_tts.model import DiT
        self.device = device or pick_device()
        for f in ("teacher_f5.safetensors", "vocab.txt", "voice_prompt.wav", "voice_prompt.txt"):
            if not (MODEL_DIR / f).exists():
                sys.exit(f"Missing {MODEL_DIR / f}. The final_model/ folder must contain the model files.")
        print(f"Loading the model on {self.device} (about 20 seconds, only once)...", flush=True)
        arch = OmegaConf.to_container(OmegaConf.load(HERE / "F5-TTS/src/f5_tts/configs/F5TTS_v1_Base.yaml").model.arch)
        with contextlib.redirect_stdout(io.StringIO()):
            self.vocoder = load_vocoder(vocoder_name="vocos", device=self.device)
            # the saved file holds the raw fine-tuned weights, so use_ema=False
            self.model = load_model(DiT, arch, str(MODEL_DIR / "teacher_f5.safetensors"), mel_spec_type="vocos",
                                    vocab_file=str(MODEL_DIR / "vocab.txt"), use_ema=False, device=self.device)
            self.ref_audio, self.ref_text = preprocess_ref_audio_text(
                str(MODEL_DIR / "voice_prompt.wav"), (MODEL_DIR / "voice_prompt.txt").read_text(encoding="utf-8").strip(),
                show_info=lambda *a, **k: None)

    def speak(self, text, out_path):
        """Generate `text` in the teacher's voice, write it to out_path, return (spoken_text, seconds_of_audio)."""
        from f5_tts.infer.utils_infer import infer_process
        spoken = spoken_form(text)
        torch.manual_seed(SEED)
        with contextlib.redirect_stdout(io.StringIO()):
            wave, sr, _ = infer_process(self.ref_audio, self.ref_text, spoken, self.model, self.vocoder,
                                        mel_spec_type="vocos", nfe_step=NFE_STEPS, device=self.device,
                                        show_info=lambda *a, **k: None)
        sf.write(str(out_path), wave, sr)
        return spoken, len(wave) / sr


def out_name(text):
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "audio"
    return OUT_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{slug}.wav"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="sentence to speak (omit to be asked interactively)")
    ap.add_argument("--no-play", action="store_true", help="do not play the audio after generating it")
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    voice = TeacherVoice()
    play = (sys.platform == "darwin") and not args.no_play

    def run_one(text):
        path = out_name(text)
        t0 = time.time()
        spoken, secs = voice.speak(text, path)
        if spoken != text:
            print(f"  spoken as: {spoken}")
        print(f"  saved: {path}  ({secs:.1f} s of audio, made in {time.time() - t0:.0f} s)")
        if play:
            subprocess.run(["afplay", str(path)])

    if args.text:
        run_one(args.text)
        return
    print("\nReady. Type a sentence and press Enter. Type q to quit.\n")
    while True:
        try:
            text = input("Sentence> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in ("q", "quit", "exit"):
            break
        if text:
            run_one(text)


if __name__ == "__main__":
    main()
