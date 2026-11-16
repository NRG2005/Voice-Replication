# Teacher voice (F5-TTS, fine-tuned)

Type a sentence, get a .wav of it in the teacher's voice.

## Run it
```bash
./venv/bin/python script.py                 # asks for a sentence, again and again (q to quit)
./venv/bin/python script.py "One sentence." # one sentence, then exit
```
Wavs are saved in `outputs/`. On a Mac the audio also plays automatically (`--no-play` turns that off).
Symbols and numbers are converted to words first: `0x41` -> "zero X four one", `%x` -> "percent X", `100%` -> "one hundred percent".

## First-time setup (once per computer)
```bash
./setup.sh      # creates ./venv and installs everything (needs Python 3.11; takes a few minutes)
```
Tested on an Apple Silicon Mac. It also runs on CPU or an NVIDIA GPU, but CPU is slow (a minute or two per sentence;
about 15 seconds on a Mac, faster on a GPU).

## What is in this folder
| Item | What it is |
|---|---|
| `final_model/teacher_f5.safetensors` | the fine-tuned model (643 MB). This is the actual result of the training |
| `final_model/vocab.txt` | the characters the model understands |
| `final_model/voice_prompt.wav` + `.txt` | a short real clip of the teacher, used as the voice example every time |
| `F5-TTS/` | the F5-TTS code (open source, pinned version) |
| `script.py` | the program you run |
| `requirements.txt`, `setup.sh` | the exact package versions and the installer |

The first run also downloads the Vocos vocoder (about 50 MB) from HuggingFace, so it needs internet once.

## Rules
- The F5-TTS pretrained weights are licensed CC-BY-NC, so this is for non-commercial (course/research) use only.
- This model can imitate a real person's voice. Share it only inside the team, and do not publish the model or its audio without the teacher's permission.
