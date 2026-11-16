# Voice-Replication

Replicate a lecturer's voice with deep learning, so that an AI-generated lecture summary can be **spoken in the teacher's own voice**.
This is the voice module of a capstone project, *AI-Powered Lecture Video Summarizer* (chatbot, summaries, voice agent).

Starting from ~24 minutes of one teacher's lecture audio, we compared three voice-cloning approaches, fine-tuned the best one
([F5-TTS](https://github.com/SWivid/F5-TTS)), and measured the result against real held-out recordings of the teacher.

> **Not included:** the teacher's recordings, the dataset clips, generated audio and the trained model weights. They contain a real
> person's voice, so they are not published. Everything needed to repeat the work with your own speaker is here.

## Results

**1. Zero-shot comparison** (no training; each method imitates the voice from a short clip; 20 test sentences, `compare_out/summary.csv`)

| Method | Voice match (speaker similarity, higher is better) | Word errors (WER, lower is better) | Naturalness (UTMOS, higher is better) |
|---|---|---|---|
| F5-TTS | **0.662** | **3.4%** | 2.24 |
| kNN-VC | 0.625 | 4.3% | 2.29 |
| XTTS v2 | 0.380 | 7.7% | **2.59** |

F5-TTS was clearly the closest to the teacher, so it was the one we fine-tuned.

**2. Fine-tuning F5-TTS** (4,032 updates on 200 clips = 23.9 min; scored on 10 held-out sentences the model never trained on, plus 3 technical sentences)

| Model | Voice match | Word errors | Naturalness |
|---|---|---|---|
| Real teacher recordings (ceiling) | 0.863 | 1.4% | 2.08 |
| F5-TTS, no fine-tuning | 0.656 | 1.9% | 1.72 |
| F5-TTS, fine-tuned (EMA weights) | 0.666 | 1.8% | 1.77 |
| **F5-TTS, fine-tuned (raw weights)** | **0.821** | 5.1% | 1.54 |

- Fine-tuning moved the voice match from 0.656 to 0.821, close to the 0.863 real recordings score against each other.
- The EMA weights barely changed: F5-TTS stores a step counter of 1,250,000 for them, so over only ~4,000 updates they move just a few percent. For short fine-tunes use the raw weights.
- Word errors are inflated because the held-out transcripts contain spoken symbols ("percentile x"); on the three technical sentences they were 0-5%.
- Naturalness is lower than the baseline because the model also copies the classroom recording noise.
- Caveats: small test set (13 sentences), one random seed, machine metrics and not a listening test. The two tables use different sentence sets, so compare within a table, not across.

Metrics: speaker similarity = cosine similarity of ECAPA-TDNN speaker embeddings against the teacher's held-out recordings; WER = Whisper
transcribes the output and compares it to the intended text; UTMOS = a model that predicts naturalness on a 1-5 scale.

## How it works

```
text ─► spoken_form (digits/symbols -> words) ─► F5-TTS (337M params) ─► mel spectrogram ─► Vocos vocoder ─► .wav
                                                        ▲
                                  ~6 s teacher clip + its transcript (the voice example)
```

F5-TTS is a non-autoregressive text-to-speech model: a Diffusion Transformer trained with *flow matching* on a speech-infilling task
(hide part of a spectrogram, predict it from the rest and the text). At generation time the voice example is the visible part, so the
model continues it in the same voice. Fine-tuning continues that same training on the target speaker.

## Repository layout

| Path | What it is |
|---|---|
| `prepare_from_transcript.py` | cuts lecture audio into 3-8 s clips at word pauses using a word-timestamped transcript, flags risky clips |
| `prepare_dataset.py` | `prepare` (video -> clips + Whisper transcripts) and `build` (reviewed clips -> `dataset_for_training.zip` + held-out set) |
| `review_clips.py` | fast by-ear review tool (macOS) to correct transcripts before training |
| `python notebooks/1_xtts.ipynb`, `2_f5tts.ipynb`, `3_knn_vc.ipynb` | Colab notebooks for the three zero-shot methods |
| `python notebooks/5_f5tts_finetune.ipynb` | **Colab notebook that fine-tunes F5-TTS** and scores every checkpoint (needs a GPU; a free T4 is enough) |
| `compare_local.py`, `compare_out/` | scores the three methods locally; result tables |
| `knn_vc_local*.py` | local kNN-VC experiments |
| `teacher_voice/` | the inference tool: `script.py` asks for a sentence and writes a wav in the trained voice |
| `dataset/summary.txt` | size of the dataset we built (200 train clips / 17 held-out clips) |

## Reproduce it with your own speaker

1. **Build a dataset** from a lecture video with a word-timestamped transcript: `prepare_from_transcript.py`, review with
   `review_clips.py`, then `python prepare_dataset.py build`. It produces `dataset_for_training.zip` and `held_out.zip`.
   Aim for 20+ minutes of clean speech, and keep a few minutes held out.
2. **Fine-tune on Colab:** open `python notebooks/5_f5tts_finetune.ipynb`, set a GPU runtime, upload the two zips into the Files panel
   (or a Drive folder), and run the cells in order. It trains, scores every checkpoint and exports the best model.
   A free T4 session can run out of GPU quota, so the notebook copies the final checkpoint to Drive.
3. **Use the model locally:** put `teacher_f5.safetensors`, `vocab.txt`, `voice_prompt.wav` and `voice_prompt.txt` (a clean ~6 s clip
   of the speaker and its exact transcript) in `teacher_voice/final_model/`, then:

```bash
cd teacher_voice
./setup.sh                          # once: creates ./venv and installs everything (Python 3.11)
./venv/bin/python script.py         # asks for a sentence, writes a wav to outputs/ (q to quit)
./venv/bin/python script.py "One sentence."
```

It runs fully offline after the first run (the Vocos vocoder downloads once). Speed: about 15 s per sentence on an Apple Silicon Mac;
a CPU alone is much slower. Tested on macOS only.

## Licences and responsible use

- This repository bundles a lightly patched copy of F5-TTS (`teacher_voice/F5-TTS`, MIT licence, see its `LICENSE`).
- The pretrained F5-TTS weights are **CC-BY-NC**: models fine-tuned from them are for **non-commercial** use.
  XTTS v2 (Coqui Public Model Licence) is also non-commercial.
- A voice model can imitate a real person. Only clone voices you have permission to clone, do not publish a cloned voice or its
  audio without that person's consent, and label generated audio as synthetic.
