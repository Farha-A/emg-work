# EMG Data Collection & Fine-Tuning Pipeline

A two-stage pipeline for collecting EMG (electromyography) jaw/facial muscle data via a pygame game, then fine-tuning a neural network classifier on the collected data.

## Overview

```
python main.py
    │
    ├── Stage 1: Data Collection (pygame game)
    │       └── emg_stream.csv  ← raw EMG samples per level
    │
    └── Stage 2: Fine-Tuning
            ├── Loads:   final_model.h5  (Keras, for training)
            ├── Trains on emg_stream.csv
            └── Saves:   finetuned_model.tflite  (TFLite, quantized)
```

## Requirements

- Python 3.8+
- Arduino on COM5 at 115200 baud, sending `filtered,envelope` CSV lines
- Dependencies:

```
pip install pygame pyserial tensorflow numpy pandas scikit-learn imbalanced-learn statsmodels
```

> TFLite is bundled with TensorFlow — no extra install needed. The fine-tuned model is saved with default dynamic-range quantization, reducing file size ~4× vs `.h5`.

## Usage

Run the full pipeline with a single command:

```bash
python main.py
```

Or run each stage independently:

```bash
python game.py        # data collection only
python finetuning.py  # fine-tuning only (uses emg_stream.csv + final_model.h5 → finetuned_model.tflite)
```

## Game Levels

Participants complete 7 levels in order. Each level label maps to a binary class:

| # | Level | Class |
|---|-------|-------|
| 1 | Do Nothing | 0 (rest) |
| 2 | Strong Bite | 1 (active) |
| 3 | Teeth in Contact | 0 (rest) |
| 4 | Preferred Trigger Bite | 1 (active) |
| 5 | Smile | 0 (rest) |
| 6 | Raise Both Eyebrows | 0 (rest) |
| 7 | Gulp (3× swallow) | 0 (rest) |

## Output Files

| File | Description |
|------|-------------|
| `emg_stream.csv` | Raw EMG samples: `timestamp, session_id, level_number, value` |
| `info.csv` | Participant info: name, age, medical history, level durations |
| `finetuned_model.tflite` | Fine-tuned model (TFLite, quantized) saved after pipeline completes |

## Hardware

- EMG sensor connected to Arduino, which streams `filtered,envelope` pairs at 115200 baud over COM5
- To test without hardware, swap `EMGReader` for `EMGStub` in `game.py:24`

## Model

The pipeline fine-tunes `final_model.h5` (Keras) using 4 extracted features per 20-sample segment:

- `filt_AR_2`, `filt_AR_3` — autoregressive coefficients of the filtered signal
- `env_AR_3` — autoregressive coefficient of the envelope
- `env_WAMP` — Willison Amplitude of the envelope

Training uses BorderlineSMOTE to balance classes, then 5 epochs of fine-tuning. The result is exported via `TFLiteConverter` with dynamic-range quantization and saved as `finetuned_model.tflite`.
