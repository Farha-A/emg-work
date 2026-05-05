# EMG Binary Classification Pipeline

An end-to-end machine-learning pipeline for classifying **clenching activity** from surface electromyography (sEMG) signals. Raw sensor data is collected via an Arduino, cleaned, segmented, and fed through feature engineering and neural-network training stages to produce a real-time binary classifier (clench vs. no-clench).

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline Stages](#pipeline-stages)
  - [1. Data Collection & Logging](#1-data-collection--logging)
  - [2. Data Cleaning](#2-data-cleaning)
  - [3. Feature Engineering](#3-feature-engineering)
  - [4. Model Training](#4-model-training)
  - [5. Hyperparameter Optimization](#5-hyperparameter-optimization)
  - [6. Feature Selection](#6-feature-selection)
  - [7. Fine-tuning & Live Inference](#7-fine-tuning--live-inference)
- [Models Explored](#models-explored)
- [Experiment Tracking](#experiment-tracking)
- [Configuration](#configuration)
- [Getting Started](#getting-started)
- [Key Results](#key-results)

---

## Overview

| Item | Detail |
| --- | --- |
| **Signal source** | sEMG via Arduino (serial / keyboard simulation) |
| **Channels used** | `filtered` and `envelope` values from the EMG shield |
| **Label scheme** | Levels 2, 4 → **1** (clench); Levels 1, 3, 5, 6, 7 → **0** (no clench) |
| **Segment size** | 50 samples per window (configurable) |
| **Primary model** | Dense neural network (Keras / TensorFlow) |
| **Class balancing** | Borderline SMOTE / SMOTE / random oversampling |
| **Experiment tracking** | MLflow (SQLite backend) |

---

## Project Structure

```text
├── config.py                 # Environment config (.env loader, input mode, serial port)
├── constants.py              # Screen dims, FPS, test duration, logging interval
├── emg.py                    # EMGReader – serial/keyboard data acquisition
├── faux_data_gen.py          # Arduino EMG signal simulator with live matplotlib plot
├── feature_engineering.py    # FeatureEngineer class – segmentation, feature extraction, SMOTE
├── finetuning.py             # Fine-tune a saved model on new raw CSV data
├── livestream.py             # Real-time inference via sliding buffer + feature extraction
├── logger.py                 # Logger class – session metadata & high-frequency stream CSV logging
├── model.py                  # EMGModel class – build, train, evaluate, save/load Keras models
├── .env                      # Local environment overrides (EMG_INPUT_MODE, SIMULATION_DURATION)
│
├── Data/
│   ├── Clean Stream/         # Cleaned & grouped CSVs (session × level, averaged segments)
│   ├── Features/             # Pre-computed feature CSVs (DASDV/MYOP, CC, all-features, etc.)
│   ├── Info/                 # Session metadata CSVs (participant info, ratings)
│   └── Stream/               # Raw EMG stream CSVs (timestamp, session_id, level, value)
│
├── EDA/
│   ├── 4lvls.ipynb           # Exploratory analysis – 4-level experiment
│   ├── 7lvls.ipynb           # Exploratory analysis – 7-level experiment
│   └── features.ipynb        # Feature analysis, correlation removal, zero-variance filtering
│
├── Mains/
│   ├── clean_data_main.py    # CLI entry point: raw CSV → cleaned DataFrame → output CSV
│   ├── feature_engineering_main.py   # CLI entry point: cleaned CSV → feature CSV
│   └── model_main.py         # CLI entry point: feature CSV → train NN → MLflow logging
│
├── Models/
│   ├── best_model.h5         # Best-performing saved model
│   ├── bg_model.h5           # Default training output model
│   ├── finetuned_model.h5    # Model after fine-tuning on new data
│   └── optimized_model.h5    # Model from GWO-optimized hyperparameters
│
├── Optimizing Model Performance/
│   ├── model_optimization.py         # Grey Wolf Optimizer for NN hyperparameters
│   ├── choosing_features.py          # SFS feature selection (accuracy-based)
│   ├── choosing_features_auc.py      # SFS feature selection (AUC-ROC + accuracy)
│   ├── hyperparameters.txt           # GWO output: best hidden_layers, neurons, dropout
│   ├── choosing_features_output.txt  # SFS run logs
│   └── choosing_features_auc_output.txt  # AUC-SFS run logs
│
├── Testing Different Approaches/
│   ├── Features Sets/
│   │   └── all_features_sfs.py       # Full time + frequency domain SFS/BFS pipeline
│   ├── Models - Raw Stream/
│   │   ├── lstm.py                   # LSTM on raw sequences (with focal loss)
│   │   └── rnn.py                    # RNN on raw sequences
│   ├── Models - Time Domain Features/
│   │   ├── knn.py                    # K-Nearest Neighbors classifier
│   │   ├── logistic_regression.py    # Logistic Regression classifier
│   │   └── random_forest.py          # Random Forest classifier
│   └── Models - Time-Freq/
│       ├── nn_time_freq.py           # Dense NN on selected time + frequency features
│       ├── nn_dasdv_myop.py          # Dense NN on DASDV + MYOP features
│       └── cnnLstm.py               # CNN-LSTM hybrid model
│
├── One timers/
│   ├── clean_ids.py          # Remove broken session IDs from data
│   ├── extract_features.py   # CLI wrapper to extract features from any CSV
│   ├── map_levels.py         # Utility to remap level numbers
│   ├── read_h5_model.py      # Inspect saved .h5 model weights/architecture
│   ├── remove_specific_instance.py  # Remove specific data instances
│   ├── verify_loading.py     # Verify model loads and predicts correctly
│   └── _merge_mlflow.py      # Merge MLflow databases
│
└── Text Notes/
    ├── notes                 # Research notes, experiment observations
    └── what files do.txt     # File-by-file documentation
```

---

## Pipeline Stages

### 1. Data Collection & Logging

**Modules:** `emg.py`, `logger.py`, `constants.py`, `config.py`

- `EMGReader` connects to the Arduino via serial port (or simulates input via spacebar) and reads `filtered` and `envelope` EMG values in a background thread.
- `Logger` handles two log files:
  - **`info.csv`** — session metadata (participant name, age, medical history, level, duration, ease rating).
  - **`emg_stream.csv`** — high-frequency time-series data (timestamp, session_id, level_number, value tuple).
- Logging interval is configurable (default: `0.01s` = 100 Hz).

### 2. Data Cleaning

**Entry point:** `Mains/clean_data_main.py`

```bash
python Mains/clean_data_main.py Data/Stream/emg_stream_h.csv Data/Clean\ Stream/emg_cleaned.csv
```

- Parses raw `(filtered, envelope)` tuples from the stream CSV.
- Groups data by `session_id` and `level_number`.
- Splits each group into 3 segments and computes element-wise averages.
- Outputs a cleaned DataFrame with columns: `session_id`, `level_number`, `filtered_values`, `envelope_values`.

### 3. Feature Engineering

**Entry point:** `Mains/feature_engineering_main.py`  
**Core module:** `feature_engineering.py`

Extracts features from 50-sample windows of both filtered and envelope signals:

| Category | Features |
| --- | --- |
| **Time-domain** | WL, AAC, DASDV, IEMG, MAV, MAV1, MAV2, SSI, VAR, TM3–TM5, RMS, V-Order, LOG, ZC, MYOP, WAMP, SSC, MAVSLP, MHW, MTW |
| **AR / Cepstral** | AR(1)–AR(4), CC(1)–CC(4) |
| **Frequency-domain** | MNF, MDF, PKF, MNP, TTP, SM1–SM3, FR, PSR, VCF |

Class imbalance is handled via **Borderline SMOTE** (`feature_engineering.py`) or random oversampling (`all_features_sfs.py`).

### 4. Model Training

**Entry point:** `Mains/model_main.py`  
**Core module:** `model.py`

```bash
python Mains/model_main.py Data/Features/emg_features.csv
```

`EMGModel` wraps a Keras `Sequential` network:

- Configurable hidden layers, neurons per layer, and dropout rate.
- Binary cross-entropy loss with Adam optimizer.
- Early stopping on accuracy (patience = 5).
- Automatic train/test split (80/20) and training graph generation.
- All runs are logged to MLflow.

### 5. Hyperparameter Optimization

**Script:** `Optimizing Model Performance/model_optimization.py`

Uses the **Grey Wolf Optimization (GWO)** metaheuristic to search over:

| Parameter | Search Range |
| --- | --- |
| Hidden layers | 1 – 2 |
| Neurons | 8, 16, 32, 64 |
| Dropout | 0.0 – 0.5 |

Best result is appended to `hyperparameters.txt` and used by downstream scripts.

**Best hyperparameters found:** 2 hidden layers, 8 neurons, 0.41 dropout → **87.25% accuracy**.

### 6. Feature Selection

**Scripts:**

- `Optimizing Model Performance/choosing_features.py` — Sequential Forward Selection (SFS) using accuracy.
- `Optimizing Model Performance/choosing_features_auc.py` — SFS using AUC-ROC + accuracy combined score.
- `Testing Different Approaches/Features Sets/all_features_sfs.py` — Full pipeline with SFS, BFS, and free SFS on all time + frequency features.

**Latest AUC-based SFS result:**

- Best features: `['filt_AR_2', 'filt_MAVSLP', 'filt_AR_3']`
- AUC-ROC: **0.8330** | Accuracy: **79.41%** | Combined: **1.6272**

### 7. Fine-tuning & Live Inference

**Fine-tuning:** `finetuning.py`

- Loads a pre-trained model and new raw data, extracts features on the fly, and fine-tunes with a fresh optimizer.

**Live inference:** `livestream.py`

- Maintains a 50-sample sliding buffer of incoming EMG values.
- Extracts features from the buffer and runs the model's `predict()` method.
- Clears the buffer upon detecting a clench event.
- Supports both serial COM input and keyboard simulation.

```bash
# Real-time classification (COM port)
python livestream.py

# Keyboard simulation mode (set EMG_INPUT_MODE=KEYBOARD in .env)
python livestream.py
```

---

## Models Explored

| Model | Input Type | Script |
| --- | --- | --- |
| **Dense NN** (primary) | Time-domain features | `model.py`, `Mains/model_main.py` |
| **Dense NN** | Time + frequency features | `Testing Different Approaches/Models - Time-Freq/nn_time_freq.py` |
| **Dense NN** | DASDV + MYOP only | `Testing Different Approaches/Models - Time-Freq/nn_dasdv_myop.py` |
| **LSTM** | Raw sequences | `Testing Different Approaches/Models - Raw Stream/lstm.py` |
| **RNN** | Raw sequences | `Testing Different Approaches/Models - Raw Stream/rnn.py` |
| **CNN-LSTM** | Raw sequences | `Testing Different Approaches/Models - Time-Freq/cnnLstm.py` |
| **KNN** | Time-domain features | `Testing Different Approaches/Models - Time Domain Features/knn.py` |
| **Logistic Regression** | Time-domain features | `Testing Different Approaches/Models - Time Domain Features/logistic_regression.py` |
| **Random Forest** | Time-domain features | `Testing Different Approaches/Models - Time Domain Features/random_forest.py` |

---

## Experiment Tracking

All training runs are logged to **MLflow** using a local SQLite backend (`mlflow.db`).

Logged artifacts include:

- Hyperparameters (architecture, learning rate, dropout, etc.)
- Per-epoch training/validation accuracy and loss
- Test accuracy, per-class recall, AUC-ROC
- Saved model files (`.h5` / `.pkl`)
- Training graphs and confusion matrices

---

## Configuration

### Environment Variables (`.env`)

| Variable | Default | Description |
| --- | --- | --- |
| `EMG_INPUT_MODE` | `COM` | `COM` for serial port, `KEYBOARD` for spacebar simulation |
| `EMG_PORT` | `COM4` | Serial port for the Arduino |
| `EMG_BAUD` | `115200` | Serial baud rate |
| `SIMULATION_DURATION` | `0.6` | Duration of simulated signal burst (seconds) |

### Constants (`constants.py`)

| Constant | Value | Description |
| --- | --- | --- |
| `WIDTH` | 960 | Screen width |
| `HEIGHT` | 640 | Screen height |
| `FPS` | 60 | Frames per second |
| `TEST_DURATION` | 3 | Test duration in seconds |
| `LOGGING_INTERVAL` | 0.01 | Logging interval (seconds) |

---

## Getting Started

### Prerequisites

- Python 3.9+
- Arduino with EMG shield (or use keyboard simulation mode)

### Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate    # Windows

# Install dependencies
pip install numpy pandas scikit-learn tensorflow matplotlib
pip install imbalanced-learn statsmodels mlflow pyserial scipy joblib
```

### Quick Start

```bash
# 1. Clean raw stream data
python Mains/clean_data_main.py Data/Stream/emg_stream_combined.csv Data/Clean\ Stream/emg_cleaned.csv

# 2. Extract features
python Mains/feature_engineering_main.py Data/Clean\ Stream/emg_cleaned.csv

# 3. Train model (with MLflow logging)
python Mains/model_main.py emg_features.csv

# 4. Run hyperparameter optimization (optional)
python "Optimizing Model Performance/model_optimization.py"

# 5. Run feature selection (optional)
python "Optimizing Model Performance/choosing_features_auc.py"

# 6. Live inference
python livestream.py
```

---

## Key Results

| Metric | Value | Context |
| --- | --- | --- |
| **Best NN Accuracy** | 87.25% | GWO-optimized hyperparameters |
| **Best AUC-ROC** | 0.8330 | SFS with 3 features (filt_AR_2, filt_MAVSLP, filt_AR_3) |
| **Optimal Architecture** | 2 layers × 8 neurons, 0.41 dropout | Found via Grey Wolf Optimization |
