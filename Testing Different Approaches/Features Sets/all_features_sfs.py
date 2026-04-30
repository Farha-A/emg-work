"""
All Time-Domain + Frequency-Domain Features + SFS Pipeline
==========================================================
1. Extracts ALL available time-domain AND frequency-domain (spectral)
   features from emg_stream_combined.csv (filtered + envelope).
2. Upsamples minority class (label 1) via random oversampling.
3. Builds a neural network using hyperparameters from hyperparameters.txt.
4. Trains on the full feature set; saves accuracy & per-class recall.
5. Runs free SFS, BFS, and SFS on the combined feature set.
   All results are appended to results.txt with time-domain and
   frequency-domain features listed separately.
"""

import numpy as np
import pandas as pd
import ast
import os
import sys
import datetime

# Suppress TensorFlow oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from scipy.signal import welch
from statsmodels.tsa.ar_model import AutoReg
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, classification_report
from sklearn.utils import resample
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

# Reproducibility
np.random.seed(13)
tf.random.set_seed(13)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    SCRIPT_DIR, os.pardir, os.pardir,
    'Data', 'Stream', 'emg_stream_combined.csv'
)
HYPERPARAMS_PATH = os.path.join(
    SCRIPT_DIR, os.pardir, os.pardir,
    'Optimizing Model Performance', 'hyperparameters.txt'
)
RESULTS_PATH = os.path.join(SCRIPT_DIR, 'results.txt')


# ===========================  FEATURE EXTRACTION  ===========================

def calculate_emg_features(signal, ar_order=4, threshold=0.01):
    """
    Calculates ALL available time-domain EMG features from a 1-D signal.
    Returns a dict of feature_name -> value.
    """
    x = np.array(signal, dtype=float)
    N = len(x)
    if N == 0:
        return {}
    i_idx = np.arange(1, N + 1)

    # --- Base features ---
    wl = np.sum(np.abs(np.diff(x)))
    aac = wl / N
    dasdv = np.sqrt(np.sum(np.diff(x)**2) / max(N - 1, 1))

    # AR coefficients
    try:
        res = AutoReg(x, lags=ar_order).fit()
        ar_coeffs = res.params[1:]
    except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
        ar_coeffs = np.zeros(ar_order)

    # Cepstral Coefficients (CC)
    cc = np.zeros(ar_order)
    if len(ar_coeffs) > 0:
        cc[0] = -ar_coeffs[0]
        for p in range(2, ar_order + 1):
            sum_val = 0
            for l in range(1, p):
                sum_val += (1 - l / p) * ar_coeffs[l - 1] * cc[p - l - 1]
            cc[p - 1] = -ar_coeffs[p - 1] - sum_val

    # IEMG, MAV variants, SSI, VAR, Moments, RMS, V-Order, LOG
    iemg = np.sum(np.abs(x))
    mav = (1 / N) * np.sum(np.abs(x))

    w1 = np.where((i_idx >= 0.25 * N) & (i_idx <= 0.75 * N), 1.0, 0.5)
    mav1 = (1 / N) * np.sum(w1 * np.abs(x))

    w2 = np.ones(N)
    w2[i_idx < 0.25 * N] = 4 * i_idx[i_idx < 0.25 * N] / N
    w2[i_idx > 0.75 * N] = 4 * (i_idx[i_idx > 0.75 * N] - N) / N
    mav2 = (1 / N) * np.sum(w2 * np.abs(x))

    ssi = np.sum(x**2)
    var = (1 / max(N - 1, 1)) * np.sum(x**2)
    tm3 = np.abs((1 / N) * np.sum(x**3))
    tm4 = (1 / N) * np.sum(x**4)
    tm5 = np.abs((1 / N) * np.sum(x**5))
    rms = np.sqrt((1 / N) * np.sum(x**2))
    v_val = ((1 / N) * np.sum(np.abs(x)**2))**0.5
    log_det = np.exp((1 / N) * np.sum(np.log(np.abs(x) + 1e-9)))

    # ZC: Zero Crossing
    zc = 0
    for j in range(N - 1):
        if (x[j] * x[j + 1] < 0) and (np.abs(x[j] - x[j + 1]) >= threshold):
            zc += 1

    # MYOP: Myopulse Percentage Rate
    myop = (1 / N) * np.sum(np.abs(x) >= threshold)

    # WAMP: Willison Amplitude
    wamp = np.sum(np.abs(np.diff(x)) >= threshold)

    # SSC: Slope Sign Change
    ssc = 0
    for j in range(1, N - 1):
        if ((x[j] - x[j - 1]) * (x[j] - x[j + 1])) >= threshold:
            ssc += 1

    # MAVSLP: MAV Slope
    mavslp = np.mean(np.diff(np.abs(x)))

    # MHW & MTW: Modified Hamming / Trapezoidal windows
    mhw = np.sum((x * np.hamming(N))**2)
    mtw = np.sum((x**2) * np.blackman(N))

    features = {
        "WL": wl, "AAC": aac, "DASDV": dasdv, "IEMG": iemg, "MAV": mav,
        "MAV1": mav1, "MAV2": mav2, "SSI": ssi, "VAR": var, "TM3": tm3,
        "TM4": tm4, "TM5": tm5, "RMS": rms, "V_Order": v_val, "LOG": log_det,
        "ZC": zc, "MYOP": myop, "WAMP": wamp, "SSC": ssc,
        "MAVSLP": mavslp, "MHW": mhw, "MTW": mtw
    }
    for i, val in enumerate(ar_coeffs):
        features[f"AR_{i + 1}"] = val
    for i, val in enumerate(cc):
        features[f"CC_{i + 1}"] = val

    return features


def calculate_emg_spectral_features(signal, fs=1000, n=20):
    """
    Calculates frequency-domain EMG features from a 1-D signal.
    Uses Welch's method to estimate the PSD, then derives spectral features.

    Parameters
    ----------
    signal : array-like
        1-D EMG signal segment.
    fs : int
        Sampling frequency in Hz.
    n : int
        Half-bandwidth around PKF for Power Spectrum Ratio (PSR).

    Returns
    -------
    dict : feature_name -> value
    """
    x = np.array(signal, dtype=float)
    if len(x) < 4:
        return {}

    # Estimate PSD via Welch's method
    frequencies, psd = welch(x, fs=fs, nperseg=min(len(x), 256))

    M = len(psd)
    if M == 0 or np.sum(psd) == 0:
        return {}

    # SM0 / TTP (Total Power)
    sm0 = np.sum(psd)
    ttp = sm0

    # Spectral Moments
    sm1 = np.sum(psd * frequencies)
    sm2 = np.sum(psd * (frequencies ** 2))
    sm3 = np.sum(psd * (frequencies ** 3))

    # Mean Frequency (MNF)
    mnf = sm1 / sm0

    # Median Frequency (MDF)
    cumulative_power = np.cumsum(psd)
    mdf_idx = np.where(cumulative_power >= sm0 / 2)[0]
    mdf = frequencies[mdf_idx[0]] if len(mdf_idx) > 0 else 0.0

    # Peak Frequency (PKF)
    pkf_idx = np.argmax(psd)
    pkf = frequencies[pkf_idx]

    # Mean Power (MNP)
    mnp = sm0 / M

    # Frequency Ratio (FR): Low (30-250 Hz) / High (250-1000 Hz)
    low_mask = (frequencies >= 30) & (frequencies <= 250)
    high_mask = (frequencies >= 250) & (frequencies <= 1000)
    fr = (np.sum(psd[low_mask]) / np.sum(psd[high_mask])) if np.any(high_mask) and np.sum(psd[high_mask]) > 0 else 0.0

    # Power Spectrum Ratio (PSR): energy around PKF ± n Hz / total
    psr_mask = (frequencies >= pkf - n) & (frequencies <= pkf + n)
    psr = np.sum(psd[psr_mask]) / sm0

    # Variance of Central Frequency (VCF)
    vcf = (sm2 / sm0) - (sm1 / sm0) ** 2

    return {
        "MNF": mnf, "MDF": mdf, "PKF": pkf, "MNP": mnp,
        "TTP": ttp, "SM1": sm1, "SM2": sm2, "SM3": sm3,
        "FR": fr, "PSR": psr, "VCF": vcf
    }


# ===========================  DATA LOADING  =================================

def load_and_extract_features(file_path, segment_size=50):
    """
    Loads emg_stream_combined.csv (columns: timestamp, session_id,
    level_number, value).  'value' is a string like "(filtered, envelope)".
    Groups consecutive samples by session_id and level_number, segments them
    into windows of `segment_size`, extracts ALL features for both filtered
    and envelope signals, and maps output labels.
    """
    print(f"Loading data from {file_path} ...")
    df = pd.read_csv(file_path)
    print(f"  Raw rows: {df.shape[0]}")

    # Parse the tuple-like 'value' column into two floats
    def parse_value(v):
        try:
            return ast.literal_eval(str(v))
        except (ValueError, SyntaxError):
            return (0.0, 0.0)

    parsed = df['value'].apply(parse_value)
    df['filtered'] = parsed.apply(lambda t: t[0])
    df['envelope'] = parsed.apply(lambda t: t[1])

    # Map level_number -> binary label (same as choosing_features.py)
    label_map = {2: 1, 4: 1, 1: 0, 3: 0, 5: 0, 6: 0, 7: 0}
    df['label'] = df['level_number'].map(label_map)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    # Group by session + level to get contiguous signal chunks
    all_features = []
    for (sid, lvl), grp in df.groupby(['session_id', 'level_number'], sort=False):
        filt_vals = grp['filtered'].values
        env_vals = grp['envelope'].values
        output_label = grp['label'].iloc[0]

        n_samples = len(filt_vals)
        for start in range(0, n_samples, segment_size):
            filt_seg = filt_vals[start:start + segment_size]
            env_seg = env_vals[start:start + segment_size]

            if len(filt_seg) < 5:  # skip tiny tail segments
                continue

            combined = {}

            # --- Time-domain features ---
            filt_feats = calculate_emg_features(filt_seg)
            for k, v in filt_feats.items():
                combined[f'filt_{k}'] = v

            env_feats = calculate_emg_features(env_seg)
            for k, v in env_feats.items():
                combined[f'env_{k}'] = v

            # --- Frequency-domain features ---
            filt_spec = calculate_emg_spectral_features(filt_seg)
            for k, v in filt_spec.items():
                combined[f'filt_spec_{k}'] = v

            env_spec = calculate_emg_spectral_features(env_seg)
            for k, v in env_spec.items():
                combined[f'env_spec_{k}'] = v

            combined['Output'] = output_label
            all_features.append(combined)

    feat_df = pd.DataFrame(all_features)
    # Replace any inf/nan with 0
    feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat_df.fillna(0, inplace=True)

    print(f"  Extracted {len(feat_df)} segments  |  "
          f"Features per segment: {len(feat_df.columns) - 1}")
    print(f"  Class distribution: {dict(feat_df['Output'].value_counts())}")
    return feat_df


# ===========================  UPSAMPLING  ====================================

def upsample_minority(df, target_col='Output'):
    """Random oversampling of the minority class to balance the dataset."""
    counts = df[target_col].value_counts()
    majority_label = counts.idxmax()
    minority_label = counts.idxmin()

    df_majority = df[df[target_col] == majority_label]
    df_minority = df[df[target_col] == minority_label]

    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )
    df_balanced = pd.concat([df_majority, df_minority_upsampled]).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    print(f"  After upsampling: {dict(df_balanced[target_col].value_counts())}")
    return df_balanced


# ===========================  HYPERPARAMETERS  ===============================

def parse_hyperparameters(file_path):
    """
    Reads the LAST set of hyperparameters from the file
    (file may contain multiple runs).
    """
    params = {'hidden_layers': 2, 'neurons': 8, 'dropout': 0.2}
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        # Walk backwards to find the last occurrence of each key
        for line in reversed(lines):
            line = line.strip()
            if 'Hidden Layers:' in line and 'hidden_layers' not in params.get('_found', set()):
                params['hidden_layers'] = int(line.split(':')[-1].strip())
                params.setdefault('_found', set()).add('hidden_layers')
            elif 'Neurons:' in line and 'neurons' not in params.get('_found', set()):
                params['neurons'] = int(line.split(':')[-1].strip())
                params.setdefault('_found', set()).add('neurons')
            elif 'Dropout:' in line and 'dropout' not in params.get('_found', set()):
                params['dropout'] = float(line.split(':')[-1].strip())
                params.setdefault('_found', set()).add('dropout')
        params.pop('_found', None)
    except FileNotFoundError:
        print(f"  Warning: {file_path} not found – using defaults.")
    print(f"  Hyperparameters: {params}")
    return params


# ===========================  MODEL  =========================================

def build_model(input_dim, hp):
    """Build a Keras Sequential NN from the given hyperparameters dict."""
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    for _ in range(hp['hidden_layers']):
        model.add(Dense(hp['neurons'], activation='relu'))
        model.add(Dropout(hp['dropout']))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


def train_and_evaluate(X_train, X_test, y_train, y_test, hp):
    """
    Trains the NN on (X_train, y_train), evaluates on (X_test, y_test).
    Returns (accuracy, recall_class_0, recall_class_1).
    """
    model = build_model(X_train.shape[1], hp)
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    model.fit(
        X_train, y_train,
        epochs=50, batch_size=32, verbose=0,
        validation_split=0.2, callbacks=[es]
    )
    y_pred = (model.predict(X_test, verbose=0) >= 0.5).astype(int).ravel()
    acc = accuracy_score(y_test, y_pred)
    rec_0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    rec_1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    return acc, rec_0, rec_1


# ===========================  FEATURE SELECTION  ============================

def bfs_step(X_train, X_test, y_train, y_test, feature_names, hp, min_features, step_size=5):
    """
    Sequential Backward Selection dropping `step_size` features at a time.
    Returns dict mapping: k -> (selected_feature_names, best_accuracy, recall_0, recall_1)
    """
    selected = list(feature_names)
    results = {}

    while len(selected) > min_features:
        feature_scores = []
        for feat in selected:
            subset = [f for f in selected if f != feat]
            idxs = [feature_names.index(f) for f in subset]
            acc, r0, r1 = train_and_evaluate(X_train[:, idxs], X_test[:, idxs], y_train, y_test, hp)
            feature_scores.append((feat, acc))

        # Highest accuracy when removed -> feature is least useful
        feature_scores.sort(key=lambda x: x[1], reverse=True)

        to_remove = min(step_size, len(selected) - min_features)
        feats_to_remove = [x[0] for x in feature_scores[:to_remove]]

        for f in feats_to_remove:
            selected.remove(f)

        # Re-evaluate the new combined subset
        idxs = [feature_names.index(f) for f in selected]
        acc, r0, r1 = train_and_evaluate(X_train[:, idxs], X_test[:, idxs], y_train, y_test, hp)

        print(f"    BFS Step: dropped {to_remove} features. New size: {len(selected)}, acc: {acc:.4f}")
        results[len(selected)] = (list(selected), acc, r0, r1)

    return results

def sfs_step(X_train, X_test, y_train, y_test, feature_names, hp, max_features, step_size=5):
    """
    Sequential Forward Selection adding `step_size` features at a time.
    Returns dict mapping: k -> (selected_feature_names, best_accuracy, recall_0, recall_1)
    """
    selected = []
    results = {}

    while len(selected) < max_features:
        remaining = [f for f in feature_names if f not in selected]
        if not remaining:
            break

        feature_scores = []
        for feat in remaining:
            subset = selected + [feat]
            idxs = [feature_names.index(f) for f in subset]
            acc, r0, r1 = train_and_evaluate(X_train[:, idxs], X_test[:, idxs], y_train, y_test, hp)
            feature_scores.append((feat, acc))

        # Highest accuracy when added -> feature is most useful
        feature_scores.sort(key=lambda x: x[1], reverse=True)

        to_add = min(step_size, max_features - len(selected))
        if to_add > len(feature_scores):
            to_add = len(feature_scores)

        feats_to_add = [x[0] for x in feature_scores[:to_add]]

        for f in feats_to_add:
            selected.append(f)

        # Re-evaluate the new combined subset
        idxs = [feature_names.index(f) for f in selected]
        acc, r0, r1 = train_and_evaluate(X_train[:, idxs], X_test[:, idxs], y_train, y_test, hp)

        print(f"    SFS Step: added {to_add} features. New size: {len(selected)}, acc: {acc:.4f}")
        results[len(selected)] = (list(selected), acc, r0, r1)

    return results

def free_sfs(X_train, X_test, y_train, y_test, feature_names, hp):
    """
    Standard Free SFS adding 1 feature at a time until accuracy stops improving.
    """
    selected = []
    best_overall_acc = 0.0
    best_overall_r0 = 0.0
    best_overall_r1 = 0.0

    while len(selected) < len(feature_names):
        remaining = [f for f in feature_names if f not in selected]
        if not remaining:
            break

        step_best_feat = None
        step_best_acc = -1.0
        step_best_r0 = 0.0
        step_best_r1 = 0.0

        for feat in remaining:
            subset = selected + [feat]
            idxs = [feature_names.index(f) for f in subset]
            acc, r0, r1 = train_and_evaluate(X_train[:, idxs], X_test[:, idxs], y_train, y_test, hp)
            if acc > step_best_acc:
                step_best_acc = acc
                step_best_feat = feat
                step_best_r0 = r0
                step_best_r1 = r1

        if step_best_acc <= best_overall_acc:
            print(f"    No improvement – stopping at {len(selected)} features.")
            break

        selected.append(step_best_feat)
        best_overall_acc = step_best_acc
        best_overall_r0 = step_best_r0
        best_overall_r1 = step_best_r1

        print(f"    Free SFS Step {len(selected):2d}: +{step_best_feat:<25s}  acc={step_best_acc:.4f}")

    return list(selected), best_overall_acc, best_overall_r0, best_overall_r1


# ===========================  RESULTS I/O  ===================================

def write_result(f, label, n_features, feature_list, acc, rec_0, rec_1,
                 time_domain_features=None, freq_domain_features=None):
    """
    Write one result block to the open file handle.
    If time_domain_features and freq_domain_features lists are provided,
    the selected features are split and printed under separate headings.
    """
    f.write(f"\n{'─'*60}\n")
    f.write(f"  Feature Set : {label}\n")
    f.write(f"  # Features  : {n_features}\n")
    f.write(f"  Accuracy    : {acc * 100:.2f}%\n")
    f.write(f"  Recall (0)  : {rec_0 * 100:.2f}%\n")
    f.write(f"  Recall (1)  : {rec_1 * 100:.2f}%\n")

    if time_domain_features is not None and freq_domain_features is not None:
        td_selected = [ft for ft in feature_list if ft in time_domain_features]
        fd_selected = [ft for ft in feature_list if ft in freq_domain_features]
        f.write(f"  Time-Domain Features  ({len(td_selected)}): {td_selected}\n")
        f.write(f"  Freq-Domain Features  ({len(fd_selected)}): {fd_selected}\n")
    else:
        f.write(f"  Features    : {feature_list}\n")


# ===========================  MAIN  ==========================================

def main():
    print("=" * 60)
    print(f"  Run started at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)

    # 1. Load hyperparameters (latest entry)
    hp = parse_hyperparameters(HYPERPARAMS_PATH)

    # 2. Extract ALL features (time-domain + frequency-domain)
    df = load_and_extract_features(DATA_PATH)
    if df is None or df.empty:
        print("ERROR: no data extracted.")
        return

    # 3. Upsample minority class
    df = upsample_minority(df, 'Output')

    feature_cols = [c for c in df.columns if c != 'Output']
    total_features = len(feature_cols)

    # --- Build separate time-domain / frequency-domain feature name lists ---
    time_domain_features = [c for c in feature_cols if '_spec_' not in c]
    freq_domain_features = [c for c in feature_cols if '_spec_' in c]

    print(f"\n  Total features (all)       : {total_features}")
    print(f"  Time-domain features       : {len(time_domain_features)}")
    print(f"  Frequency-domain features  : {len(freq_domain_features)}")

    X = df[feature_cols].values
    y = df['Output'].values

    # Single train/test split shared across all experiments
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results_file = open(RESULTS_PATH, 'a', encoding='utf-8')
    results_file.write("=" * 60 + "\n")
    results_file.write(f"  Time + Frequency Domain Feature Selection Results\n")
    results_file.write(f"  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
    results_file.write("=" * 60 + "\n")
    results_file.write(f"  Time-domain features  ({len(time_domain_features)}): {time_domain_features}\n")
    results_file.write(f"  Freq-domain features  ({len(freq_domain_features)}): {freq_domain_features}\n")

    # ------- 4. Full feature set (time + frequency) -------
    print("\n>>> 1. Training on ALL features (time + frequency) ...")
    acc, r0, r1 = train_and_evaluate(X_train, X_test, y_train, y_test, hp)
    print(f"    Accuracy={acc:.4f}  Recall(0)={r0:.4f}  Recall(1)={r1:.4f}")
    write_result(results_file, "All Features (Time + Freq)", total_features,
                 feature_cols, acc, r0, r1,
                 time_domain_features, freq_domain_features)

    # ------- 5. Free SFS -------
    print("\n>>> 2. Free SFS (no limit, adding 1 by 1) ...")
    sel, acc, r0, r1 = free_sfs(X_train, X_test, y_train, y_test, feature_cols, hp)
    write_result(results_file, "SFS (free)", len(sel), sel, acc, r0, r1,
                 time_domain_features, freq_domain_features)

    # ------- 6. BFS dropping 5 -------
    half_features = total_features // 2
    print(f"\n>>> 3. BFS starting from {total_features} down to {half_features} (decreasing by 5) ...")
    bfs_results = bfs_step(X_train, X_test, y_train, y_test, feature_cols, hp, min_features=half_features, step_size=5)

    for k in sorted(bfs_results.keys(), reverse=True):
        sel, acc, r0, r1 = bfs_results[k]
        write_result(results_file, f"BFS ({k} features)", len(sel), sel, acc, r0, r1,
                     time_domain_features, freq_domain_features)

    # ------- 7. SFS adding 5 -------
    sfs_max = half_features - 5 if half_features % 5 == 0 else half_features - (half_features % 5)
    print(f"\n>>> 4. SFS starting from 0 up to {sfs_max} (adding 5 at a time) ...")
    sfs_results = sfs_step(X_train, X_test, y_train, y_test, feature_cols, hp, max_features=sfs_max, step_size=5)

    for k in sorted(sfs_results.keys(), reverse=True):
        sel, acc, r0, r1 = sfs_results[k]
        write_result(results_file, f"SFS ({k} features)", len(sel), sel, acc, r0, r1,
                     time_domain_features, freq_domain_features)

    results_file.write(f"\n{'='*60}\n")
    results_file.close()
    print(f"\n✓ All results saved to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
