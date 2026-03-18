import numpy as np
from statsmodels.tsa.ar_model import AutoReg


class FeatureEngineer:
    """Centralises all EMG data-processing and feature-extraction logic."""

    # ------------------------------------------------------------------ #
    #  Low-level helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def average_segments(lst, n_segments=3):
        """Split *lst* into *n_segments* equal parts and return element-wise means."""
        n = len(lst) // n_segments
        if n == 0:
            return []
        parts = [lst[i * n : (i + 1) * n] for i in range(n_segments)]
        return [sum(float(x) for x in group) / n_segments for group in zip(*parts)]

    @staticmethod
    def parse_array_string(s):
        """Parse a string like '[0.1 0.2 ...]' into a numpy array."""
        try:
            clean = s.replace('[', '').replace(']', '').replace('\n', ' ')
            return np.fromstring(clean, sep=' ')
        except Exception:
            return np.zeros(4)

    @staticmethod
    def get_output_label(level_number):
        """Map *level_number* → binary label (gulp-last mapping).

        Swallow levels (2, 4) → 1, all others → 0.
        Returns -1 for unexpected values.
        """
        if level_number in [2, 4]:
            return 1
        elif level_number in [1, 3, 5, 6, 7]:
            return 0
        return -1

    # ------------------------------------------------------------------ #
    #  Feature extraction
    # ------------------------------------------------------------------ #
    @staticmethod
    def calculate_emg_features(signal, ar_order=4):
        """Calculate EMG features (AAC, WL, AR_2) from *signal*.

        Returns a dict with keys ``'AAC'``, ``'WL'``, ``'AR_2'``.
        """
        x = np.array(signal)
        N = len(x)

        # WL – Waveform Length
        if N > 1:
            wl = np.sum(np.abs(np.diff(x)))
        else:
            wl = 0.0

        # AAC – Average Amplitude Change (WL / N)
        if N > 0:
            aac = wl / N
        else:
            aac = 0.0

        # AR_2 – 2nd Autoregressive coefficient
        try:
            res = AutoReg(x, lags=ar_order).fit()
            ar_coeffs = res.params[1:]  # [a1, a2, a3, a4]
            ar_2 = ar_coeffs[1] if len(ar_coeffs) > 1 else 0.0
        except (ValueError, Exception):
            ar_2 = 0.0

        return {"AAC": aac, "WL": wl, "AR_2": ar_2}

    # ------------------------------------------------------------------ #
    #  High-level pipelines
    # ------------------------------------------------------------------ #
    @staticmethod
    def load_raw_csv(csv_path, n_segments=3):
        """Load a raw EMG CSV, parse value tuples, group & average segments.

        Returns a processed ``DataFrame`` with columns:
        ``session_id, level_number, filtered_values, envelope_values``.
        """
        import pandas as pd
        import ast
        import os

        if not os.path.exists(csv_path):
            print(f"Error: File '{csv_path}' not found.")
            return pd.DataFrame()

        file = pd.read_csv(csv_path)

        if 'value' not in file.columns:
            print("Error: 'value' column missing.")
            return pd.DataFrame()

        try:
            file['parsed_value'] = file['value'].apply(ast.literal_eval)
        except Exception as e:
            print(f"Error parsing values: {e}")
            return pd.DataFrame()

        file['filtered_value'] = file['parsed_value'].apply(lambda x: x[0])
        file['envelope_value'] = file['parsed_value'].apply(lambda x: x[1])

        data = (
            file
            .groupby(['session_id', 'level_number'])[['filtered_value', 'envelope_value']]
            .agg(list)
            .reset_index()
        )
        data.columns = ['session_id', 'level_number', 'filtered_values', 'envelope_values']

        avg = FeatureEngineer.average_segments
        data['filtered_values'] = data['filtered_values'].apply(lambda v: avg(v, n_segments))
        data['envelope_values'] = data['envelope_values'].apply(lambda v: avg(v, n_segments))

        return data

    @staticmethod
    def extract_features_from_df(df, segment_size=50):
        """Extract features from a processed DataFrame.

        Returns a ``DataFrame`` with columns ``['filt_AAC', 'env_WL', 'filt_AR_2', 'Output']``.
        """
        import pandas as pd

        if df.empty:
            return pd.DataFrame()

        extracted = []
        for _, row in df.iterrows():
            filtered_values = row['filtered_values']
            envelope_values = row['envelope_values']
            label = FeatureEngineer.get_output_label(row['level_number'])

            if label == -1:
                print(f"Warning: Unexpected level_number {row['level_number']}")

            max_len = max(len(filtered_values), len(envelope_values))
            for i in range(0, max_len, segment_size):
                filtered_segment = filtered_values[i : i + segment_size]
                envelope_segment = envelope_values[i : i + segment_size]

                if len(filtered_segment) > 0 or len(envelope_segment) > 0:
                    filtered_feats = (
                        FeatureEngineer.calculate_emg_features(filtered_segment)
                        if len(filtered_segment) > 0
                        else {"AAC": 0.0, "WL": 0.0, "AR_2": 0.0}
                    )
                    envelope_feats = (
                        FeatureEngineer.calculate_emg_features(envelope_segment)
                        if len(envelope_segment) > 0
                        else {"AAC": 0.0, "WL": 0.0, "AR_2": 0.0}
                    )
                    extracted.append({
                        "filt_AAC": filtered_feats["AAC"],
                        "env_WL": envelope_feats["WL"],
                        "filt_AR_2": filtered_feats["AR_2"],
                        "Output": label,
                    })

        features_df = pd.DataFrame(extracted)
        if not features_df.empty:
            features_df = features_df[['filt_AAC', 'env_WL', 'filt_AR_2', 'Output']]
        return features_df