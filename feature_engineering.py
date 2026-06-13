import numpy as np


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
    def calculate_emg_features(signal, threshold=0.01):
        """Calculate EMG features (DASDV, MYOP) from *signal*.

        Returns a dict with keys ``'DASDV'``, ``'MYOP'``.
        """
        x = np.array(signal)
        N = len(x)

        # DASDV – Difference Absolute Standard Deviation Value
        if N > 1:
            dasdv = np.sqrt(np.sum(np.diff(x)**2) / (N - 1))
        else:
            dasdv = 0.0

        # MYOP – Myopulse Percentage Rate
        if N > 0:
            myop = (1 / N) * np.sum(np.abs(x) >= threshold)
        else:
            myop = 0.0

        return {"DASDV": dasdv, "MYOP": myop}

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

        Returns a ``DataFrame`` with columns ``['DASDV', 'MYOP', 'Output']``.
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
                        else {"DASDV": 0.0, "MYOP": 0.0}
                    )
                    extracted.append({
                        "DASDV": filtered_feats["DASDV"],
                        "MYOP": filtered_feats["MYOP"],
                        "Output": label,
                    })

        features_df = pd.DataFrame(extracted)
        if not features_df.empty:
            features_df = features_df[['DASDV', 'MYOP', 'Output']]
            features_df = FeatureEngineer.apply_borderline_smote(features_df)
        return features_df

    # ------------------------------------------------------------------ #
    #  Class-balance (Borderline SMOTE)
    # ------------------------------------------------------------------ #
    @staticmethod
    def apply_borderline_smote(df, target_col='Output', random_state=42):
        """Upsample the minority class (1) to match the majority class (0)
        using Borderline SMOTE.

        Parameters
        ----------
        df : pandas DataFrame
            Must contain feature columns and *target_col*.
        target_col : str
            Name of the binary label column (default ``'Output'``).
        random_state : int
            Seed for reproducibility.

        Returns
        -------
        pandas DataFrame
            Resampled DataFrame with balanced classes.
        """
        import pandas as pd
        from imblearn.over_sampling import BorderlineSMOTE

        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols]
        y = df[target_col]

        smote = BorderlineSMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X, y)

        resampled_df = pd.DataFrame(X_res, columns=feature_cols)
        resampled_df[target_col] = y_res
        print(f"Borderline SMOTE applied: {dict(y.value_counts())} → {dict(y_res.value_counts())}")
        return resampled_df