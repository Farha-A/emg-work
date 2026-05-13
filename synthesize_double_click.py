"""Synthesize level-8 (Double Click) EMG sequences from existing stream data.

Double-click shape per sequence (150 samples @ 100 Hz ≈ 1.5 s):
  [20 relaxed] [30 burst] [10 relaxed] [30 burst] [60 relaxed]

Both bursts in a sequence come from the same contraction class,
chosen uniformly at random from whichever of {2, 4} are available
for that session.  Output is appended to a copy of the input CSV.
"""

import argparse
import math
import sys

import numpy as np
import pandas as pd

from constants import LOGGING_INTERVAL

LEAD   = 20
BURST  = 30
GAP    = 10
TRAIL  = 60
SEQ_LEN = LEAD + BURST + GAP + BURST + TRAIL  # 150


def _sample_window(pool: list, size: int, rng: np.random.Generator, start: int | None = None) -> list:
    """Return *size* consecutive elements from *pool*, wrapping if needed."""
    n = len(pool)
    if n == 0:
        return ["(0.0, 0.0)"] * size
    if start is None:
        start = int(rng.integers(0, n))
    indices = [(start + i) % n for i in range(size)]
    return [pool[i] for i in indices]


def synthesize_session(
    sid: int,
    relaxed: list,
    burst_pools: dict,
    target_rows: int,
    base_ts: float,
    global_seed: int,
) -> list:
    """Return a list of raw row dicts for session *sid* with level_number=8."""
    rng = np.random.default_rng(seed=global_seed ^ sid)
    available_classes = [c for c in (2, 4) if burst_pools.get(c)]
    n_sequences = math.ceil(target_rows / SEQ_LEN)

    rows = []
    ts = base_ts + 1.0
    class_counts = {2: 0, 4: 0}

    for _ in range(n_sequences):
        c = int(rng.choice(available_classes))
        class_counts[c] += 1

        # start offsets guaranteed different for burst1 vs burst2 when pool is large enough
        burst_pool = burst_pools[c]
        b1_start = int(rng.integers(0, len(burst_pool)))
        b2_offset = max(BURST, len(burst_pool) // 2)
        b2_start = (b1_start + b2_offset) % len(burst_pool)

        sequence = (
            _sample_window(relaxed, LEAD, rng)
            + _sample_window(burst_pool, BURST, rng, b1_start)
            + _sample_window(relaxed, GAP, rng)
            + _sample_window(burst_pool, BURST, rng, b2_start)
            + _sample_window(relaxed, TRAIL, rng)
        )

        for val in sequence:
            rows.append({
                "timestamp": f"{ts:.3f}",
                "session_id": sid,
                "level_number": 8,
                "value": val,
            })
            ts += LOGGING_INTERVAL

    return rows, class_counts


def main():
    parser = argparse.ArgumentParser(description="Synthesize double-click (level 8) EMG data.")
    parser.add_argument("--input",  default=r"Data\Stream\emg_stream_combined.csv")
    parser.add_argument("--output", default=r"Data\Stream\emg_stream_combined_with_double_click.csv")
    parser.add_argument("--seed",   type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} rows from {args.input}")

    all_new_rows = []
    total_added = 0

    for sid, grp in df.groupby("session_id"):
        levels = set(grp["level_number"].unique())

        if 1 not in levels:
            print(f"  Session {sid}: skipped (no level 1 / relaxed data)")
            continue
        if not ({2, 4} & levels):
            print(f"  Session {sid}: skipped (no level 2 or 4 data)")
            continue

        relaxed = grp.loc[grp["level_number"] == 1, "value"].tolist()
        burst_pools = {
            c: grp.loc[grp["level_number"] == c, "value"].tolist()
            for c in (2, 4)
        }

        # target ≈ level-2 count; fall back to level-4 if 2 absent
        target = len(burst_pools[2]) if burst_pools[2] else len(burst_pools[4])

        base_ts = grp["timestamp"].max()

        new_rows, class_counts = synthesize_session(
            sid, relaxed, burst_pools, target, base_ts, args.seed
        )
        all_new_rows.extend(new_rows)
        n = len(new_rows)
        total_added += n
        print(f"  Session {sid}: +{n} rows  (class usage: {class_counts})")

    if not all_new_rows:
        print("No sequences generated. Check that qualifying sessions exist.")
        sys.exit(1)

    new_df = pd.DataFrame(all_new_rows, columns=["timestamp", "session_id", "level_number", "value"])
    out_df = pd.concat([df, new_df], ignore_index=True)
    out_df.to_csv(args.output, index=False)

    print(f"\nTotal new level-8 rows: {total_added:,}")
    print(f"Output written to {args.output}  ({len(out_df):,} rows total)")


if __name__ == "__main__":
    main()
