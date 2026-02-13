import csv
import threading
import time
from pathlib import Path
from constants import LOGGING_INTERVAL


class Logger:
    def __init__(self, path):
        self.path = Path(path)
        self.error = ""
        # stream logging state
        self._stream_thread = None
        self._stream_stop = None
        self._stream_path = self.path.with_name("emg_stream.csv")
        self.info_path = self.path.with_name("info.csv")
        

    def ensure_log_header(self):
        try:
            if not self.info_path.exists():
                with self.info_path.open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(
                        [
                            "session_id",
                            "name",
                            "age",
                            "medical_history",
                            "level_number",
                            "duration_seconds",
                            "ease_rating_1to5",
                        ]
                    )
        except OSError as e:
            self.error = str(e)

    def log_level_result(
        self,
        session_id,
        name,
        age,
        medical_history,
        level_number,
        duration_seconds,
        rating,
    ):
        if self.error:
            return
        try:
            with self.info_path.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        session_id,
                        name,
                        age,
                        medical_history,
                        level_number,
                        f"{duration_seconds:.3f}",
                        rating,
                    ]
                )
        except OSError as e:
            self.error = str(e)

    def _stream_worker(self, session_id, level_number, get_value, interval, stop_event):
        try:
            # ensure header
            if not self._stream_path.exists():
                with self._stream_path.open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["timestamp", "session_id", "level_number", "value"])

            # buffer rows in memory, write them when level finishes
            buffer = []
            while not stop_event.is_set():
                ts = time.time()
                try:
                    val = get_value()
                except Exception:
                    val = ""
                buffer.append([f"{ts:.3f}", session_id, level_number, val])
                # wait allows early exit on stop_event
                stop_event.wait(interval)

            # level finished: flush buffered rows to disk once
            try:
                if buffer:
                    with self._stream_path.open("a", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerows(buffer)
            except OSError as e:
                self.error = str(e)
        except OSError as e:
            self.error = str(e)

    def start_stream(self, session_id, level_number, get_value_callable):
        """Start background logging of short-interval values.

        get_value_callable: zero-arg callable returning the current value to log.
        interval: seconds between samples.
        """
        self.stop_stream()
        stop_event = threading.Event()
        th = threading.Thread(
            target=self._stream_worker,
            args=(session_id, level_number, get_value_callable, LOGGING_INTERVAL, stop_event),
            daemon=True,
        )
        self._stream_stop = stop_event
        self._stream_thread = th
        th.start()

    def stop_stream(self, timeout=1.0):
        if self._stream_stop is None:
            return
        self._stream_stop.set()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout)
        self._stream_thread = None
        self._stream_stop = None

    def live_stream_generator(self, session_id, level_number, get_value_callable, interval=LOGGING_INTERVAL):
        """Generates values for livestreaming while logging them synchronously."""
        # ensure header
        try:
            if not self._stream_path.exists():
                with self._stream_path.open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["timestamp", "session_id", "level_number", "value"])
        except OSError as e:
            self.error = str(e)
            # Proceeding might be dangerous if logging is critical, but for livestream display we might continue.
            # However, we'll yield anyway.

        try:
            with self._stream_path.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                
                while True:
                    ts = time.time()
                    try:
                        val = get_value_callable()
                    except Exception:
                        val = "" # or 0? keeping consistent with _stream_worker
                    
                    # Log to file immediately (or buffer? buffering is set by OS/file object, we can flush if needed)
                    # For safety, let's trust the file object buffering or flush if critical.
                    # Given high frequency, allowing default buffering is better for performance.
                    w.writerow([f"{ts:.3f}", session_id, level_number, val])
                    
                    yield val
                    
                    time.sleep(interval)
        except OSError as e:
            self.error = str(e)
            # If we can't write to file, we might still want to yield live data?
            # But the loop is inside the 'with open'. If that fails, the generator stops.
            # Fallback: yield data without logging if logging fails?
            # For now, let's just abort if logging fails, or handle logic to continue.
            # Simpler to just let it raise or stop.
            pass

