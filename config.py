from pathlib import Path


def _load_env_file(env_path=".env"):
    env_values = {}
    path = Path(env_path)
    if not path.exists():
        return env_values

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        env_values[key.strip()] = value.strip().strip('"').strip("'")

    return env_values


_ENV = _load_env_file()

INPUT_DIM = 2
EMG_INPUT_MODE = _ENV.get("EMG_INPUT_MODE", "COM").upper()
EMG_PORT = _ENV.get("EMG_PORT", "COM4")
EMG_BAUD = int(_ENV.get("EMG_BAUD", "115200"))
SIMULATION_DURATION = float(_ENV.get("SIMULATION_DURATION", "0.6"))
