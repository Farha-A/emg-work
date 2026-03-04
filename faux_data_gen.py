import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants from your Arduino code
SAMPLE_RATE = 500  # Hz
WINDOW_SIZE = 20   # ENVELOPE_WINDOW / FILTER_WINDOW
BUFFER_SIZE = 500  # Number of points to show on screen

class ArduinoEMGSimulator:
    def __init__(self):
        self.env_min, self.env_max = 1023, 0
        self.filt_min, self.filt_max = 1023, 0
        self.env_buffer = np.zeros(WINDOW_SIZE)
        self.filt_buffer = np.zeros(WINDOW_SIZE)
        self.buf_idx = 0
        
    def map_value(self, x, in_min, in_max, out_min, out_max):
        if in_max <= in_min: return 0
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def get_next_sample(self, frame):
        # 1. Generate Raw Signal Logic (Imitating sEMG.processSignal)
        # Create a contraction burst every ~2 seconds
        is_contracting = (frame % 200) > 50 and (frame % 200) < 150
        base_noise = np.random.normal(0, 0.01)
        
        if is_contracting:
            # High frequency burst
            raw_filt = np.random.normal(0, 0.5) * np.sin(frame * 0.2) 
            raw_env = abs(raw_filt) * 1.5
        else:
            # Baseline noise
            raw_filt = base_noise
            raw_env = abs(base_noise)

        # 2. Update Circular Buffers (Moving Average)
        self.filt_buffer[self.buf_idx] = raw_filt
        self.env_buffer[self.buf_idx] = raw_env
        self.buf_idx = (self.buf_idx + 1) % WINDOW_SIZE

        # 3. Compute Smoothed Values
        smooth_filt = np.mean(self.filt_buffer)
        smooth_env = np.mean(self.env_buffer)

        # 4. Auto-scaling logic
        self.filt_min = min(self.filt_min, smooth_filt)
        self.filt_max = max(self.filt_max, smooth_filt)
        self.env_min = min(self.env_min, smooth_env)
        self.env_max = max(self.env_max, smooth_env)

        # 5. Map to 0-1023
        scaled_filt = self.map_value(smooth_filt, self.filt_min, self.filt_max, 0, 1023)
        scaled_env = self.map_value(smooth_env, self.env_min, self.env_max, 0, 1023)

        return scaled_filt, scaled_env

# Visualization Setup
sim = ArduinoEMGSimulator()
fig, ax = plt.subplots(figsize=(10, 5))
filt_line, = ax.plot([], [], label="Scaled Filtered (Filt)", color='cyan', alpha=0.6)
env_line, = ax.plot([], [], label="Scaled Envelope (Env)", color='orange', lw=2)

filt_data, env_data = [], []

def init():
    ax.set_xlim(0, BUFFER_SIZE)
    ax.set_ylim(-50, 1100)
    ax.set_title("Simulated Arduino Serial Plotter (EMG)")
    ax.legend(loc="upper right")
    return filt_line, env_line

def update(frame):
    sf, se = sim.get_next_sample(frame)
    filt_data.append(sf)
    env_data.append(se)

    if len(filt_data) > BUFFER_SIZE:
        filt_data.pop(0)
        env_data.pop(0)

    filt_line.set_data(range(len(filt_data)), filt_data)
    env_line.set_data(range(len(env_data)), env_data)
    return filt_line, env_line

ani = FuncAnimation(fig, update, frames=None, init_func=init, blit=True, interval=2)
plt.show()