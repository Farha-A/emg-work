import math
import threading

class EMGReader:
    def __init__(self, port="COM5", baud=115200):
        import serial  # imported here so stub works without pyserial

        self.ser = serial.Serial(port, baud, timeout=1)
        # print(f"Connected to Arduino on {port}") -> properly connected
        self.filtered = 0
        self.envelope = 0
        self.detect = 0
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            try:
                line = self.ser.readline().decode(errors="ignore").strip()
                # print(line) # -> values are correctly read
                filt, env = map(float, line.split(","))
                # print(f"EMG Raw: {raw}, Filtered: {filt}, Envelope: {env}, Detect: {det}") # -> values not even read
                self.filtered = filt
                self.envelope = env
            except Exception as e:
                print(f"Error parsing line: '{line}' -> {e}")

    def stop(self):
        self.running = False
        self.ser.close()
        
    def update(self, dt):
        # compatibility with EMGStub: background thread updates envelope/detect,
        # so there's nothing to do per-frame here.
        pass
        
class EMGStub:
    def __init__(self):
        self.t = 0.0
        self.envelope = 0
        self.filtered = 0
        self.detect = 1

    def update(self, dt):
        self.t += dt

        wave = (math.sin(2 * math.pi * 0.5 * self.t) + 1) / 2

        self.envelope = int(30 + wave * 170)
        self.filtered = int(20 + wave * 150)
