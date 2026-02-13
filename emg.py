import threading

class EMGReader:
    def __init__(self, port="COM6", baud=115200):
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
