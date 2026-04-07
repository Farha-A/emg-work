import math
import threading
import time


class EMGReader:
    def __init__(
        self,
        port="COM4",
        baud=115200,
        simulate_with_space=False,
        simulation_duration=0.6,
    ):
        self.filtered = 0.0
        self.envelope = 0.0
        self.detect = 0
        self.running = True
        self.simulate_with_space = simulate_with_space
        self.simulation_duration = simulation_duration
        self._simulation_until = 0.0
        self._simulation_phase = 0.0
        self.ser = None

        if self.simulate_with_space:
            print("Keyboard simulation enabled. Press Space to inject a signal burst.")
            threading.Thread(target=self._keyboard_loop, daemon=True).start()
        else:
            import serial

            self.ser = serial.Serial(port, baud, timeout=1)
            threading.Thread(target=self._serial_loop, daemon=True).start()

    def _serial_loop(self):
        while self.running:
            line = ""
            try:
                line = self.ser.readline().decode(errors="ignore").strip()
                filt, env = map(float, line.split(","))
                self.filtered = filt
                self.envelope = env
            except Exception as e:
                if self.running:
                    print(f"Error parsing line: '{line}' -> {e}")

    def _keyboard_loop(self):
        import msvcrt

        while self.running:
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                if key == " ":
                    self.trigger_simulation()
                elif key.lower() == "q":
                    self.running = False
                    break

            self._update_simulated_signal()
            time.sleep(0.01)

    def trigger_simulation(self):
        self._simulation_until = time.time() + self.simulation_duration
        self.detect = 1

    def _update_simulated_signal(self):
        now = time.time()
        if now < self._simulation_until:
            self._simulation_phase += 0.45
            burst = abs(math.sin(self._simulation_phase)) * 650 + 250
            ripple = abs(math.sin(self._simulation_phase * 2.4)) * 120
            self.filtered = burst + ripple
            self.envelope = burst
        else:
            self.filtered = 0.0
            self.envelope = 0.0
            self.detect = 0

    def stop(self):
        self.running = False
        if self.ser is not None:
            self.ser.close()
