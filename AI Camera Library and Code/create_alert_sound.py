import numpy as np
from scipy.io import wavfile

# Generate a simple beep sound
sample_rate = 44100
duration = 0.3  # seconds
t = np.linspace(0, duration, int(sample_rate * duration))
frequency = 880  # Hz
amplitude = 0.5
beep = amplitude * np.sin(2 * np.pi * frequency * t)

# Save as WAV file
wavfile.write('sounds/alert.wav', sample_rate, beep.astype(np.float32)) 