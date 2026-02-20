import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import lfilter
import time

# =========================
# Load audio file (instead of load handel)
# =========================
Fs, y = wavfile.read("file_wav_1M.wav")   # put wav in same folder

# Convert to float (important for filtering)
y = y.astype(np.float32)

# =========================
# Play original sound
# =========================
sd.play(y, Fs)
time.sleep(10)

# =========================
# Echo generation
# =========================
alpha = 0.9
D = 4096

# Create filter coefficients
b = np.zeros(D + 2)
b[0] = 1
b[-1] = alpha

# Apply FIR filter
x = lfilter(b, 1, y)

# Play sound with echo
sd.play(x, Fs)
time.sleep(10)

# =========================
# Echo removal
# =========================
w = lfilter([1], b, x)

sd.play(w, Fs)
time.sleep(10)
