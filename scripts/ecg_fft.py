import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# -----------------------------------
# Load ECG from MIMIC-IV WFDB record
# -----------------------------------
record_path = "mimic-iv-ecg/record_name"  # change to actual path

record = wfdb.rdrecord(record_path)
signal = record.p_signal

# choose ECG lead (e.g., lead II)
ecg = signal[:, 1]

fs = record.fs   # sampling frequency (usually 500 Hz)

# -----------------------------------
# Time domain
# -----------------------------------
t = np.arange(len(ecg)) / fs

plt.figure()
plt.plot(t, ecg)
plt.title("ECG Signal (Lead II)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# -----------------------------------
# FFT
# -----------------------------------
N = len(ecg)
yf = fft(ecg)
xf = fftfreq(N, 1/fs)

# Only positive frequencies
idx = np.where(xf >= 0)

plt.figure()
plt.plot(xf[idx], np.abs(yf[idx]))
plt.title("ECG Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()
