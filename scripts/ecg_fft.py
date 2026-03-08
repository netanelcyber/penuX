import os
import wfdb
import numpy as np
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d

# -----------------------------------
# CONFIG
# -----------------------------------

root_dir = "/workspaces/penuX/mimic-ecg"   # root directory of ECG files
lead_index = 1                # usually Lead II
interp_fs = 4                 # HRV interpolation frequency

# -----------------------------------
# PROCESS RECORD
# -----------------------------------

def process_record(record_path):

    try:
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal
        fs = record.fs

        if len(signal.shape) > 1:
            ecg = signal[:, lead_index]
        else:
            ecg = signal

        # R peak detection
        peaks, _ = find_peaks(ecg, distance=fs*0.4)

        if len(peaks) < 3:
            return None

        # RR intervals
        rr = np.diff(peaks) / fs
        rr_time = peaks[1:] / fs

        # Heart rate
        hr = 60 / np.mean(rr)

        # interpolate RR series
        t_interp = np.arange(rr_time[0], rr_time[-1], 1/interp_fs)

        if len(t_interp) < 4:
            return None

        interp_func = interp1d(rr_time, rr, kind="cubic")
        rr_interp = interp_func(t_interp)

        # FFT
        N = len(rr_interp)
        yf = fft(rr_interp - np.mean(rr_interp))
        xf = fftfreq(N, 1/interp_fs)

        mask = xf >= 0
        xf = xf[mask]
        yf = np.abs(yf[mask])

        # HRV bands
        def band_power(low, high):
            m = (xf >= low) & (xf <= high)
            return np.sum(yf[m])

        vlf = band_power(0.003,0.04)
        lf = band_power(0.04,0.15)
        hf = band_power(0.15,0.4)

        lfhf = lf/hf if hf > 0 else 0

        return hr, vlf, lf, hf, lfhf

    except Exception as e:
        return None


# -----------------------------------
# WALK DATASET
# -----------------------------------

for root, dirs, files in os.walk(root_dir):

    for file in files:

        if file.endswith(".hea"):

            record_path = os.path.join(root, file[:-4])

            result = process_record(record_path)

            if result:

                hr, vlf, lf, hf, lfhf = result

                print("Record:", record_path)
                print(" HR:", round(hr,2), "bpm")
                print(" VLF:", round(vlf,2),
                      "LF:", round(lf,2),
                      "HF:", round(hf,2),
                      "LF/HF:", round(lfhf,2))
                print()