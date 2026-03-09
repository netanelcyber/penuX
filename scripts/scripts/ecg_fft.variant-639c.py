import os
import wfdb
import numpy as np
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from scipy.stats import entropy

# -----------------------------------
# CONFIG
# -----------------------------------

root_dir = "/workspaces/penuX/mimic-ecg"
lead_index = 1
interp_fs = 4

# -----------------------------------
# ECG FFT CLASS
# -----------------------------------


class ecg_fft:
    @staticmethod
    def dominant_frequency(ecg, fs):
        N = len(ecg)

        yf = np.abs(fft(ecg))
        xf = fftfreq(N, 1 / fs)

        mask = xf > 0
        xf = xf[mask]
        yf = yf[mask]

        return xf[np.argmax(yf)]

    @staticmethod
    def compute_hrv(rr, rr_time):
        t_interp = np.arange(rr_time[0], rr_time[-1], 1 / interp_fs)

        if len(t_interp) < 4:
            return None

        interp_func = interp1d(rr_time, rr, kind="cubic")
        rr_interp = interp_func(t_interp)

        N = len(rr_interp)

        yf = fft(rr_interp - np.mean(rr_interp))
        xf = fftfreq(N, 1 / interp_fs)

        mask = xf >= 0
        xf = xf[mask]
        yf = np.abs(yf[mask])

        def band_power(low, high):
            m = (xf >= low) & (xf <= high)
            return np.sum(yf[m])

        vlf = band_power(0.003, 0.04)
        lf = band_power(0.04, 0.15)
        hf = band_power(0.15, 0.4)

        lfhf = lf / hf if hf > 0 else 0

        return vlf, lf, hf, lfhf

    @staticmethod
    def signal_entropy(ecg):
        hist, _ = np.histogram(ecg, bins=50, density=True)
        hist += 1e-8
        return entropy(hist)

    @staticmethod
    def signal_variance(ecg):
        return np.var(ecg)

    @staticmethod
    def detect_arrhythmias(hr, rr, ecg, fs):
        alerts = []

        rr_std = np.std(rr) if len(rr) > 0 else 0
        dom_freq = ecg_fft.dominant_frequency(ecg, fs)

        if hr < 40:
            alerts.append("SEVERE_BRADYCARDIA")

        if hr > 150:
            alerts.append("SEVERE_TACHYCARDIA")

        if hr > 120 and rr_std < 0.05 and 2 < dom_freq < 5:
            alerts.append("VENTRICULAR_TACHYCARDIA")

        if rr_std > 0.15:
            alerts.append("POSSIBLE_AFIB")

        ent = ecg_fft.signal_entropy(ecg)
        var = ecg_fft.signal_variance(ecg)

        peaks, _ = find_peaks(ecg, distance=fs * 0.2)

        if ent > 3.5 and len(peaks) < 3 and var > 0.02:
            alerts.append("VENTRICULAR_FIBRILLATION")

        if len(peaks) < 2:
            alerts.append("ASYSTOLE")
        else:
            rr_pause = np.diff(peaks) / fs
            if np.max(rr_pause) > 4:
                alerts.append("ASYSTOLE")

        return alerts, dom_freq

    @staticmethod
    def process_record(record_path):
        try:
            record = wfdb.rdrecord(record_path)

            signal = record.p_signal
            fs = record.fs

            if len(signal.shape) > 1:
                ecg = signal[:, lead_index]
            else:
                ecg = signal

            peaks, _ = find_peaks(ecg, distance=fs * 0.4)

            if len(peaks) < 3:
                return None

            rr = np.diff(peaks) / fs
            rr_time = peaks[1:] / fs

            hr = 60 / np.mean(rr)

            hrv = ecg_fft.compute_hrv(rr, rr_time)
            if hrv is None:
                return None

            vlf, lf, hf, lfhf = hrv
            alerts, dom_freq = ecg_fft.detect_arrhythmias(hr, rr, ecg, fs)

            return hr, vlf, lf, hf, lfhf, dom_freq, alerts

        except Exception:
            return None


# -----------------------------------
# DATASET WALKER
# -----------------------------------


def run_dataset():
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".hea"):
                record_path = os.path.join(root, file[:-4])

                result = ecg_fft.process_record(record_path)

                if result:
                    hr, vlf, lf, hf, lfhf, dom_freq, alerts = result

                    print("Record:", record_path)
                    print("HR:", round(hr, 2), "bpm")
                    print("Dominant ECG Frequency:", round(dom_freq, 2), "Hz")
                    print(
                        "VLF:", round(vlf, 2),
                        "LF:", round(lf, 2),
                        "HF:", round(hf, 2),
                        "LF/HF:", round(lfhf, 2)
                    )

                    if alerts:
                        print("🚨 LIFE-THREATENING ALERT:", ",".join(alerts))

                    print()


if __name__ == "__main__":
    run_dataset()