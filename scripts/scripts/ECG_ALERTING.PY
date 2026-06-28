import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

from ecg_fft import ecg_fft

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------
# CONFIG
# -----------------------------------

DEFAULT_ROOT_DIR = "/workspaces/penuX/mimic-ecg"
DEFAULT_LEADS = [0, 1, 2]   # use multiple leads, e.g. I / II / III if present
DEFAULT_INPUT_LENGTH = 2048


# -----------------------------------
# CNN MODEL
# -----------------------------------

class ECGCNN(nn.Module):
    def __init__(self, input_length=2048, num_classes=5):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)

        reduced_len = input_length // 8
        self.fc1 = nn.Linear(64 * reduced_len, 128)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


# -----------------------------------
# MULTI-LEAD MONITOR + IMAGE DISPLAY
# -----------------------------------

class ECGMultiLeadMonitor:
    LABELS = {
        0: "NORMAL",
        1: "VT",
        2: "VF",
        3: "AFIB",
        4: "OTHER",
    }

    def __init__(
        self,
        root_dir=DEFAULT_ROOT_DIR,
        leads=None,
        model_path=None,
        input_length=DEFAULT_INPUT_LENGTH,
        num_classes=5,
        device=None,
        save_dir="monitor_output",
    ):
        self.root_dir = root_dir
        self.leads = leads if leads is not None else DEFAULT_LEADS
        self.input_length = input_length
        self.save_dir = save_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.save_dir, exist_ok=True)

        self.model = ECGCNN(
            input_length=input_length,
            num_classes=num_classes,
        ).to(self.device)

        self.model.eval()

        if model_path is not None and os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()

    # -----------------------------------
    # IO
    # -----------------------------------

    def load_record(self, record_path):
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal
        fs = record.fs

        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)

        return signal, fs

    def available_leads(self, signal):
        n_leads = signal.shape[1]
        return [idx for idx in self.leads if idx < n_leads]

    # -----------------------------------
    # CNN
    # -----------------------------------

    def preprocess_ecg(self, ecg):
        ecg = np.nan_to_num(ecg).astype(np.float32)

        if len(ecg) < self.input_length:
            pad = self.input_length - len(ecg)
            ecg = np.pad(ecg, (0, pad))
        elif len(ecg) > self.input_length:
            ecg = resample(ecg, self.input_length)

        mean = np.mean(ecg)
        std = np.std(ecg)

        if std > 0:
            ecg = (ecg - mean) / std
        else:
            ecg = ecg - mean

        return ecg

    def predict_class(self, ecg):
        x = self.preprocess_ecg(ecg)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())
            pred_prob = float(probs[0, pred_idx].item())

        return self.LABELS.get(pred_idx, "UNKNOWN"), pred_prob

    # -----------------------------------
    # ANALYSIS
    # -----------------------------------

    def analyze_lead(self, ecg, fs):
        """
        Reuses functions from ecg_fft without calling process_record(),
        so we can analyze any lead directly from the loaded signal.
        """
        from scipy.signal import find_peaks

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

        return {
            "hr": float(hr),
            "vlf": float(vlf),
            "lf": float(lf),
            "hf": float(hf),
            "lfhf": float(lfhf),
            "dom_freq": float(dom_freq),
            "alerts": alerts,
            "rr_count": int(len(rr)),
        }

    def analyze_record(self, record_path):
        signal, fs = self.load_record(record_path)
        lead_ids = self.available_leads(signal)

        if not lead_ids:
            return None

        lead_results = []
        all_alerts = []

        for lead_idx in lead_ids:
            ecg = signal[:, lead_idx]
            result = self.analyze_lead(ecg, fs)

            if result is None:
                continue

            cnn_label, cnn_conf = self.predict_class(ecg)

            result["lead"] = lead_idx
            result["cnn_label"] = cnn_label
            result["cnn_confidence"] = cnn_conf

            if cnn_label == "VT":
                result["alerts"] = result["alerts"] + ["CNN_VT"]
            elif cnn_label == "VF":
                result["alerts"] = result["alerts"] + ["CNN_VF"]
            elif cnn_label == "AFIB":
                result["alerts"] = result["alerts"] + ["CNN_AFIB"]

            all_alerts.extend(result["alerts"])
            lead_results.append(result)

        if not lead_results:
            return None

        hr_values = [r["hr"] for r in lead_results]
        dom_values = [r["dom_freq"] for r in lead_results]

        summary = {
            "record_path": record_path,
            "fs": fs,
            "lead_results": lead_results,
            "summary_hr_mean": float(np.mean(hr_values)),
            "summary_dom_freq_mean": float(np.mean(dom_values)),
            "combined_alerts": sorted(list(set(all_alerts))),
        }

        return summary

    # -----------------------------------
    # DISPLAY AS IMAGE
    # -----------------------------------

    def _safe_name(self, record_path):
        return record_path.replace("/", "_").replace("\\", "_")

    def plot_monitor_image(self, record_path, seconds=10, save=True, show=True):
        signal, fs = self.load_record(record_path)
        result = self.analyze_record(record_path)

        if result is None:
            return None

        lead_ids = self.available_leads(signal)
        n_samples = min(signal.shape[0], int(seconds * fs))
        time_axis = np.arange(n_samples) / fs

        rows = len(lead_ids) + 1
        fig, axes = plt.subplots(
            rows, 1, figsize=(14, 3 * rows), constrained_layout=True
        )

        if rows == 2:
            axes = list(axes)

        # top text panel
        axes[0].axis("off")

        summary_lines = [
            f"Record: {record_path}",
            f"Sampling rate: {result['fs']} Hz",
            f"Mean HR: {result['summary_hr_mean']:.2f} bpm",
            f"Mean dominant frequency: {result['summary_dom_freq_mean']:.2f} Hz",
            f"Combined alerts: {', '.join(result['combined_alerts']) if result['combined_alerts'] else 'NONE'}",
            "",
            "Per-lead summary:",
        ]

        for lead_result in result["lead_results"]:
            summary_lines.append(
                f"Lead {lead_result['lead']}: "
                f"HR={lead_result['hr']:.2f} bpm | "
                f"DomFreq={lead_result['dom_freq']:.2f} Hz | "
                f"CNN={lead_result['cnn_label']} ({lead_result['cnn_confidence']:.3f}) | "
                f"Alerts={','.join(lead_result['alerts']) if lead_result['alerts'] else 'NONE'}"
            )

        axes[0].text(
            0.01,
            0.98,
            "\n".join(summary_lines),
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
        )
        axes[0].set_title("ECG Multi-Lead Monitor", fontsize=14)

        # waveform panels
        for i, lead_idx in enumerate(lead_ids, start=1):
            ecg = signal[:n_samples, lead_idx]
            axes[i].plot(time_axis, ecg, linewidth=0.9)
            axes[i].set_ylabel(f"Lead {lead_idx}")
            axes[i].set_xlim(time_axis[0], time_axis[-1])
            axes[i].grid(True, alpha=0.3)

            lead_result = next(
                (x for x in result["lead_results"] if x["lead"] == lead_idx), None
            )

            if lead_result is not None:
                axes[i].set_title(
                    f"Lead {lead_idx} | HR {lead_result['hr']:.2f} bpm | "
                    f"DomFreq {lead_result['dom_freq']:.2f} Hz | "
                    f"CNN {lead_result['cnn_label']}"
                )

        axes[-1].set_xlabel("Time (s)")

        image_path = None
        if save:
            safe_name = self._safe_name(record_path)
            image_path = os.path.join(self.save_dir, f"{safe_name}_monitor.png")
            plt.savefig(image_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return image_path

    # -----------------------------------
    # TEXT REPORT
    # -----------------------------------

    def print_report(self, result):
        if result is None:
            return

        print("Record:", result["record_path"])
        print("Mean HR:", round(result["summary_hr_mean"], 2), "bpm")
        print("Mean Dominant ECG Frequency:", round(result["summary_dom_freq_mean"], 2), "Hz")

        if result["combined_alerts"]:
            print("🚨 COMBINED ALERTS:", ",".join(result["combined_alerts"]))
        else:
            print("🚨 COMBINED ALERTS: NONE")

        for lead_result in result["lead_results"]:
            print(
                f"  Lead {lead_result['lead']} | "
                f"HR={lead_result['hr']:.2f} bpm | "
                f"DomFreq={lead_result['dom_freq']:.2f} Hz | "
                f"CNN={lead_result['cnn_label']} ({lead_result['cnn_confidence']:.3f}) | "
                f"Alerts={','.join(lead_result['alerts']) if lead_result['alerts'] else 'NONE'}"
            )

        print()

    # -----------------------------------
    # RUNNERS
    # -----------------------------------

    def run_dataset(self, seconds=10, save_images=True, show_images=False):
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".hea"):
                    record_path = os.path.join(root, file[:-4])

                    result = self.analyze_record(record_path)
                    self.print_report(result)

                    if result is not None:
                        image_path = self.plot_monitor_image(
                            record_path,
                            seconds=seconds,
                            save=save_images,
                            show=show_images,
                        )
                        if image_path is not None:
                            print("Saved image:", image_path)
                            print()

    def run_single_record(self, record_path, seconds=10, save_image=True, show_image=True):
        result = self.analyze_record(record_path)
        self.print_report(result)

        if result is None:
            return None

        return self.plot_monitor_image(
            record_path,
            seconds=seconds,
            save=save_image,
            show=show_image,
        )


# -----------------------------------
# EXAMPLE MAIN
# -----------------------------------

if __name__ == "__main__":
    monitor = ECGMultiLeadMonitor(
        root_dir="/workspaces/penuX/mimic-ecg",
        leads=[0, 1, 2, 3],   # analyze more leads
        model_path=None,      # put your .pth here if available
        input_length=2048,
        num_classes=5,
        save_dir="monitor_output",
    )

    # Run whole dataset, save monitor images
    monitor.run_dataset(
        seconds=8,
        save_images=True,
        show_images=False,
    )

    # Example for one record:
    # img_path = monitor.run_single_record(
    #     "/workspaces/penuX/mimic-ecg/some_record",
    #     seconds=8,
    #     save_image=True,
    #     show_image=True,
    # )
    # print("Image path:", img_path)