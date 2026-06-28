# Accelerate epochs using FFT features (full cohorts)

For large/full ICU cohorts, reduce per-epoch training cost by transforming long time-series into compact FFT features before model training.

## Practical flow
1. Build/collect full eICU features:
   - `python -m scripts.eicu_dataset --autofetch --output clinical_eicu.csv`
2. Run FFT extraction (see `scripts/ecg_fft.py`) to create compact spectral features.
3. Train on FFT vectors instead of raw long waveforms.
4. Optionally reduce `steps_per_epoch` proportional to retained FFT bins.
