import mne
import matplotlib.pyplot as plt

file_path = "dataset/S001/S001R04.edf"

print(f"Loading {file_path}...")

raw = mne.io.read_raw_edf(file_path, preload=True)

print("\n--- EEG Metadata ---")
print(f"Number of channels: {len(raw.ch_names)}")
print(f"Sampling frequency: {raw.info['sfreq']} Hz")
print(f"Duration: {raw.times[-1]:.2f} seconds")

# 1. Bandpass Filter (1 Hz to 50 Hz)
#? 1 Hz high-pass removes slow signal drifts (like sweat artifacts)
#? 50 Hz low-pass removes high-frequency muscle noise and power-line interference
print("\nApplying bandpass filter (1-50 Hz)...")
raw.filter(l_freq=1.0, h_freq=50.0)

# 2. Notch Filter (at 50 Hz to remove power-line noise)
print("Applying notch filter at 60 Hz...")
raw.notch_filter(freqs=60.0)

# 3. Extract Events (The Annotations)
events, event_dict = mne.events_from_annotations(raw)
print("\n--- Event Dictionary ---")
print("Tasks mapped to event IDs:", event_dict)
print(f"Total events found in this run: {len(events)}")

print("Generating visualizations...")
fig_time = raw.plot(duration=5.0, n_channels=20, scalings='auto', title="Filtered EEG Signals (5-Second Window)", block=False)
fig_psd = raw.compute_psd(fmax=50.0).plot()
plt.show()