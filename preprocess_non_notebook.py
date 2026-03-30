import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
from mne.datasets import eegbci

file_path = "dataset/S001/S001R04.edf"

print(f"Loading {file_path}...")

raw = mne.io.read_raw_edf(file_path, preload=True)

print("\n--- EEG Metadata ---")
print(f"Number of channels: {len(raw.ch_names)}")
print(f"Sampling frequency: {raw.info['sfreq']} Hz")
print(f"Duration: {raw.times[-1]:.2f} seconds")

print("\nApplying bandpass filter (1-50 Hz)...")
raw.filter(l_freq=1.0, h_freq=50.0)

print("Applying notch filter at 60 Hz...")
raw.notch_filter(freqs=60.0)

events, event_dict = mne.events_from_annotations(raw)
print("\n--- Event Dictionary ---")
print("Tasks mapped to event IDs:", event_dict)
print(f"Total events found in this run: {len(events)}")

print("Generating visualizations...")
fig_time = raw.plot(duration=5.0, n_channels=20, scalings='auto', title="Filtered EEG Signals (5-Second Window)", block=False)
fig_psd = raw.compute_psd(fmax=50.0).plot()
plt.show()

print("Standardizing PhysioNet channel names...")
# This single built-in function fixes the dots and the capitalization perfectly
eegbci.standardize(raw)

print("Applying standard 1005 montage...")
montage = mne.channels.make_standard_montage('standard_1005')

raw.set_montage(montage)

ica = ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw)

ica.plot_components()
ica.plot_sources(raw, block=True)