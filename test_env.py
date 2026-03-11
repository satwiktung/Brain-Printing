import mne
import torch
import sklearn
import numpy as np

print("--- Environment Check ---")
print(f"MNE Version: {mne.__version__}")
print(f"PyTorch Version: {torch.__version__}")
print(f"Scikit-Learn Version: {sklearn.__version__}")
print(f"NumPy Version: {np.__version__}")
print("All systems go! Ready to process brainwaves.")
