"""Identification showcase — script version of `EMA Showcase.ipynb`.

Run as a script (`uv run python EMA_showcase.py`) or cell by cell (`# %%`).
Set the options below to switch the pole estimator or skip the stability chart.
"""
# %%

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sdypy import EMA

# Options
METHOD = "lscf"                 # "lscf", "lsce", "rfp" or "rfp segment"
USE_STABILITY_CHART = True      # False: pick poles automatically near APPROX_NAT_FREQ
APPROX_NAT_FREQ = [176, 476, 932, 1534, 2258, 3161, 4180]
SELECTED_RESPONSE = 1           # accelerometer index used as the response
SELECT_LOC = 0                  # excitation location shown in the FRF plot

try:
    REPO_ROOT = Path(__file__).resolve().parent
except NameError:               # running interactively, no __file__
    REPO_ROOT = Path.cwd()

# %%

# Beam excited at 6 locations with an impact hammer, response measured at
# 7 locations with piezo accelerometers (see data/experiment_1.jpg).
# H1_main has shape (n_inputs, n_outputs, n_freq); the zero frequency is truncated.
freq, H1_main = np.load(REPO_ROOT / "data" / "acc_data.npy", allow_pickle=True)
FRF = H1_main[:, SELECTED_RESPONSE, :]
print(f"FRF shape: {FRF.shape}, freq: {freq[0]:.1f}-{freq[-1]:.1f} Hz")

plt.figure()
plt.semilogy(freq, np.abs(FRF.T))
plt.xlim(0, 1000)
plt.xlabel("Frequency [Hz]")
plt.title("Measured FRFs")

# %%

acc = EMA.Model(frf=FRF, freq=freq, lower=10, upper=5000, pol_order_high=60)
acc.get_poles(method=METHOD)

# %%

if USE_STABILITY_CHART:
    acc.select_poles()          # click to pick poles (SHIFT+click in the Tk window), close when done
else:
    acc.select_closest_poles(APPROX_NAT_FREQ)

# %%

frf_rec, modal_const = acc.get_constants(whose_poles="own", upper_r=False)
acc.print_modal_data()
print(f"Modal constants shape (n_locations, n_modes): {acc.A.shape}")

# %%

plt.figure()
plt.plot(acc.normal_mode()[:, :3])
plt.xlabel("Location index")
plt.title("Normal modes (first three)")

plt.matshow(np.abs(acc.autoMAC()))
plt.colorbar()
plt.title("Auto-MAC")

# %%

plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.semilogy(freq, np.abs(FRF[SELECT_LOC]), label="Experiment")
plt.semilogy(acc.freq, np.abs(frf_rec[SELECT_LOC]), "--", label=METHOD.upper())
plt.xlim(0, freq[-1])
plt.ylabel(r"abs($\alpha$)")
plt.legend(loc="best")

plt.subplot(212)
plt.plot(freq, np.angle(FRF[SELECT_LOC], deg=True), label="Experiment")
plt.plot(acc.freq, np.angle(frf_rec[SELECT_LOC], deg=True), "--", label=METHOD.upper())
plt.xlim(0, freq[-1])
plt.xlabel("Frequency [Hz]")
plt.ylabel(r"angle($\alpha$) [deg]")
plt.legend(loc="best")

plt.show()
