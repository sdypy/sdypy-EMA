"""Poly-reference (MIMO) identification showcase — script version of `ModelPoly Showcase.ipynb`.

Run as a script (`uv run python ModelPoly_showcase.py`) or cell by cell (`# %%`).
Needs the `sdypy` package (`pip install sdypy`) for the model, excitation and FRF estimation.
Set the options below to change the test or to pick the poles on the stability chart.
"""
# %%

import numpy as np
import matplotlib.pyplot as plt
from sdypy import model, excitation, FRF, EMA

# Options
NOISE = 0.02                    # noise rms relative to the rms of each measured signal
N_BLOCKS = 40                   # periodic random blocks averaged in the H1 estimate
USE_STABILITY_CHART = False     # True: pick poles on the Qt chart (pip install "sdypy-EMA[qt]")
SELECT_OUTPUT = 2               # output shown in the FRF plots

# %%

# Lumped mass-spring-damper model (sdypy.model.lumped): two equal 3-DOF chains,
# fixed at both ends and joined by a weak spring. Each mode of a single chain
# becomes a pair of closely spaced modes; the weaker the coupling spring, the
# closer the pair.
k, m, c, coupling = 1e5, 1.0, 6.0, 0.05
structure = model.lumped.Model(
    n_dof=6, mass=m, boundaries="both",
    stiffness=np.array([k, k, k, coupling * k, k, k, k]),
    damping=np.array([c, c, c, coupling * c, c, c, c]),
)
natural_freq = structure.get_eig_freq()
damping_ratio = structure.get_damping_ratios()
mode_shape_exact = structure.get_eig_vec()[0]       # unit modal A
print("natural frequencies [Hz]:", np.round(natural_freq, 2))
print("damping ratios [%]:      ", np.round(100 * damping_ratio, 3))

# %%

# Simulated poly-reference test: DOFs 0 and 5 are excited at the same time
# with independent periodic random forces (sdypy.excitation), the
# displacements of all six DOFs are the responses. Noise is added to the
# measured forces and responses, and the FRFs are estimated with the
# multiple-input H1 estimator (sdypy.FRF).
fs, n_samples = 512, 4096
ref_dofs = [0, 5]
rng = np.random.default_rng(0)

forces, responses = [], []
for _ in range(N_BLOCKS):
    force = np.array([excitation.pseudo_random(n_samples, rg=rng) for _ in ref_dofs])
    response = structure.get_response(exc_dof=ref_dofs, exc=force, sampling_rate=fs)
    forces.append(force + NOISE * force.std(axis=1, keepdims=True) * rng.standard_normal(force.shape))
    responses.append(response + NOISE * response.std(axis=1, keepdims=True) * rng.standard_normal(response.shape))

frf_estimate = FRF.FRF(fs, exc=np.array(forces), resp=np.array(responses),
                       exc_type="f", resp_type="d", frf_estimator="H1")
freq = frf_estimate.get_f_axis()
frf = frf_estimate.get_FRF(frf_form="receptance").transpose(0, 2, 1)   # (n_outputs, n_freq, n_inputs)
print(f"FRF shape (n_outputs, n_freq, n_inputs): {frf.shape}")

plt.figure()
plt.semilogy(freq, np.abs(frf[SELECT_OUTPUT]), label=[f"reference {r}" for r in ref_dofs])
plt.xlim(5, 150)
plt.xlabel("Frequency [Hz]")
plt.ylabel("|H| [m/N]")
plt.legend()
plt.title(f"H1 estimate at output {SELECT_OUTPUT}")

# %%

# Single reference: stacking the FRFs of both references into `Model` gives
# one common denominator for all of them. It has one pole for each close pair
# and misses the 22.4 Hz mode here (the closest pole is at 0 Hz).
single = EMA.Model(frf.transpose(0, 2, 1).reshape(-1, len(freq)), freq,
                   freq_lower=5, freq_upper=150, pol_order_high=30)
single.get_poles(show_progress=False)
single.select_closest_poles(natural_freq)
print("exact:    ", np.round(natural_freq, 2), "Hz")
print("Model:    ", np.round(single.natural_freq, 2), "Hz")

# %%

# Poly-reference: the denominator is a 2 x 2 matrix polynomial, so every pole
# comes with a participation factor (one value per reference). The driving
# points (output index at each reference) scale the mode shapes to unit modal A.
poly = EMA.ModelPoly(frf, freq, freq_lower=5, freq_upper=150, pol_order_high=30,
                     driving_point=ref_dofs)
poly.get_poles(show_progress=False)

if USE_STABILITY_CHART:
    poly.select_poles()         # click to pick poles, close the window when done
else:
    poly.select_closest_poles(natural_freq)

print("exact:    ", np.round(natural_freq, 2), "Hz")
print("ModelPoly:", np.round(poly.natural_freq, 2), "Hz")
print("natural frequency error [%]:", np.round(100 * (poly.natural_freq / natural_freq - 1), 4))
print("damping ratio error [%]:    ", np.round(100 * (poly.damping_ratio / damping_ratio - 1), 2))

# %%

# Mode shapes and residuals by LSFD with the poles and participation factors fixed.
frf_reconstructed, mode_shape = poly.get_constants()
print(f"Mode shapes (n_outputs, n_modes): {mode_shape.shape}")
print("Participation factors (n_inputs, n_modes):")
print(np.round(poly.participation_factor, 4))
print("MAC with the exact mode shapes:",
      np.round(np.diag(EMA.tools.MAC(mode_shape, mode_shape_exact.astype(complex))).real, 4))

dof = np.arange(structure.n_dof)
fig, axes = plt.subplots(1, len(natural_freq), figsize=(14, 3), sharey=True)
for r, ax in enumerate(axes):
    sign = np.sign(np.real(np.vdot(mode_shape_exact[:, r], mode_shape[:, r])))
    ax.plot(dof, mode_shape_exact[:, r].real, "k-", label="exact")
    ax.plot(dof, sign * mode_shape[:, r].real, "o", label="ModelPoly")
    ax.set_title(f"{poly.natural_freq[r]:.2f} Hz")
    ax.set_xlabel("DOF")
axes[0].legend()
fig.suptitle("Mode shapes (unit modal A, real part)")
fig.tight_layout()

# %%

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for i, ref in enumerate(ref_dofs):
    axes[i].semilogy(freq, np.abs(frf[SELECT_OUTPUT, :, i]), label="H1 estimate")
    axes[i].semilogy(freq, np.abs(frf_reconstructed[SELECT_OUTPUT, :, i]), "--", label="ModelPoly")
    axes[i].set_ylabel(f"|H| [m/N], reference {ref}")
    axes[i].legend(loc="best")
axes[1].set_xlim(5, 150)
axes[1].set_xlabel("Frequency [Hz]")
fig.suptitle(f"Reconstructed FRFs at output {SELECT_OUTPUT}")
fig.tight_layout()

plt.show()
