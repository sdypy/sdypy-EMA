"""Synthetic poly-reference (MIMO) test data from a pyLump model.

The model is a chain of two equal 3-DOF sub-systems joined by a weak spring
and fixed at both ends. The symmetry gives pairs of closely spaced modes; the
weaker the coupling spring, the closer the pair.

Measured data are simulated as in a periodic random test: every block is an
independent periodic random realization of all excitation signals, the
response is its steady state. Noise is added to the measured excitation and
response time signals, and the FRFs are estimated with the multiple-input H1
estimator ``H = G_yx G_xx^-1``.

``pyLump.Model.get_response`` is not used: its frequency axis starts at
``df`` instead of 0, so it evaluates the FRF one frequency bin too high.
"""
from dataclasses import dataclass

import numpy as np
import pyLump


@dataclass
class SynthMIMO:
    freq: np.ndarray            # (n_freq,)
    frf: np.ndarray             # (n_outputs, n_freq, n_inputs), H1 estimate
    frf_exact: np.ndarray       # (n_outputs, n_freq, n_inputs)
    ref_dofs: list
    natural_freq: np.ndarray    # (n_modes,)
    damping_ratio: np.ndarray   # (n_modes,)
    mode_shape: np.ndarray      # (n_dof, n_modes), A-normalized, complex
    model: pyLump.Model


def lumped_two_subsystems(k=1e5, m=1.0, c=6.0, coupling=0.05):
    """Two equal 3-DOF chains joined by a spring ``coupling * k``."""
    kc = coupling * k
    stiffness = np.array([k, k, k, kc, k, k, k])
    damping = np.array([c, c, c, c * coupling, c, c, c])
    return pyLump.Model(n_dof=6, mass=m, stiffness=stiffness,
                        damping=damping, boundaries='both')


def generate_synth_mimo(ref_dofs=(0, 5), noise=0.02, fs=512, n_samples=4096,
                        n_blocks=40, freq_lower=5., freq_upper=150.,
                        seed=42, **model_kwargs):
    """Simulate a periodic random MIMO test and estimate the FRFs with H1.

    :param ref_dofs: excited DOFs (inputs); responses are taken at all DOFs
    :param noise: noise rms relative to the rms of each measured signal
    :param fs: sampling frequency in Hz
    :param n_samples: samples per block (one period)
    :param n_blocks: number of independent periodic random blocks
    :param freq_lower, freq_upper: returned frequency band in Hz
    :param seed: random seed
    :return: :class:`SynthMIMO`
    """
    rng = np.random.default_rng(seed)
    model = lumped_two_subsystems(**model_kwargs)
    ref_dofs = list(ref_dofs)
    n_in = len(ref_dofs)

    freq_full = np.fft.rfftfreq(n_samples, 1 / fs)
    # receptance at the FFT frequencies; 0 Hz is finite (fixed chain)
    h = model.get_FRF_matrix(freq_full)[:, ref_dofs, :]     # (n_dof, n_in, n_f)

    g_yx = 0
    g_xx = 0
    for _ in range(n_blocks):
        x = rng.standard_normal((n_in, n_samples))
        y = np.fft.irfft(np.einsum('oif,if->of', h, np.fft.rfft(x)), n=n_samples)
        x_meas = x + noise * x.std(axis=1, keepdims=True) * rng.standard_normal(x.shape)
        y_meas = y + noise * y.std(axis=1, keepdims=True) * rng.standard_normal(y.shape)
        x_f = np.fft.rfft(x_meas)
        y_f = np.fft.rfft(y_meas)
        g_yx = g_yx + np.einsum('of,if->foi', y_f, x_f.conj())
        g_xx = g_xx + np.einsum('if,jf->fij', x_f, x_f.conj())

    # H g_xx = g_yx  ->  g_xx^T H^T = g_yx^T
    h1 = np.linalg.solve(g_xx.transpose(0, 2, 1), g_yx.transpose(0, 2, 1)).transpose(2, 0, 1)

    band = (freq_full >= freq_lower) & (freq_full <= freq_upper)
    return SynthMIMO(
        freq=freq_full[band],
        frf=h1[:, band, :],
        frf_exact=h[:, :, band].transpose(0, 2, 1),
        ref_dofs=ref_dofs,
        natural_freq=model.get_eig_freq(),
        damping_ratio=model.get_damping_ratios(),
        mode_shape=model.get_eig_vec()[0],
        model=model,
    )
