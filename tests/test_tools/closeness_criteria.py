import numpy as np

def all_close_frequency(measured_freqs, true_freqs):
    """
    Check if measured natural frequencies are close to true values.
    Uses criterion of all absolute tolerances < 1 Hz or relative tolerance < 2%.
    """
    return _all_close_abs_or_rel(measured_freqs, true_freqs, atol=1, rtol=2e-2)

def all_close_damping(measured_damping, true_damping):
    """
    Check if measured damping ratios are close to true values.
    Uses a relative tolerance of 2%.
    """
    return np.allclose(measured_damping, true_damping, rtol=2e-2)

def _all_close_abs_or_rel(measured_vals, true_vals, atol=1, rtol=2e-2):
    """
    Check if measured values are close to true values using either absolute or relative tolerance.
    """
    return np.allclose(measured_vals, true_vals, atol=atol) or np.allclose(measured_vals, true_vals, rtol=rtol)