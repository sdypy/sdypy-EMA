"""
Basic unit tests for the EMA module using synthetic FRFs generated via pyLump.
"""

import pytest
import numpy as np
import sys, os

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
from sdypy import EMA as pyEMA

from test_tools.generate_synth_frf import generate_synth_frf
from test_tools.closeness_criteria import all_close_frequency, all_close_damping

@pytest.fixture(params=["lscf", "lsce"])
def ema_setup(request):
    method = request.param

    n_dof = 5
    m = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    k = np.array([500e3, 1.3e6, 3.8e6, 7.8e6, 7.9e6])
    c = np.array([12, 25, 55, 90, 160])

    #freq, FRF, f_nat_true, zeta_true, _ = generate_synth_frf(n_dof, m, k, c)
    freq, FRF, f_nat_true, zeta_true, _ = generate_synth_frf(n_dof)

    # pol_order reduced to 30 for CI runtime; suite runtime reduced from 96s to 16s; accuracy assertions unchanged
    acc = pyEMA.Model(frf=FRF, freq=freq, freq_lower=freq[0], freq_upper=freq[-1], pol_order_high=30)
    acc.get_poles(method=method, show_progress=False)

    acc.select_closest_poles(f_nat_true)
    H, A = acc.get_constants(whose_poles='own', FRF_ind='all')
    
    return acc, A, H, f_nat_true, zeta_true


def test_data_shape(ema_setup):
    acc, A, H, _, _ = ema_setup

    assert A.shape[0] == 5
    assert A.shape[1] == 5
    assert H.shape[0] == 5
    assert H.shape[1] == len(acc.freq)
        
    assert acc.A.shape[0] == A.shape[0]
    assert acc.A.shape[1] == A.shape[1]
    assert acc.frf_reconstructed.shape[0] == H.shape[0]
    assert acc.frf_reconstructed.shape[1] == H.shape[1]


def test_natural_frequencies(ema_setup):
    acc, _, _, f_nat_true, _ = ema_setup
    nat_freq = np.array(acc.natural_freq)

    # Check if the natural frequencies are close to the true values
    assert all_close_frequency(nat_freq, f_nat_true)

def test_damping_ratios(ema_setup):
    acc, _, _, _, zeta_true = ema_setup
    zeta = np.array(acc.damping_ratio)

    # Check if the damping ratios are close to the true values
    # Using np.allclose with a 2% relative tolerance
    assert all_close_damping(zeta, zeta_true)

# Manual testing requires removal of pytest fixture decorator
#if __name__ == "__main__":
#
#    class TestMethod:
#        param = "lsce"
#
#    acc, A, H, f_nat_true, zeta_true = ema_setup(TestMethod())
#
#    print(f"acc.A.shape: {acc.A.shape} vs A.shape: {A.shape}")
#    print(f"acc.frf_reconstructed.shape: {acc.frf_reconstructed.shape} vs H.shape: {H.shape}")
#
#    print(f"acc.natural_freq:{acc.natural_freq} vs f_nat_true: {f_nat_true}")
#    print(f"error = {(acc.natural_freq - f_nat_true)/ f_nat_true * 100}%")
#    print(f"combined error = {np.allclose(acc.natural_freq, f_nat_true, atol=1) or np.allclose(acc.natural_freq, f_nat_true, rtol=1e-2)}")
#
#    print(f"acc.zeta: {acc.damping_ratio} vs zeta_true: {zeta_true}")
#    print(f"error = {(acc.damping_ratio - zeta_true) / zeta_true * 100}%")
#
#    print(f"combined error = {np.allclose(acc.damping_ratio, zeta_true, rtol=2e-2)}")
