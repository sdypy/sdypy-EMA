"""
Basic unit tests for the EMA module using synthetic FRFs generated via sdypy-model.
"""

import pytest
import numpy as np
import sys, os

# Add the path to the LOCAL sdypy-EMA module, ensures there is no naming conflict with the installed sdypy namespace coming from sdypy-model package
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
sys.path.insert(0, my_path + '/../sdypy/EMA/')
import EMA as pyEMA

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

    acc = pyEMA.Model(frf=FRF, freq=freq, lower=freq[0], upper=freq[-1], pol_order_high=60)
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
    assert acc.H.shape[0] == H.shape[0]
    assert acc.H.shape[1] == H.shape[1]


def test_natural_frequencies(ema_setup):
    acc, _, _, f_nat_true, _ = ema_setup
    nat_freq = np.array(acc.nat_freq)

    # Check if the natural frequencies are close to the true values
    assert all_close_frequency(nat_freq, f_nat_true)

def test_damping_ratios(ema_setup):
    acc, _, _, _, zeta_true = ema_setup
    zeta = np.array(acc.nat_xi)

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
#    print(f"acc.H.shape: {acc.H.shape} vs H.shape: {H.shape}")
#
#    print(f"acc.nat_freq:{acc.nat_freq} vs f_nat_true: {f_nat_true}")
#    print(f"error = {(acc.nat_freq - f_nat_true)/ f_nat_true * 100}%")
#    print(f"combined error = {np.allclose(acc.nat_freq, f_nat_true, atol=1) or np.allclose(acc.nat_freq, f_nat_true, rtol=1e-2)}")
#
#    print(f"acc.zeta: {acc.nat_xi} vs zeta_true: {zeta_true}")
#    print(f"error = {(acc.nat_xi - zeta_true) / zeta_true * 100}%")
#
#    print(f"combined error = {np.allclose(acc.nat_xi, zeta_true, rtol=2e-2)}")
