"""
Tests of ModelPoly (poly-reference identification) on synthetic data from
pyLump: periodic random excitation at two DOFs, noise on the measured time
signals, multiple-input H1 FRFs, then identification.
"""
import os
import sys

import numpy as np
import pytest

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
sys.path.insert(0, my_path)

from sdypy import EMA
from sdypy.EMA import ModelPoly

from test_tools.generate_synth_mimo import generate_synth_mimo
from test_tools.generate_synth_frf import generate_synth_frf
from test_tools.closeness_criteria import all_close_frequency, all_close_damping

POL_ORDER = 30


def mac(a, b):
    return np.abs(np.vdot(a, b))**2 / (np.vdot(a, a).real * np.vdot(b, b).real)


def identify(data, driving_point=None):
    m = ModelPoly(data.frf, data.freq, pol_order_high=POL_ORDER,
                  driving_point=driving_point)
    m.get_poles(show_progress=False)
    m.select_closest_poles(data.natural_freq)
    m.get_constants()
    return m


@pytest.fixture(scope='module')
def noisy():
    data = generate_synth_mimo(noise=0.02)
    return data, identify(data, driving_point=data.ref_dofs)


@pytest.fixture(scope='module')
def exact():
    data = generate_synth_mimo(noise=0.0)
    data.frf = data.frf_exact
    return data, identify(data)


def test_public_import():
    assert isinstance(ModelPoly, type)
    assert EMA.ModelPoly is ModelPoly
    assert 'ModelPoly' in EMA.__all__


def test_frf_shapes():
    freq = np.linspace(1, 100, 1000)
    m = ModelPoly(np.ones((6, 1000, 2)), freq)
    assert (m.n_outputs, m.n_inputs) == (6, 2)
    assert m.frf.shape == (6, 1000, 2)
    m = ModelPoly(np.ones((6, 1000)), freq)
    assert m.n_inputs == 1
    assert m.frf.shape == (6, 1000, 1)


@pytest.mark.parametrize('kwargs', [
    dict(frf=np.ones((6, 1000, 2)), freq=np.linspace(1, 100, 999)),
    dict(frf=np.ones((6, 1000, 2, 1)), freq=np.linspace(1, 100, 1000)),
    dict(frf=np.ones((6, 1000, 2)), freq=np.linspace(1, 100, 1000), freq_lower=50, freq_upper=40),
    dict(frf=np.ones((6, 1000, 2)), freq=np.linspace(1, 100, 1000), freq_upper=200),
    dict(frf=np.ones((6, 1000, 2)), freq=np.linspace(1, 100, 1000), frf_form='velocity'),
    dict(frf=np.ones((6, 1000, 2)), freq=np.linspace(1, 100, 1000), driving_point=[0]),
])
def test_invalid_input(kwargs):
    with pytest.raises(ValueError):
        ModelPoly(**kwargs)


def test_poles_per_order(noisy):
    _, m = noisy
    for name in ('all_poles', 'pole_freq', 'pole_damping_ratio', 'pole_participation_factor'):
        assert len(getattr(m, name)) == POL_ORDER
    for poles, participation in zip(m.all_poles, m.pole_participation_factor):
        assert np.all(poles.imag > 0)
        assert participation.shape == (2, len(poles))


def test_select_closest_poles(noisy):
    data, m = noisy
    assert m.natural_freq.shape == (6,)
    assert m.damping_ratio.shape == (6,)
    assert m.participation_factor.shape == (2, 6)
    assert len(set(map(tuple, m.pole_idx))) == 6


def test_same_pole_twice(noisy):
    data, m = noisy
    with pytest.raises(ValueError):
        m.select_closest_poles([data.natural_freq[0], data.natural_freq[0]])
    m.select_closest_poles(data.natural_freq)  # restore for the other tests
    m.get_constants()


def test_constants_shapes(noisy):
    _, m = noisy
    assert m.mode_shape.shape == (6, 6)
    assert m.residual_lower.shape == (6, 2)
    assert m.residual_upper.shape == (6, 2)
    assert m.frf_reconstructed.shape == m.frf.shape


def test_driving_point_scaling(noisy):
    data, m = noisy
    for r in range(6):
        psi, phi = m.mode_shape[:, r], data.mode_shape[:, r]
        err = min(np.linalg.norm(psi - phi), np.linalg.norm(psi + phi)) / np.linalg.norm(phi)
        assert err < 0.05, f'mode {r}: {err:.3f}'
    # the scaling leaves the residues unchanged
    residues = np.einsum('or,ir->roi', m.mode_shape, m.participation_factor)
    unscaled = ModelPoly(data.frf, data.freq, pol_order_high=POL_ORDER)
    unscaled.get_poles(show_progress=False)
    unscaled.select_closest_poles(data.natural_freq)
    unscaled.get_constants()
    residues_unscaled = np.einsum('or,ir->roi', unscaled.mode_shape, unscaled.participation_factor)
    np.testing.assert_allclose(residues, residues_unscaled, rtol=1e-8, atol=1e-12)


def test_noisy_mimo_accuracy(noisy):
    data, m = noisy
    np.testing.assert_allclose(m.natural_freq, data.natural_freq, rtol=1e-3)
    np.testing.assert_allclose(m.damping_ratio, data.damping_ratio, rtol=2e-2)
    for r in range(6):
        assert mac(m.mode_shape[:, r], data.mode_shape[:, r]) >= 0.995
        assert mac(m.participation_factor[:, r], data.mode_shape[data.ref_dofs, r]) >= 0.995


def test_close_modes_separated(noisy):
    data, m = noisy
    fn, zeta = data.natural_freq, data.damping_ratio
    r = int(np.argmin(np.diff(fn)))
    bandwidth = 2 * zeta[r] * fn[r]
    assert fn[r + 1] - fn[r] < 0.1 * bandwidth   # the model really has such a pair
    assert m.pole_idx[r, 1] != m.pole_idx[r + 1, 1]
    np.testing.assert_allclose(m.natural_freq[r:r + 2], fn[r:r + 2], rtol=1e-3)
    np.testing.assert_allclose(m.damping_ratio[r:r + 2], zeta[r:r + 2], rtol=2e-2)
    for i in (r, r + 1):
        assert mac(m.mode_shape[:, i], data.mode_shape[:, i]) >= 0.995


def test_exact_data(exact):
    data, m = exact
    np.testing.assert_allclose(m.natural_freq, data.natural_freq, rtol=1e-6)
    np.testing.assert_allclose(m.damping_ratio, data.damping_ratio, rtol=1e-5)
    err = np.linalg.norm(m.frf_reconstructed - data.frf) / np.linalg.norm(data.frf)
    assert err < 1e-5


def test_single_reference():
    freq, frf, f_nat_true, zeta_true, _ = generate_synth_frf(5)
    m = ModelPoly(frf, freq, pol_order_high=POL_ORDER)
    m.get_poles(show_progress=False)
    m.select_closest_poles(f_nat_true)
    m.get_constants()
    assert all_close_frequency(m.natural_freq, f_nat_true)
    assert all_close_damping(m.damping_ratio, zeta_true)
