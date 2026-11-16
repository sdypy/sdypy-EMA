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


def test_normal_mode(noisy):
    data, m = noisy
    real_modes = m.normal_mode()
    assert real_modes.shape == (6, 6)
    assert np.isrealobj(real_modes)
    # the synthetic model has proportional damping: its modes are real
    for r in range(6):
        assert mac(real_modes[:, r], data.mode_shape[:, r].real) >= 0.995


def test_from_uff(noisy, tmp_path):
    import pyuff
    data, _ = noisy
    nodes, ref_nodes = np.arange(1, 7), [1, 6]
    filename = tmp_path / 'mimo.uff'
    uff = pyuff.UFF(str(filename))
    for i, ref in enumerate(ref_nodes):
        for o, node in enumerate(nodes):
            uff.write_sets(pyuff.prepare_58(
                func_type=4, rsp_node=int(node), rsp_dir=3, ref_node=ref, ref_dir=3,
                ordinate_spec_data_type=8, data=data.frf[o, :, i], x=data.freq,
                id1='FRF', rsp_ent_name='name', ref_ent_name='name',
                abscissa_spacing=1, abscissa_spec_data_type=18, orddenom_spec_data_type=13,
                abscissa_min=data.freq[0], abscissa_inc=data.freq[1] - data.freq[0]),
                mode='add')
    m = ModelPoly.from_uff(str(filename), pol_order_high=POL_ORDER)
    assert m.outputs == [(int(n), 3) for n in nodes]
    assert m.inputs == [(1, 3), (6, 3)]
    assert m.frf_form == 'receptance'
    np.testing.assert_array_equal(m.driving_point, [0, 5])
    np.testing.assert_allclose(m.frf, data.frf, rtol=1e-6)
    m.get_poles(show_progress=False)
    m.select_closest_poles(data.natural_freq)
    np.testing.assert_allclose(m.natural_freq, data.natural_freq, rtol=1e-3)


def test_from_uff_missing_pair(tmp_path):
    import pyuff
    freq = np.linspace(1, 100, 100)
    filename = tmp_path / 'incomplete.uff'
    uff = pyuff.UFF(str(filename))
    for rsp, ref in [(1, 1), (2, 1), (1, 2)]:   # (2, 2) is missing
        uff.write_sets(pyuff.prepare_58(
            func_type=4, rsp_node=rsp, rsp_dir=3, ref_node=ref, ref_dir=3,
            ordinate_spec_data_type=8, data=np.ones(100, dtype=complex), x=freq,
            id1='FRF', rsp_ent_name='name', ref_ent_name='name',
            abscissa_spacing=1, abscissa_spec_data_type=18, orddenom_spec_data_type=13,
            abscissa_min=freq[0], abscissa_inc=freq[1] - freq[0]), mode='add')
    with pytest.raises(ValueError):
        ModelPoly.from_uff(str(filename))


def test_stabilization_many_poles_per_order():
    from sdypy.EMA import stabilization
    # 8 identical poles at each of 3 orders: more than the 2 * n_orders = 6
    # rows a single-reference model needs
    poles = 2 * np.pi * np.arange(10., 90., 10.) * (-0.01 + 1j)
    fn, xi, test_fn, test_xi = stabilization._stabilization([poles] * 3, 3, err_fn=1e-3, err_xi=1e-2)
    assert fn.shape[0] >= 8
    # _stabilization compares from the third order on and stores that result
    # in column 1; every pole repeats one of the previous order
    assert np.all(test_fn[:8, 1] > 0) and np.all(test_xi[:8, 1] > 0)



def near_repeated_model(split=0.0005, pol_order_high=20):
    """Noise-free FRFs of two inputs, two modes ``split`` Hz apart at 50 Hz."""
    freq = np.linspace(30, 70, 4001)
    omega = 2 * np.pi * freq
    zeta = 0.005
    x = np.arange(6)
    phi_a, phi_b = np.cos(np.pi * x / 6), np.sin(np.pi * x / 6)
    frf = np.zeros((6, len(freq), 2), dtype=complex)
    for f0, phi in [(50.0, phi_a), (50.0 + split, phi_b), (62.0, phi_a + phi_b)]:
        wr = 2 * np.pi * f0
        s = -zeta * wr + 1j * wr * np.sqrt(1 - zeta**2)
        a = np.outer(phi, phi[[1, 4]])[:, None, :]
        jw = 1j * omega[None, :, None]
        frf += a / (jw - s) + a / (jw - s.conjugate())
    m = ModelPoly(frf, freq, freq_lower=35, freq_upper=65, pol_order_high=pol_order_high)
    m.get_poles(show_progress=False)
    return m


def test_stabilization_near_repeated_poles():
    from sdypy.EMA import stabilization
    m = near_repeated_model()
    n_orders = len(m.all_poles)
    near = {}
    for conjugate_pairs in (True, False):
        fn, *_ = stabilization._stabilization(m.all_poles, n_orders, err_fn=0.01, err_xi=0.05,
                                              conjugate_pairs=conjugate_pairs)
        # the last column of the stabilization matrix stays empty
        near[conjugate_pairs] = np.sum(np.abs(fn[:, -2] - 50) < 0.2)
    assert near[False] == 2
    assert near[True] == 1      # the single-reference conjugate filter merges them


def test_select_poles_without_qt(noisy, monkeypatch):
    _, m = noisy
    # a None entry in sys.modules makes the import raise ImportError
    monkeypatch.setitem(sys.modules, 'matplotlib.backends.qt_compat', None)
    with pytest.raises(ImportError, match='sdypy-EMA\\[qt\\]'):
        m.select_poles()
