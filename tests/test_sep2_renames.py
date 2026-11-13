"""
SEP 2 renames keep working deprecated aliases (sdypy hub, public-api spec,
"Evidenced divergences carry deprecated aliases to the canonical names").

Every old name emits DeprecationWarning and gives the same value as its new
name; the new name and positional calls emit no DeprecationWarning.
"""

import os
import warnings

import numpy as np
import pytest

import sdypy.EMA as EMA
from sdypy.EMA.EMA import LSFD, LSFD_proportional, LSFD_old

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "acc_data.npy")

# (old, new) attribute names of Model
MODEL_ATTRIBUTES = [
    ("lower", "freq_lower"),
    ("upper", "freq_upper"),
    ("pole_xi", "pole_damping_ratio"),
    ("pole_ind", "pole_idx"),
    ("nat_freq", "natural_freq"),
    ("nat_xi", "damping_ratio"),
    ("H", "frf_reconstructed"),
]


@pytest.fixture(scope="module")
def model():
    freq, H1_main = np.load(DATA_FILE, allow_pickle=True)
    m = EMA.Model(frf=H1_main[:, 1, :], freq=freq, freq_lower=10, freq_upper=5000, pol_order_high=20)
    m.get_poles(method="lscf", show_progress=False)
    m.select_closest_poles([176, 476, 932, 1534, 2258, 3161, 4180])
    m.get_constants()
    return m


def _assert_same(a, b):
    if isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
    else:
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Model attributes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("old, new", MODEL_ATTRIBUTES)
def test_old_attribute_warns_and_returns_new_value(model, old, new):
    with pytest.warns(DeprecationWarning, match=new):
        value = getattr(model, old)
    _assert_same(value, getattr(model, new))


@pytest.mark.parametrize("old, new", MODEL_ATTRIBUTES)
def test_new_attribute_does_not_warn(model, old, new):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        getattr(model, new)


@pytest.mark.parametrize("old, new", [("lower", "freq_lower"), ("nat_freq", "natural_freq")])
def test_writing_old_attribute_sets_new(old, new):
    frf = np.ones((1, 500), dtype=complex)
    freq = np.linspace(50, 200, 500)
    m = EMA.Model(frf=frf, freq=freq, freq_lower=60, freq_upper=190)
    with pytest.warns(DeprecationWarning, match=new):
        setattr(m, old, 70.0)
    assert getattr(m, new) == 70.0


# ---------------------------------------------------------------------------
# Model(lower=, upper=)
# ---------------------------------------------------------------------------

def _frf_freq():
    return np.ones((1, 500), dtype=complex), np.linspace(50, 200, 500)


def test_model_old_keywords_warn_and_give_same_limits():
    frf, freq = _frf_freq()
    with pytest.warns(DeprecationWarning, match=r"freq_(lower|upper)"):
        m_old = EMA.Model(frf=frf, freq=freq, lower=60, upper=190)
    m_new = EMA.Model(frf=frf, freq=freq, freq_lower=60, freq_upper=190)
    assert (m_old.freq_lower, m_old.freq_upper) == (m_new.freq_lower, m_new.freq_upper)
    np.testing.assert_array_equal(m_old.freq, m_new.freq)


def test_model_positional_limits_do_not_warn():
    frf, freq = _frf_freq()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        m = EMA.Model(frf, freq, None, 60, 190)
    assert (m.freq_lower, m.freq_upper) == (60.0, 190.0)


def test_model_old_and_new_keyword_together_raise():
    frf, freq = _frf_freq()
    with pytest.raises(TypeError), warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        EMA.Model(frf=frf, freq=freq, lower=60, freq_lower=70)


# ---------------------------------------------------------------------------
# Model.get_constants(f_lower=, f_upper=)
# ---------------------------------------------------------------------------

def test_get_constants_old_keywords_warn_and_give_same_result(model):
    frf_new, A_new = model.get_constants(freq_lower=100, freq_upper=4500)
    with pytest.warns(DeprecationWarning, match=r"freq_(lower|upper)"):
        frf_old, A_old = model.get_constants(f_lower=100, f_upper=4500)
    np.testing.assert_array_equal(frf_old, frf_new)
    np.testing.assert_array_equal(A_old, A_new)


def test_get_constants_old_and_new_keyword_together_raise(model):
    with pytest.raises(TypeError), warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model.get_constants(f_lower=100, freq_lower=100)


# ---------------------------------------------------------------------------
# LSFD functions (lower_ind=, upper_ind=)
# ---------------------------------------------------------------------------

_LSFD_FUNCTIONS = [LSFD, LSFD_proportional, LSFD_old]


def _lsfd_args():
    freq = np.linspace(1, 100, 50)
    poles = np.array([5 + 30j, 8 + 60j])
    rng = np.random.default_rng(1)
    frf = rng.standard_normal((1, 50)) + 1j * rng.standard_normal((1, 50))
    return poles, frf, freq, True, False


@pytest.mark.parametrize("fn", _LSFD_FUNCTIONS, ids=lambda f: f.__name__)
def test_lsfd_old_keywords_warn_and_give_same_result(fn):
    args = _lsfd_args()
    result_new = fn(*args, lower_idx=0, upper_idx=50, frf_form="receptance")
    with pytest.warns(DeprecationWarning, match=r"(lower|upper)_idx"):
        result_old = fn(*args, lower_ind=0, upper_ind=50, frf_form="receptance")
    for a, b in zip(result_old, result_new):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("fn", _LSFD_FUNCTIONS, ids=lambda f: f.__name__)
def test_lsfd_old_and_new_keyword_together_raise(fn):
    with pytest.raises(TypeError), warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        fn(*_lsfd_args(), lower_idx=0, upper_idx=50, frf_form="receptance", lower_ind=0)


@pytest.mark.parametrize("fn", _LSFD_FUNCTIONS, ids=lambda f: f.__name__)
def test_lsfd_missing_index_raises(fn):
    with pytest.raises(TypeError):
        fn(*_lsfd_args(), frf_form="receptance")
