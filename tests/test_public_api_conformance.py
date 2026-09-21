"""
Public-API conformance tests for sdypy-EMA.

Tests:
  (a) auto_mac callable without DeprecationWarning
  (b) autoMAC callable, emits DeprecationWarning, same result as auto_mac
  (c) Model(frf_form=...) no warning; Model(frf_type=...) emits DeprecationWarning, same result
  (d) np, tqdm, warnings not in sdypy.EMA.__all__
  (e) every entry in sdypy.EMA.__all__ resolves via getattr
"""

import sys
import os
import warnings
import pytest
import numpy as np

# Insert the package root so the local source is used when running from %TEMP%
# If the package is installed (wheel), this sys.path insert is a no-op because
# the installed version will be found first when this file is not present in cwd.
_here = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.join(_here, "..")
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import sdypy.EMA as EMA


# ---------------------------------------------------------------------------
# Minimal model fixture: uses acc_data.npy from data/ (repo root), following
# the same pattern as test_basic.py.  We keep pol_order_high very low to
# make the fixture fast.
# ---------------------------------------------------------------------------

DATA_FILE = os.path.join(_here, "..", "data", "acc_data.npy")


def _build_model_with_modes():
    """Return a Model that has self.A populated (needed for auto_mac)."""
    freq, H1_main = np.load(DATA_FILE, allow_pickle=True)
    FRF = H1_main[:, 1, :]
    model = EMA.Model(frf=FRF, freq=freq, lower=10, upper=5000, pol_order_high=20)
    model.get_poles(method="lscf", show_progress=False)
    # Use known approximate natural frequencies from test_data
    n_freq = [176, 476, 932, 1534, 2258, 3161, 4180]
    model.select_closest_poles(n_freq)
    model.get_constants(whose_poles="own", FRF_ind="all")
    return model


@pytest.fixture(scope="module")
def model_with_modes():
    return _build_model_with_modes()


# ---------------------------------------------------------------------------
# (a) auto_mac callable without DeprecationWarning
# ---------------------------------------------------------------------------

def test_auto_mac_no_deprecation_warning(model_with_modes):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = model_with_modes.auto_mac()
    assert result is not None
    assert result.ndim == 2


# ---------------------------------------------------------------------------
# (b) autoMAC emits DeprecationWarning and returns the same result as auto_mac
# ---------------------------------------------------------------------------

def test_autoMAC_emits_deprecation_warning(model_with_modes):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = model_with_modes.autoMAC()
    dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep_warnings) >= 1
    assert "auto_mac" in str(dep_warnings[0].message)


def test_autoMAC_same_result_as_auto_mac(model_with_modes):
    ref = model_with_modes.auto_mac()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = model_with_modes.autoMAC()
    np.testing.assert_array_equal(result, ref)


# ---------------------------------------------------------------------------
# (c) frf_form / frf_type constructor behaviour
# ---------------------------------------------------------------------------

def _minimal_frf():
    """Return (frf, freq) small enough to just construct a Model."""
    freq = np.linspace(50, 200, 500)
    frf = np.ones((1, 500), dtype=complex)
    return frf, freq


def test_frf_form_no_warning():
    frf, freq = _minimal_frf()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        m = EMA.Model(frf=frf, freq=freq, lower=50, upper=200, frf_form="receptance")
    assert m.frf_form == "receptance"


def test_frf_type_emits_deprecation_warning():
    frf, freq = _minimal_frf()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = EMA.Model(frf=frf, freq=freq, lower=50, upper=200, frf_type="receptance")
    dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep_warnings) >= 1
    assert "frf_form" in str(dep_warnings[0].message)


def test_frf_type_behaves_identically_to_frf_form():
    frf, freq = _minimal_frf()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        m_type = EMA.Model(frf=frf, freq=freq, lower=50, upper=200, frf_type="mobility")
    m_form = EMA.Model(frf=frf, freq=freq, lower=50, upper=200, frf_form="mobility")
    assert m_type.frf_form == m_form.frf_form == "mobility"


def test_frf_type_property_warns():
    frf, freq = _minimal_frf()
    m = EMA.Model(frf=frf, freq=freq, lower=50, upper=200, frf_form="accelerance")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        val = m.frf_type
    dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep_warnings) >= 1
    assert val == "accelerance"


# ---------------------------------------------------------------------------
# (d) np, tqdm, warnings not in sdypy.EMA.__all__
# ---------------------------------------------------------------------------

def test_leaked_names_not_in_all():
    for name in ("np", "tqdm", "warnings"):
        assert name not in EMA.__all__, f"{name!r} must not appear in sdypy.EMA.__all__"


# ---------------------------------------------------------------------------
# (e) every entry in __all__ resolves via getattr
# ---------------------------------------------------------------------------

def test_all_entries_resolvable():
    for name in EMA.__all__:
        assert hasattr(EMA, name), f"sdypy.EMA.{name} not found but listed in __all__"
        obj = getattr(EMA, name)
        assert obj is not None, f"sdypy.EMA.{name} resolved to None"
