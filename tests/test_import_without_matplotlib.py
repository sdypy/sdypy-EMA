"""
sdypy.EMA must import, and its numerical path must run, without matplotlib.

The checks run in a subprocess in which ``import matplotlib`` is blocked, so
they do not depend on what the test session has already imported.
"""

import os
import subprocess
import sys

_repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

_SCRIPT = r"""
import sys
sys.modules['matplotlib'] = None  # every `import matplotlib...` raises ImportError

import numpy as np
from sdypy import EMA

assert EMA.pole_picking.SelectPoles is not None

# Synthetic 3-DOF receptance (proportional damping)
f_nat = np.array([71.0, 199.0, 287.0])
xi = np.array([0.01, 0.008, 0.012])
phi = np.array([[1.0, 0.8, 0.5],
                [0.9, -0.3, -0.8],
                [0.6, -0.9, 0.7]])
freq = np.linspace(0, 500, 2001)
w = 2 * np.pi * freq
w_n = 2 * np.pi * f_nat
frf = np.zeros((3, freq.size), dtype=complex)
for r in range(3):
    frf += np.outer(phi[:, r] * phi[0, r],
                    1 / (w_n[r]**2 - w**2 + 2j * xi[r] * w_n[r] * w))

m = EMA.Model(frf=frf, freq=freq, freq_lower=5, freq_upper=500, pol_order_high=30)
m.get_poles(show_progress=False)
m.select_closest_poles([70, 200, 290])
H, A = m.get_constants()
np.testing.assert_allclose(m.natural_freq, f_nat, rtol=1e-3)
np.testing.assert_allclose(m.damping_ratio, xi, rtol=5e-2)
assert EMA.MAC(A[:, :3], A[:, :3]).shape == (3, 3)
assert m.auto_mac().shape == (3, 3)

for call in (m.select_poles, lambda: EMA.pole_picking.SelectPoles(m)):
    try:
        call()
    except ImportError as err:
        assert 'matplotlib is required' in str(err), err
    else:
        raise AssertionError('ImportError not raised')

assert not [name for name in sys.modules if name.startswith('matplotlib.')]
print('OK')
"""


def test_numerical_path_without_matplotlib():
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        cwd=_repo_root, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"
