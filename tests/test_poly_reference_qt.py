"""
ModelPoly on the Qt stability chart (skipped when no Qt binding is installed).
"""
import os
import sys

import numpy as np
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
pytest.importorskip("matplotlib.backends.qt_compat", exc_type=ImportError)

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path)

from sdypy.EMA import ModelPoly
from sdypy.EMA.poly_reference import _StabilityChartView
from sdypy.EMA.pole_picking_qt import SelectPolesQt, QtWidgets

from test_tools.generate_synth_mimo import generate_synth_mimo
from test_poly_reference import near_repeated_model


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_pick_poles_on_chart(qapp):
    data = generate_synth_mimo(noise=0.02)
    model = ModelPoly(data.frf, data.freq, pol_order_high=30, driving_point=data.ref_dofs)
    model.get_poles(show_progress=False)

    view = _StabilityChartView(model)
    window = SelectPolesQt(view)
    window.canvas.draw()
    assert window.H is None
    picked = data.natural_freq[:4]     # the 90 Hz pair is closer than a pixel
    for f in picked:
        assert window.pick_pole(f, 25)
    # the chart shows every output-input pair and its reconstruction
    assert window.H.shape == (model.n_outputs * model.n_inputs, len(model.freq))
    window.close()

    view.apply_to_model()
    np.testing.assert_allclose(model.natural_freq, picked, rtol=1e-3)
    assert model.participation_factor.shape == (2, 4)
    assert model.mode_shape.shape == (6, 4)
    assert model.frf_reconstructed.shape == model.frf.shape


def test_existing_selection_is_shown(qapp):
    data = generate_synth_mimo(noise=0.0)
    model = ModelPoly(data.frf_exact, data.freq, pol_order_high=20)
    model.get_poles(show_progress=False)
    model.select_closest_poles(data.natural_freq)
    window = SelectPolesQt(_StabilityChartView(model))
    assert len(window.selected) == 6
    window.close()


def test_chart_shows_near_repeated_poles(qapp):
    model = near_repeated_model()
    window = SelectPolesQt(_StabilityChartView(model))
    highest = window._k == len(model.all_poles) - 1
    assert np.sum(highest & (np.abs(window._f - 50) < 0.2)) == 2
    window.close()
