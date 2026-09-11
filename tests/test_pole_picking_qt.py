"""
Tests for the Qt stability chart (skipped when no Qt binding is installed).
"""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("matplotlib.backends.qt_compat")

from matplotlib.backend_bases import MouseEvent

from sdypy import EMA
from sdypy.EMA.pole_picking_qt import PICK_RADIUS, SelectPolesQt, QtCore, QtWidgets

APPROX_NAT_FREQ = [176, 476, 932, 1534, 2258, 3161, 4180]


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def make_model(method="lscf", pol_order_high=60):
    freq, H1_main = np.load("./data/acc_data.npy", allow_pickle=True)
    model = EMA.Model(frf=H1_main[:, 1, :], freq=freq, lower=10, upper=5000, pol_order_high=pol_order_high)
    model.get_poles(method=method, show_progress=False)
    return model


@pytest.fixture
def window(qapp):
    win = SelectPolesQt(make_model())
    win.canvas.draw()  # lay out the figure so pixel positions are final
    yield win
    win.close()


def test_pick_all_modes(window):
    for f in APPROX_NAT_FREQ:
        assert window.pick_pole(f, 45)

    model = window.Model
    assert len(model.nat_freq) == len(APPROX_NAT_FREQ)
    np.testing.assert_allclose(model.nat_freq, APPROX_NAT_FREQ, rtol=0.02)
    assert np.all(np.diff(model.nat_freq) > 0)
    assert window.table.rowCount() == len(APPROX_NAT_FREQ)
    assert window.H.shape == model.frf.shape

    H, A = model.get_constants(whose_poles="own")
    assert A.shape == (6, len(APPROX_NAT_FREQ))


def test_no_pick_far_from_poles(window):
    # find the chart position furthest (on screen) from every stable pole
    window.canvas.draw()
    to_px = window.ax_poles.transData.transform
    stable = window._category == 0
    poles_px = to_px(np.column_stack((window._f[stable], window._y()[stable])))
    grid = np.array([(f, y) for f in np.linspace(100, 4900, 49) for y in np.linspace(2, 58, 29)])
    diff = to_px(grid)[:, None, :] - poles_px[None, :, :]
    dist = np.hypot(diff[..., 0], diff[..., 1]).min(axis=1)
    assert dist.max() > PICK_RADIUS

    assert not window.pick_pole(*grid[np.argmax(dist)])
    assert window.selected == []


def test_remove_and_undo(window):
    window.pick_pole(476, 45)
    window.pick_pole(932, 45)
    window.pick_pole(176, 45)

    assert window.remove_pole(480)
    np.testing.assert_allclose(window.Model.nat_freq, [176, 932], rtol=0.02)

    window.undo_last_pick()  # last pick was 176 Hz
    np.testing.assert_allclose(window.Model.nat_freq, [932], rtol=0.02)

    window.clear_selection()
    assert window.Model.pole_ind == [] and window.H is None


def send(window, name, x, y, **kwargs):
    window.canvas.callbacks.process(name, MouseEvent(name, window.canvas, x, y, **kwargs))


def drag(window, start, end, button=1):
    send(window, "button_press_event", *start, button=button)
    send(window, "motion_notify_event", *end, button=button)
    send(window, "button_release_event", *end, button=button)


def pole_pixel(window, freq):
    """Screen position of a stable pole near ``freq``."""
    p = np.flatnonzero((window._category == 0) & (np.abs(window._f - freq) < 10))[0]
    return tuple(window.ax_poles.transData.transform((window._f[p], window._y()[p])))


def plot_point(window, fx, fy):
    """Screen position at fractions ``fx``, ``fy`` of the plot area."""
    bbox = window.ax_poles.bbox
    return bbox.x0 + fx * bbox.width, bbox.y0 + fy * bbox.height


def test_click_picks_and_removes_pole(window):
    xy = pole_pixel(window, 932)
    drag(window, xy, xy)
    assert len(window.selected) == 1

    drag(window, xy, xy, button=3)
    assert window.selected == []


def test_toolbar_has_no_pan_zoom_modes(window):
    texts = [action.text() for action in window.toolbar.actions()]
    assert "Home" in texts
    assert "Pan" not in texts and "Zoom" not in texts


def test_drag_zooms_without_picking(window):
    xlim, ylim = window.ax_poles.get_xlim(), window.ax_poles.get_ylim()
    frf_ylim = window.ax_frf.get_ylim()

    drag(window, pole_pixel(window, 932), plot_point(window, 0.6, 0.9))
    assert window.selected == []
    assert np.diff(window.ax_poles.get_xlim()) < np.diff(xlim)
    assert np.diff(window.ax_poles.get_ylim()) < np.diff(ylim)
    assert window.ax_frf.get_ylim() != frf_ylim

    window.toolbar.home()
    np.testing.assert_allclose(window.ax_poles.get_xlim(), xlim)
    np.testing.assert_allclose(window.ax_poles.get_ylim(), ylim)


def test_horizontal_drag_zooms_frequency_only(window):
    ylim = window.ax_poles.get_ylim()
    x0, y0 = plot_point(window, 0.2, 0.5)
    drag(window, (x0, y0), (plot_point(window, 0.5, 0.5)[0], y0 + 3))
    np.testing.assert_allclose(window.ax_poles.get_xlim(), [1000, 2500], rtol=0.02)
    np.testing.assert_allclose(window.ax_poles.get_ylim(), ylim)


def test_right_drag_pans_without_removing(window):
    window.pick_pole(932, 45)
    xlim = np.array(window.ax_poles.get_xlim())
    start = pole_pixel(window, 932)
    shift = 0.1 * window.ax_poles.bbox.width
    drag(window, start, (start[0] + shift, start[1]), button=3)

    assert len(window.selected) == 1
    one_pixel = np.diff(xlim)[0] / window.ax_poles.bbox.width  # mouse events round to whole pixels
    np.testing.assert_allclose(window.ax_poles.get_xlim(), xlim - 0.1 * np.diff(xlim), atol=one_pixel)


def test_double_click_resets_view(window):
    xlim = window.ax_poles.get_xlim()
    x0, y0 = plot_point(window, 0.1, 0.5)
    drag(window, (x0, y0), (plot_point(window, 0.5, 0.5)[0], y0))
    assert not np.allclose(window.ax_poles.get_xlim(), xlim)

    window.pick_pole(1534, 45)
    xy = pole_pixel(window, 932)
    drag(window, xy, xy)  # first click of the double-click picks a pole ...
    assert len(window.selected) == 2
    send(window, "button_press_event", *xy, button=1, dblclick=True)
    send(window, "button_release_event", *xy, button=1)

    # ... which the double-click takes back, keeping the earlier pick
    np.testing.assert_allclose(window.Model.nat_freq, [1534], rtol=0.02)
    np.testing.assert_allclose(window.ax_poles.get_xlim(), xlim)


def test_wheel_zoom(window):
    xlim, ylim = window.ax_poles.get_xlim(), window.ax_poles.get_ylim()
    send(window, "scroll_event", *plot_point(window, 0.5, 0.5), step=1)
    np.testing.assert_allclose(np.diff(window.ax_poles.get_xlim()), np.diff(xlim) / 1.2)
    np.testing.assert_allclose(np.diff(window.ax_poles.get_ylim()), np.diff(ylim) / 1.2)

    # over the frequency axis only the frequency range changes
    ylim = window.ax_poles.get_ylim()
    bbox = window.ax_poles.bbox
    send(window, "scroll_event", bbox.x0 + 0.5 * bbox.width, bbox.y0 - 10, step=-1)
    np.testing.assert_allclose(np.diff(window.ax_poles.get_xlim()), np.diff(xlim))
    np.testing.assert_allclose(window.ax_poles.get_ylim(), ylim)


def test_views_and_criteria_keep_selection(window):
    window.pick_pole(1534, 45)
    n_stable = np.sum(window._category == 0)

    window.xi_spin.setValue(20.0)
    window._apply_criteria()
    assert np.sum(window._category == 0) > n_stable

    window.cluster_radio.setChecked(True)
    window.unstable_check.setChecked(True)
    window.legend_check.setChecked(True)
    window.frf_combo.setCurrentIndex(1)
    window.canvas.draw()

    assert len(window.selected) == 1
    np.testing.assert_allclose(window.Model.nat_freq, [1534], rtol=0.02)


def test_loads_existing_selection(qapp):
    model = make_model()
    model.select_closest_poles(APPROX_NAT_FREQ)
    win = SelectPolesQt(model)
    assert len(win.selected) == len(APPROX_NAT_FREQ)
    win.close()


def test_ignores_stale_selection(qapp):
    model = make_model()
    model.pole_ind = [[40, 10_000]]
    model.nat_freq = [123.0]
    win = SelectPolesQt(model)
    assert win.selected == []
    win.close()


def test_rfp_order_cap(qapp):
    # rfp caps pol_order_high at 20; the chart must use the orders actually computed
    with pytest.warns(UserWarning):
        model = make_model(method="rfp", pol_order_high=25)
    win = SelectPolesQt(model)
    assert np.sum(win._category == 0) > 0
    win.close()


def test_export_poles(window, tmp_path):
    window.pick_pole(476, 45)
    window.pick_pole(932, 45)
    path = tmp_path / "poles.csv"
    window.export_poles(path)
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    assert data.shape == (2, 4)
    np.testing.assert_allclose(data[:, 0], window.Model.nat_freq)


def test_select_poles_blocks_until_closed(qapp):
    model = make_model()

    def pick_and_close():
        for w in QtWidgets.QApplication.topLevelWidgets():
            if isinstance(w, SelectPolesQt) and w.isVisible():
                w.pick_pole(932, 45)
                w.close()

    QtCore.QTimer.singleShot(500, pick_and_close)
    model.select_poles(gui="qt")
    assert len(model.nat_freq) == 1


def test_select_poles_rejects_unknown_gui(qapp):
    with pytest.raises(ValueError):
        make_model().select_poles(gui="wx")
