"""Qt stability chart for interactive pole picking.

Built on matplotlib's Qt compatibility layer, so it runs with whichever Qt
binding is installed (PySide6 or PyQt6). Opened by ``Model.select_poles()``.
"""
import contextlib
import io
import sys
import warnings

import numpy as np
from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from . import stabilization

PICK_RADIUS = 25     # [px] clicks further than this from every pole pick nothing
DRAG_THRESHOLD = 5   # [px] mouse movement below this is a click, above it a drag
AXIS_LOCK = 15       # [px] a zoom drag narrower than this in one direction zooms the other axis only
WHEEL_ZOOM = 1.2     # view scale per wheel step

# (legend label, marker, colour, marker size) of the pole categories; index 0 are the stable poles
_CATEGORIES = [
    ('Stable frequency, stable damping', 'X', 'g', 7),
    ('Stable frequency, unstable damping', 'X', 'b', 4),
    ('Unstable frequency, stable damping', '*', 'r', 4),
    ('Unstable frequency, unstable damping', '.', 'r', 4),
]

HELP_TEXT = """
<b>Mouse</b>
<ul>
<li><b>Click</b> near a pole to pick it; <b>right-click</b> near a picked pole to remove it.</li>
<li><b>Drag</b> to zoom to a rectangle. A mostly horizontal drag zooms the frequency axis only,
a mostly vertical drag the vertical axes only.</li>
<li><b>Right-drag</b> (or middle-drag) to pan.</li>
<li><b>Wheel</b> zooms around the cursor; over an axis it zooms that axis only.</li>
<li><b>Double-click</b> resets the view; Home, Back and Forward in the toolbar step through the views.</li>
</ul>
<b>Keyboard</b>
<ul>
<li>Ctrl+Z undoes the last pick; Delete removes the poles selected in the table.</li>
</ul>
<b>Charts</b>
<ul>
<li><i>Stability chart:</i> pole frequency against polynomial order.</li>
<li><i>Cluster diagram:</i> pole frequency against damping ratio.</li>
</ul>
A pole is stable when its frequency and damping ratio match a pole of the previous
polynomial order within the tolerances set under <i>Stability tolerances</i>.
"""


class _Toolbar(NavigationToolbar2QT):
    """Toolbar without the pan and zoom modes; the mouse zooms and pans directly."""
    toolitems = [item for item in NavigationToolbar2QT.toolitems if item[0] not in ('Pan', 'Zoom', 'Subplots')]


class SelectPolesQt(QtWidgets.QMainWindow):
    """Stability chart window for picking the poles of a ``Model``.

    The picked poles are written to ``Model.pole_ind``, ``Model.nat_freq`` and
    ``Model.nat_xi`` after every change, so the selection is kept when the
    window closes. Poles already selected on the model (e.g. by
    ``select_closest_poles()``) are loaded as the starting selection.

    :param Model: model with poles computed by ``get_poles()``
    :type Model: sdypy.EMA.Model
    :param fn_tol: relative natural frequency tolerance of a stable pole
    :type fn_tol: float, optional
    :param xi_tol: relative damping tolerance of a stable pole
    :type xi_tol: float, optional
    """
    def __init__(self, Model, fn_tol=0.001, xi_tol=0.05, parent=None):
        super().__init__(parent)
        if not getattr(Model, 'all_poles', None):
            raise RuntimeError('No poles to show; call Model.get_poles() first.')

        self.Model = Model
        self.fn_tol = fn_tol
        self.xi_tol = xi_tol
        self.chart_type = 'stability'   # or 'cluster'
        self.frf_plot_type = 'abs'      # or 'all'
        self.show_unstable = False
        self.show_legend = False
        self.selected = []              # (order index, pole index), sorted by frequency
        self.H = None                   # reconstructed FRFs
        self._pick_history = []         # selected poles in picking order, for undo
        self._loop = None
        self._drag = None               # mouse press being tracked as a click or drag
        self._click_pick = None         # pole picked by the latest click, undone if it turns into a double-click
        self._home_view_saved = False
        self._pole_artists = []
        self._frf_artists = []
        self._rec_artists = []

        # rfp methods cap the polynomial order, so count the computed orders
        self._n_orders = len(Model.all_poles)
        self._n_per_band = max(self._n_orders // getattr(Model, 'n_bands', 1), 1)

        self._build_ui()
        self._compute_stability()
        self._redraw_poles(reset_view=True)
        self._redraw_frf()
        self._set_selection(self._existing_selection())

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        self.setWindowTitle('Stability Chart')

        self.fig = Figure(constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.ax_frf = self.fig.add_subplot(111)
        self.ax_frf.set_yscale('log')
        self.ax_frf.set_xlabel('Frequency [Hz]')
        self.ax_frf.set_ylabel('Magnitude')
        self.ax_poles = self.ax_frf.twinx()
        self.ax_poles.grid(True, alpha=0.4)

        style = dict(ls='none', zorder=3)
        self._sel_markers, = self.ax_poles.plot([], [], marker='o', ms=11, mfc='none', mec='k', mew=1.5, **style)
        self._sel_top, = self.ax_poles.plot([], [], marker='v', ms=9, color='k', clip_on=False,
                                            transform=self.ax_poles.get_xaxis_transform(), **style)
        self._highlight, = self.ax_poles.plot([], [], marker='o', ms=17, mfc='none', mec='tab:orange', mew=2.5, **style)
        # Keep these out of constrained layout: an empty line reports the figure origin as its
        # extent, which would shrink the axes on every redraw.
        for artist in (self._sel_markers, self._sel_top, self._highlight):
            artist.set_in_layout(False)
        for name, handler in [('button_press_event', self._on_press),
                              ('motion_notify_event', self._on_motion),
                              ('button_release_event', self._on_release),
                              ('scroll_event', self._on_scroll)]:
            self.canvas.mpl_connect(name, handler)

        self.toolbar = _Toolbar(self.canvas, self)
        self.addToolBar(self.toolbar)
        # record the view once the wheel stops, so Back skips over a whole wheel zoom
        self._wheel_timer = QtCore.QTimer(self)
        self._wheel_timer.setSingleShot(True)
        self._wheel_timer.setInterval(300)
        self._wheel_timer.timeout.connect(self.toolbar.push_current)

        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(280)
        side = QtWidgets.QVBoxLayout(panel)

        # Selected poles
        sel_box = QtWidgets.QGroupBox('Selected poles')
        sel_layout = QtWidgets.QVBoxLayout(sel_box)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['f [Hz]', 'ζ [%]', 'Order'])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._update_highlight)
        sel_layout.addWidget(self.table)
        buttons = QtWidgets.QHBoxLayout()
        for text, slot in [('Remove', self._remove_table_selection), ('Clear all', self.clear_selection)]:
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda *_, slot=slot: slot())
            buttons.addWidget(button)
        sel_layout.addLayout(buttons)
        side.addWidget(sel_box, stretch=1)

        # Stability criteria
        crit_box = QtWidgets.QGroupBox('Stability tolerances')
        crit_box.setToolTip('Largest relative change from the previous polynomial order for a pole to count as stable')
        grid = QtWidgets.QGridLayout(crit_box)
        self._criteria_timer = QtCore.QTimer(self)
        self._criteria_timer.setSingleShot(True)
        self._criteria_timer.setInterval(400)
        self._criteria_timer.timeout.connect(self._apply_criteria)
        self.fn_spin = self._percent_spin(100 * self.fn_tol, 0.001, 10.0, 0.05)
        self.xi_spin = self._percent_spin(100 * self.xi_tol, 0.1, 100.0, 1.0)
        grid.addWidget(QtWidgets.QLabel('Frequency [%]'), 0, 0)
        grid.addWidget(self.fn_spin, 0, 1)
        grid.addWidget(QtWidgets.QLabel('Damping [%]'), 1, 0)
        grid.addWidget(self.xi_spin, 1, 1)
        self.stable_label = QtWidgets.QLabel()
        grid.addWidget(self.stable_label, 2, 0, 1, 2)
        side.addWidget(crit_box)

        # View
        view_box = QtWidgets.QGroupBox('View')
        view = QtWidgets.QVBoxLayout(view_box)
        self.stability_radio = QtWidgets.QRadioButton('Stability chart')
        self.stability_radio.setChecked(True)
        self.cluster_radio = QtWidgets.QRadioButton('Cluster diagram')
        self.cluster_radio.toggled.connect(self._on_chart_type)
        self.frf_combo = QtWidgets.QComboBox()
        self.frf_combo.addItems(['Mean FRF magnitude', 'All FRFs'])
        self.frf_combo.currentIndexChanged.connect(self._on_frf_type)
        self.unstable_check = QtWidgets.QCheckBox('Show unstable poles')
        self.unstable_check.toggled.connect(self._on_show_unstable)
        self.legend_check = QtWidgets.QCheckBox('Show legend')
        self.legend_check.toggled.connect(self._on_show_legend)
        for widget in (self.stability_radio, self.cluster_radio, self.frf_combo,
                       self.unstable_check, self.legend_check):
            view.addWidget(widget)
        side.addWidget(view_box)

        done = QtWidgets.QPushButton('Done')
        done.setToolTip('Close the chart and keep the selected poles')
        done.clicked.connect(lambda *_: self.close())
        side.addWidget(done)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self.canvas)
        splitter.addWidget(panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        self.setCentralWidget(splitter)

        # Menus
        file_menu = self.menuBar().addMenu('&File')
        self._add_action(file_menu, 'Save figure…', self.save_figure_dialog, QtGui.QKeySequence.StandardKey.Save)
        self._add_action(file_menu, 'Export selected poles…', self.export_poles_dialog, 'Ctrl+E')
        file_menu.addSeparator()
        self._add_action(file_menu, 'Done', self.close, QtGui.QKeySequence.StandardKey.Close)
        edit_menu = self.menuBar().addMenu('&Edit')
        self._add_action(edit_menu, 'Undo last pick', self.undo_last_pick, QtGui.QKeySequence.StandardKey.Undo)
        self._add_action(edit_menu, 'Remove selected rows', self._remove_table_selection, QtGui.QKeySequence.StandardKey.Delete)
        self._add_action(edit_menu, 'Clear all', self.clear_selection)
        help_menu = self.menuBar().addMenu('&Help')
        self._add_action(help_menu, 'Pole picking help', self.show_help, QtGui.QKeySequence.StandardKey.HelpContents)

        self.statusBar().addPermanentWidget(QtWidgets.QLabel(
            'Click: pick  ·  Right-click: remove  ·  Drag: zoom  ·  Right-drag: pan  ·  Wheel: zoom  ·  '
            'Double-click: reset view  ·  F1: help'))

        geometry = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(int(0.8 * geometry.width()), int(0.8 * geometry.height()))
        splitter.setSizes([int(0.6 * geometry.width()), int(0.2 * geometry.width())])

    def _add_action(self, menu, text, slot, shortcut=None):
        action = menu.addAction(text)
        if shortcut is not None:
            action.setShortcut(QtGui.QKeySequence(shortcut))
        action.triggered.connect(lambda *_: slot())
        return action

    def _percent_spin(self, value, lo, hi, step):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(3)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.valueChanged.connect(lambda *_: self._criteria_timer.start())
        return spin

    # ------------------------------------------------------------ stability
    def _compute_stability(self):
        """Classify every pole under the current stability tolerances."""
        # _stabilization prints a progress bar and warns on zero entries;
        # neither is useful while the chart is open.
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter('ignore', RuntimeWarning)
            fn, xi, test_fn, test_xi = stabilization._stabilization(
                self.Model.all_poles, self._n_orders, err_fn=self.fn_tol, err_xi=self.xi_tol)

        rows, cols = np.nonzero(fn > 1e-6)  # skip the zero padding
        stable_fn = test_fn[rows, cols] > 0
        stable_xi = (test_xi[rows, cols] > 0) & (xi[rows, cols] > 0)
        self._f = fn[rows, cols]
        self._xi = xi[rows, cols]
        # column c of the stabilization matrices holds the poles of order index c + 1
        self._k = (cols + 1) % self._n_orders
        self._category = np.select([stable_fn & stable_xi, stable_fn, stable_xi], [0, 1, 2], default=3)
        self.stable_label.setText(f'{np.sum(self._category == 0)} stable poles')

    def _y(self):
        """Chart y-values of all poles: polynomial order or damping ratio."""
        if self.chart_type == 'stability':
            return self._k % self._n_per_band
        return self._xi

    def _cluster_ylim(self):
        in_band = (self._category == 0) & (self._f > self.Model.lower) & (self._f < self.Model.upper)
        xi = self._xi[in_band]
        if xi.size == 0:
            return 0, 0.1
        return 0, 1.1 * min(xi.max(), xi.mean() + 2 * xi.std())

    # -------------------------------------------------------------- drawing
    def _redraw_poles(self, reset_view=False):
        ax = self.ax_poles
        xlim = (self.Model.lower, self.Model.upper) if reset_view else ax.get_xlim()
        for artist in self._pole_artists:
            artist.remove()
        self._pole_artists = []

        y = self._y()
        for cat, (label, marker, color, size) in enumerate(_CATEGORIES):
            if cat > 0 and not self.show_unstable:
                continue
            mask = self._category == cat
            line, = ax.plot(self._f[mask], y[mask], ls='none', marker=marker, color=color, ms=size, label=label)
            self._pole_artists.append(line)

        if self.chart_type == 'stability':
            ax.set_title('Stability chart')
            ax.set_ylabel('Polynomial order')
            ax.set_ylim(0, self._n_per_band + 1)
        else:
            ax.set_title('Cluster diagram')
            ax.set_ylabel('Damping ratio')
            ax.set_ylim(*self._cluster_ylim())
        ax.set_xlim(xlim)

        if self.show_legend:
            ax.legend(loc='lower center', ncol=2, fontsize='small', framealpha=0.9)
        elif ax.get_legend() is not None:
            ax.get_legend().remove()

        self._update_selection_artists()

    def _redraw_frf(self):
        xlim = self.ax_frf.get_xlim()
        for artist in self._frf_artists:
            artist.remove()
        freq, frf = self.Model.freq, np.abs(self.Model.frf)
        if self.frf_plot_type == 'abs':
            self._frf_artists = self.ax_frf.plot(freq, frf.mean(axis=0), color='k', alpha=0.7, lw=1)
        else:
            self._frf_artists = self.ax_frf.plot(freq, frf.T, color='k', alpha=0.3, lw=0.8)

        # Scale y to the measurement, then fix it so the reconstruction doesn't rescale it
        self.ax_frf.relim()
        self.ax_frf.set_autoscaley_on(True)
        self.ax_frf.autoscale_view(scalex=False)
        self.ax_frf.set_ylim(self.ax_frf.get_ylim())
        self.ax_frf.set_xlim(xlim)
        self._redraw_reconstruction()

    def _redraw_reconstruction(self):
        for artist in self._rec_artists:
            artist.remove()
        self._rec_artists = []
        if self.H is None:
            return
        H = np.abs(self.H)
        if self.frf_plot_type == 'abs':
            self._rec_artists = self.ax_frf.plot(self.Model.freq, H.mean(axis=0), color='r', lw=2)
        else:
            self._rec_artists = self.ax_frf.plot(self.Model.freq, H.T, color='r', lw=1)

    def _selection_xy(self, poles):
        if not poles:
            return [], []
        f, xi = zip(*(self._pole(k, i) for k, i in poles))
        if self.chart_type == 'stability':
            return list(f), [k % self._n_per_band for k, _ in poles]
        return list(f), list(xi)

    def _update_selection_artists(self):
        f, y = self._selection_xy(self.selected)
        self._sel_markers.set_data(f, y)
        self._sel_top.set_data(f, [1.0] * len(f))
        self._update_highlight()

    def _update_highlight(self):
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        self._highlight.set_data(*self._selection_xy([self.selected[r] for r in rows if r < len(self.selected)]))
        self.canvas.draw_idle()

    def _refresh_table(self):
        self.table.blockSignals(True)
        self.table.clearSelection()
        self.table.setRowCount(len(self.selected))
        align = QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        for row, (k, i) in enumerate(self.selected):
            f, xi = self._pole(k, i)
            for col, text in enumerate((f'{f:.2f}', f'{100 * xi:.3f}', str(k % self._n_per_band))):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(align)
                self.table.setItem(row, col, item)
        self.table.blockSignals(False)

    # ------------------------------------------------------------ selection
    def _pole(self, k, i):
        """Natural frequency and damping ratio of pole ``i`` at order index ``k``."""
        return float(np.ravel(self.Model.pole_freq[k])[i]), float(np.ravel(self.Model.pole_xi[k])[i])

    def _pole_index(self, k, f, xi):
        """Index in ``Model.all_poles[k]`` of the pole with frequency ``f`` and damping ``xi``."""
        pf = np.ravel(self.Model.pole_freq[k])
        px = np.ravel(self.Model.pole_xi[k])
        cost = np.abs(pf - f) + np.abs(px - xi) * f
        near = np.flatnonzero(cost <= cost.min() + 1e-9 * max(f, 1.0))
        # a conjugate pair shares f and xi; take the positive-imaginary pole, as select_closest_poles does
        upper = near[np.ravel(self.Model.all_poles[k])[near].imag > 0]
        return int(upper[0] if upper.size else near[0])

    def _existing_selection(self):
        """Poles already selected on the model, if they still match its current poles."""
        pole_ind = getattr(self.Model, 'pole_ind', None)
        nat_freq = getattr(self.Model, 'nat_freq', None)
        if pole_ind is None or nat_freq is None or len(pole_ind) == 0 or len(pole_ind) != len(nat_freq):
            return []
        poles = []
        for (k, i), f in zip(pole_ind, nat_freq):
            k, i = int(k), int(i)
            if not (0 <= k < self._n_orders and 0 <= i < np.size(self.Model.pole_freq[k])):
                return []
            # stale indices from an earlier get_poles() call point at unrelated frequencies
            if not np.isclose(self._pole(k, i)[0], f, rtol=5e-3):
                return []
            poles.append((k, i))
        return poles

    def _set_selection(self, poles):
        self.selected = sorted(set(poles), key=lambda p: self._pole(*p)[0])
        self._pick_history = ([p for p in self._pick_history if p in self.selected]
                              + [p for p in self.selected if p not in self._pick_history])

        self.Model.pole_ind = [[k, i] for k, i in self.selected]
        self.Model.nat_freq = [self._pole(k, i)[0] for k, i in self.selected]
        self.Model.nat_xi = [self._pole(k, i)[1] for k, i in self.selected]

        self._reconstruct()
        self._refresh_table()
        self._update_selection_artists()

    def _reconstruct(self):
        self.H = None
        if self.selected:
            freq = self.Model.freq
            try:
                self.H, _ = self.Model.get_constants(
                    whose_poles='own',
                    f_lower=max(self.Model.lower, freq[0]),
                    f_upper=min(self.Model.upper, freq[-1]))
            except Exception as err:
                self.statusBar().showMessage(f'FRF reconstruction failed: {err}', 8000)
        self._redraw_reconstruction()

    def _pixel_ratio(self):
        return getattr(self.canvas, 'device_pixel_ratio', 1)

    def _pick_at_pixel(self, x_px, y_px):
        y = self._y()
        pickable = (self._category == 0) | self.show_unstable
        x0, x1 = sorted(self.ax_poles.get_xlim())
        y0, y1 = sorted(self.ax_poles.get_ylim())
        pickable &= (self._f >= x0) & (self._f <= x1) & (y >= y0) & (y <= y1)
        idx = np.flatnonzero(pickable)
        if idx.size == 0:
            self.statusBar().showMessage('No poles in view.', 3000)
            return False

        xy = self.ax_poles.transData.transform(np.column_stack((self._f[idx], y[idx])))
        dist = np.hypot(xy[:, 0] - x_px, xy[:, 1] - y_px)
        j = int(np.argmin(dist))
        if dist[j] > PICK_RADIUS * self._pixel_ratio():
            self.statusBar().showMessage('No pole close to the click.', 3000)
            return False

        p = idx[j]
        k = int(self._k[p])
        pole = (k, self._pole_index(k, self._f[p], self._xi[p]))
        f, xi = self._pole(*pole)
        if pole in self.selected:
            self.statusBar().showMessage(f'Pole at {f:.2f} Hz is already selected.', 3000)
            return False
        self._set_selection(self.selected + [pole])
        self.statusBar().showMessage(f'Picked pole at {f:.2f} Hz, ζ = {100 * xi:.3f} %', 5000)
        return True

    def _remove_at_pixel(self, x_px):
        if not self.selected:
            return False
        f = np.array([self._pole(k, i)[0] for k, i in self.selected])
        xs = self.ax_poles.transData.transform(np.column_stack((f, np.zeros_like(f))))[:, 0]
        j = int(np.argmin(np.abs(xs - x_px)))
        if abs(xs[j] - x_px) > PICK_RADIUS * self._pixel_ratio():
            self.statusBar().showMessage('No selected pole close to the click.', 3000)
            return False
        self._remove([self.selected[j]])
        self.statusBar().showMessage(f'Removed pole at {f[j]:.2f} Hz', 5000)
        return True

    def _remove(self, poles):
        self._set_selection([p for p in self.selected if p not in poles])

    def _remove_table_selection(self):
        rows = {index.row() for index in self.table.selectionModel().selectedRows()}
        self._remove([self.selected[r] for r in rows if r < len(self.selected)])

    def pick_pole(self, freq, y):
        """Pick the pole shown nearest to a chart position, as a click there would.

        :param freq: frequency [Hz]
        :param y: polynomial order (stability chart) or damping ratio (cluster diagram)
        :return: True if a pole was added to the selection
        """
        self.canvas.draw()
        return self._pick_at_pixel(*self.ax_poles.transData.transform((freq, y)))

    def remove_pole(self, freq):
        """Remove the selected pole nearest to ``freq``, as a right-click there would.

        :return: True if a pole was removed
        """
        self.canvas.draw()
        return self._remove_at_pixel(self.ax_poles.transData.transform((freq, 0))[0])

    def undo_last_pick(self):
        """Remove the most recently picked pole."""
        if self._pick_history:
            self._remove([self._pick_history[-1]])

    def clear_selection(self):
        """Remove all selected poles."""
        self._set_selection([])

    def export_poles(self, path):
        """Write the selected poles to a CSV file.

        Columns: natural frequency [Hz], damping ratio, and the two ``Model.pole_ind`` indices.
        """
        rows = np.array([(*self._pole(k, i), k, i) for k, i in self.selected], dtype=float).reshape(-1, 4)
        np.savetxt(path, rows, delimiter=',', fmt=['%.6f', '%.6e', '%d', '%d'],
                   header='frequency_hz,damping_ratio,order_index,pole_index', comments='')

    # --------------------------------------------------------------- events
    def _zoom_to_pixels(self, targets):
        """Set axis limits to pixel ranges.

        :param targets: ``(axes, (x0, x1) or None, (y0, y1) or None)`` tuples. All ranges
            are converted with the current transforms before any limit changes,
            because the two axes share the frequency axis.
        """
        limits = []
        for ax, x_px, y_px in targets:
            inv = ax.transData.inverted()
            bbox = ax.bbox
            xlim = None if x_px is None else inv.transform([(x_px[0], bbox.y0), (x_px[1], bbox.y0)])[:, 0]
            ylim = None if y_px is None else inv.transform([(bbox.x0, y_px[0]), (bbox.x0, y_px[1])])[:, 1]
            limits.append((ax, xlim, ylim))
        for ax, xlim, ylim in limits:
            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)

    def _zoom_ranges(self, drag, event):
        """Pixel ranges ``(x_px, y_px)`` of a zoom drag.

        A range is None for an axis the drag doesn't zoom (a mostly horizontal drag
        zooms frequency only); the result is None if the drag is too small both ways.
        """
        bbox = self.ax_poles.bbox
        x_px = tuple(sorted((drag['x'], np.clip(event.x, bbox.x0, bbox.x1))))
        y_px = tuple(sorted((drag['y'], np.clip(event.y, bbox.y0, bbox.y1))))
        lock = AXIS_LOCK * self._pixel_ratio()
        x_small = x_px[1] - x_px[0] < lock
        y_small = y_px[1] - y_px[0] < lock
        if x_small and y_small:
            return None
        return (None if x_small else x_px), (None if y_small else y_px)

    def _save_home_view(self):
        """Record the view before the first zoom or pan, so Home returns to it."""
        if not self._home_view_saved:
            self.toolbar.push_current()
            self._home_view_saved = True

    def _on_press(self, event):
        if event.dblclick and event.button == 1:
            # the first click of the double-click may have picked a pole; take it back
            if self._click_pick in self.selected:
                self._remove([self._click_pick])
            self._click_pick = None
            self._drag = None
            self.toolbar.home()
            self.canvas.draw_idle()
            return
        if self._drag is not None or event.inaxes is None or event.button not in (1, 2, 3):
            return
        self._drag = {
            'button': int(event.button), 'x': event.x, 'y': event.y, 'moved': False,
            'limits': [(ax, ax.get_xlim(), ax.get_ylim()) for ax in (self.ax_poles, self.ax_frf)],
        }

    def _on_motion(self, event):
        drag = self._drag
        if drag is None:
            return
        dx, dy = event.x - drag['x'], event.y - drag['y']
        if not drag['moved']:
            if np.hypot(dx, dy) < DRAG_THRESHOLD * self._pixel_ratio():
                return
            drag['moved'] = True
            self._save_home_view()

        if drag['button'] == 1:
            ranges = self._zoom_ranges(drag, event)
            if ranges is None:
                self.toolbar.remove_rubberband()
            else:
                bbox = self.ax_poles.bbox
                (x0, x1), (y0, y1) = ranges[0] or (bbox.x0, bbox.x1), ranges[1] or (bbox.y0, bbox.y1)
                self.toolbar.draw_rubberband(event, x0, y0, x1, y1)
        else:
            # pan from the view at the press, so the data follows the cursor exactly
            for ax, xlim, ylim in drag['limits']:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            bbox = self.ax_poles.bbox
            self._zoom_to_pixels([(ax, (bbox.x0 - dx, bbox.x1 - dx), (bbox.y0 - dy, bbox.y1 - dy))
                                  for ax in (self.ax_poles, self.ax_frf)])
            self.canvas.draw_idle()

    def _on_release(self, event):
        drag, self._drag = self._drag, None
        if drag is None:
            return
        if not drag['moved']:
            if drag['button'] == 1:
                picked = self._pick_at_pixel(drag['x'], drag['y'])
                self._click_pick = self._pick_history[-1] if picked else None
            elif drag['button'] == 3:
                self._remove_at_pixel(drag['x'])
        elif drag['button'] == 1:
            self.toolbar.remove_rubberband()
            ranges = self._zoom_ranges(drag, event)
            if ranges is not None:
                self._zoom_to_pixels([(ax, *ranges) for ax in (self.ax_poles, self.ax_frf)])
                self.toolbar.push_current()
        else:
            self.toolbar.push_current()
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        bbox = self.ax_poles.bbox
        x, y = event.x, event.y
        in_x = bbox.x0 <= x <= bbox.x1
        in_y = bbox.y0 <= y <= bbox.y1
        if in_x and in_y:
            zoom_x, zoom_y = [self.ax_poles], [self.ax_poles, self.ax_frf]
        elif in_x and y < bbox.y0:      # over the frequency axis
            zoom_x, zoom_y = [self.ax_poles], []
        elif in_y and x < bbox.x0:      # over the magnitude axis
            zoom_x, zoom_y = [], [self.ax_frf]
        elif in_y and x > bbox.x1:      # over the order / damping axis
            zoom_x, zoom_y = [], [self.ax_poles]
        else:
            return

        self._save_home_view()
        scale = WHEEL_ZOOM ** -event.step
        x_px = (x - (x - bbox.x0) * scale, x + (bbox.x1 - x) * scale)
        y_px = (y - (y - bbox.y0) * scale, y + (bbox.y1 - y) * scale)
        self._zoom_to_pixels([(ax, x_px if ax in zoom_x else None, y_px if ax in zoom_y else None)
                              for ax in (self.ax_poles, self.ax_frf)])
        self._wheel_timer.start()
        self.canvas.draw_idle()

    def _on_chart_type(self, *_):
        self.chart_type = 'cluster' if self.cluster_radio.isChecked() else 'stability'
        self._redraw_poles()
        self.toolbar.update()  # restart the zoom history; y means something else now
        self._home_view_saved = False
        self.canvas.draw_idle()

    def _on_frf_type(self, index):
        self.frf_plot_type = 'abs' if index == 0 else 'all'
        self._redraw_frf()
        self.canvas.draw_idle()

    def _on_show_unstable(self, checked):
        self.show_unstable = checked
        self._redraw_poles()
        self.canvas.draw_idle()

    def _on_show_legend(self, checked):
        self.show_legend = checked
        self._redraw_poles()
        self.canvas.draw_idle()

    def _apply_criteria(self):
        self.fn_tol = self.fn_spin.value() / 100
        self.xi_tol = self.xi_spin.value() / 100
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))
        try:
            self._compute_stability()
            self._redraw_poles()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.canvas.draw_idle()

    def save_figure_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save figure', 'stability_chart.png', 'Images (*.png *.pdf *.svg)')
        if path:
            self.fig.savefig(path, dpi=200)
            self.statusBar().showMessage(f'Saved {path}', 5000)

    def export_poles_dialog(self):
        if not self.selected:
            self.statusBar().showMessage('No poles selected.', 3000)
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Export selected poles', 'selected_poles.csv', 'CSV files (*.csv)')
        if path:
            self.export_poles(path)
            self.statusBar().showMessage(f'Exported {len(self.selected)} poles to {path}', 5000)

    def show_help(self):
        QtWidgets.QMessageBox.information(self, 'Pole picking help', HELP_TEXT)

    def closeEvent(self, event):
        if self._loop is not None:
            self._loop.quit()
        super().closeEvent(event)


_app = None  # keeps the QApplication alive between calls


def select_poles_qt(Model, **kwargs):
    """Open the Qt stability chart and block until it is closed.

    :param Model: model with poles computed by ``get_poles()``
    :param kwargs: passed on to :class:`SelectPolesQt`
    """
    global _app
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])

    window = SelectPolesQt(Model, **kwargs)
    loop = QtCore.QEventLoop()
    window._loop = loop
    window.show()
    window.raise_()
    window.activateWindow()
    (loop.exec if hasattr(loop, 'exec') else loop.exec_)()
    window.deleteLater()
