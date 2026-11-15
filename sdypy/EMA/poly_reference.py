"""Poly-reference (MIMO) modal identification.

:class:`ModelPoly` identifies poles, modal participation factors and mode
shapes from FRFs measured with several references (inputs). Poles come from
the poly-reference least-squares complex frequency-domain method (pLSCF,
PolyMAX), the mode shapes from a least-squares frequency-domain fit (LSFD)
that uses the participation factors.

Literature:
    [1] Peeters, B., Van der Auweraer, H., Guillaume, P., Leuridan, J.,
        "The PolyMAX Frequency-Domain Method: A New Standard for Modal
        Parameter Estimation?", Shock and Vibration, 11 (2004), 395-409.
    [2] Guillaume, P., Verboven, P., Vanlanduit, S., Van der Auweraer, H.,
        Peeters, B., "A Poly-Reference Implementation of the Least-Squares
        Complex Frequency-Domain Estimator", Proceedings of IMAC XXI (2003).
"""

import importlib
import warnings

import numpy as np
import scipy.linalg
from tqdm import tqdm

from . import tools
from . import normal_modes

_FRF_FORMS = ('receptance', 'mobility', 'accelerance')


class ModelPoly():
    """
    Modal model of poly-reference frequency response functions.

    The FRF array has the shape ``(n_outputs, n_freq, n_inputs)``: the first
    two axes are as in :class:`Model`, the last axis holds the references
    (inputs). A 2-D array ``(n_outputs, n_freq)`` is a single reference.

    Typical use::

        m = ModelPoly(frf, freq, freq_lower=10, freq_upper=500)
        m.get_poles()
        m.select_closest_poles([52.1, 54.0, 130.2])
        frf_reconstructed, mode_shape = m.get_constants()
    """

    def __init__(self,
                 frf,
                 freq,
                 freq_lower=None,
                 freq_upper=None,
                 pol_order_high=50,
                 frf_form='receptance',
                 driving_point=None):
        """
        :param frf: FRF array of shape ``(n_outputs, n_freq, n_inputs)``, or
            ``(n_outputs, n_freq)`` for a single input
        :type frf: array_like
        :param freq: frequency vector in Hz, length ``n_freq``, ascending
        :type freq: array_like
        :param freq_lower: lower limit of the identification band in Hz,
            defaults to the first frequency above 0
        :type freq_lower: float, optional
        :param freq_upper: upper limit of the identification band in Hz,
            defaults to the last frequency
        :type freq_upper: float, optional
        :param pol_order_high: highest order of the denominator matrix
            polynomial; poles are computed for every order from 1 to this
            value
        :type pol_order_high: int, optional
        :param frf_form: ``'receptance'``, ``'mobility'`` or ``'accelerance'``
        :type frf_form: str, optional
        :param driving_point: for every input, the index of the output measured
            at the excitation point; when given, mode shapes and participation
            factors are scaled to unit modal A (see :meth:`get_constants`)
        :type driving_point: int or sequence of int, optional
        """
        frf = np.asarray(frf)
        if frf.ndim == 2:
            frf = frf[:, :, np.newaxis]
        if frf.ndim != 3:
            raise ValueError(
                f'frf must be a 2-D or 3-D array, got {frf.ndim} dimensions')
        self.frf = frf.astype(complex)
        self.n_outputs, _, self.n_inputs = self.frf.shape

        self.freq = np.asarray(freq, dtype=float)
        if self.freq.ndim != 1:
            raise ValueError('freq must be a 1-D array')
        if len(self.freq) != self.frf.shape[1]:
            raise ValueError(
                f'frf has {self.frf.shape[1]} frequency points, '
                f'freq has {len(self.freq)}')
        if len(self.freq) < 2 or np.any(np.diff(self.freq) <= 0):
            raise ValueError('freq must be strictly ascending')

        if freq_lower is None:
            freq_lower = self.freq[self.freq > 0][0]
        if freq_upper is None:
            freq_upper = self.freq[-1]
        freq_lower, freq_upper = float(freq_lower), float(freq_upper)
        if not (freq_lower < freq_upper):
            raise ValueError('freq_lower must be smaller than freq_upper')
        if freq_lower < self.freq[0] or freq_upper > self.freq[-1]:
            raise ValueError(
                f'[freq_lower, freq_upper] = [{freq_lower}, {freq_upper}] is '
                f'outside the frequency range [{self.freq[0]}, {self.freq[-1]}]')
        if freq_lower <= 0:
            raise ValueError('freq_lower must be greater than zero')
        self.freq_lower = freq_lower
        self.freq_upper = freq_upper

        if isinstance(pol_order_high, bool) or not isinstance(pol_order_high, (int, np.integer)):
            raise ValueError('pol_order_high must be an integer')
        if pol_order_high < 1:
            raise ValueError('pol_order_high must be at least 1')
        self.pol_order_high = int(pol_order_high)

        if frf_form not in _FRF_FORMS:
            raise ValueError(f'frf_form must be one of {_FRF_FORMS}, not {frf_form!r}')
        self.frf_form = frf_form

        if driving_point is not None:
            driving_point = np.atleast_1d(np.asarray(driving_point, dtype=int))
            if driving_point.shape != (self.n_inputs,):
                raise ValueError(
                    f'driving_point must give one output index per input '
                    f'({self.n_inputs}), got {driving_point.tolist()}')
            if np.any(driving_point < 0) or np.any(driving_point >= self.n_outputs):
                raise ValueError('driving_point index out of range')
        self.driving_point = driving_point

    @property
    def _band(self):
        return (self.freq >= self.freq_lower) & (self.freq <= self.freq_upper)

    def get_poles(self, show_progress=True):
        """Estimate poles and participation factors with pLSCF.

        For every model order ``n`` from 1 to ``pol_order_high`` the right
        matrix-fraction model ``H(z) = B(z) A(z)^-1`` with an
        ``n_inputs x n_inputs`` denominator matrix polynomial of order ``n``
        is fitted to the FRFs in ``[freq_lower, freq_upper]``. The numerators
        are eliminated with an orthonormal polynomial basis (as in the
        single-reference pLSCF of sdypy-EMA PR #17); poles and participation
        factors are the eigenvalues and eigenvectors of the block companion
        matrix of ``A``.

        Sets ``all_poles``, ``pole_freq``, ``pole_damping_ratio`` and
        ``pole_participation_factor``, one entry per model order; only poles
        with a positive imaginary part are kept. Participation factors have
        the shape ``(n_inputs, n_poles)`` and unit maximum magnitude.

        :param show_progress: show a progress bar
        :type show_progress: bool, optional
        """
        band = self._band
        freq = self.freq[band]
        frf = self.frf[:, band, :]
        n_out, n_freq, n_in = frf.shape
        n_max = self.pol_order_high
        if 2 * n_freq < n_max + 1:
            raise ValueError(
                'too few frequency points in [freq_lower, freq_upper] for '
                f'pol_order_high={n_max}')

        self.sampling_time = 1 / (2 * self.freq_upper)
        frf = frf / np.max(np.abs(frf))
        z = np.exp(1j * 2 * np.pi * freq * self.sampling_time)
        basis = z[:, np.newaxis] ** np.arange(n_max + 1)[np.newaxis, :]
        # orthonormal basis of the real-stacked numerator space; the first
        # n+1 columns of q span the numerator of order n
        q, _ = np.linalg.qr(np.vstack([basis.real, basis.imag]))

        # regression columns ordered (order r, input j): slicing the first
        # (n+1)*n_in columns gives the problem of order n
        size = (n_max + 1) * n_in
        t = np.zeros((size, size))
        s = np.zeros((n_out, n_max + 1, size))
        for o in range(n_out):
            x = -(frf[o][:, np.newaxis, :] * basis[:, :, np.newaxis]).reshape(n_freq, size)
            t += (x.conj().T @ x).real
            s[o] = q.T @ np.vstack([x.real, x.imag])

        self.all_poles = []
        self.pole_freq = []
        self.pole_damping_ratio = []
        self.pole_participation_factor = []

        orders = range(1, n_max + 1)
        if show_progress:
            orders = tqdm(orders, ncols=100)

        correction = np.einsum('ok,ol->kl', s[:, 0, :], s[:, 0, :])
        for n in orders:
            correction += np.einsum('ok,ol->kl', s[:, n, :], s[:, n, :])
            k = (n + 1) * n_in
            m = t[:k, :k] - correction[:k, :k]
            poles, participation = _poles_from_reduced_matrix(
                m, n, n_in, self.sampling_time)
            f_pole, damping_ratio = tools.complex_freq_to_freq_and_damp(poles)
            self.all_poles.append(poles)
            self.pole_freq.append(f_pole)
            self.pole_damping_ratio.append(damping_ratio)
            self.pole_participation_factor.append(participation)

    def select_closest_poles(self, approx_natural_freq, model_order=None,
                             freq_tol=0.01, damping_tol=0.05):
        """Select the stable poles closest to approximate natural frequencies.

        A pole of ``model_order`` is stable when the previous order has a pole
        within ``freq_tol`` relative frequency and ``damping_tol`` relative
        damping ratio, it has a positive damping ratio and lies in
        ``[freq_lower, freq_upper]``.

        Sets ``natural_freq``, ``damping_ratio``, ``poles``, ``pole_idx``
        (rows ``[order index, pole index]``) and ``participation_factor``
        (shape ``(n_inputs, n_modes)``), in the order of
        ``approx_natural_freq``.

        :param approx_natural_freq: approximate natural frequencies in Hz
        :type approx_natural_freq: array_like
        :param model_order: model order to select from, defaults to
            ``pol_order_high``
        :type model_order: int, optional
        :param freq_tol: relative frequency tolerance of the stability check
        :type freq_tol: float, optional
        :param damping_tol: relative damping tolerance of the stability check
        :type damping_tol: float, optional
        :raises ValueError: if two approximate frequencies resolve to the same
            pole or no stable pole exists
        """
        if not hasattr(self, 'all_poles'):
            raise RuntimeError('call get_poles() first')
        if model_order is None:
            model_order = self.pol_order_high
        if not 2 <= model_order <= self.pol_order_high:
            raise ValueError(
                f'model_order must be in [2, {self.pol_order_high}]')
        i_order = model_order - 1

        f = self.pole_freq[i_order]
        xi = self.pole_damping_ratio[i_order]
        f_prev = self.pole_freq[i_order - 1]
        xi_prev = self.pole_damping_ratio[i_order - 1]
        stable = (xi > 0) & (f >= self.freq_lower) & (f <= self.freq_upper)
        if len(f_prev):
            df = np.abs(f[:, None] - f_prev[None, :]) / f[:, None]
            dxi = np.abs(xi[:, None] - xi_prev[None, :]) / np.abs(xi[:, None])
            stable &= np.any((df < freq_tol) & (dxi < damping_tol), axis=1)
        else:
            stable[:] = False
        candidates = np.flatnonzero(stable)
        if len(candidates) == 0:
            raise ValueError(f'no stable poles at model order {model_order}')

        approx_natural_freq = np.atleast_1d(np.asarray(approx_natural_freq, dtype=float))
        selected = [candidates[np.argmin(np.abs(f[candidates] - fr))]
                    for fr in approx_natural_freq]
        if len(set(selected)) != len(selected):
            raise ValueError(
                'two approximate natural frequencies resolve to the same pole: '
                f'{approx_natural_freq.tolist()} -> {f[selected].tolist()} Hz')

        self._apply_selection([[i_order, i] for i in selected])

    def _apply_selection(self, pole_idx):
        """Set the selected modes from ``[order index, pole index]`` rows."""
        pole_idx = np.asarray(pole_idx, dtype=int).reshape(-1, 2)
        self.pole_idx = pole_idx
        self.poles = np.array([self.all_poles[k][i] for k, i in pole_idx], dtype=complex)
        self.natural_freq = np.array([self.pole_freq[k][i] for k, i in pole_idx])
        self.damping_ratio = np.array([self.pole_damping_ratio[k][i] for k, i in pole_idx])
        self.participation_factor = np.array(
            [self.pole_participation_factor[k][:, i] for k, i in pole_idx],
            dtype=complex).reshape(-1, self.n_inputs).T

    def select_poles(self):
        """Select poles interactively on the Qt stability chart.

        Opens the stability chart of :class:`Model` (needs PySide6 or PyQt6,
        e.g. ``pip install "sdypy-EMA[qt]"``) and blocks until it is closed.
        The FRFs of all output-input pairs are shown; the reconstruction
        shown while picking uses :meth:`get_constants`. On closing, the
        picked poles are selected as by :meth:`select_closest_poles`
        (``natural_freq``, ``damping_ratio``, ``poles``, ``pole_idx``,
        ``participation_factor``) and the modal constants are computed.
        """
        if not hasattr(self, 'all_poles'):
            raise RuntimeError('call get_poles() first')
        try:
            # imported here so that EMA imports without them; qt_compat raises
            # ImportError when no Qt binding is installed
            importlib.import_module('matplotlib')
            importlib.import_module('matplotlib.backends.qt_compat')
        except ImportError as err:
            raise ImportError(
                'ModelPoly.select_poles() needs matplotlib and PySide6 or PyQt6; '
                'install them, e.g. pip install "sdypy-EMA[qt]", or use '
                'ModelPoly.select_closest_poles() instead.') from err
        from .pole_picking_qt import select_poles_qt

        view = _StabilityChartView(self)
        select_poles_qt(view)
        view.apply_to_model()

    @classmethod
    def from_uff(cls, uff_filename, **kwargs):
        """Create a model from the FRFs (dataset 58) in a UFF file.

        Every FRF is placed by its response node and direction (output) and
        its reference node and direction (input). Outputs and inputs keep
        the order of their first appearance in the file and are stored in
        ``outputs`` and ``inputs`` as ``(node, direction)`` tuples. Unless
        given, ``frf_form`` is taken from the file and ``driving_point`` is
        set when every input is also an output.

        :param uff_filename: path to the UFF file
        :param kwargs: passed on to :class:`ModelPoly`
        :raises ValueError: if the file holds no FRFs, the FRFs have different
            frequency vectors or an output-input pair is missing
        """
        import pyuff

        sets = pyuff.UFF(uff_filename).read_sets()
        if isinstance(sets, dict):
            sets = [sets]
        frfs = [d for d in sets if d.get('type') == 58 and d.get('func_type') == 4]
        if not frfs:
            raise ValueError(f'no FRFs (dataset 58, function type 4) in {uff_filename}')

        freq = np.asarray(frfs[0]['x'], dtype=float)
        outputs, inputs = [], []
        for d in frfs:
            if len(d['x']) != len(freq) or not np.allclose(d['x'], freq):
                raise ValueError('the FRFs in the UFF file have different frequency vectors')
            out, inp = (int(d['rsp_node']), int(d['rsp_dir'])), (int(d['ref_node']), int(d['ref_dir']))
            if out not in outputs:
                outputs.append(out)
            if inp not in inputs:
                inputs.append(inp)

        frf = np.full((len(outputs), len(freq), len(inputs)), np.nan, dtype=complex)
        for d in frfs:
            o = outputs.index((int(d['rsp_node']), int(d['rsp_dir'])))
            i = inputs.index((int(d['ref_node']), int(d['ref_dir'])))
            frf[o, :, i] = d['data']
        missing = np.argwhere(np.isnan(frf[:, 0, :]))
        if len(missing):
            pairs = [(outputs[o], inputs[i]) for o, i in missing[:5]]
            raise ValueError(f'missing FRFs for (output, input) pairs, e.g. {pairs}')

        if 'frf_form' not in kwargs:
            forms = {8: 'receptance', 11: 'mobility', 12: 'accelerance'}
            kwargs['frf_form'] = forms.get(frfs[0].get('ordinate_spec_data_type'), 'receptance')
        if 'driving_point' not in kwargs and all(inp in outputs for inp in inputs):
            kwargs['driving_point'] = [outputs.index(inp) for inp in inputs]

        model = cls(frf, freq, **kwargs)
        model.outputs = outputs
        model.inputs = inputs
        return model

    def get_constants(self, freq_lower=None, freq_upper=None):
        """Mode shapes and residuals by least squares (LSFD).

        With the selected poles ``lambda_r`` and participation factors ``l_r``
        fixed, the model

        .. math::

            H(\\omega) = \\sum_r \\left(
                \\frac{\\psi_r l_r^T}{\\mathrm{j}\\omega - \\lambda_r} +
                \\frac{\\psi_r^* l_r^H}{\\mathrm{j}\\omega - \\lambda_r^*}\\right)
                - \\frac{LR}{\\omega^2} + UR

        (receptance; mobility and accelerance are converted first) is fitted
        to the FRFs in ``[freq_lower, freq_upper]`` for every output, giving
        the complex mode shapes ``psi_r`` and real residuals ``LR`` and
        ``UR``.

        With ``driving_point`` set, every mode is rescaled so that its
        participation factor equals its mode shape at the driving points in
        the least-squares sense (unit modal A scaling for a reciprocal
        system). The residues ``psi_r l_r^T`` do not depend on the scaling.

        Sets ``mode_shape`` ``(n_outputs, n_modes)``, ``residual_lower`` and
        ``residual_upper`` ``(n_outputs, n_inputs)`` and ``frf_reconstructed``
        (shape of ``frf``, in ``frf_form``; at 0 Hz it is not finite when the
        lower residual is non-zero).

        :param freq_lower: lower limit of the fit in Hz, defaults to
            ``self.freq_lower``
        :param freq_upper: upper limit of the fit in Hz, defaults to
            ``self.freq_upper``
        :return: ``(frf_reconstructed, mode_shape)``
        """
        if not hasattr(self, 'poles'):
            raise RuntimeError('select the poles first (select_closest_poles)')
        freq_lower = self.freq_lower if freq_lower is None else float(freq_lower)
        freq_upper = self.freq_upper if freq_upper is None else float(freq_upper)
        band = (self.freq >= freq_lower) & (self.freq <= freq_upper) & (self.freq > 0)

        poles = self.poles
        participation = self.participation_factor
        n_modes = len(poles)
        n_in = self.n_inputs

        omega = 2 * np.pi * self.freq
        with np.errstate(divide='ignore', invalid='ignore'):
            receptance = self.frf / _form_factor(omega, self.frf_form)[None, :, None]

        def design(omega):
            jw = 1j * omega[:, None]
            p = 1 / (jw - poles[None, :])
            pc = 1 / (jw - poles.conj()[None, :])
            a = np.zeros((len(omega), n_in, 2 * n_modes + 2 * n_in), dtype=complex)
            with np.errstate(divide='ignore', invalid='ignore'):
                lower = -1 / omega**2
            for i in range(n_in):
                l_i = participation[i]
                a[:, i, :n_modes] = p * l_i + pc * l_i.conj()
                a[:, i, n_modes:2 * n_modes] = 1j * (p * l_i - pc * l_i.conj())
                a[:, i, 2 * n_modes + i] = lower
                a[:, i, 2 * n_modes + n_in + i] = 1
            return a.reshape(len(omega) * n_in, -1)

        a = design(omega[band])
        b = receptance[:, band, :].reshape(self.n_outputs, -1).T
        x, *_ = np.linalg.lstsq(np.vstack([a.real, a.imag]),
                                np.vstack([b.real, b.imag]), rcond=None)

        mode_shape = (x[:n_modes] + 1j * x[n_modes:2 * n_modes]).T
        self.residual_lower = x[2 * n_modes:2 * n_modes + n_in].T
        self.residual_upper = x[2 * n_modes + n_in:].T

        with np.errstate(divide='ignore', invalid='ignore'):
            rec = (design(omega) @ x).T.reshape(self.n_outputs, len(omega), n_in)
            self.frf_reconstructed = rec * _form_factor(omega, self.frf_form)[None, :, None]

        if self.driving_point is not None:
            psi_dp = mode_shape[self.driving_point, :]
            c2 = (np.sum(psi_dp.conj() * participation, axis=0)
                  / np.sum(np.abs(psi_dp)**2, axis=0))
            c = np.sqrt(c2)
            mode_shape = mode_shape * c[None, :]
            self.participation_factor = participation / c[None, :]

        self.mode_shape = mode_shape
        return self.frf_reconstructed, self.mode_shape

    def normal_mode(self):
        """Real (normal) mode shapes closest to the complex ``mode_shape``.

        See :func:`sdypy.EMA.normal_modes.complex_to_normal_mode`.

        :return: real mode shapes, shape ``(n_outputs, n_modes)``
        """
        if not hasattr(self, 'mode_shape'):
            raise RuntimeError('mode shapes not computed; call get_constants() first')
        return normal_modes.complex_to_normal_mode(self.mode_shape)


class _StabilityChartView:
    """A :class:`ModelPoly` as the single-reference model the Qt chart shows.

    The chart reads the poles of every order and the FRFs as a 2-D array
    ``(n_outputs * n_inputs, n_freq)``; it writes the picked poles back as
    ``[order index, pole index]`` rows and calls ``get_constants`` for the
    reconstruction. Both the pre- and post-SEP 2 attribute spellings of
    :class:`Model` are provided, so that the chart works with either.
    """

    def __init__(self, model):
        self._model = model
        self.all_poles = model.all_poles
        self.pole_freq = model.pole_freq
        self.pole_xi = self.pole_damping_ratio = model.pole_damping_ratio
        self.freq = model.freq
        self.frf = model.frf.transpose(0, 2, 1).reshape(-1, len(model.freq))
        self.lower = self.freq_lower = model.freq_lower
        self.upper = self.freq_upper = model.freq_upper
        self.pol_order_high = model.pol_order_high
        self.n_bands = 1
        self.len_band = len(model.freq)
        self._selection = []
        self._natural_freq = []
        self._damping_ratio = []
        if hasattr(model, 'pole_idx'):
            self._selection = np.asarray(model.pole_idx).tolist()
            self._natural_freq = list(model.natural_freq)
            self._damping_ratio = list(model.damping_ratio)

    def _alias(name):
        return property(lambda self: getattr(self, name),
                        lambda self, value: setattr(self, name, list(value)))

    pole_ind = pole_idx = _alias('_selection')
    nat_freq = natural_freq = _alias('_natural_freq')
    nat_xi = damping_ratio = _alias('_damping_ratio')
    del _alias

    def get_constants(self, whose_poles='own', f_lower=None, f_upper=None,
                      freq_lower=None, freq_upper=None, **kwargs):
        lower = freq_lower if freq_lower is not None else f_lower
        upper = freq_upper if freq_upper is not None else f_upper
        self._model._apply_selection(self._selection)
        frf_reconstructed, mode_shape = self._model.get_constants(lower, upper)
        return frf_reconstructed.transpose(0, 2, 1).reshape(-1, len(self.freq)), mode_shape

    def apply_to_model(self):
        """Select the picked poles on the model and compute its constants."""
        if self._selection:
            self._model._apply_selection(self._selection)
            self._model.get_constants()


def _form_factor(omega, frf_form):
    """Factor ``(j omega)^k`` that turns receptance into ``frf_form``."""
    k = _FRF_FORMS.index(frf_form)
    return (1j * omega)**k


def _poles_from_reduced_matrix(m, n, n_in, sampling_time):
    """Poles and participation factors from the reduced normal matrix.

    :param m: reduced normal matrix of order ``n``, shape
        ``((n+1)*n_in, (n+1)*n_in)``, columns ordered (order, input)
    :return: s-domain poles with positive imaginary part and their
        participation factors ``(n_in, n_poles)`` with unit maximum magnitude
    """
    m = 0.5 * (m + m.T)
    # highest coefficient fixed to the identity; an over-parametrized model
    # (order above what the data supports) is singular: minimum-norm solution
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', scipy.linalg.LinAlgWarning)
            alpha = scipy.linalg.solve(m[:-n_in, :-n_in], -m[:-n_in, -n_in:],
                                       assume_a='sym')
    except (scipy.linalg.LinAlgError, scipy.linalg.LinAlgWarning):
        alpha = np.linalg.lstsq(m[:-n_in, :-n_in], -m[:-n_in, -n_in:], rcond=None)[0]

    # last block row [-alpha_0^T, ..., -alpha_{n-1}^T]: the eigenvectors start
    # with the left null vectors of A(z_k), the participation factors
    companion = np.zeros((n * n_in, n * n_in))
    companion[:-n_in, n_in:] = np.eye((n - 1) * n_in)
    companion[-n_in:, :] = -alpha.reshape(n, n_in, n_in).transpose(2, 0, 1).reshape(n_in, n * n_in)
    z_poles, vectors = scipy.linalg.eig(companion)

    with np.errstate(divide='ignore', invalid='ignore'):
        poles = np.log(z_poles) / sampling_time
    keep = np.isfinite(poles) & (poles.imag > 0)
    participation = vectors[:n_in, keep]
    idx = np.argmax(np.abs(participation), axis=0)
    participation = participation / participation[idx, np.arange(participation.shape[1])]
    return poles[keep], participation
