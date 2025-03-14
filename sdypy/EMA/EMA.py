import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import time
import scipy.linalg
from tqdm import tqdm
from scipy.linalg import toeplitz, companion
from scipy.optimize import least_squares, leastsq

import warnings

try:
    import tkinter as tk
except:
    print('WARNING: tkinter is not istalled or not accessible. Stability chart is not available.')


warnings.filterwarnings('ignore', category=RuntimeWarning)

import pyuff

from .pole_picking import SelectPoles
from . import tools
from . import stabilization
from . import normal_modes

class Model():
    """
    Modal model of frequency response functions.
    """

    def __init__(self,
                 frf=None,
                 freq=None,
                 dt=None,
                 lower=50,
                 upper=10000,
                 pol_order_high=100,
                 pyfrf=False,
                 get_partfactors=False,
                 driving_point=None,
                 frf_type='accelerance'):
        """
        :param frf: Frequency response function matrix
            A ndarray with shape `(n_locations, n_frequency_points)`.
        :type frf: ndarray
        :param freq: Frequency array
        :type freq: array
        :param lower: Lower limit for pole determination [Hz]
        :type lower: int, float
        :param upper: Upper limit for pole determination [Hz]
        :type upper: int, float
        :param pol_order_high: Highest order of the polynomial
        :type pol_order_high: int
        :param pyfrf: add FRFs directly from the pyFRF object
        :type pyfrf: bool
        :param get_partfactors: calculate the participation factors.
        :type get_partfactors: bool
        :param driving point: the index of the driving point (used to scale
            the modal constants to modal shapes)
        :type driving_point: int, defaults to None
        :param frf_type: type of the Frequency Response Function. Must be 'receptance',
            'mobility' or 'accelerance'. The correct FRF type selection is important for
            the LSFD algorithm.
        """
        try:
            self.lower = float(lower)
        except:
            raise Exception('lower must be float or integer')
        if self.lower < 0:
            raise Exception('lower must be more than or equal to zero')

        try:
            self.upper = float(upper)
        except:
            raise Exception('upper must be float or integer')
        if self.upper < self.lower:
            raise Exception('upper must be greater than lower')

        if pyfrf:
            self.frf = 0
        elif not pyfrf and frf is not None and freq is not None:
            try:
                self.frf = np.asarray(frf)
            except:
                raise Exception('cannot convert frf to numpy ndarray')
            if self.frf.ndim == 1:
                self.frf = np.array([self.frf])

            try:
                self.freq = np.asarray(freq)
            except:
                raise Exception('cannot convert freq to a numpy array')
            if self.freq.ndim != 1:
                raise Exception(f'ndim of freq is not equal to 1 ({self.freq.ndim})')

            # Cut off the frequencies above 'upper' argument
            cutoff_ind = np.argmin(np.abs(self.freq - self.upper))
            self.frf = self.frf[:, :cutoff_ind]
            self.freq = self.freq[:cutoff_ind]
        else:
            # The FRFs will be added from UFF
            pass

        if not isinstance(pol_order_high, int):
            raise Exception('pol_order_high must be an integer')
        self.pol_order_high = pol_order_high

        if self.pol_order_high <= 0:
            raise Exception('pol_order_high must be more than zero')

        if not pyfrf and frf is not None and freq is not None:
            self.omega = 2 * np.pi * self.freq
            if dt is None:
                self.sampling_time = 1/(2*self.freq[-1])
            else:
                self.sampling_time = dt

        if not isinstance(driving_point, int) and driving_point is not None:
            raise Exception('"driving_point" must be an integer')
        if driving_point is not None:
            if driving_point > self.frf.shape[0]:
                raise Exception('"driving_point" must be an index of the FRF matrix. "driving_point" too large.')
        self.driving_point = driving_point

        if frf_type not in ['receptance', 'mobility', 'accelerance']:
            raise Exception('"frf_type" must be "receptance", "mobility" or "accelerance".')
        else:
            self.frf_type = frf_type
       
        self.get_participation_factors = get_partfactors

    def add_frf(self, pyfrf_object):
        """
        Add an FRF at the next location.

        This method can be used in relation to pyFRF from Open Modal (https://github.com/openmodal)

        :param pyfrf_object: FRF object from pyFRF
        :type pyfrf_object: object
        """
        freq = pyfrf_object.get_f_axis()

        self.freq = freq
        self.omega = 2 * np.pi * self.freq
        self.sampling_time = 1/(2*self.freq[-1])

        new_frf = np.vstack(pyfrf_object.get_FRF(form='receptance'))

        if isinstance(self.frf, int):
            self.frf = new_frf.T
        else:
            self.frf = np.concatenate((self.frf, new_frf.T), axis=0)

    def read_uff(self, uff_filename):
        """
        Add FRFs from a UFF file.

        :param uff_filename: The file to read.
        :type uff_filename: str
        """
        try:
            uff_file = pyuff.UFF(uff_filename)
        except:
            raise Exception('cannot open uff file')
        
        set_types = uff_file.get_set_types()
        if 58 not in set_types:
            raise Exception('no FRFs in uff file')
        
        ind58 = np.argwhere(set_types==58)[:, 0]
        
        uff_data = uff_file.read_sets()
        uffFRF = np.asarray([uff_data[ind]['data'] for ind in ind58 if uff_data[ind]['func_type'] == 4])
        ufffreq = np.asarray([uff_data[ind]['x'] for ind in ind58 if uff_data[ind]['func_type'] == 4])[0]

        cutoff_ind = np.argmin(np.abs(ufffreq - self.upper))

        self.freq = ufffreq
        self.freq = self.freq[:cutoff_ind]
        self.omega = 2 * np.pi * self.freq
        self.sampling_time = 1/(2*self.freq[-1])

        self.frf = uffFRF[:, :cutoff_ind]

        if uff_data[ind58[0]]['ordinate_spec_data_type'] == 8:
            self.frf_type = 'receptance'
        elif uff_data[ind58[0]]['ordinate_spec_data_type'] == 11:
            self.frf_type = 'mobility'
        elif uff_data[ind58[0]]['ordinate_spec_data_type'] == 12:
            self.frf_type = 'accelerance'
        else:
            print('Warning: frf_type cannot be obtained from the uff file. Assuming "receptance".')
            self.frf_type = 'receptance'

    def get_poles(self, method='lscf', show_progress=True):
        """Compute poles based on polynomial approximation of FRF.

        :param method: The method of poles calculation ("lscf", "rfp" or "rfp segment").
        :param show_progress: Show progress bar

        LSCF
        ====

        Source: https://github.com/openmodal/OpenModal/blob/master/OpenModal/analysis/lscf.py

        The LSCF method is an frequency-domain Linear Least Squares
        estimator optimized for modal parameter estimation. The choice of
        the most important algorithm characteristics is based on the
        results in [1] (Section 5.3.3.) and can be summarized as:

            - Formulation: the normal equations [1]
              (Eq. 5.26: [sum(Tk - Sk.H * Rk^-1 * Sk)]*ThetaA=D*ThetaA = 0)
              are constructed for the common denominator discrete-time
              model in the Z-domain. Consequently, by looping over the
              outputs and inputs, the submatrices Rk, Sk, and Tk are
              formulated through the use of the FFT algorithm as Toeplitz
              structured (n+1) square matrices. Using complex coefficients,
              the FRF data within the frequency band of interest (FRF-zoom)
              is projected in the Z-domain in the interval of [0, 2*pi] in
              order to improve numerical conditioning. (In the case that
              real coefficients are used, the data is projected in the
              interval of [0, pi].) The projecting on an interval that does
              not completely describe the unity circle, say [0, alpha*2*pi]
              where alpha is typically 0.9-0.95. Deliberately over-modeling
              is best applied to cope with discontinuities. This is
              justified by the use of a discrete time model in the Z-domain,
              which is much more robust for a high order of the transfer
              function polynomials.

            - Solver: the normal equations can be solved for the
              denominator coefficients ThetaA by computing the Least-Squares
              (LS) or mixed Total-Least-Squares (TLS) solution. The inverse
              of the square matrix D for the LS solution is computed by
              means of a pseudo inverse operation for reasons of numerical
              stability, while the mixed LS-TLS solution is computed using
              an SVD (Singular Value Decomposition).

        Literature:
            [1] Verboven, P., Frequency-domain System Identification for
                Modal Analysis, Ph. D. thesis, Mechanical Engineering Dept.
                (WERK), Vrije Universiteit Brussel, Brussel, (Belgium),
                May 2002, (http://mech.vub.ac.be/avrg/PhD/thesis_PV_web.pdf)
            [2] Verboven, P., Guillaume, P., Cauberghe, B., Parloo, E. and
                Vanlanduit S., Stabilization Charts and Uncertainty Bounds
                For Frequency-Domain Linear Least Squares Estimators, Vrije
                Universiteit Brussel(VUB), Mechanical Engineering Dept.
                (WERK), Acoustic and Vibration Research Group (AVRG),
                Pleinlaan 2, B-1050 Brussels, Belgium,
                e-mail: Peter.Verboven@vub.ac.be, url:
                (http://sem-proceedings.com/21i/sem.org-IMAC-XXI-Conf-s02p01
                -Stabilization-Charts-Uncertainty-Bounds-Frequency-Domain-
                Linear-Least.pdf)
            [3] P. Guillaume, P. Verboven, S. Vanlanduit, H. Van der
                Auweraer, B. Peeters, A Poly-Reference Implementation of the
                Least-Squares Complex Frequency-Domain Estimator, Vrije
                Universiteit Brussel, LMS International

        RFP
        ===

        The (Global) Rational Fracion Polynomial method is a SIMO MDOF frequency
        domain modal parameter estimation method based on the rational fraction
        formulation of the FRF. In the frequency band of interest each measured
        FRF is approximated with its analytical rational fraction form, then by
        solving for the roots of both the numerator and denominator polynomial,
        the modal parameters are determined. The Least Squares method is used,
        making the system of equations, which is used to calculate the polynomial
        coefficients of the analytical rational fraction function, a linear one.
        To improve numerical stability, orthogonal polynomials are used insted of
        ordinary ones. The former are generated using the Forsythe method.
        
        Literature:
            [1] Richardson, M. H., Formenti, D. L., Parameter Estimation from
                Frequency Response Measurements using Rational Fraction Polynomials,
                Proceedings of the 1st International Modal Analysis Conference
                (IMAC 1), Orlando, Florida, U. S. A., 1982, pp. 167-181
            [2] Richardson, M. H., Formenti, D. L., Global Curve Fitting of Frequency
                Response Measurements using the Rational Fraction Polynomial Method,
                Proceedings of the 3rd International Modal Analysis Conference
                (IMAC 3), Orlando, Florida, U. S. A., 1985, pp. 390-397
            [3] Maia, N. M. M., Silva, J. M. M., Theoretical and Experimental Modal
                Analysis, Wiley, 1997

        """

        if show_progress:
            def tqdm_range(x): return tqdm(x, ncols=100)
        else:
            def tqdm_range(x): return x

        self.all_poles = []
        self.pole_freq = []
        self.pole_xi = []
        self.partfactors = []

        if method == 'lscf':
            self._get_poles_lscf(tqdm_range)

        elif method == 'rfp':
            self._get_poles_rfp(tqdm_range)

        elif method == 'rfp segment':
            self._get_poles_rfp_segment(tqdm_range)

        else:
            raise Exception(
                f'''no method "{method}". Currently only the "lscf" method, 
                the "rfp" method and the "rfp segment" are implemented.''')

    def _get_poles_lscf(self, tqdm_range):
        self.n_bands = 1
        self.len_band = len(self.freq)

        if self.freq[0] != 0:
            df = self.freq[1] - self.freq[0]
            freq_start = np.arange(0, self.freq[0], df)
            self.freq = np.hstack((freq_start, self.freq))
            self.frf = np.column_stack((np.zeros((len(freq_start), self.frf.shape[0])).T, self.frf))

        lower_ind = np.argmin(np.abs(self.freq - self.lower))
        n = self.pol_order_high * 2
        nf = 2 * (self.frf.shape[1] - 1)
        nr = self.frf.shape[0]

        indices_s = np.arange(-n, n+1)
        indices_t = np.arange(n+1)

        sk = -_irfft_adjusted_lower_limit(self.frf, lower_ind, indices_s)
        t = _irfft_adjusted_lower_limit(
            self.frf.real**2 + self.frf.imag**2, lower_ind, indices_t)
        r = -(np.fft.irfft(np.ones(lower_ind), n=nf))[indices_t]*nf
        r[0] += nf

        s = []
        for i in range(nr):
            s.append(toeplitz(sk[i, n:], sk[i, :n+1][::-1]))
        t = toeplitz(np.sum(t[:, :n+1], axis=0))
        r = toeplitz(r)

        # Ascending polynomial order pole computation
        for j in tqdm_range(range(2, n+1, 2)):
            d = 0
            rinv = np.linalg.inv(r[:j+1, :j+1])
            for i in range(nr):
                snew = s[i][:j+1, :j+1]
                d -= np.dot(np.dot(snew[:j+1, :j+1].T,
                                    rinv), snew[:j+1, :j+1])   # sum
            d += t[:j+1, :j+1]

            a0an1 = np.linalg.solve(-d[0:j, 0:j], d[0:j, j])
            # the numerator coefficients
            sr = np.roots(np.append(a0an1, 1)[::-1])

            # Z-domain (for discrete-time domain model)
            poles = -np.log(sr) / self.sampling_time

            if self.get_participation_factors:
                _t = companion(np.append(a0an1, 1)[::-1])
                _v, _w = np.linalg.eig(_t)
                self.partfactors.append(_w[-1, :])

            f_pole, ceta = tools.complex_freq_to_freq_and_damp(poles)

            self.all_poles.append(poles)
            self.pole_freq.append(f_pole)
            self.pole_xi.append(ceta)

    def _get_poles_rfp(self, tqdm_range):
        self.n_bands = 1
        self.len_band = len(self.freq)

        if self.pol_order_high > 20:
            self.pol_order_high = 20
            print('rfp and rfp segment methods currently not optimized for high polynomial order. pol_order_high = 20 will be used.')
        n = self.pol_order_high

        lower_ind = np.argmin(np.abs(self.freq - self.lower))
        self.frf = self.frf[:, lower_ind:]
        self.freq = self.freq[lower_ind:]
        self.omega = 2 * np.pi * self.freq

        # Frequency scaling
        freq_scal_fact = np.max(self.omega)
        self.freq /= freq_scal_fact
        self.omega = 2 * np.pi * self.freq
        frf = self.frf

        # Ascending polynomial order pole computation
        for j in tqdm_range(range(1, n+1)):
            m = 20

            # Izračun polov v skaliranem fr. območju
            sB, B, poles = RFP_poles(self.omega, frf, m, j)

            # Izračun skaliranih lastnih frevenc
            f_pole, ceta = tools.complex_freq_to_freq_and_damp(poles)

            f_pole *= freq_scal_fact
            omega_pole = 2 * np.pi * f_pole

            # Izračun "pravih" polov
            poles = -omega_pole * ceta + 1j * omega_pole * np.sqrt(1 - ceta**2)

            self.all_poles.append(poles)
            self.pole_freq.append(f_pole)
            self.pole_xi.append(ceta)

        self.freq *= freq_scal_fact
        self.omega *= freq_scal_fact

    def _get_poles_rfp_segment(self, tqdm_range):
        if self.pol_order_high > 20:
            self.pol_order_high = 20
            print('rfp and rfp segment methods currently not optimized for high polynomial order. pol_order_high = 20 will be used.')
        n = self.pol_order_high
        self.len_band = 800

        lower_ind = np.argmin(np.abs(self.freq - self.lower))
        self.frf = self.frf[:, lower_ind:]
        self.freq = self.freq[lower_ind:]
        self.omega = 2 * np.pi * self.freq

        n_full_bands = len(self.freq) // self.len_band
        self.n_bands = n_full_bands + 1
        cutoff_ind = n_full_bands * self.len_band
        n_frf = np.shape(self.frf)[0]

        omega_full_bands = np.reshape(self.omega[:cutoff_ind], (n_full_bands, self.len_band))
        frf_full_bands = np.reshape(self.frf[:, :cutoff_ind], (n_frf, n_full_bands, self.len_band))

        omega_part_band = self.omega[cutoff_ind:]
        frf_part_band = self.frf[:, cutoff_ind:]

        if n_full_bands > 0:
            for ii in tqdm_range(range(n_full_bands)): #zanka čez vse polne frekvenčne pasove

                # Frequency scaling
                freq_scal_fact = np.max(omega_full_bands[ii, :])
                omega_band = omega_full_bands[ii, :] / freq_scal_fact

                frf_band = frf_full_bands[:, ii, :]

                # Ascending polynomial order pole computation
                for j in range(1, n+1):
                    m = 20 #2 * j

                    # Izračun polov v skaliranem fr. območju
                    sB, B, poles = RFP_poles(omega_band, frf_band, m, j)

                    # Izračun skaliranih lastnih frevenc
                    f_pole, ceta = tools.complex_freq_to_freq_and_damp(poles)

                    f_pole *= freq_scal_fact
                    omega_pole = 2 * np.pi * f_pole

                    # Izračun "pravih" polov
                    poles = -omega_pole * ceta + 1j * omega_pole * np.sqrt(1 - ceta**2)

                    self.all_poles.append(poles)
                    self.pole_freq.append(f_pole)
                    self.pole_xi.append(ceta)

        # Izračun polov delnega frekvenčnega pasu
        # Frequency scaling
        freq_scal_fact = np.max(omega_part_band)
        omega_part_band_scal = omega_part_band / freq_scal_fact

        # Ascending polynomial order pole computation
        for j in tqdm_range(range(1, n+1)):
            m = 20 #2 * j

            # Izračun polov v skaliranem fr. območju
            sB, B, poles = RFP_poles(omega_part_band_scal, frf_part_band, m, j)

            # Izračun skaliranih lastnih frevenc
            f_pole, ceta = tools.complex_freq_to_freq_and_damp(poles)

            f_pole *= freq_scal_fact
            omega_pole = 2 * np.pi * f_pole

            # Izračun "pravih" polov
            poles = -omega_pole * ceta + 1j * omega_pole * np.sqrt(1 - ceta**2)

            self.all_poles.append(poles)
            self.pole_freq.append(f_pole)
            self.pole_xi.append(ceta)

    def select_poles(self):
        """Select stable poles from stability chart.
       
        Interactive pole selection is possible. Identification of natural
        frequency and damping coefficients is executed on-the-fly,
        as well as computing the reconstructed FRF and modal constants.

        The identification can be done in two ways:
        
        1. Using stability chart

        >>> a.select_poles() # pick poles
        >>> a.nat_freq # natural frequencies
        >>> a.nat_xi # damping coefficients
        >>> a.H # reconstructed FRF matrix
        >>> a.A # modal constants (a.A[:, -2:] are Lower and Upper residual)

        2. Using approximate natural frequencies

        >>> approx_nat_freq = [234, 545]
        >>> a.select_closest_poles(approx_nat_freq)
        >>> a.nat_freq # natural frequencies
        >>> a.nat_xi # damping coefficients
        >>> H, A = a.get_constants(whose_poles='own', FRF_ind='all) # reconstruction
        """
        root = tk.Tk()
        _ = SelectPoles(self, root)
        root.mainloop()
        root.destroy()

    def _select_closest_poles_on_the_fly(self):
        """
        On-the-fly selection of the closest poles.        
        """
        y_ind = int(np.argmin(np.abs(np.arange(0, len(self.pole_freq)
                                               )-self.y_data_pole)))  # Find closest pole order
        # Find cloeset frequency
        sel = np.argmin(np.abs(self.pole_freq[y_ind] - self.x_data_pole))

        self.pole_ind.append([y_ind, sel])
        self.nat_freq.append(self.pole_freq[y_ind][sel])
        self.nat_xi.append(self.pole_xi[y_ind][sel])

    def select_closest_poles(self, approx_nat_freq, f_window=50, fn_temp=0.001, xi_temp=0.05):
        """
        Identification of natural frequency and damping.

        If `approx_nat_freq` is used, the method finds closest poles of the polynomial.

        :param approx_nat_freq: Approximate natural frequency value
        :type approx_nat_freq: list
        :param f_window: width of the optimization frequency window when searching for stable poles
        :type f_window: float, int
        """
        pole_ind = []
        sel_ind = []

        Nmax = self.pol_order_high
        poles = self.all_poles
        fn_temp, xi_temp, test_fn, test_xi = stabilization._stabilization(
            poles, self.n_bands * Nmax, err_fn=fn_temp, err_xi=xi_temp)
        # select the stable poles
        b = np.argwhere((test_fn > 0) & ((test_xi > 0) & (xi_temp > 0)))

        mask = np.zeros_like(fn_temp)
        mask[b[:, 0], b[:, 1]] = 1  # mask the unstable poles
        f_stable = fn_temp * mask
        xi_stable = xi_temp * mask
        f_stable[f_stable != f_stable] = 0
        xi_stable[xi_stable != xi_stable] = 0

        self.f_stable = f_stable
        f_windows = [
            f_window//i for i in range(2, 100) if f_window//i > 3] + [2]
        for i, fr in enumerate(approx_nat_freq):
            # Optimize the approximate frequency
            def fun(x, f_step):
                f = x[0]
                _f_stable = f_stable[(f_stable > (fr - f_step))
                                     & (f_stable < (fr + f_step))]
                return _f_stable.flatten() - f

            for f_w in f_windows:
                sol = least_squares(lambda x: fun(x, f_w), x0=[fr])
                fr = sol.x[0]

            # Select the closest frequency
            f_sel = np.argmin(np.abs(f_stable - fr))
            f_sel = np.unravel_index(f_sel, f_stable.shape)

            # The pole index is known (f_sel[1])
            # The frequency index for this pole order is not known
            # A reconstructed pole is compared with existing poles to
            # get the index of the pole.
            sel = np.argmin(np.abs(self.pole_freq[f_sel[1]] - fr))
            selected_pole = -xi_temp[f_sel]*(2*np.pi*fn_temp[f_sel]) + 1j*(
                2*np.pi*fn_temp[f_sel])*np.sqrt(1-xi_temp[f_sel]**2)
            _sel = np.argmin(np.abs(self.all_poles[f_sel[1]] - selected_pole))

            pole_ind.append([f_sel[1], _sel])
            sel_ind.append([f_sel[1], f_sel[0]])

        sel_ind = np.asarray(sel_ind, dtype=int)
        self.pole_ind = np.asarray(pole_ind, dtype=int)

        self.nat_freq = f_stable[sel_ind[:, 1], sel_ind[:, 0]]
        self.nat_xi = xi_stable[sel_ind[:, 1], sel_ind[:, 0]]

    def get_constants(self, method='lsfd', whose_poles='own', FRF_ind='all',
                      f_lower=None, f_upper=None, complex_mode=True, upper_r=True, lower_r=True, least_squares_type='new'):
        """
        Least square frequency domain 1D (Participation factor excluded)

        :param whose_poles: Whose poles to use, defaults to 'own'
        :type whose_poles: object or string ('own'), optional
        :param FRF_ind: FRF at which location to reconstruct, defaults to 'all'
        :type FRF_ind: int or 'all', optional
        :param f_lower: lower limit on frequency for reconstruction. If None, self.lower is used, defaults to None
        :type f_lower: float, optional
        :param f_upper: upper limit on frequency for reconstruction. If None, self.lower is used, defaults to None
        :type f_upper: float, optional
        :param complex_mode: Return complex modes, defaults to True
        :type complex_mode: bool, optional
        :param upper_r: Compute upper residual, defaults to True
        :type upper_r: bool, optional
        :param lower_r: Compute lower residual, defaults to True
        :type lower_r: bool, optional
        :return: modal constants if ``FRF_ind=None``, otherwise reconstructed FRFs and modal constants
        """
        if method not in ['lsfd', 'lsfd_proportional']:
            raise Exception(
                f'no method "{method}".')

        if whose_poles == 'own':
            whose_poles = self

        pole_ind = np.asarray(whose_poles.pole_ind, dtype=int)
        n_poles = pole_ind.shape[0]
        poles = []
        for i in range(n_poles):
            poles.append(whose_poles.all_poles[pole_ind[i, 0]][pole_ind[i, 1]])
        poles = np.asarray(poles)

        # concatenate frequency and FRF array
        if f_lower == None:
            f_lower = self.lower

        if f_upper == None:
            f_upper = self.upper

        lower_ind = np.argmin(np.abs(self.freq - f_lower))
        upper_ind = np.argmin(np.abs(self.freq - f_upper))

        # Modal constant identification
        if method == 'lsfd':
            self.A, self.H, self.LR, self.UR = LSFD(poles, self.frf, self.freq, lower_r, upper_r, lower_ind, upper_ind, self.frf_type)
        elif method == 'lsfd_proportional':
            self.A, self.H, self.LR, self.UR = LSFD_proportional(poles, self.frf, self.freq, lower_r, upper_r, lower_ind, upper_ind, self.frf_type)

        # Scale with the driving point to obtain the modal shapes
        if self.driving_point is not None:
            scale = self.A[self.driving_point]**(0.5)
            self.phi = self.A/scale

        return self.H, self.A
       
    def FRF_reconstruct(self, FRF_ind):
        """
        Reconstruct FRF based on modal constants.

        .. deprecated:: 0.28.0
            Use the :func:`get_constants` method instead.

        :param FRF_ind: Reconstruct FRF on location with this index, int
        :return: Reconstructed FRF
        """
        warnings.warn("FRF_reconstruct() is deprecated and will be removed in a future version. Use the 'get_constants' method instead.", DeprecationWarning)
        if not hasattr(self, 'omega'):
            self.omega = 2*np.pi*self.freq
        if not hasattr(self, 'poles'):
            self.poles = self.all_poles[self.pole_ind[:,0], self.pole_ind[:,1]]

        FRF_true = np.zeros(len(self.omega), complex)
        for n in range(self.A.shape[1]):
            FRF_true += (self.A[FRF_ind, n] /
                         (1j*self.omega - self.poles[n])) + \
                (np.conjugate(self.A[FRF_ind, n]) /
                 (1j*self.omega - np.conjugate(self.poles[n])))

        FRF_true += -self.LR[FRF_ind] / \
            (self.omega**2) + self.UR[FRF_ind]
        return FRF_true

    def autoMAC(self):
        """
        Auto Modal Assurance Criterion.

        :return: autoMAC matrix
        """
        if not hasattr(self, 'A'):
            raise Exception('Mode shape matrix not defined.')
        return tools.MAC(self.A, self.A)

    def normal_mode(self):
        """Transform the complex mode shape matrix self.A to normal mode shape.
       
        The real mode shape should have the maximum correlation with
        the original complex mode shape. The vector that is most correlated
        with the complex mode, is the real part of the complex mode when it is
        rotated so that the norm of its real part is maximized. [1]
       
        Literature:
            [1] Gladwell, H. Ahmadian GML, and F. Ismail.
                "Extracting Real Modes from Complex Measured Modes."
       
        :return: normal mode shape
        """
        if not hasattr(self, 'A'):
            raise Exception('Mode shape matrix not defined.')
       
        return normal_modes.complex_to_normal_mode(self.A)

    def print_modal_data(self):
        """
        Show modal data in a table-like structure.
        """
        print('   Nat. f.      Damping')
        print(23*'-')
        for i, f in enumerate(self.nat_freq):
            print(f'{i+1}) {f:6.1f}\t{self.nat_xi[i]:5.4f}')
    
    def write_uff(self, filename, name, rsp_dir, ref_dir):
        """
        Export modal data to a UFF file.

        :param filename: The file to write to (.uff)
        :type filename: str
        :param name: Response and reference entity name
        :param rsp_dir: Response direction:
            - 1 - +X Translation       4 - +X Rotation
            - -1 - -X Translation      -4 - -X Rotation
            - 2 - +Y Translation       5 - +Y Rotation
            - -2 - -Y Translation      -5 - -Y Rotation
            - 3 - +Z Translation       6 - +Z Rotation
            - -3 - -Z Translation      -6 - -Z Rotation
        :param ref_dir: Reference direction:
             same options as for parameter rsp_dir  
        """
        print('Warning: This is an experimental feature and will be further developed.')

        uffwrite = pyuff.UFF(filename)

        eigval_data_to_write = {'type':58,
                'func_type': 1, #0 - general; 22 - eigenvalue
                'rsp_node': 1, #
                'rsp_dir': rsp_dir,  #0 - scalar
                'ref_dir': ref_dir,  #0 - scalar
                'ref_node': 1, #
                'data': self.nat_freq,
                'x': np.arange(np.shape(self.nat_freq)[0]) + 1,
                'id1': 'id1', 
                'rsp_ent_name': name,
                'ref_ent_name': name,
                'abscissa_spacing':1,
                'abscissa_spec_data_type':1, #general
                'ordinate_spec_data_type':18, #frequency
                'orddenom_spec_data_type':13 #
                }
        uffwrite.write_sets(eigval_data_to_write, 'add')

        damp_data_to_write = {'type':58,
                'func_type': 1, #0 - general;
                'rsp_node': 1, #
                'rsp_dir': rsp_dir,  #0 - scalar
                'ref_dir': ref_dir,  #0 - scalar
                'ref_node': 1, #
                'data': self.nat_xi,
                'x': np.arange(np.shape(self.nat_xi)[0]) + 1,
                'id1': 'id1', 
                'rsp_ent_name': name,
                'ref_ent_name': name,
                'abscissa_spacing':1,
                'abscissa_spec_data_type':1, #general
                'ordinate_spec_data_type':1, #general
                'orddenom_spec_data_type':13 #
                }
        uffwrite.write_sets(damp_data_to_write, 'add')

        for i in range(np.shape(self.normal_mode())[1]):
            eigvec_data_to_write = {'type':58,
                    'func_type': 1, #0 - general; 23 = eigenvector
                    'rsp_node': 1, #
                    'rsp_dir': rsp_dir,  #0 - scalar
                    'ref_dir': ref_dir,  #0 - scalar
                    'ref_node':1, # 
                    'data': self.normal_mode()[:, i],
                    'x': np.arange(np.shape(self.normal_mode())[0]),
                    'id1': 'id1', 
                    'rsp_ent_name': name,
                    'ref_ent_name': name,
                    'abscissa_spacing':1,
                    'abscissa_spec_data_type':1, #general
                    'ordinate_spec_data_type':1, #general
                    'orddenom_spec_data_type':13 #
                    }
            uffwrite.write_sets(eigvec_data_to_write, 'add')


def _irfft_adjusted_lower_limit(x, low_lim, indices):
    """
    Compute the ifft of real matrix x with adjusted summation limits:
    ::
        y(j) = sum[k=-n-2, ... , -low_lim-1, low_lim, low_lim+1, ... n-2, n-1] x[k] * exp(sqrt(-1)*j*k* 2*pi/n),
        j =-n-2, ..., -low_limit-1, low_limit, low_limit+1, ... n-2, n-1

    :param x: Single-sided real array to Fourier transform.
    :param low_lim: lower limit index of the array x.
    :param indices: list of indices of interest
    :return: Fourier transformed two-sided array x with adjusted lower limit.
             Retruns values.

    Source: https://github.com/openmodal/OpenModal/blob/master/OpenModal/fft_tools.py
    """

    nf = 2 * (x.shape[1] - 1)
    a = (np.fft.irfft(x, n=nf)[:, indices]) * nf
    b = (np.fft.irfft(x[:, :low_lim], n=nf)[:, indices]) * nf

    return a - b


def LSFD_old(poles, frf, freq, lower_r, upper_r, lower_ind, upper_ind, frf_type):
    """Identification of the modal constants using the Least-Squares Frequency Domain method.
   
    :param poles: poles, identified with the LSCF
    :param frf: the measured Frequeny Response Functions
    :param freq: the frequency vector [Hz]
    :param lower_r: bool, include the lower residuals
    :param upper_r: bool, include the upper residuals
    :param lower_ind: the lower frequency limit
    :param upper_ind: the upper frequency limit
    """
    nr_poles = len(poles)
    frf_ = frf[:, lower_ind:upper_ind]
    freq_ = freq[lower_ind:upper_ind]

    TA = TA_construction(poles, freq_, lower_r, upper_r)
    AT = np.linalg.pinv(TA)
    FRF_r_i = np.concatenate([np.real(frf_.T),np.imag(frf_.T)])
    A_LSFD = AT @ FRF_r_i

    A = (A_LSFD[0:2*nr_poles:2, :] + 1.j*A_LSFD[1:2*nr_poles+1:2, :]).T

    # FRF reconstruction
    FRF_rec_ = TA_construction(poles, freq, lower_r, upper_r) @ A_LSFD
    FRF_rec = (FRF_rec_[:len(freq),:] + FRF_rec_[len(freq):,:]*1.j).T


    # Get the upper and lower residuals
    if upper_r and lower_r:
        LR = A_LSFD[-4, :]+1.j*A_LSFD[-3, :]
        UR = A_LSFD[-2, :]+1.j*A_LSFD[-1, :]

    elif lower_r:
        LR = A_LSFD[-2, :]+1.j*A_LSFD[-1, :]
        UR = 0

    elif upper_r:
        LR = 0
        UR = A_LSFD[-2, :]+1.j*A_LSFD[-1, :]

    return A, FRF_rec, LR, UR


def TA_construction(poles, freq, lower_r, upper_r):
    """Construct a matrix for the modal constant identification.

    The real and imaginary parts must be separated.
    """
    nr_poles = len(poles)
    nr_freq = len(freq)

    omega = 2*np.pi*freq

    if omega[0] == 0:
        omega[0] = 1.e-2

    omega_ = omega[:, None]

    if upper_r and lower_r:
        TA = np.zeros([2*nr_freq, 2*nr_poles + 4])
    elif upper_r:
        TA = np.zeros([2*nr_freq, 2*nr_poles + 2])
    elif lower_r:
        TA = np.zeros([2*nr_freq, 2*nr_poles + 2])
    else:
        TA = np.zeros([2*nr_freq, 2*nr_poles])

    # Initialization
    TA = np.zeros([2*nr_freq, 2*nr_poles + 4])

    # Eigenmodes contribution
    TA[:nr_freq, 0:2*nr_poles:2] =    (-np.real(poles))/(np.real(poles)**2+(omega_-np.imag(poles))**2)+\
                                (-np.real(poles))/(np.real(poles)**2+(omega_+np.imag(poles))**2)
    TA[nr_freq:, 0:2*nr_poles:2] =    (-(omega_-np.imag(poles)))/(np.real(poles)**2+(omega_-np.imag(poles))**2)+\
                                (-(omega_+np.imag(poles)))/(np.real(poles)**2+(omega_+np.imag(poles))**2)
    TA[:nr_freq, 1:2*nr_poles+1:2] =  ((omega_-np.imag(poles)))/(np.real(poles)**2+(omega_-np.imag(poles))**2)+\
                                (-(omega_+np.imag(poles)))/(np.real(poles)**2+(omega_+np.imag(poles))**2)
    TA[nr_freq:, 1:2*nr_poles+1:2] =  (-np.real(poles))/(np.real(poles)**2+(omega_-np.imag(poles))**2)+\
                                (np.real(poles))/(np.real(poles)**2+(omega_+np.imag(poles))**2)

    # Lower and upper residuals contribution
    if upper_r and lower_r:
        TA[:nr_freq, -4] = -1/(omega**2)
        TA[nr_freq:, -3] = -1/(omega**2)
        TA[:nr_freq, -2] = np.ones(nr_freq)
        TA[nr_freq:, -1] = np.ones(nr_freq)
    elif lower_r:
        TA[:nr_freq, -2] = -1/(omega**2)
        TA[nr_freq:, -1] = -1/(omega**2)
    elif upper_r:
        TA[:nr_freq, -2] = np.ones(nr_freq)
        TA[nr_freq:, -1] = np.ones(nr_freq)

    return TA


def LSFD_proportional(poles, frf, freq, lower_r, upper_r, lower_ind, upper_ind, frf_type):
    """Identification of the modal constants using the Least-Squares Frequency Domain
    method, where the real-valued modal constants (proportional damping) are assumed.
   
    :param poles: poles, identified with the LSCF
    :param frf: the measured Frequeny Response Functions
    :param freq: the frequency vector [Hz]
    :param lower_r: bool, include the lower residuals
    :param upper_r: bool, include the upper residuals
    :param lower_ind: the lower frequency limit
    :param upper_ind: the upper frequency limit
    """
    frf = frf.T

    omega = 2*np.pi*freq[:, None]

    f, xi = tools.complex_freq_to_freq_and_damp(poles)
    w = 2*np.pi*f

    # flexible terms
    if frf_type == 'receptance':
        p1F = (w**2 - omega**2) / (4 * xi**2 * omega**2 * w**2 + (-omega**2 + w**2)**2) # real part
        p2F = (-2 * xi * omega * w) / (4 * xi**2 * omega**2 * w**2 + (-omega**2 + w**2)**2) # imag part
    elif frf_type == 'mobility':
        p1F = (2 * xi * omega**2 * w) / (4 * xi**2 * omega**2 * w**2 + (-omega**2 + w**2)**2) # real part
        p2F = (-omega**3 + omega * w**2) / (4 * xi**2 * omega**2 * w**2 + (-omega**2 + w**2)**2) # imag part
    elif frf_type == 'accelerance':
        p1F = (omega**4 - omega**2 * w**2) / (4 * xi**2 * omega**2 * w**2 + (-omega**2 + w**2)**2) # real part
        p2F = (2 * xi * omega**3 * w) / (4 * xi**2 * omega**2 * w**2 + (-omega**2 + w**2)**2) # imag part

    if lower_r == True:
        # lower residuals
        if frf_type == 'receptance':
            p1L = np.kron(np.array([1, 0]), -1/omega**2)
            p2L = np.kron(np.array([0, 1]), -1/omega**2)
        elif frf_type == 'mobility':
            p1L = np.kron(np.array([1, 0]), 1/omega) # real and imag part is switched because of frequency-domain derivation
            p2L = np.kron(np.array([0, 1]), -1/omega)
        elif frf_type == 'accelerance':
            p1L = np.kron(np.array([1, 0]), np.ones(freq.shape[0])[:, np.newaxis])
            p2L = np.kron(np.array([0, 1]), np.ones(freq.shape[0])[:, np.newaxis])


    if upper_r == True:
        if frf_type == 'receptance':
            p1U = np.kron(np.array([1, 0]), np.ones(freq.shape[0])[:, np.newaxis])
            p2U = np.kron(np.array([0, 1]), np.ones(freq.shape[0])[:, np.newaxis])
        elif frf_type == 'mobility':
            p1U = np.kron(np.array([1, 0]), -omega)
            p2U = np.kron(np.array([0, 1]), omega)
        elif frf_type == 'accelerance':
            p1U = np.kron(np.array([1, 0]), -omega**2)
            p2U = np.kron(np.array([0, 1]), -omega**2)
       
    if lower_r == True and upper_r == True:
        P = np.block([[p1F, p1L, p1U], [p2F, p2L, p2U]])
    elif lower_r == True and upper_r == False:
        P = np.block([[p1F, p1L], [p2F, p2L]])
    elif lower_r == False and upper_r == True:
        P = np.block([[p1F, p1U], [p2F, p2U]])
    elif lower_r == False and upper_r == False:
        P = np.block([[p1F], [p2F]])
       
    Y = np.block([[frf.real], [frf.imag]])
   
    # Lower and upper frequency limit mask
    mask = np.zeros(Y.shape[0], dtype=bool)
    mask[lower_ind:upper_ind] = True
    mask[frf.shape[0]+lower_ind:frf.shape[0]+upper_ind] = True

    A_ = np.linalg.lstsq(P[mask], Y[mask])[0].T
    # modal constants
    A = A_[:, :w.shape[0]]
   
    # residuals
    if lower_r == True and upper_r == True:
        if frf_type == 'mobility':
            LR = A_[:, -3] + 1j*A_[:, -4]
            UR = A_[:, -1] + 1j*A_[:, -2]
        else:
            LR = A_[:, -4] + 1j*A_[:, -3]
            UR = A_[:, -2] + 1j*A_[:, -1]

    elif lower_r == True and upper_r == False:
        if frf_type == 'mobility':
            LR = A_[:, -1] + 1j*A_[:, -2]
        else:
            LR = A_[:, -2] + 1j*A_[:, -1]
        UR = np.zeros(frf.shape[1], dtype=complex)
    elif lower_r == False and upper_r == True:
        LR = np.zeros(frf.shape[1], dtype=complex)
        if frf_type == 'mobility':
            UR = A_[:, -1] + 1j*A_[:, -2]
        else:
            UR = A_[:, -2] + 1j*A_[:, -1]
    elif lower_r == False and upper_r == False:
        LR = np.zeros(frf.shape[1], dtype=complex)
        UR = np.zeros(frf.shape[1], dtype=complex)
   
    # FRF reconstruction
    FRF_rec_ = np.einsum("fp,op", P, A_)
    FRF_rec_r, FRF_rec_i = np.split(FRF_rec_, 2, axis=0)
    FRF_rec = (FRF_rec_r + 1.j*FRF_rec_i).T
   
    return A, FRF_rec, LR, UR


def LSFD(poles, frf, freq, lower_r, upper_r, lower_ind, upper_ind, frf_type):
    """Identification of the modal constants using the Least-Squares Frequency Domain
    method, where the real-valued modal constants (proportional damping) are assumed.
   
    :param poles: poles, identified with the LSCF or RFP
    :param frf: the measured Frequeny Response Functions
    :param freq: the frequency vector [Hz]
    :param lower_r: bool, include the lower residuals
    :param upper_r: bool, include the upper residuals
    :param lower_ind: the lower frequency limit
    :param upper_ind: the upper frequency limit
    """
    frf = frf.T

    omega = 2*np.pi*freq[:, None]

    f, xi = tools.complex_freq_to_freq_and_damp(poles)
    w = 2*np.pi*f

    sr = poles.real
    si = poles.imag

    # flexible terms
    if frf_type == 'receptance':
        p11 = -(sr) / (sr**2 + (-si + omega)**2) - (sr) / (sr**2 + (si + omega)**2)
        p12 = (-si + omega) / (sr**2 + (-si + omega)**2) - (si + omega) / (sr**2 + (si + omega)**2)
        p21 = (si - omega) / (sr**2 + (-si + omega)**2) - (si + omega) / (sr**2 + (si + omega)**2)
        p22 = -(sr) / (sr**2 + (-si + omega)**2) + (sr) / (sr**2 + (si + omega)**2)
    elif frf_type == 'mobility':
        p11 = (-si * omega + omega**2) / (sr**2 + (-si + omega)**2) + (si * omega + omega**2) / (sr**2 + (si + omega)**2)
        p12 = (sr * omega) / (sr**2 + (-si + omega)**2) - (sr * omega) / (sr**2 + (si + omega)**2)
        p21 = -(sr * omega) / (sr**2 + (-si + omega)**2) - (sr * omega) / (sr**2 + (si + omega)**2)
        p22 = (-si * omega + omega**2) / (sr**2 + (-si + omega)**2) - (si * omega + omega**2) / (sr**2 + (si + omega)**2)
    elif frf_type == 'accelerance':
        p11 = (sr * omega**2) / (sr**2 + (-si + omega)**2) + (sr * omega**2) / (sr**2 + (si + omega)**2)
        p12 = (si * omega**2 - omega**3) / (sr**2 + (-si + omega)**2) + (si * omega**2 + omega**3) / (sr**2 + (si + omega)**2)
        p21 = (-si * omega**2 + omega**3) / (sr**2 + (-si + omega)**2) + (si * omega**2 + omega**3) / (sr**2 + (si + omega)**2)
        p22 = (sr * omega**2) / (sr**2 + (-si + omega)**2) - (sr * omega**2) / (sr**2 + (si + omega)**2)

    if lower_r == True:
        # lower residuals
        if frf_type == 'receptance':
            p1L = np.kron(np.array([1, 0]), -1/omega**2)
            p2L = np.kron(np.array([0, 1]), -1/omega**2)
        elif frf_type == 'mobility':
            p1L = np.kron(np.array([1, 0]), 1/omega) # real and imag part is switched because of frequency-domain derivation
            p2L = np.kron(np.array([0, 1]), -1/omega)
        elif frf_type == 'accelerance':
            p1L = np.kron(np.array([1, 0]), np.ones(freq.shape[0])[:, np.newaxis])
            p2L = np.kron(np.array([0, 1]), np.ones(freq.shape[0])[:, np.newaxis])


    if upper_r == True:
        if frf_type == 'receptance':
            p1U = np.kron(np.array([1, 0]), np.ones(freq.shape[0])[:, np.newaxis])
            p2U = np.kron(np.array([0, 1]), np.ones(freq.shape[0])[:, np.newaxis])
        elif frf_type == 'mobility':
            p1U = np.kron(np.array([1, 0]), -omega)
            p2U = np.kron(np.array([0, 1]), omega)
        elif frf_type == 'accelerance':
            p1U = np.kron(np.array([1, 0]), -omega**2)
            p2U = np.kron(np.array([0, 1]), -omega**2)
       
    if lower_r == True and upper_r == True:
        P = np.block([[p11, p12, p1L, p1U], [p21, p22, p2L, p2U]])
    elif lower_r == True and upper_r == False:
        P = np.block([[p11, p12, p1L], [p21, p22, p2L]])
    elif lower_r == False and upper_r == True:
        P = np.block([[p11, p12, p1U], [p21, p22, p2U]])
    elif lower_r == False and upper_r == False:
        P = np.block([[p11, p12], [p21, p22]])
       
    Y = np.block([[frf.real], [frf.imag]])
   
    # Lower and upper frequency limit mask
    mask = np.zeros(Y.shape[0], dtype=bool)
    mask[lower_ind:upper_ind] = True
    mask[frf.shape[0]+lower_ind:frf.shape[0]+upper_ind] = True

    # weighted least squares in works
    # weights = np.tile(np.sqrt(np.arange(lower_ind, upper_ind)), 2) # with sqrt
    # W = np.diag(weights/np.max(weights))
    # weights = np.tile(np.sum(1/((freq[:, None] - f)**2 + 1e5), axis=1), 2)[mask]
    # W = np.diag(weights/np.max(weights))
    # A_ = np.linalg.lstsq(W@P[mask], W@Y[mask])[0].T


    A_ = np.linalg.lstsq(P[mask], Y[mask], rcond=None)[0].T
    # modal constants
    Ar, Ai = np.split(A_[:, :2*w.shape[0]], 2, axis=1)
    A = Ar + 1j*Ai

   
    # residuals
    if lower_r == True and upper_r == True:
        if frf_type == 'mobility':
            LR = A_[:, -3] + 1j*A_[:, -4]
            UR = A_[:, -1] + 1j*A_[:, -2]
        else:
            LR = A_[:, -4] + 1j*A_[:, -3]
            UR = A_[:, -2] + 1j*A_[:, -1]

    elif lower_r == True and upper_r == False:
        if frf_type == 'mobility':
            LR = A_[:, -1] + 1j*A_[:, -2]
        else:
            LR = A_[:, -2] + 1j*A_[:, -1]
        UR = np.zeros(frf.shape[1], dtype=complex)
    elif lower_r == False and upper_r == True:
        LR = np.zeros(frf.shape[1], dtype=complex)
        if frf_type == 'mobility':
            UR = A_[:, -1] + 1j*A_[:, -2]
        else:
            UR = A_[:, -2] + 1j*A_[:, -1]
    elif lower_r == False and upper_r == False:
        LR = np.zeros(frf.shape[1], dtype=complex)
        UR = np.zeros(frf.shape[1], dtype=complex)
   
    # FRF reconstruction
    FRF_rec_ = np.einsum("fp,op", P, A_)
    FRF_rec_r, FRF_rec_i = np.split(FRF_rec_, 2, axis=0)
    FRF_rec = (FRF_rec_r + 1.j*FRF_rec_i).T
   
    return A, FRF_rec, LR, UR






def generate_forsythe_polynomials(x, pol_order, weight): 
    """
    Generate orthogonal polynomials using the Forsythe method.

    Generate orthogonal polynomials using the Forsythe method. The function
    returns a matrix `matGamma` where the element `gamma_ji` means an
    orthogonal polynomial of order `i` evaluated at the jth element of `x`.

    :param x: the independent value vector
    :param pol_order: the highest order of generated Forsythe polynomilas
    :param weight: the vector of the weighting function at each element of `x`
    """

    len_x = len(x)

    Rfp = np.zeros((len_x, pol_order + 1))
    vfp = np.zeros(pol_order + 1)
    dfp = np.zeros(pol_order + 1)

    # k = 0:
    Rfp[:, 0] = 1
    vfp[0] = 0
    dfp[0] = 2 * np.sum((Rfp[:, 0])**2 * weight)

    # k = 1:
    Rfp[:, 1] = x * Rfp[:, 0]
    vfp[1] = (2 * np.sum(x * Rfp[:, 1] * Rfp[:, 0] * weight)) / dfp[0]
    dfp[1] = 2 * np.sum((Rfp[:, 1])**2 * weight)

    # k >= 2:
    for k in range(2, pol_order + 1):
        Rfp[:, k] = x * Rfp[:, k-1] - vfp[k-1] * Rfp[:, k-2]
        vfp[k] = (2 * np.sum(x * Rfp[:, k] * Rfp[:, k-1] * weight)) / dfp[k-1]
        dfp[k] = 2 * np.sum((Rfp[:, k])**2 * weight)

    Rfp /= (dfp)**(1/2)
    matGamma = (1j)**(np.arange(pol_order + 1)) * Rfp
    
    return matGamma


def RFP_poles(omega, frf, num_pol_order, denom_pol_order):
    """
    Calculate complex poles of the FRFs using the RFP method.

    Calculate the denominator polynomial coefficients and complex poles
    of the approximated FRFs using the RFP method.

    :param omega: the frequency (omega) vector
    :param frf: the measured Frequeny Response Functions
    :param num_pol_order: the numerator polynomial order
    :param denom_pol_order: the denominator polynomial order
    """

    len_omega = len(omega)
    n_frf = np.shape(frf)[0]

    sB = np.zeros((len_omega, denom_pol_order + 1), dtype=complex)
    
    for k in range(denom_pol_order + 1):
        sB[:, k] = (1j * omega)**k

    # Uteži za generiranje ortogonalnih polinomov
    qp = np.ones_like(omega, dtype=int)
    qt = (np.abs(frf))**2

    I = np.identity(denom_pol_order)

    # Generiranje ortogonalnih polinomov števca (enaki za vse FRF-je, odvisni le od frekvenc)
    matPhi = generate_forsythe_polynomials(omega, num_pol_order, qp)
    matP = matPhi

    for j in range(n_frf): #zanka čez vse FRF-je

        # Generiranje ortogonalnih polinomov imenovalca (za vsak FRF)
        matTheta = generate_forsythe_polynomials(omega, denom_pol_order, qt[j])
        matT_ = np.diag(frf[j]) @ matTheta
        matT = matT_[:, :-1]
        vecW = matT_[:, -1]

        X = -2 * np.real(np.conj(matP).T @ matT)
        H = 2 * np.real(np.conj(matP).T @ vecW)

        Tbd = np.linalg.pinv(sB[:, :-1]) @ matTheta[:, :-1]
        R = np.linalg.pinv(sB[:, :-1]) @ (matTheta[:, -1] - sB[:, -1])

        Ug = (I - X.T @ X) @ np.linalg.inv(Tbd)
        Vg = Ug @ R - X.T @ H

        if j == 0:
            Ut = Ug
            Vt = Vg
        else:
            Ut = np.vstack((Ut, Ug))
            Vt = np.hstack((Vt, Vg))

    # Koeficienti polinoma v imenovalcu
    B = np.linalg.inv(Ut.T @ Ut) @ Ut.T @ Vt
    B = np.append(B, 1)

    # Poli racionalne funkcije
    sr = np.roots(B[::-1])

    return sB, B, sr