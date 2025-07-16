import pyLump as lumped
import numpy as np
import numpy.typing as npt

def generate_synth_frf(n_dof: int = 5, M: npt.ArrayLike = None, K: npt.ArrayLike = None, C: npt.ArrayLike = None, verbose: bool = False) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, lumped.Model]:
    """
    Generate a synthetic Frequency Response Function (FRF) for a lumped mass-spring-damper system.
    If no matrices are provided, random values will be generated within specified ranges.
    """

    np.random.seed(6020) # For reproducibility

    for mat in [M, K, C]:
        if mat is not None:
            if mat.shape != (n_dof, n_dof) and mat.shape != (n_dof,):
                raise ValueError(f"Input matrix must be of shape ({n_dof}, {n_dof}) or ({n_dof},) - got {mat.shape}.")

    masses = np.random.uniform(0.1, 1, n_dof) if M is None else M
    stiffnesses = np.random.uniform(500e3, 1e6, n_dof) if K is None else K
    damping = np.random.uniform(10, 100, n_dof) if C is None else C

    lumped_model = lumped.Model(n_dof=n_dof, mass=masses, stiffness=stiffnesses, damping=damping, boundaries="left")

    if verbose:
        print(f"M = \n {lumped_model.get_mass_matrix()}\n")
        print(f"K = \n {lumped_model.get_stiffness_matrix()}\n")
        print(f"C = \n {lumped_model.get_damping_matrix()}\n")

    f_nat = lumped_model.get_eig_freq()
    if verbose:
        print(f"f_nat = {f_nat} Hz")

    zeta = lumped_model.get_damping_ratios()

    if verbose:
        print(f"zeta = {zeta}")

    f_min = np.round(f_nat[0] * 0.25 / 10, 0) * 10    # round to nearest 10 Hz
    f_max = np.round(f_nat[-1] * 5.0 / 100, 0) * 100 # round to nearest 100 Hz

    if verbose:
        print(f"f_min = {f_min} Hz")
        print(f"f_max = {f_max} Hz")

    freq = np.linspace(f_min, f_max, 5000)
    frf_matrix = lumped_model.get_FRF_matrix(freq=freq, frf_method="f")
    FRF = frf_matrix[0, :, :]

    return freq, FRF, f_nat, zeta, lumped_model


if __name__ == "__main__":
    freq, FRF, f_nat, zeta, _ = generate_synth_frf(verbose=True)