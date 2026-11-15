## Context

`Model` (single reference, FRF `(n_outputs, n_freq)`) stays as it is. Issue #7
and draft PR #17 settled on a separate class for poly-reference data; PR #17
contributed a single-reference pLSCF whose numerator elimination with an
orthonormal polynomial basis is kept here.

## Goals / Non-Goals

**Goals:** pLSCF poles with participation factors, pole selection, LSFD with
participation factors, validated on noisy synthetic MIMO data, including a pair
of closely spaced modes.

**Non-Goals:** the Tk stability chart, `add_frf`, frequency weighting; changing
`Model`'s behaviour.

## Decisions

### Right matrix-fraction model in the z-domain

`H_o(ω) = B_o(z) A(z)^-1` for every output row `o` (1 × `n_inputs`), with
`z = exp(jωΔt)`, `Δt = 1/(2 freq_upper)` and real coefficients. `A(z)` is a
matrix polynomial of order `n` with `n_inputs × n_inputs` coefficients and the
highest coefficient fixed to the identity. Only the frequencies inside
`[freq_lower, freq_upper]` enter the fit; no zero-padding down to DC (PR #17 and
`Model`'s LSCF pad with zeros, which adds fake data below `freq_lower`).

### Numerator elimination with an orthonormal basis (from PR #17)

The numerator of every output is eliminated in closed form. The real-stacked
basis `[Re z^r; Im z^r]`, `r = 0..n_max`, is orthonormalized once by a QR
factorization (Householder, instead of PR #17's Gram-Schmidt loop, so numba is
not needed). Because the first `n+1` columns of `Q` span the first `n+1`
monomials, the reduced normal matrix of every order is a slice of matrices
built once at `n_max`, plus a running sum of the per-order correction. Columns
are ordered order-major, input-minor, so slicing gives the lower orders.

Alternative considered: PR #17's two-sided spectrum. Real-stacking the
one-sided band gives the same real normal equations without doubling the
frequency axis.

### Poles and participation factors from a block companion matrix

With `A(z) = Σ α_r z^r`, the companion matrix carries `-α_r^T` in its last
block row; its eigenvalues are the poles `z_k` and the first `n_inputs` rows of
its eigenvectors are the left null vectors of `A(z_k)`, i.e. the participation
factors (`H ≈ ψ_k l_k^T / (s - λ_k)`). Poles in the s-domain are
`λ = ln(z)/Δt`; only `Im λ > 0` is kept.

### LSFD with fixed poles and participation factors

For every output, one real linear least-squares problem over all inputs and
frequencies gives the complex mode-shape row (two real unknowns per mode) and
real lower (`-LR/ω²`) and upper (`UR`) residuals per input. FRFs given as
mobility or accelerance are converted to receptance first.

### Scaling

Participation factors are normalized to unit maximum magnitude per mode. With
`driving_point` set, each mode is rescaled by a complex `c` with
`c² = ψ_dpᴴ l / ψ_dpᴴ ψ_dp` (least squares of `c² ψ_dp = l`), which gives
unit modal A scaling for a reciprocal system and leaves the residues unchanged.

### Pole selection

Poles are taken from the highest model order (or a given order). A pole is
stable when the previous order has a pole within `freq_tol` relative frequency
and `damping_tol` relative damping; for each approximate frequency the closest
stable pole with positive damping is chosen.

### Test data

A lumped chain of two equal 3-DOF sub-systems joined by a weak spring, fixed at
both ends (pyLump `boundaries="both"`), excited at DOFs 0 and 5. Each of
`n_blocks` blocks is an independent periodic random realization; the response
is the steady state `irfft(H(f) X(f))` with `H` from
`pyLump.Model.get_FRF_matrix` at the FFT frequencies. Noise is added to the
measured time signals; FRFs come from a multiple-input H1
(`H = G_yx G_xx^-1`). `pyLump.Model.get_response` is not used: its frequency
axis starts at `df` instead of 0 (`np.arange(1, N//2 + 2) * fs/N`), so it
evaluates the FRF one bin too high (ladisk/pyLump#3).

### Stability chart

The Qt chart of `Model` is reused unchanged: `ModelPoly.select_poles()` hands
it a private view that looks like a single-reference model (poles per order,
FRFs flattened to `(n_outputs * n_inputs, n_freq)`, `get_constants` returning
the flattened reconstruction) and provides both the old and the SEP 2
attribute spellings, so it keeps working when the chart moves to the new
names. The Tk chart depends on more `Model` internals and is not offered.

`stabilization._stabilization` had `2 * n_orders` rows and compared a pole only
with the first `2 n` poles of the previous order. A poly-reference order `n`
has up to `n * n_inputs / 2` poles with positive imaginary part, which exceeds
both for `n_inputs > 4`. The rows are now sized by the largest pole count and
every pole of the previous order is compared; empty rows hold `1e-10` and never
match, so the matrices for `Model` are identical (checked for lscf, lsce, rfp
and rfp segment on `data/acc_data.npy`).

### Cross-check with other implementations

On the synthetic data set (order 30, 60 poles in every package) `ModelPoly`
was compared with pyFBS 1.0.7 (`ModalID.p_lscf`, `p_lsfd`, `normalize`) and
sdynpy 0.24.2 (`PolyPy`). On exact FRFs the three agree to round-off: poles
within 1e-7 relative, participation-factor MAC 1.000000, residues and
A-normalized mode shapes within 1e-6. With 2 % and 5 % noise all three
separate the close pair; `ModelPoly` has the smallest frequency and damping
errors, and its residue error for the close pair is within about 1.2 times
the others' over 10 seeds. A deliberately wrong companion matrix (without the
transpose) leaves the poles unchanged but drops the participation-factor MAC
to 0.12, so the left-null-vector convention is checked by the tests. The
comparison scripts are not part of the test suite (both packages are heavy
optional dependencies).

## Risks / Trade-offs

- Memory grows as `((n_max+1) n_inputs)^2` per model; adequate for typical
  orders (≤ 100) and references (≤ 4).
- Unweighted least squares: outputs with large FRF magnitude dominate; FRFs are
  scaled by their overall maximum only.
