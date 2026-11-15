## Why

`Model` identifies modes from FRFs of a single reference: its FRF array is
`(n_outputs, n_freq)` and every pole estimator fits one common scalar
denominator. Poly-reference (MIMO) tests are the norm for structures with
closely spaced or repeated modes, and a scalar denominator cannot separate such
a pair, however many references are stacked into it (issue #7). The agreed
direction (#7, draft PR #17) is a separate class on a 3-D FRF array, leaving
`Model` unchanged.

Change type: **non-trivial** (new public name `ModelPoly` and new behaviour).

## What Changes

- New class `ModelPoly` in `sdypy.EMA` for poly-reference FRFs of shape
  `(n_outputs, n_freq, n_inputs)`; a 2-D array is a single input.
- Pole estimation with the poly-reference least-squares complex frequency
  method (pLSCF): matrix-polynomial denominator of size
  `n_inputs × n_inputs`, so that every pole comes with its modal participation
  factors. Builds on the single-reference pLSCF of draft PR #17.
- Selection of the physical poles closest to given approximate natural
  frequencies, with their participation factors.
- Modal constants by a poly-reference LSFD that uses the participation factors:
  mode shapes, lower and upper residuals and the reconstructed FRF array. With
  driving points given, mode shapes and participation factors are scaled to
  unit modal A (as the residue `mode_shape * participation_factor` of each
  mode).
- Validation on a synthetic poly-reference model: a lumped mass-spring-damper
  system (pyLump) excited at several DOFs with periodic random signals; measured
  excitation and response carry added noise; FRFs come from a multiple-input H1
  estimate. The model has a pair of closely spaced modes.
- Interactive pole selection on the existing Qt stability chart
  (`select_poles()`), FRFs from a UFF file (`ModelPoly.from_uff`) and real
  mode shapes (`normal_mode()`).
- The stability classification (`stabilization._stabilization`) compares all
  poles of consecutive orders and sizes its matrices by the number of poles,
  so that it handles more poles per order than a single-reference model has.
  Its results for `Model` are unchanged.
- Not in this change: the Tk stability chart and `add_frf` (pyFRF) for
  poly-reference data, frequency weighting.
- `Model` is not changed. No **BREAKING** change.

New public names (declared for SEP 2, see the hub's
`docs/source/dev/nomenclature.rst`): `ModelPoly`, `participation_factor`,
`pole_participation_factor`, `n_inputs`, `n_outputs`, `residual_lower`,
`residual_upper`, `from_uff`, `outputs`, `inputs`. All other attribute names follow the SEP 2 table
(`freq_lower`, `freq_upper`, `natural_freq`, `damping_ratio`,
`pole_damping_ratio`, `pole_idx`, `mode_shape`, `frf_form`,
`frf_reconstructed`).

## Capabilities

### New Capabilities

- `poly-reference-identification`: identification of poles, participation
  factors and mode shapes from poly-reference FRFs (`ModelPoly`).

### Modified Capabilities

None in this repository. Adding `ModelPoly` to `sdypy.EMA.__all__` changes the
hub's `public-api` spec, which lists that `__all__` exactly; that delta is
proposed in the hub (task 4.1).

## Impact

- New module `sdypy/EMA/poly_reference.py`; `sdypy/EMA/__init__.py` exports
  `ModelPoly`. `sdypy/EMA/stabilization.py`: classification of many poles per
  order. The Qt chart (`pole_picking_qt.py`) is not changed: `ModelPoly` hands
  it a private view in the shape of a single-reference model.
- Tests: `tests/test_poly_reference.py`, `tests/test_poly_reference_qt.py`, synthetic data helper in
  `tests/test_tools/`. pyLump is already a test dependency; no new runtime
  dependency.
- Docs: code documentation and tutorial section.
- Hub: `openspec/specs/public-api/spec.md` (`sdypy.EMA.__all__` list).
