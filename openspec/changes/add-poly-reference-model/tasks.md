# Tasks

## 1. Implementation

- [x] 1.1 `ModelPoly` input handling: 3-D and 2-D FRF arrays, frequency limits, `frf_form`, `driving_point`, `ValueError` on invalid input
- [x] 1.2 pLSCF `get_poles()`: poles, natural frequencies, damping ratios and participation factors per model order
- [x] 1.3 `select_closest_poles()`: stable poles, distinct selection, `ValueError` on a repeated pole
- [x] 1.4 `get_constants()`: LSFD with participation factors, residuals, reconstructed FRF, driving-point scaling
- [x] 1.5 Export `ModelPoly` from `sdypy.EMA`
- [x] 1.6 `select_poles()` on the Qt stability chart through a private single-reference view
- [x] 1.7 `ModelPoly.from_uff()`: FRF array, `outputs`, `inputs`, `frf_form`, `driving_point`
- [x] 1.8 `normal_mode()`
- [x] 1.9 `stabilization._stabilization`: rows sized by the pole count, all previous poles compared; results for `Model` unchanged

## 2. Tests (verifiers for REQUIREMENTS.md)

- [x] 2.1 Synthetic MIMO data helper in `tests/test_tools/` (pyLump, periodic random excitation, noise, multiple-input H1)
- [x] 2.2 Poly-reference FRF array, invalid input → `tests/test_poly_reference.py::test_frf_shapes`, `::test_invalid_input`
- [x] 2.3 Poles and participation factors per order → `::test_poles_per_order`
- [x] 2.4 Pole selection → `::test_select_closest_poles`, `::test_same_pole_twice`
- [x] 2.5 Modal constants and driving-point scaling → `::test_constants_shapes`, `::test_driving_point_scaling`
- [x] 2.6 Noisy MIMO accuracy and close pair → `::test_noisy_mimo_accuracy`, `::test_close_modes_separated`
- [x] 2.7 Exact data → `::test_exact_data`
- [x] 2.8 Single-reference data → `::test_single_reference`
- [x] 2.9 Public import → `::test_public_import`
- [x] 2.10 Interactive pole selection, existing selection → `tests/test_poly_reference_qt.py::test_pick_poles_on_chart`, `::test_existing_selection_is_shown`; chart without Qt → `tests/test_poly_reference.py::test_select_poles_without_qt`
- [x] 2.11 FRFs from a UFF file and its defaults → `::test_from_uff`; missing FRF → `::test_from_uff_missing_pair`
- [x] 2.12 Normal modes → `::test_normal_mode`
- [x] 2.13 Stability classification of many poles per order → `::test_stabilization_many_poles_per_order`

## 3. Documentation

- [x] 3.1 Code documentation entry and a tutorial section for `ModelPoly`
- [x] 3.2 Changelog entry

## 4. Hub and follow-ups

- [x] 4.1 Hub change `add-ema-model-poly` (hub branch `ema-model-poly`): `ModelPoly` in the `sdypy.EMA.__all__` list of `openspec/specs/public-api/spec.md` and in `tools/check_public_api.py`
- [ ] 4.2 Declare the new names (`ModelPoly`, `participation_factor`, `pole_participation_factor`, `n_inputs`, `n_outputs`, `residual_lower`, `residual_upper`, `from_uff`, `outputs`, `inputs`) in the PR for SEP 2
- [x] 4.3 `pyLump.Model.get_response` frequency-axis offset: already reported as ladisk/pyLump#3
