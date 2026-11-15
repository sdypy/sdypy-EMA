# Tasks

## 1. Implementation

- [ ] 1.1 `ModelPoly` input handling: 3-D and 2-D FRF arrays, frequency limits, `frf_form`, `driving_point`, `ValueError` on invalid input
- [ ] 1.2 pLSCF `get_poles()`: poles, natural frequencies, damping ratios and participation factors per model order
- [ ] 1.3 `select_closest_poles()`: stable poles, distinct selection, `ValueError` on a repeated pole
- [ ] 1.4 `get_constants()`: LSFD with participation factors, residuals, reconstructed FRF, driving-point scaling
- [ ] 1.5 Export `ModelPoly` from `sdypy.EMA`

## 2. Tests (verifiers for REQUIREMENTS.md)

- [ ] 2.1 Synthetic MIMO data helper in `tests/test_tools/` (pyLump, periodic random excitation, noise, multiple-input H1)
- [ ] 2.2 Poly-reference FRF array, invalid input → `tests/test_poly_reference.py::test_frf_shapes`, `::test_invalid_input`
- [ ] 2.3 Poles and participation factors per order → `::test_poles_per_order`
- [ ] 2.4 Pole selection → `::test_select_closest_poles`, `::test_same_pole_twice`
- [ ] 2.5 Modal constants and driving-point scaling → `::test_constants_shapes`, `::test_driving_point_scaling`
- [ ] 2.6 Noisy MIMO accuracy and close pair → `::test_noisy_mimo_accuracy`, `::test_close_modes_separated`
- [ ] 2.7 Exact data → `::test_exact_data`
- [ ] 2.8 Single-reference data → `::test_single_reference`
- [ ] 2.9 Public import → `::test_public_import`

## 3. Documentation

- [ ] 3.1 Code documentation entry and a tutorial section for `ModelPoly`
- [ ] 3.2 Changelog entry

## 4. Hub and follow-ups

- [ ] 4.1 Hub change: add `ModelPoly` to the `sdypy.EMA.__all__` list in `openspec/specs/public-api/spec.md`
- [ ] 4.2 Declare the new names (`ModelPoly`, `participation_factor`, `pole_participation_factor`, `n_inputs`, `n_outputs`, `residual_lower`, `residual_upper`) in the PR for SEP 2
- [ ] 4.3 Report the `pyLump.Model.get_response` frequency-axis offset to pyLump
