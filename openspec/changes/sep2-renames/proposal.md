## Why

The sdypy hub's nomenclature checker (`tools/check_nomenclature.py`) reports
17 public names in sdypy-EMA 0.31.0 that diverge from SEP 2, which keeps the
umbrella's `test_nomenclature` gate red for EMA (issue sdypy/sdypy-EMA#25).

Non-trivial: it renames public names. The contract is the hub's
`public-api` requirement *Evidenced divergences carry deprecated aliases to
the canonical names*, which already lists these names; this change has no
spec delta of its own (`skip_specs`) and carries the tasks.

## What Changes

- `Model(lower=, upper=)` and the attributes → `freq_lower`, `freq_upper`;
  `Model.get_constants(f_lower=, f_upper=)` → `freq_lower`, `freq_upper`.
- `Model.nat_freq` → `natural_freq`, `Model.nat_xi` → `damping_ratio`,
  `Model.pole_xi` → `pole_damping_ratio`, `Model.pole_ind` → `pole_idx`.
- `SelectPolesQt.H` and `Model.H` (reconstructed FRFs) → `frf_reconstructed`.
  The checker flags only `SelectPolesQt.H`; `Model.H` holds the same matrix
  and is renamed with it.
- `LSFD`, `LSFD_proportional`, `LSFD_old`: `lower_ind=`, `upper_ind=` →
  `lower_idx`, `upper_idx`.
- Every old name stays as a deprecated alias: reading or writing an old
  attribute, or passing an old keyword, emits `DeprecationWarning` and acts on
  the new name. Positional calls are unchanged. Passing an old and a new
  keyword together raises `TypeError`.
- The package, tests, README, tutorial and showcase use the new names.

`pole_xi` and `nat_xi` are both on `Model`, so they cannot both become
`damping_ratio`: `nat_xi` (selected modes) takes `damping_ratio`, paired with
`natural_freq`; `pole_xi` (every pole of the stability chart) takes
`pole_damping_ratio`, paired with the existing `pole_freq`. The hub's
inventory, which maps both to `damping_ratio`, needs the same correction.

## Impact

- `sdypy/EMA/EMA.py`, `pole_picking.py`, `pole_picking_qt.py`; new private
  module `sdypy/EMA/_deprecation.py`.
- Users see `DeprecationWarning` for old names; nothing breaks before v2.0.
- Hub follow-up: `tests/test_interop.py` moves from `nat_freq` to
  `natural_freq` once this is on PyPI; the `public-api` inventory gets
  `pole_xi` → `pole_damping_ratio`.
