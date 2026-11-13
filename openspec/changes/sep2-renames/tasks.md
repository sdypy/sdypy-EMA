## 1. Renames with deprecated aliases

- [x] 1.1 `_deprecation.py`: `DeprecatedAttribute` (class-level alias, warns on read, write and delete) and `reject_both`
- [x] 1.2 `Model`: `freq_lower`/`freq_upper` (arguments and attributes), `natural_freq`, `damping_ratio`, `pole_damping_ratio`, `pole_idx`, `frf_reconstructed`; old names as aliases
- [x] 1.3 `Model.get_constants`: `freq_lower`/`freq_upper`; `f_lower`/`f_upper` keyword-only, warn
- [x] 1.4 `LSFD`, `LSFD_proportional`, `LSFD_old`: `lower_idx`/`upper_idx`; `lower_ind`/`upper_ind` keyword-only, warn
- [x] 1.5 `SelectPolesQt.frf_reconstructed`, `H` as alias
- [x] 1.6 Package internals, tests, README, tutorial, showcase (script and notebook) use the new names; changelog entry

## 2. Verification

- [x] 2.1 `tests/test_sep2_renames.py`: every old name warns and equals the new one; new names and positional calls do not warn; old and new together raise. Qt alias in `tests/test_pole_picking_qt.py`
- [x] 2.2 `pytest` on Python 3.13 with `-W error::DeprecationWarning:sdypy` (no internal use of an old name): 87 passed
- [x] 2.3 Hub checkers against this clone: `check_nomenclature.py` and `check_public_api.py` pass
- [ ] 2.4 CI green on the PR

## 3. After merge

- [ ] 3.1 Release to PyPI
- [ ] 3.2 Hub: `tests/test_interop.py` uses `natural_freq`; `public-api` inventory corrected for `pole_xi`
