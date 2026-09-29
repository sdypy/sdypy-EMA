Change log
==========

Version 0.31
------------

- `frf_type` renamed to `frf_form` in `Model`, and in the `LSFD`, `LSFD_proportional` and `LSFD_old` functions. `autoMAC` renamed to `auto_mac`. The old names still work but raise a `DeprecationWarning`.
- `Model.add_frf` works with pyFRF 1.5: it now calls pyFRF's `get_FRF` positionally.
- Curated the public API to match the SDyPy public-api contract (SEP 2).
- Documentation moved to the unified SDyPy docs standard.


Version 0.30
------------

- Added a Qt-based stability chart alongside the Tkinter one.
- `EMA` now imports without tkinter installed.
- Input validation and scoped warnings added across `Model`.
- Switched PyPI releases to trusted publishing.


Version 0.29
------------

- Added the **LSCE** method for parameter estimation, alongside LSCF.
- Added participation factor calculation to LSCE, matching LSCF.
- Fixed the LSCE lower frequency bound handling.
- Fixed stability chart markers not showing for stable poles.


Version 0.28
------------

- Improved the Tkinter stability chart: layout and toolbar placement.
- Deprecated `FRF_reconstruct`; use `get_constants` instead.


Version 0.27
------------

- **UFF** support: Import FRFs directly from UFF files. The `pyUFF <https://pypi.org/project/pyuff/>`_ package is used to read the UFF files.
- **Stability chart** upgrade: Show/hide unstable poles to improve the clearity of the chart.
- Documentation update.


Version 0.26
------------

- Include/exclude upper and lower **residuals**.
- **Driving point** implementation (scaling modal constants to modal shapes).
- Implementation of the **LSFD** method that assumes **proportional damping** (modal constants are real-valued).
- **FRF type** implementation (enables the use of accelerance, mobility or receptance).