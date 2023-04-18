Change log
==========

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