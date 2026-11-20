|pytest| |Docs Status| |DOI|


sdypy-EMA
=========

Experimental and operational modal analysis

This project is successor of the `pyEMA`_ project. pyEMA is no longer developed after version 0.26.

Basic usage
-----------

Import ``EMA`` module:
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from sdypy import EMA


Make an instance of ``Model`` class:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   a = EMA.Model(
       frf_matrix,
       frequency_array,
       freq_lower=50,
       freq_upper=10000,
       pol_order_high=60
       )

Compute poles:
~~~~~~~~~~~~~~

.. code:: python

   a.get_poles()

Determine correct poles:
~~~~~~~~~~~~~~~~~~~~~~~~

The stable poles can be determined in two ways: 

1. Display **stability chart**

.. code:: python
    
    a.select_poles()

The stability chart displayes calculated poles and the user can hand-pick the stable ones.
The chart opens in a Qt window when PySide6 or PyQt6 is installed (``pip install sdypy-EMA[qt]``), otherwise in a Tk window.

2. If the approximate values of natural frequencies are already known, it is not necessary to display the stability chart:

.. code:: python

    approx_nat_freq = [314, 864]     
    a.select_closest_poles(approx_nat_freq)

After the stable poles are selected, the natural frequencies and damping coefficients can now be accessed:

.. code:: python

   a.natural_freq # natrual frequencies
   a.damping_ratio # damping coefficients

Reconstruction:
~~~~~~~~~~~~~~~

There are two types of reconstruction possible: 

1. Reconstruction using **own** poles (the default option):

.. code:: python

    H, A = a.get_constants(whose_poles='own')

where **H** is reconstructed FRF matrix and **A** is a matrix of modal constants.

2. Reconstruction on **c** using poles from **a**:

.. code:: python

    c = EMA.Model(frf_matrix, frequency_array, freq_lower=50, freq_upper=10000, pol_order_high=60)

    H, A = c.get_constants(whose_poles=a)

Poly-reference (MIMO) identification
------------------------------------

FRFs measured with several references (inputs) at the same time are identified
with the ``ModelPoly`` class. The FRF array has the shape
``(n_outputs, n_freq, n_inputs)``:

.. code:: python

    p = EMA.ModelPoly(
        frf_array,              # (n_outputs, n_freq, n_inputs)
        frequency_array,
        freq_lower=10,
        freq_upper=500,
        pol_order_high=40,
        driving_point=[0, 5],   # output index at each reference, optional
    )
    p.get_poles()
    p.select_closest_poles([22.4, 25.2, 62.8, 63.5, 90.7, 90.8])  # or p.select_poles()
    frf_reconstructed, mode_shape = p.get_constants()

The poles come from the poly-reference LSCF method (pLSCF, PolyMAX), each with
its modal participation factors (``p.participation_factor``, shape
``(n_inputs, n_modes)``). This lets ``ModelPoly`` separate closely spaced modes
that ``Model`` merges into one. The mode shapes come from an LSFD fit that uses
the participation factors; with ``driving_point`` given, they are scaled to
unit modal A. FRFs stored in a UFF file are read with
``EMA.ModelPoly.from_uff(filename)``.

Showcases
---------

- ``EMA Showcase.ipynb`` (script version ``EMA_showcase.py``): single-reference
  identification of a measured beam with ``Model``.
- ``ModelPoly Showcase.ipynb`` (script version ``ModelPoly_showcase.py``):
  poly-reference identification of a simulated MIMO test with closely spaced
  modes, compared with ``Model``. The test is simulated with the other ``sdypy``
  packages (lumped model, excitation, FRF estimation), so it needs
  ``pip install sdypy``.

.. |Docs Status| image:: https://readthedocs.org/projects/sdypy-ema/badge/
   :target: https://sdypy-ema.readthedocs.io/
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4016671.svg?
   :target: https://doi.org/10.5281/zenodo.4016671
.. |pytest| image:: https://github.com/sdypy/sdypy-EMA/actions/workflows/python-package.yml/badge.svg
    :target: https://github.com/sdypy/sdypy-EMA/actions


.. _sdypy: https://github.com/sdypy/sdypy

.. _pyEMA: https://github.com/ladisk/pyEMA
