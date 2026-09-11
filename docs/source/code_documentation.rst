Code documentation
==================

.. _code-doc-Model:

The ``Model`` class
-------------------

.. autoclass:: sdypy.EMA.Model
    :members:


Stable pole selection
---------------------

The ``SelectPoles`` class is called from the ``Model`` class by calling ``select_poles()`` method:

.. code:: python

    # Initiating the ``Model`` class
    m = Model(...)
    m.get_poles()

    # Selecting the poles
    m.select_poles()

In this case, the ``Model`` argument is passed automatically.

When PySide6 or PyQt6 is installed, ``select_poles()`` opens the Qt chart
(``sdypy.EMA.pole_picking_qt.SelectPolesQt``) instead of the Tk chart documented below.
Pass ``gui='qt'`` or ``gui='tk'`` to choose explicitly.

.. autoclass:: sdypy.EMA.pole_picking.SelectPoles
    :members:


Tools
-----

.. automodule:: sdypy.EMA.tools
    :members:


Normal modes
------------

.. autofunction:: sdypy.EMA.normal_modes.complex_to_normal_mode

.. autofunction:: sdypy.EMA.normal_modes._large_normal_mode_approx
