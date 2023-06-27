More examples
=============

Motorcyle data
^^^^^^^^^^^^^^

This is a classic dataset featuring unknown heteroscedastic noise.

.. figure:: ../../example/motorcycle_data.png
   :scale: 70%
   :align: center
   :alt: Motorcycle dataset

   Motorcycle dataset.

The noise increases near 10 and 30. We model it with our TGP.

.. literalinclude:: ../../example/motorcycle.py
  :language: python
  
The result correctly models the heteroscedastic noise in the distinct regions.

.. figure:: ../../example/motorcycle_tgp.png
   :scale: 70%
   :align: center
   :alt: TGP model

   TGP model of motorcyle dataset. The grey bars show the locations of partitions.

  
Step-functions
^^^^^^^^^^^^^^

The data contains step-functions, which ordinary GPs struggle with.

.. literalinclude:: ../../example/step_functions.py
  :language: python
  
The step is modeled appropriately by the TGP:

.. figure:: ../../example/step_functions_tgp.png
   :scale: 70%
   :align: center
   :alt: TGP model

   TGP model of step functions. The grey bars show the locations of partitions.
