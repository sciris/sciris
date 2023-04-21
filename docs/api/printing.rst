=====================
Printing and plotting
=====================

.. currentmodule:: sciris

Do you ever do ``print(obj)`` and get some unhelpful result like ``<object at 0x7f4d0b1ea190>``? :func:`sc.pr(obj) <sc_printing.pr>` will tell you exactly what's in the object, including attributes (including their values), properties, and methods.

Have you ever wanted to plot something in 3D but given up because it seems like it would take too long? With Sciris, it will work out of the box: try :func:`sc.surf3d(np.random.randn(10,10)) <sc_plotting.surf3d>`. We'll wait.

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:
   :nosignatures:

   sciris.sc_printing
   sciris.sc_plotting
   sciris.sc_colors