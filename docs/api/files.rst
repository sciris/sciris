=====================
Files and versioning
=====================

.. currentmodule:: sciris

Saving and loading data can be a pain. Sciris tries to make it easier with the should-just-work functions :func:`sc.save() <sc_fileio.save>` and :func:`sc.load() <sc_fileio.load>`.† But Sciris also makes it easier if you have a particular format in mind, such as :func:`sc.savejson() <sc_fileio.savejson>`, or if you want to store metadata along with your results to improve reproducibility (:func:`sc.savewithmetadata() <sc_versioning.savewithmetadata>`).

† Never load a data file you don't trust -- Sciris won't save you if you `get yourself into a pickle <https://docs.python.org/3/library/pickle.html>`_.

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:
   :nosignatures:

   sciris.sc_fileio
   sciris.sc_versioning