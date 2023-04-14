=============================
Parallelization and profiling
=============================

.. currentmodule:: sciris

Scientific computing workflows are often `embarrassingly parallel <https://en.wikipedia.org/wiki/Embarrassingly_parallel>`_, and yet it can be hard to do in practice. With Sciris, you can do :func:`sc.parallelize(my_func, 10) <sc_parallel.parallelize>` to run your function 10 times.

It's also often hard to know where your code is being slow. While there are great tools like `Austin <https://austin-python.readthedocs.io/en/latest/>`_ for deep dives into performance, you can use :func:`sc.profile(my_func) <sc_profiling.profile>` for a quick glance – which is often all you need.

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:
   :nosignatures:

   sciris.sc_parallel
   sciris.sc_profiling