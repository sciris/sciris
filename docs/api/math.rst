====================
Math and array tools
====================

.. currentmodule:: sciris

Quiz: how do you return the indices of a vector ``v = np.random.rand(100)`` that are greater than 0.4 but less than 0.6? If you answered ``((v>0.4)*(v<0.6)).nonzero()[0]``, you're right! But with Sciris, you can also just do :func:`sc.findinds(v>0.4, v\<0.6) <sc_math.findinds>`, which is a little easier.

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:
   :nosignatures:

   sciris.sc_math
   sciris.sc_asd