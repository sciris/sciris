=============
Containers
=============

.. currentmodule:: sciris

Sciris includes several container classes for making it easier to work with data:

- Dictionaries are great, right? :class:`sc.odict() <sc_odict.odict>` is a drop-in replacement for a dictionary that has lots of extra features (such as retrieving items by index).
- Pandas DataFrames are great, right? :class:`sc.dataframe() <sc_dataframe.dataframe>` is a drop-in replacement for a DataFrame that has some additional features for ease of use (such as being able to concatenate in place).

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:
   :nosignatures:

    sciris.sc_odict
    sciris.sc_dataframe