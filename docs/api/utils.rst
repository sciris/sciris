===============
Other utilities
===============

.. currentmodule:: sciris

The string ``d = '2022-02-02'`` looks like a date, right? Without googling, do you know how to convert it to an *actual* date object? With Sciris, it's :func:`sc.date(d) <sc_datetime.date>`. Without sciris, it's ``datetime.datetime.strptime(d, '%Y-%m-%d').date()``.

Also, if you need help with anything in Sciris, you can do :func:`sc.help() <sc_settings.help>`. It doesn't use ChatGPT, but will do a full text search through the source code.

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:
   :nosignatures:

   sciris.sc_utils
   sciris.sc_datetime
   sciris.sc_nested
   sciris.sc_settings
   