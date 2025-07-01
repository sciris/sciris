Welcome to Sciris
=================

.. image:: https://badgen.net/pypi/v/sciris/?color=blue
 :target: https://pypi.org/project/sciris

.. image:: https://static.pepy.tech/personalized-badge/sciris?period=total&units=international_system&left_color=grey&right_color=yellow&left_text=Downloads
 :target: https://pepy.tech/project/sciris

.. image:: https://img.shields.io/pypi/l/sciris.svg
 :target: https://github.com/sciris/sciris/blob/main/LICENSE

.. image:: https://github.com/sciris/sciris/actions/workflows/test_sciris.yaml/badge.svg
 :target: https://github.com/sciris/sciris/actions/workflows/test_sciris.yaml?query=workflow


What is Sciris?
---------------

Sciris is a library of tools that can help make writing scientific Python code easier and more pleasant. Built on top of `NumPy <https://numpy.org/>`_ and `Matplotlib <https://matplotlib.org/>`_, Sciris provides functions covering a wide range of common math, file I/O, and plotting operations. This means you can get more done with less code, so you can spend less time looking up answers on Stack Overflow or copy-pasting dubious solutions from ChatGPT. It was originally written to help epidemiologists and neuroscientists focus on doing science, rather than on writing code, but Sciris is applicable across scientific domains (and some nonscientific ones too).

For more information, see the full `documentation <https://docs.sciris.org/en/latest/overview.html>`_, the `paper <http://paper.sciris.org>`_, or `GitHub <https://github.com/sciris/sciris>`_.

If you have questions, feature suggestions, or would like some help getting started, please reach out to us at info@sciris.org or `open an issue <https://github.com/sciris/sciris/issues/new/choose>`_.


Installation
------------

Using pip: ``pip install sciris``

Using conda: ``conda install -c conda-forge sciris``

Using uv: ``uv add sciris``

*Requires Python >= 3.7*.


Tests
-----

Sciris comes with an automated test suite covering all functions. You almost certainly don't need to run these, but if you want to, go to the ``tests`` folder and run ``pytest``. See the readme in that folder for more information.