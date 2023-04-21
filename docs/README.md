# Sciris docs

## Tutorials

Please see the `tutorials` subfolder.

## Everything else

This folder includes source code for building the docs. Users are unlikely to need to do this themselves. Instead, view the Sciris docs at http://docs.sciris.org.

To build the docs, follow these steps:

1.  Make sure dependencies are installed::
    ```
    pip install -r requirements.txt
    ```

2.  Make the documents; there are many build options. In most cases, running `./build_docs` (to rerun the tutorials; takes a few minutes) is best. Alternatively, you can call `make html` directly.

3.  The built documents will be in `./_build/html`.