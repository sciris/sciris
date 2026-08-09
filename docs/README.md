# Sciris docs

## Tutorials

Please see the `tutorials` subfolder.

## Everything else

This folder includes the source for building the docs, which are built with [Quarto](https://quarto.org). Users are unlikely to need to do this themselves; instead, view the Sciris docs at https://docs.sciris.org.

To build the docs, follow these steps:

1.  Install [Quarto](https://quarto.org/docs/get-started/), plus the Python dependencies:
    ```
    pip install -r requirements.txt
    ```

2.  Install the Quarto extensions (only needed once):
    ```
    quarto add sciris/quartopydoc
    ```

3.  Build the docs with `./render` (or `quarto render`). To preview them with live reloading, use `./preview` instead. Note that the tutorials are only re-executed if they have changed (`freeze: auto`); to force a rebuild, use `./render --cache-refresh`.

4.  The built documents will be in `./_site`; open `./_site/index.html`.

Other scripts:

- `./check_notebooks.py` runs all the tutorials as scripts, which is a quicker way of checking that they all still work.
- `./clean_outputs.py` removes the temporary files created by running the tutorials.
- `./clean_all` removes all build artifacts, including the rendered site and the cache.
- `./publish` builds and publishes the docs to GitHub Pages (usually done by CI instead).
