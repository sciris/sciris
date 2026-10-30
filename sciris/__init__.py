"""
Sciris root module

Typically just handles imports, but also sets number of threads for Numpy if SCIRIS_NUM_THREADS is set (see `sc.options`).
"""

# Handle threadcount -- the environment variables need to be set before Numpy is imported; see also sc.limit_threads()
import os as _os
_threadvars = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'] # Used by sc.limit_threads()
_threads = _os.getenv('SCIRIS_NUM_THREADS', '')
if _threads: # pragma: no cover
    _os.environ.update({key:_threads for key in _threadvars})

# Optionally allow lazy loading
_lazy = _os.getenv('SCIRIS_LAZY', False)

# Otherwise, import everything
if not _lazy:
    from .sc_version    import *
    from .sc_utils      import *
    from .sc_printing   import *
    from .sc_nested     import *
    from .sc_odict      import *
    from .sc_settings   import *
    from .sc_datetime   import *
    from .sc_math       import *
    from .sc_dataframe  import *
    from .sc_fileio     import *
    from .sc_versioning import *
    from .sc_parallel   import *
    from .sc_profiling  import *
    from .sc_asd        import *
    from .sc_plotting   import *
    from .sc_colors     import *

    # Also limit any BLAS libraries already loaded, in case Numpy was imported before Sciris
    if _threads: # pragma: no cover
        limit_threads(int(_threads))

del _os, _lazy, _threads
