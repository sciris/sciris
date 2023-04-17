"""
Sciris root module

Typically just handles imports, but also sets number of threads for Numpy if SCIRIS_NUM_THREADS is set (see :class:`sc.options <sc_settings.ScirisOptions>`).
"""

# Handle threadcount -- may require Sciris to be imported before Numpy; see https://stackoverflow.com/questions/17053671/how-do-you-stop-numpy-from-multithreading
import os as _os
_threads = _os.getenv('SCIRIS_NUM_THREADS', '')
if _threads: # pragma: no cover
  _os.environ.update(
      OMP_NUM_THREADS        = _threads,
      OPENBLAS_NUM_THREADS   = _threads,
      NUMEXPR_NUM_THREADS    = _threads,
      MKL_NUM_THREADS        = _threads,
      VECLIB_MAXIMUM_THREADS = _threads,
)

# Optionally allow lazy loading
_lazy = _os.getenv('SCIRIS_LAZY', False)

import time as pytime
D = dict(start=pytime.time())

# Otherwise, import everything
if not _lazy:
    from .sc_version    import *
    D['sc_version'] = pytime.time()
    from .sc_utils      import *
    D['sc_utils'] = pytime.time()
    from .sc_printing   import *
    D['sc_printing'] = pytime.time()
    from .sc_nested     import *
    D['sc_nested'] = pytime.time()
    from .sc_odict      import *
    D['sc_odict'] = pytime.time()
    from .sc_settings   import *
    D['sc_settings'] = pytime.time()
    from .sc_datetime   import *
    D['sc_datetime'] = pytime.time()
    from .sc_math       import *
    D['sc_math'] = pytime.time()
    from .sc_dataframe  import *
    D['sc_dataframe'] = pytime.time()
    from .sc_fileio     import *
    D['sc_fileio'] = pytime.time()
    from .sc_versioning import *
    D['sc_versioning'] = pytime.time()
    from .sc_profiling  import *
    D['sc_profiling'] = pytime.time()
    from .sc_parallel   import *
    D['sc_parallel'] = pytime.time()
    from .sc_asd        import *
    D['sc_asd'] = pytime.time()
    from .sc_plotting   import *
    D['sc_plotting'] = pytime.time()
    from .sc_colors     import *
    D['sc_colors'] = pytime.time()

del _os, _lazy, _threads
