"""
Sciris root module

Typically just handles imports, but also sets number of threads for Numpy if SCIRIS_NUM_THREADS is set (see :class:`sc.options <sciris.sc_settings.ScirisOptions>`).
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
    from .sc_profiling  import *
    from .sc_parallel   import *
    from .sc_asd        import *
    from .sc_plotting   import *
    from .sc_colors     import *

del _os, _lazy, _threads
