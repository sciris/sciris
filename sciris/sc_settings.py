'''
Options for configuring Sciris and Matplotlib.

All options should be set directly or using set(), e.g.::

    sc.options(sep='.')

To reset default options, use::

    sc.options.default()

New in version 1.3.0.
'''

import os
import re
import inspect
import collections as co
import copy as cp
import pylab as pl


__all__ = ['dictobj', 'options', 'help']

class dictobj(dict):
    '''
    Lightweight class to create an object that can also act like a dictionary.

    **Example**::

        obj = sc.dictobj()
        obj.a = 5
        obj['b'] = 10
        print(obj.items())

    For a more powerful alternative, see ``sc.objdict()``.

    (Note: ``sc.dictobj()`` is defined in ``sc_settings.py`` rather than ``sc_odict.py``
    since it's used for the options object, which needs to be the first module loaded.)

    | New in version 1.3.0.
    | New in version 1.3.1: inherit from dict
    '''

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            self.__dict__[k] = v
        return

    def __repr__(self):
        output = 'dictobj(' + self.__dict__.__repr__() + ')'
        return output

    def fromkeys(self, *args, **kwargs):
        return dictobj(self.__dict__.fromkeys(*args, **kwargs))

    # Copy default dictionary methods
    def __getitem__( self, *args, **kwargs): return self.__dict__.__getitem__( *args, **kwargs)
    def __setitem__( self, *args, **kwargs): return self.__dict__.__setitem__( *args, **kwargs)
    def __contains__(self, *args, **kwargs): return self.__dict__.__contains__(*args, **kwargs)
    def __len__(     self, *args, **kwargs): return self.__dict__.__len__(     *args, **kwargs)
    def clear(       self, *args, **kwargs): return self.__dict__.clear(       *args, **kwargs)
    def copy(        self, *args, **kwargs): return self.__dict__.copy(        *args, **kwargs)
    def get(         self, *args, **kwargs): return self.__dict__.get(         *args, **kwargs)
    def items(       self, *args, **kwargs): return self.__dict__.items(       *args, **kwargs)
    def keys(        self, *args, **kwargs): return self.__dict__.keys(        *args, **kwargs)
    def pop(         self, *args, **kwargs): return self.__dict__.pop(         *args, **kwargs)
    def popitem(     self, *args, **kwargs): return self.__dict__.popitem(     *args, **kwargs)
    def setdefault(  self, *args, **kwargs): return self.__dict__.setdefault(  *args, **kwargs)
    def update(      self, *args, **kwargs): return self.__dict__.update(      *args, **kwargs)
    def values(      self, *args, **kwargs): return self.__dict__.values(      *args, **kwargs)


class Options(dictobj):
    ''' Small derived class for the options itself '''
    def __call__(self, *args, **kwargs):
        return self.set(*args, **kwargs)

    def __repr__(self):
        output = 'Sciris options:\n'
        for k,v in self.items():
            if k not in ['set', 'default', 'help']:
                output += f'  {k:>8s}: {repr(v)}\n'
        return output









'''
Define options for Covasim, mostly plotting and Numba options. All options should
be set using set() or directly, e.g.::

    cv.options(font_size=18)

To reset default options, use::

    cv.options('default')

Note: "options" is used to refer to the choices available (e.g., DPI), while "settings"
is used to refer to the choices made (e.g., DPI=150).
'''

import os
import pylab as pl
import sciris as sc
import matplotlib.font_manager as fm

# Only the class instance is public
__all__ = ['options']


#%% General settings


# Define simple plotting options -- similar to Matplotlib default
rc_simple = {
    'axes.axisbelow':    True, # So grids show up behind
    'axes.spines.right': False,
    'axes.spines.top':   False,
    'figure.facecolor':  'white',
    'font.family':       'sans-serif', # Replaced with Mulish in load_fonts() if import succeeds
    'legend.frameon':    False,
}

# Define default plotting options -- based on Seaborn
rc_covasim = sc.mergedicts(rc_simple, {
    'axes.facecolor': '#f2f2ff',
    'axes.grid':      True,
    'grid.color':     'white',
    'grid.linewidth': 1,
})


#%% Define the options class

class Options(sc.objdict):
    '''
    Set options for Covasim.

    Use ``cv.options.set('defaults')`` to reset all values to default, or ``cv.options.set(dpi='default')``
    to reset one parameter to default. See ``cv.options.help(detailed=True)`` for
    more information.

    Options can also be saved and loaded using ``cv.options.save()`` and ``cv.options.load()``.
    See ``cv.options.context()`` and ``cv.options.with_style()`` to set options
    temporarily.

    Common options are (see also ``cv.options.help(detailed=True)``):

        - verbose:        default verbosity for simulations to use
        - style:          the plotting style to use
        - dpi:            the overall DPI (i.e. size) of the figures
        - font:           the font family/face used for the plots
        - fontsize:       the font size used for the plots
        - interactive:    convenience method to set show, close, and backend
        - jupyter:        defaults for Jupyter (change backend and figure close/return)
        - show:           whether to show figures
        - close:          whether to close the figures
        - backend:        which Matplotlib backend to use
        - warnings:       how to handle warnings (e.g. print, raise as errors, ignore)

    **Examples**::

        cv.options(dpi=150) # Larger size
        cv.options(style='simple', font='Rosario') # Change to the "simple" Covasim style with a custom font
        cv.options.set(fontsize=18, show=False, backend='agg', precision=64) # Multiple changes
        cv.options(interactive=False) # Turn off interactive plots
        cv.options(jupyter=True) # Defaults for Jupyter
        cv.options('defaults') # Reset to default options

    | New in version 3.1.1: Jupyter defaults
    | New in version 3.1.2: Updated plotting styles; refactored options as a class
    '''

    def __init__(self):
        super().__init__()
        optdesc, options = self.get_orig_options() # Get the options
        self.update(options) # Update this object with them
        self.setattribute('optdesc', optdesc) # Set the description as an attribute, not a dict entry
        self.setattribute('orig_options', sc.dcp(options)) # Copy the default options
        return


    def __call__(self, *args, **kwargs):
        '''Allow ``cv.options(dpi=150)`` instead of ``cv.options.set(dpi=150)`` '''
        return self.set(*args, **kwargs)


    def to_dict(self):
        ''' Pull out only the settings from the options object '''
        return {k:v for k,v in self.items()}


    def __repr__(self):
        ''' Brief representation '''
        output = sc.objectid(self)
        output += 'Covasim options (see also cv.options.disp()):\n'
        output += sc.pp(self.to_dict(), output=True)
        return output


    def __enter__(self):
        ''' Allow to be used in a with block '''
        return self


    def __exit__(self, *args, **kwargs):
        ''' Allow to be used in a with block '''
        try:
            reset = {}
            for k,v in self.on_entry.items():
                if self[k] != v: # Only reset settings that have changed
                    reset[k] = v
            self.set(**reset)
            self.delattribute('on_entry')
        except AttributeError as E:
            errormsg = 'Please use cv.options.context() if using a with block'
            raise AttributeError(errormsg) from E
        return


    def disp(self):
        ''' Detailed representation '''
        output = 'Covasim options (see also cv.options.help()):\n'
        keylen = 14 # Maximum key length  -- "numba_parallel"
        for k,v in self.items():
            keystr = sc.colorize(f'  {k:>{keylen}s}: ', fg='cyan', output=True)
            reprstr = sc.pp(v, output=True)
            reprstr = sc.indent(n=keylen+4, text=reprstr, width=None)
            output += f'{keystr}{reprstr}'
        print(output)
        return


    @staticmethod
    def get_orig_options():
        '''
        Set the default options for Covasim -- not to be called by the user, use
        ``cv.options.set('defaults')`` instead.
        '''

        # Options acts like a class, but is actually an objdict for simplicity
        optdesc = sc.objdict() # Help for the options
        options = sc.objdict() # The options

        optdesc.verbose = 'Set default level of verbosity (i.e. logging detail): e.g., 0.1 is an update every 10 simulated days'
        options.verbose = float(os.getenv('COVASIM_VERBOSE', 0.1))

        optdesc.style = 'Set the default plotting style -- options are "covasim" and "simple" plus those in pl.style.available; see also options.rc'
        options.style = os.getenv('COVASIM_STYLE', 'covasim')

        optdesc.dpi = 'Set the default DPI -- the larger this is, the larger the figures will be'
        options.dpi = int(os.getenv('COVASIM_DPI', pl.rcParams['figure.dpi']))

        optdesc.font = 'Set the default font family (e.g., sans-serif or Arial)'
        options.font = os.getenv('COVASIM_FONT', pl.rcParams['font.family'])

        optdesc.fontsize = 'Set the default font size'
        options.fontsize = int(os.getenv('COVASIM_FONT_SIZE', pl.rcParams['font.size']))

        optdesc.interactive = 'Convenience method to set figure backend, showing, and closing behavior'
        options.interactive = os.getenv('COVASIM_INTERACTIVE', True)

        optdesc.jupyter = 'Convenience method to set common settings for Jupyter notebooks: set to "retina" or "widget" (default) to set backend'
        options.jupyter = os.getenv('COVASIM_JUPYTER', False)

        optdesc.show = 'Set whether or not to show figures (i.e. call pl.show() automatically)'
        options.show = int(os.getenv('COVASIM_SHOW', True))

        optdesc.close = 'Set whether or not to close figures (i.e. call pl.close() automatically)'
        options.close = int(os.getenv('COVASIM_CLOSE', False))

        optdesc.returnfig = 'Set whether or not to return figures from plotting functions'
        options.returnfig = int(os.getenv('COVASIM_RETURNFIG', True))

        optdesc.backend = 'Set the Matplotlib backend (use "agg" for non-interactive)'
        options.backend = os.getenv('COVASIM_BACKEND', pl.get_backend())

        optdesc.rc = 'Matplotlib rc (run control) style parameters used during plotting -- usually set automatically by "style" option'
        options.rc = sc.dcp(rc_covasim)

        optdesc.warnings = 'How warnings are handled: options are "warn" (default), "print", and "error"'
        options.warnings = str(os.getenv('COVASIM_WARNINGS', 'warn'))

        optdesc.sep = 'Set thousands seperator for text output'
        options.sep = str(os.getenv('COVASIM_SEP', ','))

        optdesc.precision = 'Set arithmetic precision for Numba -- 32-bit by default for efficiency'
        options.precision = int(os.getenv('COVASIM_PRECISION', 32))

        optdesc.numba_parallel = 'Set Numba multithreading -- none, safe, full; full multithreading is ~20% faster, but results become nondeterministic'
        options.numba_parallel = str(os.getenv('COVASIM_NUMBA_PARALLEL', 'none'))

        optdesc.numba_cache = 'Set Numba caching -- saves on compilation time; disabling is not recommended'
        options.numba_cache = bool(int(os.getenv('COVASIM_NUMBA_CACHE', 1)))

        return optdesc, options


    def set(self, key=None, value=None, **kwargs):
        '''
        Actually change the style. See ``cv.options.help()`` for more information.

        Args:
            key    (str):    the parameter to modify, or 'defaults' to reset everything to default values
            value  (varies): the value to specify; use None or 'default' to reset to default
            kwargs (dict):   if supplied, set multiple key-value pairs

        **Example**::

            cv.options.set(dpi=50) # Equivalent to cv.options(dpi=50)
        '''

        # Reset to defaults
        if key in ['default', 'defaults']:
            kwargs = self.orig_options # Reset everything to default

        # Handle other keys
        elif key is not None:
            kwargs = sc.mergedicts(kwargs, {key:value})

        # Handle Jupyter
        if 'jupyter' in kwargs.keys() and kwargs['jupyter']:
            jupyter = kwargs['jupyter']
            kwargs['returnfig'] = False # We almost never want to return figs from Jupyter, since then they appear twice
            try: # This makes plots much nicer, but isn't available on all systems
                if not os.environ.get('SPHINX_BUILD'): # Custom check implemented in conf.py to skip this if we're inside Sphinx
                    try: # First try interactive
                        assert jupyter not in ['default', 'retina'] # Hack to intentionally go to the other part of the loop
                        from IPython import get_ipython
                        magic = get_ipython().magic
                        magic('%matplotlib widget')
                    except: # Then try retina
                        assert jupyter != 'default'
                        import matplotlib_inline
                        matplotlib_inline.backend_inline.set_matplotlib_formats('retina')
            except:
                pass

        # Handle interactivity
        if 'interactive' in kwargs.keys():
            interactive = kwargs['interactive']
            if interactive in [None, 'default']:
                interactive = self.orig_options['interactive']
            if interactive:
                kwargs['show'] = True
                kwargs['close'] = False
                kwargs['backend'] = self.orig_options['backend']
            else:
                kwargs['show'] = False
                kwargs['backend'] = 'agg'

        # Reset options
        for key,value in kwargs.items():

            # Handle deprecations
            rename = {'font_size': 'fontsize', 'font_family':'font'}
            if key in rename.keys():
                from . import misc as cvm # Here to avoid circular import
                oldkey = key
                key = rename[oldkey]
                warnmsg = f'Key "{oldkey}" is deprecated, please use "{key}" instead'
                cvm.warn(warnmsg, FutureWarning)

            if key not in self:
                keylist = self.orig_options.keys()
                keys = '\n'.join(keylist)
                errormsg = f'Option "{key}" not recognized; options are "defaults" or:\n{keys}\n\nSee help(cv.options.set) for more information.'
                raise sc.KeyNotFoundError(errormsg)
            else:
                if value in [None, 'default']:
                    value = self.orig_options[key]
                self[key] = value
                if key in 'backend':
                    pl.switch_backend(value)

        return


    def context(self, **kwargs):
        '''
        Alias to set() for non-plotting options, for use in a "with" block.

        Note: for plotting options, use ``cv.options.with_style()``, which is linked
        to Matplotlib's context manager. If you set plotting options with this,
        they won't have any effect.

        **Examples**::

            # Silence all output
            with cv.options.context(verbose=0):
                cv.Sim().run()

            # Convert warnings to errors
            with cv.options.context(warnings='error'):
                cv.Sim(location='not a location').initialize()

            # Use with_style(), not context(), for plotting options
            with cv.options.with_style(dpi=50):
                cv.Sim().run().plot()

        New in version 3.1.2.
        '''

        # Store current settings
        on_entry = {k:self[k] for k in kwargs.keys()}
        self.setattribute('on_entry', on_entry)

        # Make changes
        self.set(**kwargs)
        return self


    def get_default(self, key):
        ''' Helper function to get the original default options '''
        return self.orig_options[key]


    def changed(self, key):
        ''' Check if current setting has been changed from default '''
        if key in self.orig_options:
            return self[key] != self.orig_options[key]
        else:
            return None


    def help(self, detailed=False, output=False):
        '''
        Print information about options.

        Args:
            detailed (bool): whether to print out full help
            output (bool): whether to return a list of the options

        **Example**::

            cv.options.help(detailed=True)
        '''

        # If not detailed, just print the docstring for cv.options
        if not detailed:
            print(self.__doc__)
            return

        n = 15 # Size of indent
        optdict = sc.objdict()
        for key in self.orig_options.keys():
            entry = sc.objdict()
            entry.key = key
            entry.current = sc.indent(n=n, width=None, text=sc.pp(self[key], output=True)).rstrip()
            entry.default = sc.indent(n=n, width=None, text=sc.pp(self.orig_options[key], output=True)).rstrip()
            if not key.startswith('rc'):
                entry.variable = f'COVASIM_{key.upper()}' # NB, hard-coded above!
            else:
                entry.variable = 'No environment variable'
            entry.desc = sc.indent(n=n, text=self.optdesc[key])
            optdict[key] = entry

        # Convert to a dataframe for nice printing
        print('Covasim global options ("Environment" = name of corresponding environment variable):')
        for k, key, entry in optdict.enumitems():
            sc.heading(f'{k}. {key}', spaces=0, spacesafter=0)
            changestr = '' if entry.current == entry.default else ' (modified)'
            print(f'          Key: {key}')
            print(f'      Current: {entry.current}{changestr}')
            print(f'      Default: {entry.default}')
            print(f'  Environment: {entry.variable}')
            print(f'  Description: {entry.desc}')

        sc.heading('Methods:', spacesafter=0)
        print('''
    cv.options(key=value) -- set key to value
    cv.options[key] -- get or set key
    cv.options.set() -- set option(s)
    cv.options.get_default() -- get default setting(s)
    cv.options.load() -- load settings from file
    cv.options.save() -- save settings to file
    cv.options.to_dict() -- convert to dictionary
    cv.options.style() -- create style context for plotting
''')

        if output:
            return optdict
        else:
            return


    def load(self, filename, verbose=True, **kwargs):
        '''
        Load current settings from a JSON file.

        Args:
            filename (str): file to load
            kwargs (dict): passed to ``sc.loadjson()``
        '''
        json = sc.loadjson(filename=filename, **kwargs)
        current = self.to_dict()
        new = {k:v for k,v in json.items() if v != current[k]} # Don't reset keys that haven't changed
        self.set(**new)
        if verbose: print(f'Settings loaded from {filename}')
        return


    def save(self, filename, verbose=True, **kwargs):
        '''
        Save current settings as a JSON file.

        Args:
            filename (str): file to save to
            kwargs (dict): passed to ``sc.savejson()``
        '''
        json = self.to_dict()
        output = sc.savejson(filename=filename, obj=json, **kwargs)
        if verbose: print(f'Settings saved to {filename}')
        return output




# Create the options on module load, and load the fonts
options = Options()


































def set_default_options():
    '''
    Set the default options for Sciris -- not to be called by the user, use
    ``sc.options.set('defaults')`` instead.
    '''

    # Options acts like a class, but is actually a dictobj for ease of manipulation
    optdesc = dictobj() # Help for the options
    options = Options() # The options

    optdesc.sep = 'Set thousands seperator'
    options.sep = str(os.getenv('SCIRIS_SEP', ','))

    optdesc.aspath = 'Set whether to return Path objects instead of strings by default'
    options.aspath = bool(os.getenv('SCIRIS_ASPATH', False))

    optdesc.backend = 'Set the Matplotlib backend (use "agg" for non-interactive)'
    options.backend = os.getenv('SCIRIS_BACKEND', pl.get_backend())

    # optdesc.interactive = 'Convenience method to set figure backend'
    # options.interactive = os.getenv('SCIRIS_INTERACTIVE', True)

    # optdesc.jupyter = 'Plotting settings for Jupyter notebooks: shortcut to set backend to "widget" (default) or "retina"'
    # options.jupyter = os.getenv('SCIRIS_JUPYTER', False)

    optdesc.dpi = 'Set the default DPI -- the larger this is, the larger the figures will be'
    options.dpi = int(os.getenv('SCIRIS_DPI', pl.rcParams['figure.dpi']))

    optdesc.fontsize = 'Set the default font size'
    options.fontsize = int(os.getenv('SCIRIS_FONTSIZE', pl.rcParams['font.size']))

    optdesc.font = 'Set the default font family (e.g., Arial)'
    options.font = os.getenv('SCIRIS_FONT', pl.rcParams['font.family'])

    return options, optdesc


# Actually set the options
options, optdesc = set_default_options()
orig_options = cp.deepcopy(options) # Make a copy for referring back to later

# Specify which keys require a reload
matplotlib_keys = ['fontsize', 'font', 'dpi', 'backend']


def set_option(key=None, value=None, **kwargs):
    '''
    Set a parameter or parameters. Use ``sc.options('defaults')`` to reset all
    values to default, or ``sc.options(dpi='default')`` to reset one parameter
    to default. See ``sc.options.help()`` for more information.

    Args:
        key    (str):    the parameter to modify, or 'defaults' to reset everything to default values
        value  (varies): the value to specify; use None or 'default' to reset to default
        kwargs (dict):   if supplied, set multiple key-value pairs

    Options are (see also ``sc.options.help()``):

        - sep:       the thousands separator to use
        - aspath:    whether to return Path objects instead of strings
        - fontsize:  the font size used for the plots
        - font:      the font family/face used for the plots
        - dpi:       the overall DPI for the figure
        - backend:   which Matplotlib backend to use

    **Examples**::

        sc.options.set('fontsize', 18) # Larger font
        sc.options(fontsize=18, backend='agg') # Larger font, non-interactive plots
        sc.options('defaults') # Reset to default options

    | New in version 1.3.0.
    | New in version 2.0.0: ``interactive`` and ``jupyter`` options.
    '''

    # Reset to defaults
    if key in ['default', 'defaults']:
        kwargs = orig_options # Reset everything to default

    # Handle other keys
    elif key is not None:
        kwargs.update({key:value})

    # Reset options
    for key,value in kwargs.items():
        if key not in options:
            keylist = orig_options.keys()
            keys = ', '.join(keylist)
            errormsg = f'Option "{key}" not recognized; options are "defaults" or: {keys}. See help(sc.options.set) for more information.'
            raise KeyError(errormsg)
        else:
            if value in [None, 'default']:
                value = orig_options[key]
            options[key] = value
            if key in matplotlib_keys:
                set_matplotlib_global(key, value)
    return


def default(key=None, reset=True):
    ''' Helper function to set the original default options '''
    if key is not None:
        value = orig_options[key]
        if reset:
            options.set(key=key, value=value)
        return value
    else:
        if not reset:
            return orig_options
        else:
            options.set('defaults')
    return


def get_help(output=False):
    '''
    Print information about options.

    Args:
        output (bool): whether to return a list of the options

    **Example**::

        sc.options.help()
    '''

    optdict = dictobj()
    for key in orig_options.keys():
        entry = dictobj()
        entry.key = key
        entry.current = options[key]
        entry.default = orig_options[key]
        entry.variable = f'SCIRIS_{key.upper()}' # NB, hard-coded above!
        entry.desc = optdesc[key]
        optdict[key] = entry

    # Convert to a dataframe for nice printing
    print('Sciris global options ("Environment" = name of corresponding environment variable):')
    for key,entry in optdict.items():
        print(f'\n{key}')
        changestr = '' if entry.current == entry.default else ' (modified)'
        print(f'      Current: {entry.current}{changestr}')
        print(f'      Default: {entry.default}')
        print(f'  Environment: {entry.variable}')
        print(f'  Description: {entry.desc}')

    if output:
        return optdict
    else:
        return


def set_matplotlib_global(key, value):
    ''' Set a global option for Matplotlib -- not for users '''
    import pylab as pl # In some cases re-import is needed
    if value: # Don't try to reset any of these to a None value
        if   key == 'fontsize': pl.rcParams['font.size']   = value
        elif key == 'font':     pl.rcParams['font.family'] = value
        elif key == 'dpi':      pl.rcParams['figure.dpi']  = value
        elif key == 'backend':  pl.switch_backend(value)
        else: raise KeyError(f'Key {key} not found')
    return


# Add these here to be more accessible to the user
options.set = set_option
options.default = default
options.help = get_help


def help(pattern=None, source=False, ignorecase=True, flags=None, context=False, output=False):
    '''
    Get help on Sciris in general, or search for a word/expression.

    Args:
        pattern    (str):  the word, phrase, or regex to search for
        source     (bool): whether to search source code instead of docstrings for matches
        ignorecase (bool): whether to ignore case (equivalent to ``flags=re.I``)
        flags      (list): additional flags to pass to ``re.findall()``
        context    (bool): whether to show the line(s) of matches
        output     (bool): whether to return the dictionary of matches

    **Examples**::

        sc.help()
        sc.help('smooth')
        sc.help('JSON', ignorecase=False, context=True)
        sc.help('pickle', source=True, context=True)

    | New in version 1.3.0.
    | New in version 1.3.1: "source" argument
    '''
    defaultmsg = '''
For general help using Sciris, the best place to start is the docs:

    http://docs.sciris.org

To search for a keyword/phrase/regex in Sciris' docstrings, use e.g.:

    >>> sc.help('smooth')

See help(sc.help) for more information.
'''
    # No pattern is provided, print out default help message
    if pattern is None:
        print(defaultmsg)

    else:

        import sciris as sc # Here to avoid circular import

        # Handle inputs
        flags = sc.promotetolist(flags)
        if ignorecase:
            flags.append(re.I)

        def func_ok(f):
            ''' Skip certain functions '''
            excludes = [
                f.startswith('_'),
                f.startswith('sc_'),
                f in ['help', 'options'],
            ]
            ok = not(any(excludes))
            return ok

        # Get available functions/classes
        funcs = [f for f in dir(sc) if func_ok(f)] # Skip dunder methods and modules

        # Get docstrings or full source code
        docstrings = dict()
        for funcname in funcs:
            f = getattr(sc, funcname)
            if source: string = inspect.getsource(f)
            else:      string = f.__doc__
            docstrings[funcname] = string

        # Find matches
        matches = co.defaultdict(list)
        linenos = co.defaultdict(list)

        for k,docstring in docstrings.items():
            for l,line in enumerate(docstring.splitlines()):
                if re.findall(pattern, line, *flags):
                    linenos[k].append(str(l))
                    matches[k].append(line)

        # Assemble output
        if not len(matches):
            string = f'No matches for "{pattern}" found among {len(docstrings)} available functions.'
        else:
            string = f'Found {len(matches)} matches for "{pattern}" among {len(docstrings)} available functions:\n'
            maxkeylen = 0
            for k in matches.keys(): maxkeylen = max(len(k), maxkeylen)
            for k,match in matches.items():
                if not context:
                    keystr = f'  {k:>{maxkeylen}s}'
                else:
                    keystr = k
                matchstr = f'{keystr}: {len(match)} matches'
                if context:
                    matchstr = sc.heading(matchstr, output=True)
                else:
                    matchstr += '\n'
                string += matchstr
                if context:
                    lineno = linenos[k]
                    maxlnolen = max([len(l) for l in lineno])
                    for l,m in zip(lineno, match):
                        string += sc.colorize(string=f'  {l:>{maxlnolen}s}: ', fg='cyan', output=True)
                        string += f'{m}\n'
                    string += '—'*60 + '\n'

        # Print result and return
        print(string)
        if output:
            return string
        else:
            return
