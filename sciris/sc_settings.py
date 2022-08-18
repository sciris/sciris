'''
Define options for Sciris, mostly plotting options.

All options should be set using ``set()`` or directly, e.g.::

    sc.options(font_size=18)

To reset default options, use::

    sc.options('default')

Note: "options" is used to refer to the choices available (e.g., DPI), while "settings"
is used to refer to the choices made (e.g., ``dpi=150``).
'''

import os
import re
import inspect
import collections as co
import pylab as pl
from . import sc_utils as scu
from . import sc_odict as sco
from . import sc_printing as scp


__all__ = ['parse_env', 'options', 'help']


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
rc_fancy = scu.dcp(rc_simple)
rc_fancy.update({
    'axes.facecolor': '#f2f2ff',
    'axes.grid':      True,
    'grid.color':     'white',
    'grid.linewidth': 1,
})


def parse_env(var, default=None, which='str'):
    '''
    Simple function to parse environment variables

    Args:
        var (str): name of the environment variable to get
        default (any): default value
        which (str): what type to convert to (if None, don't convert)

    New in version 2.0.0.
    '''
    val = os.getenv(var, default)
    if which is None:
        return val
    elif which in ['str', 'string']:
        if val: out = str(val)
        else:   out = ''
    elif which == 'int':
        if val: out = int(val)
        else:   out = 0
    elif which == 'float':
        if val: out = float(val)
        else:   out = 0.0
    elif which == 'bool':
        if val:
            if isinstance(val, str):
                if val.lower() in ['false', 'f', '0', '']:
                    val = False
                else:
                    val = True
            out = bool(val)
        else:
            out = False
    else:
        errormsg = f'Could not understand type "{which}": must be None, str, int, float, or bool'
        raise ValueError(errormsg)
    return out


#%% Define the options class

class Options(sco.objdict):
    '''
    Set options for Sciris.

    Use ``sc.options.set('defaults')`` to reset all values to default, or ``sc.options.set(dpi='default')``
    to reset one parameter to default. See ``sc.options.help(detailed=True)`` for
    more information.

    Options can also be saved and loaded using ``sc.options.save()`` and ``sc.options.load()``.
    See ``sc.options.with_style()`` to set options temporarily.

    Common options are (see also ``sc.options.help(detailed=True)``):

        - dpi:            the overall DPI (i.e. size) of the figures
        - font:           the font family/face used for the plots
        - fontsize:       the font size used for the plots
        - backend:        which Matplotlib backend to use
        - interactive:    convenience method to set backend
        - jupyter:        defaults for Jupyter (change backend)
        - style:          the plotting style to use

    Each setting can also be set with an environment variable, e.g. SCIRIS_DPI.
    Note also the environment variable SCIRIS_LAZY, which imports Sciris lazily
    (i.e. does not import submodules).

    **Examples**::

        sc.options(dpi=150) # Larger size
        sc.options(style='simple', font='Rosario') # Change to the "simple" Sciris style with a custom font
        sc.options.set(fontsize=18, show=False, backend='agg', precision=64) # Multiple changes
        sc.options(interactive=False) # Turn off interactive plots
        sc.options(jupyter=True) # Defaults for Jupyter
        sc.options('defaults') # Reset to default options

    | New in version 1.3.0.
    | New in version 2.0.0: revamed with additional options ``interactive`` and ``jupyter``, plus styles
    '''

    def __init__(self):
        super().__init__()
        optdesc, options = self.get_orig_options() # Get the options
        self.update(options) # Update this object with them
        self.optdesc = optdesc # Set the description as an attribute, not a dict entry
        self.orig_options = scu.dcp(options) # Copy the default options
        return


    def __call__(self, *args, **kwargs):
        '''Allow ``sc.options(dpi=150)`` instead of ``sc.options.set(dpi=150)`` '''
        return self.set(*args, **kwargs)


    def to_dict(self):
        ''' Pull out only the settings from the options object '''
        return {k:v for k,v in self.items()}


    def __repr__(self):
        ''' Brief representation '''
        output = scp.objectid(self)
        output += 'Sciris options (see also sc.options.disp()):\n'
        output += scu.pp(self.to_dict(), output=True)
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
            del self.on_entry
        except AttributeError as E:
            errormsg = 'Please use sc.options.context() if using a with block'
            raise AttributeError(errormsg) from E
        return


    def disp(self):
        ''' Detailed representation '''
        output = 'Sciris options (see also sc.options.help()):\n'
        keylen = 14 # Maximum key length  -- "numba_parallel"
        for k,v in self.items():
            keystr = scp.colorize(f'  {k:>{keylen}s}: ', fg='cyan', output=True)
            reprstr = scu.pp(v, output=True)
            reprstr = scp.indent(n=keylen+4, text=reprstr, width=None)
            output += f'{keystr}{reprstr}'
        print(output)
        return


    @staticmethod
    def get_orig_options():
        '''
        Set the default options for Sciris -- not to be called by the user, use
        ``sc.options.set('defaults')`` instead.
        '''

        # Options acts like a class, but is actually an objdict for simplicity
        optdesc = sco.objdict() # Help for the options
        options = sco.objdict() # The options

        optdesc.sep = 'Set thousands seperator'
        options.sep = parse_env('SCIRIS_SEP', ',', 'str')

        optdesc.aspath = 'Set whether to return Path objects instead of strings by default'
        options.aspath = parse_env('SCIRIS_ASPATH', False, 'bool')

        optdesc.style = 'Set the default plotting style -- options are "simple" and "fancy" plus those in pl.style.available; see also options.rc'
        options.style = parse_env('SCIRIS_STYLE', 'simple', 'str')

        optdesc.dpi = 'Set the default DPI -- the larger this is, the larger the figures will be'
        options.dpi = parse_env('SCIRIS_DPI', pl.rcParams['figure.dpi'], 'int')

        optdesc.font = 'Set the default font family (e.g., sans-serif or Arial)'
        options.font = parse_env('SCIRIS_FONT', pl.rcParams['font.family'], 'str')

        optdesc.fontsize = 'Set the default font size'
        options.fontsize = parse_env('SCIRIS_FONT_SIZE', pl.rcParams['font.size'], 'str')

        optdesc.interactive = 'Convenience method to set figure backend'
        options.interactive = parse_env('SCIRIS_INTERACTIVE', True, 'bool')

        optdesc.jupyter = 'Convenience method to set common settings for Jupyter notebooks: set to "retina" or "widget" (default) to set backend'
        options.jupyter = parse_env('SCIRIS_JUPYTER', False, 'bool')

        optdesc.backend = 'Set the Matplotlib backend (use "agg" for non-interactive)'
        options.backend = parse_env('SCIRIS_BACKEND', pl.get_backend(), 'str')

        optdesc.rc = 'Matplotlib rc (run control) style parameters used during plotting -- usually set automatically by "style" option'
        options.rc = scu.dcp(rc_simple)

        return optdesc, options


    def set(self, key=None, value=None, use=True, **kwargs):
        '''
        Actually change the style. See ``sc.options.help()`` for more information.

        Args:
            key    (str):    the parameter to modify, or 'defaults' to reset everything to default values
            value  (varies): the value to specify; use None or 'default' to reset to default
            use    (bool):   whether to immediately apply the change (to Matplotlib)
            kwargs (dict):   if supplied, set multiple key-value pairs

        **Example**::

            sc.options.set(dpi=50) # Equivalent to sc.options(dpi=50)
        '''

        # Reset to defaults
        if key in ['default', 'defaults']:
            kwargs = self.orig_options # Reset everything to default

        # Handle other keys
        elif key is not None:
            kwargs.update({key:value})

        # Handle Jupyter
        if 'jupyter' in kwargs.keys() and kwargs['jupyter']: # pragma: no cover

            # Handle import
            matplotlib_inline = None
            try:
                from IPython import get_ipython
                import matplotlib_inline
                magic = get_ipython().magic
            except Exception as E:
                errormsg = f'Could not import IPython and matplotlib_inline; not attempting to set Jupyter ({str(E)})'
                print(errormsg)

            # Handle options
            widget_opts  = [True, 'widget', 'matplotlib', 'interactive']
            default_opts = [False, 'default']
            format_opts  = ['retina', 'pdf','png','png2x','svg','jpg']

            jupyter = kwargs['jupyter']
            if jupyter in widget_opts:
                jupyter = 'widget'
            elif jupyter in default_opts:
                jupyter = 'png'

            if matplotlib_inline:
                if jupyter == 'widget':
                    try: # First try interactive
                        magic('%matplotlib widget')
                    except Exception as E:
                        errormsg = 'Could not set backend to "widget"; try "pip install ipympl" or try "retina" instead'
                        raise RuntimeError(errormsg) from E
                elif jupyter in format_opts:
                    magic('%matplotlib inline')
                    matplotlib_inline.backend_inline.set_matplotlib_formats(jupyter)
                else:
                    errormsg = f'Could not understand Jupyter option "{jupyter}": options are widget, {scu.strjoin(format_opts)}'
                    raise ValueError(errormsg)

        # Handle interactivity
        if 'interactive' in kwargs.keys():
            interactive = kwargs['interactive']
            if interactive in [None, 'default']:
                interactive = self.orig_options['interactive']
            if interactive:
                kwargs['backend'] = self.orig_options['backend']
            else:
                kwargs['backend'] = 'agg'

        # Reset options
        for key,value in kwargs.items():

            # Handle deprecations
            rename = {'font_size': 'fontsize', 'font_family':'font'}
            if key in rename.keys(): # pragma: no cover
                oldkey = key
                key = rename[oldkey]

            if key not in self.keys(): # pragma: no cover
                keylist = self.orig_options.keys()
                keys = '\n'.join(keylist)
                errormsg = f'Option "{key}" not recognized; options are "defaults" or:\n{keys}\n\nSee help(sc.options.set) for more information.'
                raise ValueError(errormsg) from KeyError(key) # Can't use sc.KeyNotFoundError since would be a circular import
            else:
                if value in [None, 'default']:
                    value = self.orig_options[key]
                self[key] = value
                matplotlib_keys = ['fontsize', 'font', 'dpi', 'backend']
                if key in matplotlib_keys:
                    self.set_matplotlib_global(key, value)

        if use:
            self.use_style()

        return


    def context(self, **kwargs):
        '''
        Alias to set() for non-plotting options, for use in a "with" block.

        Note: for plotting options, use ``sc.options.with_style()``, which is linked
        to Matplotlib's context manager. If you set plotting options with this,
        they won't have any effect.
        '''
        # Store current settings
        self.on_entry = {k:self[k] for k in kwargs.keys()}

        # Make changes
        self.set(**kwargs)
        return self


    def set_matplotlib_global(self, key, value):
        ''' Set a global option for Matplotlib -- not for users '''
        if value: # Don't try to reset any of these to a None value
            if   key == 'fontsize': pl.rcParams['font.size']   = value
            elif key == 'font':     pl.rcParams['font.family'] = value
            elif key == 'dpi':      pl.rcParams['figure.dpi']  = value
            elif key == 'backend':  pl.switch_backend(value)
            else: raise KeyError(f'Key {key} not found')
        return


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

            sc.options.help(detailed=True)
        '''

        # If not detailed, just print the docstring for sc.options
        if not detailed:
            print(self.__doc__)
            return

        n = 15 # Size of indent
        optdict = sco.objdict()
        for key in self.orig_options.keys():
            entry = sco.objdict()
            entry.key = key
            entry.current = scp.indent(n=n, width=None, text=scu.pp(self[key], output=True)).rstrip()
            entry.default = scp.indent(n=n, width=None, text=scu.pp(self.orig_options[key], output=True)).rstrip()
            if not key.startswith('rc'):
                entry.variable = f'SCIRIS_{key.upper()}' # NB, hard-coded above!
            else:
                entry.variable = 'No environment variable'
            entry.desc = scp.indent(n=n, text=self.optdesc[key])
            optdict[key] = entry

        # Convert to a dataframe for nice printing
        print('Sciris global options ("Environment" = name of corresponding environment variable):')
        for k, key, entry in optdict.enumitems():
            scp.heading(f'{k}. {key}', spaces=0, spacesafter=0)
            changestr = '' if entry.current == entry.default else ' (modified)'
            print(f'          Key: {key}')
            print(f'      Current: {entry.current}{changestr}')
            print(f'      Default: {entry.default}')
            print(f'  Environment: {entry.variable}')
            print(f'  Description: {entry.desc}')

        scp.heading('Methods:', spacesafter=0)
        print('''
    sc.options(key=value) -- set key to value
    sc.options[key] -- get or set key
    sc.options.set() -- set option(s)
    sc.options.get_default() -- get default setting(s)
    sc.options.load() -- load settings from file
    sc.options.save() -- save settings to file
    sc.options.to_dict() -- convert to dictionary
    sc.options.style() -- create style context for plotting
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
        from . import sc_fileio as scf # To avoid circular import
        json = scf.loadjson(filename=filename, **kwargs)
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
        from . import sc_fileio as scf # To avoid circular import
        json = self.to_dict()
        output = scf.savejson(filename=filename, obj=json, **kwargs)
        if verbose: print(f'Settings saved to {filename}')
        return output


    def _handle_style(self, style=None, reset=False, copy=True):
        ''' Helper function to handle logic for different styles '''
        rc = self.rc # By default, use current
        if isinstance(style, dict): # If an rc-like object is supplied directly
            rc = scu.dcp(style)
        elif style is not None: # Usual use case
            stylestr = str(style).lower()
            if   stylestr in ['simple', 'default']: rc = scu.dcp(rc_simple)
            elif stylestr in ['fancy', 'covasim']:  rc = scu.dcp(rc_fancy)
            elif style in pl.style.library:         rc = scu.dcp(pl.style.library[style])
            else:
                errormsg = f'Style "{style}"; not found; options are "simple" (default), "fancy", plus:\n{scu.newlinejoin(pl.style.available)}'
                raise ValueError(errormsg)
        if reset:
            self.rc = rc
        if copy:
            rc = scu.dcp(rc)
        return rc


    def with_style(self, style_args=None, use=False, **kwargs):
        '''
        Combine all Matplotlib style information, and either apply it directly
        or create a style context.

        To set globally, use ``sc.options.use_style()``. Otherwise, use ``sc.options.with_style()``
        as part of a ``with`` block to set the style just for that block (using
        this function outsde of a with block and with ``use=False`` has no effect, so
        don't do that!).

        Args:
            style_args (dict): a dictionary of style arguments
            use (bool): whether to set as the global style; else, treat as context for use with "with" (default)
            kwargs (dict): additional style arguments

        Valid style arguments are:

            - ``dpi``:       the figure DPI
            - ``font``:      font (typeface)
            - ``fontsize``:  font size
            - ``grid``:      whether or not to plot gridlines
            - ``facecolor``: color of the axes behind the plot
            - any of the entries in ``pl.rParams``

        **Examples**::

            with sc.options.with_style(dpi=300): # Use default options, but higher DPI
                pl.plot([1,3,6])
        '''

        # Handle inputs
        rc = scu.dcp(self.rc) # Make a local copy of the currently used settings
        kwargs = scu.mergedicts(style_args, kwargs)

        # Handle style, overwiting existing
        style = kwargs.pop('style', None)
        rc = self._handle_style(style, reset=False)

        def pop_keywords(sourcekeys, rckey):
            ''' Helper function to handle input arguments '''
            sourcekeys = scu.tolist(sourcekeys)
            key = sourcekeys[0] # Main key
            value = None
            changed = self.changed(key)
            if changed:
                value = self[key]
            for k in sourcekeys:
                kwvalue = kwargs.pop(k, None)
                if kwvalue is not None:
                    value = kwvalue
            if value is not None:
                rc[rckey] = value
            return

        # Handle special cases
        pop_keywords('dpi', rckey='figure.dpi')
        pop_keywords(['font', 'fontfamily', 'font_family'], rckey='font.family')
        pop_keywords(['fontsize', 'font_size'], rckey='font.size')
        pop_keywords('grid', rckey='axes.grid')
        pop_keywords('facecolor', rckey='axes.facecolor')

        # Handle other keywords
        for key,value in kwargs.items():
            if key not in pl.rcParams:
                errormsg = f'Key "{key}" does not match any value in Sciris options or pl.rcParams'
                raise KeyError(errormsg)
            elif value is not None:
                rc[key] = value

        # Tidy up
        if use:
            return pl.style.use(scu.dcp(rc))
        else:
            return pl.style.context(scu.dcp(rc))


    def use_style(self, **kwargs):
        '''
        Shortcut to set Sciris's current style as the global default.

        **Example**::

            sc.options.use_style() # Set Sciris options as default
            pl.figure()
            pl.plot([1,3,7])

            pl.style.use('seaborn-whitegrid') # to something else
            pl.figure()
            pl.plot([3,1,4])
        '''
        return self.with_style(use=True, **kwargs)


# Create the options on module load
options = Options()


#%% Module help

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
            try:
                f = getattr(sc, funcname)
                if source: string = inspect.getsource(f)
                else:      string = f.__doc__
                docstrings[funcname] = string
            except OSError: # Happens for built-ins, e.g. defaultdict
                pass

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
