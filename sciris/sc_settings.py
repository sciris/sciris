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

    New in version 1.3.0.
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