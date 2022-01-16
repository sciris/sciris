'''
Printing/notification functions.

Highlights:
    - ``sc.heading()``: print text as a 'large' heading
    - ``sc.colorize()``: print text in a certain color
    - ``sc.pr()``: print full representation of an object, including methods and each attribute
    - ``sc.sigfigs()``: truncate a number to a certain number of significant figures
    - ``sc.progressbar()``: show a (text-based) progress bar
    - ``sc.capture()``: capture text output (e.g., stdout) as a variable
'''

import io
import os
import time
import colors
import pprint
import numpy as np
import collections as co
from textwrap import fill
from collections import UserString
from contextlib import redirect_stdout
from . import sc_utils as scu


# Add Windows support for colors (do this at the module level so that colorama.init() only gets called once)
if scu.iswindows(): # pragma: no cover # NB: can't use startswith() because of 'cygwin'
    try:
        import colorama
        colorama.init()
        ansi_support = True
    except:
        ansi_support = False  # print('Warning: you have called colorize() on Windows but do not have either the colorama or tendo modules.')
else:
    ansi_support = True




#%% Object display functions

__all__ = ['createcollist', 'objectid', 'objatt', 'objmeth', 'objprop', 'objrepr', 'prepr', 'pr']


def createcollist(items, title=None, strlen=18, ncol=3):
    ''' Creates a string for a nice columnated list (e.g. to use in __repr__ method) '''
    nrow = int(np.ceil(float(len(items))/ncol))
    newkeys = []
    for x in range(nrow):
        newkeys += items[x::nrow]

    attstring = title + ':' if title else ''
    c = 0
    for x in newkeys:
        if c%ncol == 0: attstring += '\n  '
        if len(x) > strlen: x = x[:strlen-3] + '...'
        attstring += '%-*s  ' % (strlen,x)
        c += 1
    attstring += '\n'
    return attstring


def objectid(obj):
    ''' Return the object ID as per the default Python __repr__ method '''
    c = obj.__class__
    output = f'<{c.__module__}.{c.__name__} at {hex(id(obj))}>\n'
    return output


def objatt(obj, strlen=18, ncol=3):
    ''' Return a sorted string of object attributes for the Python __repr__ method '''
    if   hasattr(obj, '__dict__'):  oldkeys = sorted(obj.__dict__.keys())
    elif hasattr(obj, '__slots__'): oldkeys = sorted(obj.__slots__)
    else:                           oldkeys = []
    if len(oldkeys): output = createcollist(oldkeys, 'Attributes', strlen = 18, ncol = 3)
    else:            output = ''
    return output


def objmeth(obj, strlen=18, ncol=3):
    ''' Return a sorted string of object methods for the Python __repr__ method '''
    try:
        oldkeys = sorted([method + '()' for method in dir(obj) if callable(getattr(obj, method)) and not method.startswith('__')])
    except: # pragma: no cover
        oldkeys = ['Methods N/A']
    if len(oldkeys): output = createcollist(oldkeys, 'Methods', strlen=strlen, ncol=ncol)
    else:            output = ''
    return output


def objprop(obj, strlen=18, ncol=3):
    ''' Return a sorted string of object properties for the Python __repr__ method '''
    try:
        oldkeys = sorted([prop for prop in dir(obj) if isinstance(getattr(type(obj), prop, None), property) and not prop.startswith('__')])
    except: # pragma: no cover
        oldkeys = ['Properties N/A']
    if len(oldkeys): output = createcollist(oldkeys, 'Properties', strlen=strlen, ncol=ncol)
    else:            output = ''
    return output


def objrepr(obj, showid=True, showmeth=True, showprop=True, showatt=True, dividerchar='—', dividerlen=60):
    ''' Return useful printout for the Python __repr__ method '''
    divider = dividerchar*dividerlen + '\n'
    output = ''
    if showid:
        output += objectid(obj)
        output += divider
    if showmeth:
        meths = objmeth(obj)
        if meths:
            output += objmeth(obj)
            output += divider
    if showprop:
        props = objprop(obj)
        if props:
            output += props
            output += divider
    if showatt:
        attrs = objatt(obj)
        if attrs:
            output += attrs
            output += divider
    return output


def prepr(obj, maxlen=None, maxitems=None, skip=None, dividerchar='—', dividerlen=60, use_repr=True, maxtime=3, die=False):
    '''
    Akin to "pretty print", returns a pretty representation of an object --
    all attributes (except any that are skipped), plust methods and ID. Usually
    used via the interactive sc.pr() (which prints), rather than this (which returns
    a string).

    Args:
        obj (anything): the object to be represented
        maxlen (int): maximum number of characters to show for each item
        maxitems (int): maximum number of items to show in the object
        skip (list): any properties to skip
        dividerchar (str): divider for methods, attributes, etc.
        divierlen (int): number of divider characters
        use_repr (bool): whether to use repr() or str() to parse the object
        maxtime (float): maximum amount of time to spend on trying to print the object
        die (bool): whether to raise an exception if an error is encountered
    '''

    # Decide how to handle representation function -- repr is dangerous since can lead to recursion
    repr_fn = repr if use_repr else str
    T = time.time() # Start the timer
    time_exceeded = False

    # Handle input arguments
    divider = dividerchar*dividerlen + '\n'
    if maxlen   is None: maxlen   = 80
    if maxitems is None: maxitems = 100
    if skip     is None: skip = []
    else:                skip = scu.promotetolist(skip)

    # Initialize things to print out
    labels = []
    values = []

    # Wrap entire process in a try-except in case it fails
    try:
        if not (hasattr(obj, '__dict__') or hasattr(obj, '__slots__')):
            # It's a plain object
            labels = [f'{type(obj)}']
            values = [repr_fn(obj)]
        else:
            if hasattr(obj, '__dict__'):
                labels = sorted(set(obj.__dict__.keys()) - set(skip))  # Get the dict attribute keys
            else:
                labels = sorted(set(obj.__slots__) - set(skip))  # Get the slots attribute keys

            if len(labels):
                extraitems = len(labels) - maxitems
                if extraitems>0:
                    labels = labels[:maxitems]
                values = []
                for a,attr in enumerate(labels):
                    if (time.time() - T) < maxtime:
                        try: value = repr_fn(getattr(obj, attr))
                        except: value = 'N/A'
                        values.append(value)
                    else:
                        labels = labels[:a]
                        labels.append('etc. (time exceeded)')
                        values.append(f'{len(labels)-a} entries not shown')
                        time_exceeded = True
                        break
            else:
                items = dir(obj)
                extraitems = len(items) - maxitems
                if extraitems > 0:
                    items = items[:maxitems]
                for a,attr in enumerate(items):
                    if not attr.startswith('__'):
                        if (time.time() - T) < maxtime:
                            try:    value = repr_fn(getattr(obj, attr))
                            except: value = 'N/A'
                            labels.append(attr)
                            values.append(value)
                        else:
                            labels.append('etc. (time exceeded)')
                            values.append(f'{len(labels)-a} entries not shown')
                            time_exceeded = True
            if extraitems > 0:
                labels.append('etc. (too many items)')
                values.append(f'{extraitems} entries not shown')

        # Decide how to print them
        maxkeylen = 0
        if len(labels):
            maxkeylen = max([len(label) for label in labels]) # Find the maximum length of the attribute keys
        if maxkeylen<maxlen:
            maxlen = maxlen - maxkeylen # Shorten the amount of data shown if the keys are long
        formatstr = '%'+ '%i'%maxkeylen + 's' # Assemble the format string for the keys, e.g. '%21s'
        output  = objrepr(obj, showatt=False, dividerchar=dividerchar, dividerlen=dividerlen) # Get the methods
        for label,value in zip(labels,values): # Loop over each attribute
            if len(value)>maxlen: value = value[:maxlen] + ' [...]' # Shorten it
            prefix = formatstr%label + ': ' # The format key
            output += indent(prefix, value)
        output += divider
        if time_exceeded:
            timestr = f'\nNote: the object did not finish printing within maxtime={maxtime} s.\n'
            timestr += 'To see the full object, call prepr() with increased maxtime.'
            output += timestr

    # If that failed, try progressively simpler approaches
    except Exception as E: # pragma: no cover
        if die:
            errormsg = 'Failed to create pretty representation of object'
            raise RuntimeError(errormsg) from E
        else:
            try: # Next try the objrepr, which is the same except doesn't print attribute values
                output = objrepr(obj, dividerchar=dividerchar, dividerlen=dividerlen)
                output += f'\nWarning: showing simplified output since full repr failed {str(E)}'
            except: # If that fails, try just the string representation
                output = str(obj)

    return output


def pr(obj, *args, **kwargs):
    '''
    Shortcut for printing the pretty repr for an object -- similar to prettyprint

    **Example**::

        import pandas as pd
        df = pd.DataFrame({'a':[1,2,3], 'b':[4,5,6]})
        print(df) # See just the data
        sc.pr(df) # See all the methods too
    '''
    print(prepr(obj, *args, **kwargs))
    return


#%% Spacing functions

__all__ += ['blank', 'indent']


def blank(n=3):
    ''' Tiny function to print n blank lines, 3 by default '''
    print('\n'*n)


def indent(prefix=None, text=None, suffix='\n', n=0, pretty=False, width=70, **kwargs):
    '''
    Small wrapper to make textwrap more user friendly.

    Args:
        prefix (str): text to begin with (optional)
        text (str): text to wrap
        suffix (str): what to put on the end (by default, a newline)
        n (int): if prefix is not specified, the size of the indent
        pretty (bool): whether to use pprint to format the text
        width (int): maximum width before wrapping (if None, don't wrap)
        kwargs (dict): passed to ``textwrap.fill()``

    **Examples**::

        prefix = 'and then they said: '
        text = 'blah '*100
        print(sc.indent(prefix, text))

        print('my fave is: ' + sc.indent(text=rand(100), n=12))

    New in version 1.3.1: more flexibility in arguments
    '''
    # If "prefix" is given but text isn't, swap them
    if text is None and prefix is not None:
        text, prefix = prefix, text

    # Handle prefix and width
    if prefix is None: prefix = ' '*n
    if width  is None: width = 999_999

    # Get text in the right format -- i.e. a string
    if pretty: text = pprint.pformat(text)
    else:      text = scu.flexstr(text)

    # If there is no newline in the text, process the output normally.
    if text.find('\n') == -1:
        output = fill(text, initial_indent=prefix, subsequent_indent=' '*len(prefix), width=width, **kwargs)+suffix
    # Otherwise, handle each line separately and splice together the output.
    else:
        textlines = text.split('\n')
        output = ''
        for i, textline in enumerate(textlines):
            if i == 0:
                theprefix = prefix
            else:
                theprefix = ' '*len(prefix)
            output += fill(textline, initial_indent=theprefix, subsequent_indent=' '*len(prefix), width=width, **kwargs)+suffix

    if n: output = output[n:] # Need to remove the fake prefix
    return output



#%% Data representation functions

__all__ += ['sigfig', 'printarr', 'printdata', 'printvars']


def sigfig(x, sigfigs=5, SI=False, sep=False, keepints=False):
    '''
    Return a string representation of variable x with sigfigs number of significant figures

    Args:
        x (int/float/arr): the number(s) to round
        sigfigs (int): number of significant figures to round to
        SI (bool): whether to use SI notation
        sep (bool/str): if provided, use as thousands separator
        keepints (bool): never round ints

    **Examples**::

        x = 32433.3842
        sc.sigfig(x, SI=True) # Returns 32.433k
        sc.sigfig(x, sep=True) # Returns 32,433
    '''
    output = []

    try:
        n=len(x)
        X = x
        islist = True
    except:
        X = [x]
        n = 1
        islist = False
    for i in range(n):
        x = X[i]

        suffix = ''
        formats = [(1e18,'e18'), (1e15,'e15'), (1e12,'t'), (1e9,'b'), (1e6,'m'), (1e3,'k')]
        if SI:
            for val,suff in formats:
                if abs(x)>=val:
                    x = x/val
                    suffix = suff
                    break # Find at most one match

        try:
            if x==0:
                output.append('0')
            elif sigfigs is None:
                output.append(scu.flexstr(x)+suffix)
            elif x>(10**sigfigs) and not SI and keepints: # e.g. x = 23432.23, sigfigs=3, output is 23432
                roundnumber = int(round(x))
                if sep: string = format(roundnumber, ',')
                else:   string = f'{x:0.0f}'
                output.append(string)
            else:
                magnitude = np.floor(np.log10(abs(x)))
                factor = 10**(sigfigs-magnitude-1)
                x = round(x*factor)/float(factor)
                digits = int(abs(magnitude) + max(0, sigfigs - max(0,magnitude) - 1) + 1 + (x<0) + (abs(x)<1)) # one because, one for decimal, one for minus
                decimals = int(max(0,-magnitude+sigfigs-1))
                strformat = '%' + f'{digits}.{decimals}' + 'f'
                string = strformat % x
                if sep: # To insert separators in the right place, have to convert back to a number
                    if decimals>0:  roundnumber = float(string)
                    else:           roundnumber = int(string)
                    string = format(roundnumber, ',') # Allow comma separator
                string += suffix
                output.append(string)
        except: # pragma: no cover
            output.append(scu.flexstr(x))
    if islist:
        return tuple(output)
    else:
        return output[0]



def printarr(arr, arrformat='%0.2f  '):
    '''
    Print a numpy array nicely.

    **Example**::

        sc.printarr(pl.rand(3,7,4))

    Version: 2014dec01
    '''
    if np.ndim(arr)==1:
        string = ''
        for i in range(len(arr)):
            string += arrformat % arr[i]
        print(string)
    elif np.ndim(arr)==2:
        for i in range(len(arr)):
            printarr(arr[i], arrformat)
    elif np.ndim(arr)==3:
        for i in range(len(arr)):
            print('='*len(arr[i][0])*len(arrformat % 1))
            for j in range(len(arr[i])):
                printarr(arr[i][j], arrformat)
    else: # pragma: no cover
        print(arr) # Give up
    return



def printdata(data, name='Variable', depth=1, maxlen=40, indent='', level=0, showcontents=False): # pragma: no cover
    '''
    Nicely print a complicated data structure, a la Matlab.

    Note: this function is deprecated.

    Args:
      data: the data to display
      name: the name of the variable (automatically read except for first one)
      depth: how many levels of recursion to follow
      maxlen: number of characters of data to display (if 0, don't show data)
      indent: where to start the indent (used internally)

    Version: 2015aug21
    '''
    datatype = type(data)
    def printentry(data):
        if   datatype==dict:              string = (f'dict with {len(data.keys())} keys')
        elif datatype==list:              string = (f'list of length {len(data)}')
        elif datatype==tuple:             string = (f'tuple of length {len(data)}')
        elif datatype==np.ndarray:        string = (f'array of shape {np.shape(data)}')
        elif datatype.__name__=='module': string = (f'module with {len(dir(data))} components')
        elif datatype.__name__=='class':  string = (f'class with {len(dir(data))} components')
        else: string = datatype.__name__
        if showcontents and maxlen>0:
            datastring = ' | '+scu.flexstr(data)
            if len(datastring)>maxlen: datastring = datastring[:maxlen] + ' <etc> ' + datastring[-maxlen:]
        else: datastring=''
        return string+datastring

    string = printentry(data).replace('\n',' ') # Remove newlines
    print(level*'..' + indent + name + ' | ' + string)

    if depth>0:
        level += 1
        if type(data)==dict:
            keys = data.keys()
            maxkeylen = max([len(key) for key in keys])
            for key in keys:
                thisindent = ' '*(maxkeylen-len(key))
                printdata(data[key], name=key, depth=depth-1, indent=indent+thisindent, level=level)
        elif type(data) in [list, tuple]:
            for i in range(len(data)):
                printdata(data[i], name='[%i]'%i, depth=depth-1, indent=indent, level=level)
        elif type(data).__name__ in ['module', 'class']:
            keys = dir(data)
            maxkeylen = max([len(key) for key in keys])
            for key in keys:
                if key[0]!='_': # Skip these
                    thisindent = ' '*(maxkeylen-len(key))
                    printdata(getattr(data,key), name=key, depth=depth-1, indent=indent+thisindent, level=level)
        print('\n')
    return


def printvars(localvars=None, varlist=None, label=None, divider=True, spaces=1, color=None):
    '''
    Print out a list of variables. Note that the first argument must be locals().

    Args:
        localvars: function must be called with locals() as first argument
        varlist: the list of variables to print out
        label: optional label to print out, so you know where the variables came from
        divider: whether or not to offset the printout with a spacer (i.e. ------)
        spaces: how many spaces to use between variables
        color: optionally label the variable names in color so they're easier to see

    **Example**::

    >>> a = range(5)
    >>> b = 'example'
    >>> sc.printvars(locals(), ['a','b'], color='green')

    Another useful usage case is to print out the kwargs for a function:

    >>> sc.printvars(locals(), kwargs.keys())

    Version: 2017oct28
    '''

    varlist = scu.promotetolist(varlist) # Make sure it's actually a list
    dividerstr = '-'*40

    if label:  print(f'Variables for {label}:')
    if divider: print(dividerstr)
    for varnum,varname in enumerate(varlist):
        controlstr = f'{varnum}. "{varname}": ' # Basis for the control string -- variable number and name
        if color: controlstr = colorize(color, output=True) + controlstr + colorize('reset', output=True) # Optionally add color
        if spaces>1: controlstr += '\n' # Add a newline if the variables are going to be on different lines
        try:    controlstr += f'{localvars[varname]}' # The variable to be printed
        except: controlstr += 'WARNING, could not be printed' # In case something goes wrong
        controlstr += '\n' * spaces # The number of spaces to add between variables
        print(controlstr), # Print it out
    if divider: print(dividerstr) # If necessary, print the divider again
    return



#%% Color functions

__all__ += ['colorize', 'heading', 'printred', 'printyellow', 'printgreen',
            'printcyan', 'printblue', 'printmagenta']


def colorize(color=None, string=None, doprint=None, output=False, enable=True, showhelp=False, fg=None, bg=None, style=None):
    '''
    Colorize output text.

    Args:
        color (str): the color you want (use 'bg' with background colors, e.g. 'bgblue'); alternatively, use fg, bg, and style
        string (str): the text to be colored
        doprint (bool): whether to print the string (default true unless output)
        output (bool): whether to return the modified version of the string (default false)
        enable (bool): switch to allow ``sc.colorize()`` to be easily turned off without converting to a ``print()`` statement
        showhelp (bool): show help rather than changing colors

    **Examples**::

        sc.colorize('green', 'hi') # Simple example
        sc.colorize(['yellow', 'bgblack']); print('Hello world'); print('Goodbye world'); colorize() # Colorize all output in between
        bluearray = sc.colorize(color='blue', string=str(range(5)), output=True); print("c'est bleu: " + bluearray)
        sc.colorize('magenta') # Now type in magenta for a while
        sc.colorize() # Stop typing in magenta
        sc.colorize('cat in the hat', fg='#ffa044', bg='blue', style='italic+underline') # Alternate usage example

    To get available colors, type ``sc.colorize(showhelp=True)``.

    | New in version 1.3.1: "doprint" argument; ansicolors shortcut
    '''

    # Handle short-circuit case
    if not enable: # pragma: no cover
        if output:
            return string
        else:
            print(string)
            return

    # Decide which path we'll be taking
    ansistring = ''
    alt_usage = (fg is not None) or (bg is not None) or (style is not None)

    # Handle alternate usage pattern -- string as first argument rather than color, so swap
    if alt_usage:
        if string is None and color is not None:
            string, color = color, string
        if color is not None:
            errormsg = 'You can supply either color or fg, but not both'
            raise ValueError(errormsg)

        if ansi_support: ansistring = colors.color(s=string, fg=fg, bg=bg, style=style) # Actually apply color
        else:            ansistring = str(string) # Otherwise, just return the string

    # Original use case
    else:

        # Define ANSI colors
        ansicolors = co.OrderedDict([
            ('black', '30'),
            ('red', '31'),
            ('green', '32'),
            ('yellow', '33'),
            ('blue', '34'),
            ('magenta', '35'),
            ('cyan', '36'),
            ('gray', '37'),
            ('bgblack', '40'),
            ('bgred', '41'),
            ('bggreen', '42'),
            ('bgyellow', '43'),
            ('bgblue', '44'),
            ('bgmagenta', '45'),
            ('bgcyan', '46'),
            ('bggray', '47'),
            ('reset', '0'),
        ])
        for key, val in ansicolors.items():
            ansicolors[key] = '\033[' + val + 'm'

        # Determine what color to use
        colorlist = scu.promotetolist(color)  # Make sure it's a list
        for color in colorlist:
            if color not in ansicolors.keys(): # pragma: no cover
                print(f'Color "{color}" is not available, use colorize(showhelp=True) to show options.')
                return  # Don't proceed if the color isn't found
        ansicolor = ''
        for color in colorlist:
            ansicolor += ansicolors[color]

        # Modify string, if supplied
        if string is None: ansistring = ansicolor # Just return the color
        else:              ansistring = ansicolor + str(string) + ansicolors['reset'] # Add to start and end of the string
        if not ansi_support:
            ansistring = str(string) # To avoid garbling output on unsupported systems

    if showhelp:
        print('Available colors are:')
        for key in ansicolors.keys():
            if key[:2] == 'bg':
                darks = ['bgblack', 'bgred', 'bgblue', 'bgmagenta']
                if key in darks: foreground = 'gray'
                else:            foreground = 'black'
                helpcolor = [foreground, key]
            else:
                helpcolor = key
            colorize(helpcolor, '  ' + key)

    return scu._printout(string=ansistring, doprint=doprint, output=output)


# Alias certain colors functions -- not including white and black since poor practice on light/dark terminals
def printred(s, **kwargs):
    ''' Alias to print(colors.red(s)) '''
    return print(colors.red(s, **kwargs))

def printgreen(s, **kwargs):
    ''' Alias to print(colors.green(s)) '''
    return print(colors.green(s, **kwargs))

def printblue(s, **kwargs):
    ''' Alias to print(colors.blue(s)) '''
    return print(colors.blue(s, **kwargs))

def printcyan(s, **kwargs):
    ''' Alias to print(colors.cyan(s)) '''
    return print(colors.cyan(s, **kwargs))

def printyellow(s, **kwargs):
    ''' Alias to print(colors.yellow(s)) '''
    return print(colors.yellow(s, **kwargs))

def printmagenta(s, **kwargs):
    ''' Alias to print(colors.magenta(s)) '''
    return print(colors.magenta(s, **kwargs))


def heading(string=None, *args, color=None, divider=None, spaces=None, spacesafter=None, minlength=None, maxlength=None, sep=' ', doprint=None, output=False, **kwargs):
    '''
    Create a colorful heading. If just supplied with a string (or list of inputs like print()),
    create blue text with horizontal lines above and below and 3 spaces above. You
    can customize the color, the divider character, how many spaces appear before
    the heading, and the minimum length of the divider (otherwise will expand to
    match the length of the string, up to a maximum length).

    Args:
        string      (str):  the string to print as the heading (or object to convert to a string)
        args        (list): additional strings to print
        color       (str):  color to use for the heading (default cyan)
        divider     (str):  symbol to use for the divider (default '—')
        spaces      (int):  number of spaces to put before the heading (default 3)
        spacesafter (int):  number of spaces to put after the heading (default 1)
        minlength   (int):  minimum length of the divider (default 30)
        maxlength   (int):  maximum length of the divider (default 120)
        sep         (str):  if multiple arguments are supplied, use this separator to join them
        doprint     (bool): whether to print the string (default true if no output)
        output      (bool): whether to return the string as output (else, print)
        kwargs      (dict): passed to ``sc.colorize()``

    Returns:
        Formatted string if ``output=True``

    **Examples**::
        sc.heading('This is a heading')
        sc.heading(string='This is also a heading', color='red', divider='*', spaces=0, minlength=50)

    | New in version 1.3.1.: "spacesafter"
    '''
    if string      is None: string      = ''
    if color       is None: color       = 'cyan' # Reasonable default for light and dark consoles
    if divider     is None: divider     = '—' # Em dash for a continuous line
    if spaces      is None: spaces      = 3
    if spacesafter is None: spacesafter = 1
    if minlength   is None: minlength   = 30
    if maxlength   is None: maxlength   = 120

    # Convert to single string
    args = scu.mergelists(string, list(args))
    string = sep.join([str(item) for item in args])

    # Add header and footer
    length      = int(np.median([minlength, len(string), maxlength]))
    space       = '\n'*spaces
    spaceafter  = '\n'*spacesafter
    fulldivider = divider*length
    if fulldivider:
        string = scu.newlinejoin(fulldivider, string, fulldivider)
    fullstring = space + string + spaceafter

    # Create output
    return colorize(color=color, string=fullstring, doprint=doprint, output=output, **kwargs)



#%% Other

__all__ += ['printv', 'slacknotification', 'printtologfile', 'percentcomplete', 'progressbar', 'capture']


def printv(string, thisverbose=1, verbose=2, newline=True, indent=True):
    '''
    Optionally print a message and automatically indent. The idea is that
    a global or shared "verbose" variable is defined, which is passed to
    subfunctions, determining how much detail to print out.

    The general idea is that verbose is an integer from 0-4 as follows:

    * 0 = no printout whatsoever
    * 1 = only essential warnings, e.g. suppressed exceptions
    * 2 = standard printout
    * 3 = extra debugging detail (e.g., printout on each iteration)
    * 4 = everything possible (e.g., printout on each timestep)

    Thus a very important statement might be e.g.

    >>> sc.printv('WARNING, everything is wrong', 1, verbose)

    whereas a much less important message might be

    >>> sc.printv(f'This is timestep {i}', 4, verbose)

    Version: 2016jan30
    '''
    if thisverbose>4 or verbose>4: print(f'Warning, verbosity should be from 0-4 (this message: {thisverbose}; current: {verbose})')
    if verbose>=thisverbose: # Only print if sufficiently verbose
        indents = '  '*thisverbose*bool(indent) # Create automatic indenting
        if newline: print(indents+scu.flexstr(string)) # Actually print
        else: print(indents+scu.flexstr(string)), # Actually print
    return









def slacknotification(message=None, webhook=None, to=None, fromuser=None, verbose=2, die=False):  # pragma: no cover
    '''
    Send a Slack notification when something is finished.

    The webhook is either a string containing the webhook itself, or a plain text file containing
    a single line which is the Slack webhook. By default it will look for the file
    ".slackurl" in the user's home folder. The webhook needs to look something like
    "https://hooks.slack.com/services/af7d8w7f/sfd7df9sb/lkcpfj6kf93ds3gj". Webhooks are
    effectively passwords and must be kept secure! Alternatively, you can specify the webhook
    in the environment variable SLACKURL.

    Args:
        message (str): The message to be posted.
        webhook (str): See above
        to (str): The Slack channel or user to post to. Channels begin with #, while users begin with @ (note: ignored by new-style webhooks)
        fromuser (str): The pseudo-user the message will appear from (note: ignored by new-style webhooks)
        verbose (bool): How much detail to display.
        die (bool): If false, prints warnings. If true, raises exceptions.

    **Example**::

        sc.slacknotification('Long process is finished')
        sc.slacknotification(webhook='/.slackurl', channel='@username', message='Hi, how are you going?')

    What's the point? Add this to the end of a very long-running script to notify
    your loved ones that the script has finished.

    Version: 2018sep25
    '''
    try:
        from requests import post # Simple way of posting data to a URL
        from json import dumps # For sanitizing the message
    except Exception as E:
        errormsg = f'Cannot use Slack notification since imports failed: {str(E)}'
        if die: raise ImportError(errormsg)
        else:   print(errormsg)

    # Validate input arguments
    printv('Sending Slack message', 1, verbose)
    if not webhook:  webhook    = os.path.expanduser('~/.slackurl')
    if not to:       to       = '#general'
    if not fromuser: fromuser = 'sciris-bot'
    if not message:  message  = 'This is an automated notification: your notifier is notifying you.'
    printv(f'Channel: {to} | User: {fromuser} | Message: {message}', 3, verbose) # Print details of what's being sent

    # Try opening webhook as a file
    if webhook.find('hooks.slack.com')>=0: # It seems to be a URL, let's proceed
        slackurl = webhook
    elif os.path.exists(os.path.expanduser(webhook)): # If not, look for it as a file
        with open(os.path.expanduser(webhook)) as f: slackurl = f.read()
    elif os.getenv('SLACKURL'): # See if it's set in the user's environment variables
        slackurl = os.getenv('SLACKURL')
    else:
        slackurl = webhook # It doesn't seemt to be a URL but let's try anyway
        errormsg = f'"{webhook}" does not seem to be a valid webhook string or file'
        if die: raise ValueError(errormsg)
        else:   print(errormsg)

    # Package and post payload
    try:
        payload = '{"text": %s, "channel": %s, "username": %s}' % (dumps(message), dumps(to), dumps(fromuser))
        printv(f'Full payload: {payload}', 4, verbose)
        response = post(url=slackurl, data=payload)
        printv(response, 3, verbose) # Optionally print response
        printv('Message sent.', 2, verbose) # We're done
    except Exception as E:
        errormsg = f'Sending of Slack message failed: {repr(E)}'
        if die: raise RuntimeError(errormsg)
        else:   print(errormsg)
    return


def printtologfile(message=None, filename=None):
    '''
    Append a message string to a file specified by a filename name/path.

    Note: in almost all cases, you are better off using Python's built-in logging
    system rather than this function.
    '''

    # Set defaults
    if message is None:
        return # Return immediately if nothing to append
    if filename is None:
        import tempfile
        tempdir = tempfile.gettempdir()
        filename = os.path.join(tempdir, 'logfile') # Some generic filename that should work on *nix systems

    # Try writing to file
    try:
        with open(filename, 'a') as f:
            f.write('\n'+message+'\n') # Add a newline to the message.
    except Exception as E: # pragma: no cover # Fail gracefully
        print(f'Warning, could not write to logfile {filename}: {str(E)}')

    return





def percentcomplete(step=None, maxsteps=None, stepsize=1, prefix=None):
    '''
    Display progress.

    **Example**::

        maxiters = 500
        for i in range(maxiters):
            sc.percentcomplete(i, maxiters) # will print on every 5th iteration
            sc.percentcomplete(i, maxiters, stepsize=10) # will print on every 50th iteration
            sc.percentcomplete(i, maxiters, prefix='Completeness: ') # will print e.g. 'Completeness: 1%'
    '''
    if prefix is None:
        prefix = ' '
    elif scu.isnumber(prefix):
        prefix = ' '*prefix
    onepercent = max(stepsize,round(maxsteps/100*stepsize)); # Calculate how big a single step is -- not smaller than 1
    if not step%onepercent: # Does this value lie on a percent
        thispercent = round(step/maxsteps*100) # Calculate what percent it is
        print(prefix + '%i%%'% thispercent) # Display the output
    return


def progressbar(i, maxiters, label='', every=1, length=30, empty='—', full='•', newline=False):
    '''
    Call in a loop to create terminal progress bar.

    Args:
        i        (int): current iteration
        maxiters (int): maximum number of iterations (can also use an object with length)
        label    (str): initial label to print
        every    (int/float): if int, print every "every"th iteration (if 1, print all); if float and <1, print every maxiters*every iteration
        length   (int): length of progress bar
        empty    (str): character for not-yet-completed steps
        full     (str): character for completed steps
        newline  (str): character to print at the end of the line (default none)

    **Examples**::

        for i in range(20):
            sc.progressbar(i+1, 20)
            pl.pause(0.05)

        x = np.arange(100)
        for i in x:
            sc.progressbar(i+1, x, every=0.1)
            pl.pause(0.01)

    Adapted from example by Greenstick (https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console)

    New in version 1.3.3: "every" argument
    '''
    # Handle inputs
    if hasattr(maxiters, '__len__'):
        maxiters = len(maxiters)
    ending = None if newline else '\r'
    if every < 1:
        every = max(1, int(every*maxiters)) # Don't let it go below 1

    # Calculate percent and handle zero case
    if maxiters > 0:
        pct = i/maxiters*100
    else:
        i = 1
        maxiters = 1
        pct = 100
    percent = f'{pct:0.0f}%'

    # Assemble string
    filled = int(length*i//maxiters)
    bar = full*filled + empty*(length-filled)

    # Print
    lastiter = (i == maxiters)
    if not(i%every) or lastiter:
        print(f'\r{label} {bar} {percent}', end=ending)
        if lastiter:
            print() # Newline at the end

    return


class capture(UserString, str, redirect_stdout):
    '''
    Captures stdout (e.g., from ``print()``) as a variable.

    Based on ``contextlib.redirect_stdout``, but saves the user the trouble of
    defining and reading from an IO stream. Useful for testing the output of functions
    that are supposed to print certain output.

    **Examples**::

        # Using with...as
        with sc.capture() as txt1:
            print('Assign these lines')
            print('to a variable')

        # Using start()...stop()
        txt2 = sc.capture().start()
        print('This works')
        print('the same way')
        txt2.stop()

        print('txt1:')
        print(txt1)
        print('txt2:')
        print(txt2)

    New in version 1.3.3.
    '''

    def __init__(self, seq='', *args, **kwargs):
        self._io = io.StringIO()
        UserString.__init__(self, seq=seq, *args, **kwargs)
        redirect_stdout.__init__(self, self._io)
        return

    def __enter__(self, *args, **kwargs):
        redirect_stdout.__enter__(self, *args, **kwargs)
        return self

    def __exit__(self, *args, **kwargs):
        self.data += self._io.getvalue()
        redirect_stdout.__exit__(self, *args, **kwargs)
        return

    def start(self):
        self.__enter__()
        return self

    def stop(self):
        self.__exit__(None, None, None)
        return