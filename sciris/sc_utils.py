##############################################################################
### IMPORTS FROM OTHER LIBRARIES
##############################################################################

import os
import sys
import six
import copy
import time
import json
import pprint
import hashlib
import datetime
import dateutil
import subprocess
import numbers
import numpy as np
import uuid as py_uuid
from textwrap import fill
from functools import reduce
from collections import OrderedDict as OD
from distutils.version import LooseVersion

# Handle types and Python 2/3 compatibility
if six.PY2: 
    _stringtypes = (basestring,)
    import urllib2 as urlrequester
    import cgi as htmlencoder
    import HTMLParser
    htmldecoder = HTMLParser.HTMLParser() # Old method, have to define an instance
else:       
    _stringtypes = (str, bytes)
    import urllib.request as urlrequester
    import html as htmlencoder
    htmldecoder = htmlencoder # New method, these are the same now
_numtype    = numbers.Number

# Add Windows support for colors (do this at the module level so that colorama.init() only gets called once)
if 'win' in sys.platform:
    try:
        import colorama
        colorama.init()
        ansi_support = True
    except:
        ansi_support = False  # print('Warning: you have called colorize() on Windows but do not have either the colorama or tendo modules.')
else:
    ansi_support = True


# Define the modules being loaded
__all__ = ['uuid', 'dcp', 'cp', 'pp', 'sha', 'wget', 'htmlify']


def uuid(uid=None, which=None, die=False, tostring=False):
    ''' Shortcut for creating a UUID; default is to create a UUID4. Can also convert a UUID. '''
    if which is None: which = 4
    if   which==1: uuid_func = py_uuid.uuid1
    elif which==3: uuid_func = py_uuid.uuid3
    elif which==4: uuid_func = py_uuid.uuid4
    elif which==5: uuid_func = py_uuid.uuid5
    else: raise Exception('UUID type %i not recognized; must be 1,  3, 4 [default], or 5)' % which)
    
    if uid is None: # If not supplied, create a new UUID
        output = uuid_func()
    else: # Otherwise, try converting it
        try:
            if isinstance(uid, py_uuid.UUID):
                output = uid # Use directly
            else: # Convert
                output = py_uuid.UUID(uid)
        except Exception as E: 
            errormsg = 'Could not convert "%s" to a UID (%s)' % (uid, repr(E))
            if die:
                raise Exception(errormsg)
            else:
                print(errormsg)
                output = uuid_func() # Just create a new one
    
    if tostring: output = str(output)
    return output


def dcp(obj, verbose=True, die=False):
    ''' Shortcut to perform a deep copy operation '''
    try:
        output = copy.deepcopy(obj)
    except Exception as E:
        output = cp(obj)
        errormsg = 'Warning: could not perform deep copy, performing shallow instead: %s' % str(E)
        if die: raise Exception(errormsg)
        else:   print(errormsg)
    return output


def cp(obj, verbose=True, die=True):
    ''' Shortcut to perform a shallow copy operation '''
    try:
        output = copy.copy(obj)
    except Exception as E:
        output = obj
        errormsg = 'Warning: could not perform shallow copy, returning original object: %s' % str(E)
        if die: raise Exception(errormsg)
        else:   print(errormsg)
    return output


def pp(obj, jsonify=True, verbose=False, doprint=True, *args, **kwargs):
    ''' Shortcut for pretty-printing the object '''
    # Get object
    if jsonify:
        try:
            toprint = json.loads(json.dumps(obj)) # This is to handle things like OrderedDicts
        except Exception as E:
            if verbose: print('Could not jsonify object ("%s"), printing default...' % str(E))
            toprint = obj # If problems are encountered, just return the object
    else:
        toprint = obj
    
    # Decide what to do with object
    if doprint:
        pprint.pprint(toprint, *args, **kwargs)
        return None
    else:
        output = pprint.pformat(toprint, *args, **kwargs)
        return output


def sha(string, encoding='utf-8', *args, **kwargs):
    ''' Shortcut for the standard hashing (SHA) method '''
    if not isstring(string): # Ensure it's actually a string
        string = str(string)
    needsencoding = (six.PY2 and isinstance(string, unicode)) or (six.PY3 and isinstance(string, str))
    if needsencoding: # If it's unicode, encode it to bytes first
        string = string.encode(encoding)
    output = hashlib.sha224(string, *args, **kwargs)
    return output


def wget(url, convert=True):
    ''' Download a URL '''
    output = urlrequester.urlopen(url).read()
    if convert and six.PY3: output = output.decode()
    return output
    

def htmlify(string, reverse=False, tostring=False):
    '''
    Convert a string to its HTML representation by converting unicode characters,
    characters that need to be escaped, and newlines. If reverse=True, will convert
    HTML to string. If tostring=True, will convert the bytestring back to Unicode.
    
    Examples:
        output = sc.htmlify('foo&\nbar') # Returns b'foo&amp;<br>bar'
        output = sc.htmlify('foo&\nbar', tostring=True) # Returns 'foo&amp;<br>bar'
        output = sc.htmlify('foo&amp;<br>bar', reverse=True) # Returns 'foo&\nbar'
    '''
    if not reverse: # Convert to HTML
        output = htmlencoder.escape(string).encode('ascii', 'xmlcharrefreplace') # Replace non-ASCII characters
        output = output.replace(b'\n',b'<br>') # Replace newlines with <br>
        if tostring: # Convert from bytestring to unicode
            output = output.decode()
    else: # Convert from HTML
        output = htmldecoder.unescape(string)
        output = output.replace('<br>','\n').replace('<BR>','\n')
    return output


##############################################################################
### PRINTING/NOTIFICATION FUNCTIONS
##############################################################################

__all__ += ['printv', 'blank', 'createcollist', 'objectid', 'objatt', 'objmeth', 'objrepr']
__all__ += ['prepr', 'pr', 'indent', 'sigfig', 'printarr', 'printdata', 'printvars', 'getdate']
__all__ += ['slacknotification', 'printtologfile', 'colorize']

def printv(string, thisverbose=1, verbose=2, newline=True, indent=True):
    '''
    Optionally print a message and automatically indent. The idea is that
    a global or shared "verbose" variable is defined, which is passed to
    subfunctions, determining how much detail to print out.

    The general idea is that verbose is an integer from 0-4 as follows:
        0 = no printout whatsoever
        1 = only essential warnings, e.g. suppressed exceptions
        2 = standard printout
        3 = extra debugging detail (e.g., printout on each iteration)
        4 = everything possible (e.g., printout on each timestep)
    
    Thus a very important statement might be e.g.
        printv('WARNING, everything is wrong', 1, verbose)

    whereas a much less important message might be
        printv('This is timestep %i' % i, 4, verbose)

    Version: 2016jan30
    '''
    if thisverbose>4 or verbose>4: print('Warning, verbosity should be from 0-4 (this message: %i; current: %i)' % (thisverbose, verbose))
    if verbose>=thisverbose: # Only print if sufficiently verbose
        indents = '  '*thisverbose*bool(indent) # Create automatic indenting
        if newline: print(indents+flexstr(string)) # Actually print
        else: print(indents+flexstr(string)), # Actually print
    return None


def blank(n=3):
    ''' Tiny function to print n blank lines, 3 by default '''
    print('\n'*n)


def createcollist(oldkeys, title, strlen = 18, ncol = 3):
    ''' Creates a string for a nice columnated list (e.g. to use in __repr__ method) '''
    nrow = int(np.ceil(float(len(oldkeys))/ncol))
    newkeys = []
    for x in range(nrow):
        newkeys += oldkeys[x::nrow]
    
    attstring = title + ':'
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
    output = '<%s.%s at %s>\n' % (obj.__class__.__module__, obj.__class__.__name__, hex(id(obj)))
    return output


def objatt(obj, strlen=18, ncol=3):
    ''' Return a sorted string of object attributes for the Python __repr__ method '''
    oldkeys = sorted(obj.__dict__.keys())
    if len(oldkeys): output = createcollist(oldkeys, 'Attributes', strlen = 18, ncol = 3)
    else:            output = 'No attributes\n'
    return output


def objmeth(obj, strlen=18, ncol=3):
    ''' Return a sorted string of object methods for the Python __repr__ method '''
    oldkeys = sorted([method + '()' for method in dir(obj) if callable(getattr(obj, method)) and not method.startswith('__')])
    if len(oldkeys): output = createcollist(oldkeys, 'Methods', strlen=strlen, ncol=ncol)
    else:            output = 'No methods\n'
    return output


def objrepr(obj, showid=True, showmeth=True, showatt=True):
    ''' Return useful printout for the Python __repr__ method '''
    divider = '============================================================\n'
    output = ''
    if showid:
        output += objectid(obj)
        output += divider
    if showmeth:
        output += objmeth(obj)
        output += divider
    if showatt:
        output += objatt(obj)
        output += divider
    return output


def prepr(obj, maxlen=None, skip=None):
    ''' 
    Akin to "pretty print", returns a pretty representation of an object -- 
    all attributes (except any that are skipped), plust methods and ID.
    '''
    
    # Handle input arguments
    if maxlen is None: maxlen = 80
    if skip   is None: skip   = []
    else:              skip = promotetolist(skip)
    
    # Initialize things to print out
    labels = []
    values = []
    if hasattr(obj, '__dict__'):
        if len(obj.__dict__):
            labels = sorted(set(obj.__dict__.keys()) - set(skip)) # Get the attribute keys
            values = [flexstr(getattr(obj, attr)) for attr in labels] # Get the string representation of the attribute
        else:
            items = dir(obj)
            for attr in items:
                if not attr.startswith('__'):
                    try:    value = flexstr(getattr(obj, attr))
                    except: value = 'N/A'
                    labels.append(attr)
                    values.append(value)
    else: # If it's not an object, just get its representation
        labels = ['%s' % type(obj)]
        values = [flexstr(obj)]
    
    # Decide how to print them
    maxkeylen = 0
    if len(labels):  
        maxkeylen = max([len(label) for label in labels]) # Find the maximum length of the attribute keys
    if maxkeylen<maxlen: 
        maxlen = maxlen - maxkeylen # Shorten the amount of data shown if the keys are long
    formatstr = '%'+ '%i'%maxkeylen + 's' # Assemble the format string for the keys, e.g. '%21s'
    output  = objrepr(obj, showatt=False) # Get the methods
    for label,value in zip(labels,values): # Loop over each attribute
        if len(value)>maxlen: value = value[:maxlen] + ' [...]' # Shorten it
        prefix = formatstr%label + ': ' # The format key
        output += indent(prefix, value)
    output += '============================================================\n'
    return output


def pr(obj, maxlen=None):
    ''' Shortcut for printing the pretty repr for an object -- comparable to sc.pp() '''
    print(prepr(obj, maxlen=maxlen))
    return None

    
def indent(prefix=None, text=None, suffix='\n', n=0, pretty=False, simple=True, width=70, **kwargs):
    '''
    Small wrapper to make textwrap more user friendly.
    
    Arguments:
        prefix = text to begin with (optional)
        text = text to wrap
        suffix = what to put on the end (by default, a newline)
        n = if prefix is not specified, the size of the indent
        prettify = whether to use pprint to format the text
        kwargs = anything to pass to textwrap.fill() (e.g., linewidth)
    
    Examples:
        prefix = 'and then they said:'
        text = 'blah '*100
        print(indent(prefix, text))
        
        print('my fave is: ' + indent(text=rand(100), n=14))
    
    Version: 2017feb20
    '''
    # Handle no prefix
    if prefix is None: prefix = ' '*n
    
    # Get text in the right format -- i.e. a string
    if pretty: text = pprint.pformat(text)
    else:      text = flexstr(text)

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
    


def sigfig(X, sigfigs=5, SI=False, sep=False, keepints=False):
    '''
    Return a string representation of variable x with sigfigs number of significant figures -- 
    copied from asd.py.
    
    If SI=True,  then will return e.g. 32433 as 32.433K
    If sep=True, then will return e.g. 32433 as 32,433
    '''
    output = []
    
    try: 
        n=len(X)
        islist = True
    except:
        X = [X]
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
                output.append(flexstr(x)+suffix)
            elif x>(10**sigfigs) and not SI and keepints: # e.g. x = 23432.23, sigfigs=3, output is 23432
                roundnumber = int(round(x))
                if sep: string = format(roundnumber, ',')
                else:   string = '%0.0f' % x
                output.append(string)
            else:
                magnitude = np.floor(np.log10(abs(x)))
                factor = 10**(sigfigs-magnitude-1)
                x = round(x*factor)/float(factor)
                digits = int(abs(magnitude) + max(0, sigfigs - max(0,magnitude) - 1) + 1 + (x<0) + (abs(x)<1)) # one because, one for decimal, one for minus
                decimals = int(max(0,-magnitude+sigfigs-1))
                strformat = '%' + '%i.%i' % (digits, decimals)  + 'f'
                string = strformat % x
                if sep: # To insert separators in the right place, have to convert back to a number
                    if decimals>0:  roundnumber = float(string)
                    else:           roundnumber = int(string)
                    string = format(roundnumber, ',') # Allow comma separator
                string += suffix
                output.append(string)
        except:
            output.append(flexstr(x))
    if islist:
        return tuple(output)
    else:
        return output[0]



def printarr(arr, arrformat='%0.2f  '):
    ''' 
    Print a numpy array nicely.
    
    Example:
        from utils import printarr
        from numpy import random
        printarr(rand(3,7,4))
    
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
    else:
        print(arr) # Give up
    return None



def printdata(data, name='Variable', depth=1, maxlen=40, indent='', level=0, showcontents=False):
    '''
    Nicely print a complicated data structure, a la Matlab.
    Arguments:
      data: the data to display
      name: the name of the variable (automatically read except for first one)
      depth: how many levels of recursion to follow
      maxlen: number of characters of data to display (if 0, don't show data)
      indent: where to start the indent (used internally)
    
    Version: 1.0 (2015aug21)    
    '''
    datatype = type(data)
    def printentry(data):
        if   datatype==dict:              string = ('dict with %i keys' % len(data.keys()))
        elif datatype==list:              string = ('list of length %i' % len(data))
        elif datatype==tuple:             string = ('tuple of length %i' % len(data))
        elif datatype==np.ndarray:        string = ('array of shape %s' % flexstr(np.shape(data)))
        elif datatype.__name__=='module': string = ('module with %i components' % len(dir(data)))
        elif datatype.__name__=='class':  string = ('class with %i components' % len(dir(data)))
        else: string = datatype.__name__
        if showcontents and maxlen>0:
            datastring = ' | '+flexstr(data)
            if len(datastring)>maxlen: datastring = datastring[:maxlen] + ' <etc> ' + datastring[-maxlen:]
        else: datastring=''
        return string+datastring
    
    string = printentry(data).replace('\n',' \ ') # Remove newlines
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
    return None


def printvars(localvars=None, varlist=None, label=None, divider=True, spaces=1, color=None):
    '''
    Print out a list of variables. Note that the first argument must be locals().
    
    Arguments:
        localvars = function must be called with locals() as first argument
        varlist = the list of variables to print out
        label = optional label to print out, so you know where the variables came from
        divider = whether or not to offset the printout with a spacer (i.e. ------)
        spaces = how many spaces to use between variables
        color = optionally label the variable names in color so they're easier to see
    
    Simple usage example:
        a = range(5); b = 'example'; printvars(locals(), ['a','b'], color='blue')
    
    Another useful usage case is to print out the kwargs for a function:
        printvars(locals(), kwargs.keys())
        
    Version: 2017oct28
    '''
    
    varlist = promotetolist(varlist) # Make sure it's actually a list
    dividerstr = '-'*40
    
    if label:  print('Variables for %s:' % label)
    if divider: print(dividerstr)
    for varnum,varname in enumerate(varlist):
        controlstr = '%i. "%s": ' % (varnum, varname) # Basis for the control string -- variable number and name
        if color: controlstr = colorize(color, output=True) + controlstr + colorize('reset', output=True) # Optionally add color
        if spaces>1: controlstr += '\n' # Add a newline if the variables are going to be on different lines
        try:    controlstr += '%s' % localvars[varname] # The variable to be printed
        except: controlstr += 'WARNING, could not be printed' # In case something goes wrong
        controlstr += '\n' * spaces # The number of spaces to add between variables
        print(controlstr), # Print it out
    if divider: print(dividerstr) # If necessary, print the divider again
    return None



def slacknotification(message=None, webhook=None, to=None, fromuser=None, verbose=2, die=False):
    ''' 
    Send a Slack notification when something is finished.
    
    Arguments:
        message:
            The message to be posted.
        webhook:
            This is either a string containing the webhook itself, or a plain text file containing 
            a single line which is the Slack webhook. By default it will look for the file
            ".slackurl" in the user's home folder. The webhook needs to look something like
            "https://hooks.slack.com/services/af7d8w7f/sfd7df9sb/lkcpfj6kf93ds3gj". Webhooks are 
            effectively passwords and must be kept secure! Alternatively, you can specify the webhook
            in the environment variable SLACKURL.
        to (WARNING: ignored by new-style webhooks):
            The Slack channel or user to post to. Channels begin with #, while users begin with @.
        fromuser (WARNING: ignored by new-style webhooks):
            The pseudo-user the message will appear from.
        verbose:
            How much detail to display.
        die:
            If false, prints warnings. If true, raises exceptions.
    
    Example usage:
        slacknotification('Long process is finished')
        slacknotification(webhook='/.slackurl', channel='@username', message='Hi, how are you going?')
    
    What's the point? Add this to the end of a very long-running script to notify
    your loved ones that the script has finished.
        
    Version: 2018sep25
    '''
    try:
        from requests import post # Simple way of posting data to a URL
        from json import dumps # For sanitizing the message
    except Exception as E:
        errormsg = 'Cannot use Slack notification since imports failed: %s' % str(E)
        if die: raise Exception(errormsg)
        else:   print(errormsg)
        
    # Validate input arguments
    printv('Sending Slack message', 1, verbose)
    if not webhook:  webhook    = os.path.expanduser('~/.slackurl')
    if not to:       to       = '#general'
    if not fromuser: fromuser = 'sciris-bot'
    if not message:  message  = 'This is an automated notification: your notifier is notifying you.'
    printv('Channel: %s | User: %s | Message: %s' % (to, fromuser, message), 3, verbose) # Print details of what's being sent
    
    # Try opening webhook as a file
    if webhook.find('hooks.slack.com')>=0: # It seems to be a URL, let's proceed
        slackurl = webhook
    elif os.path.exists(os.path.expanduser(webhook)): # If not, look for it sa a file
        with open(os.path.expanduser(webhook)) as f: slackurl = f.read()
    elif os.getenv('SLACKURL'): # See if it's set in the user's environment variables
        slackurl = os.getenv('SLACKURL')
    else:
        slackurl = webhook # It doesn't seemt to be a URL but let's try anyway
        errormsg = '"%s" does not seem to be a valid webhook string or file' % webhook
        if die: raise Exception(errormsg)
        else:   print(errormsg)
    
    # Package and post payload
    try:
        payload = '{"text": %s, "channel": %s, "username": %s}' % (dumps(message), dumps(to), dumps(fromuser))
        printv('Full payload: %s' % payload, 4, verbose)
        response = post(url=slackurl, data=payload)
        printv(response, 3, verbose) # Optionally print response
        printv('Message sent.', 2, verbose) # We're done
    except Exception as E:
        errormsg = 'Sending of Slack message failed: %s' % repr(E)
        if die: raise Exception(errormsg)
        else:   print(errormsg)
    return None


def printtologfile(message=None, filename=None):
    '''
    Append a message string to a file specified by a filename name/path.  This 
    is especially useful for capturing information from spawned processes not 
    so handily captured through print statements.
    Warning: If you pass a file in, existing or not, it will try to append
    text to it!
    '''
    
    # Set defaults
    if message is None:  
        return None # Return immediately if nothing to append
    if filename is None: 
        filename = '/tmp/logfile' # Some generic filename that should work on *nix systems
    
    # Try writing to file
    try:
        with open(filename, 'a') as f:
            f.write('\n'+message+'\n') # Add a newline to the message.
    except: # Fail gracefully
        print('WARNING, could not write to logfile %s' % filename)
    
    return None


def colorize(color=None, string=None, output=False, showhelp=False, enable=True):
    '''
    Colorize output text. Arguments:
        color = the color you want (use 'bg' with background colors, e.g. 'bgblue')
        string = the text to be colored
        output = whether to return the modified version of the string
        enable = switch to allow colorize() to be easily turned off

    Examples:
        colorize('green', 'hi') # Simple example
        colorize(['yellow', 'bgblack']); print('Hello world'); print('Goodbye world'); colorize() # Colorize all output in between
        bluearray = colorize(color='blue', string=str(range(5)), output=True); print("c'est bleu: " + bluearray)
        colorize('magenta') # Now type in magenta for a while
        colorize() # Stop typing in magenta

    To get available colors, type colorize(showhelp=True).

    Version: 2018sep09
    '''

    # Handle short-circuit case
    if not enable:
        if output:
            return string
        else:
            print(string)
            return None

    # Define ANSI colors
    ansicolors = OD([
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
    for key, val in ansicolors.items(): ansicolors[key] = '\033[' + val + 'm'

    # Determine what color to use
    colorlist = promotetolist(color)  # Make sure it's a list
    for color in colorlist:
        if color not in ansicolors.keys():
            print('Color "%s" is not available, use colorize(showhelp=True) to show options.' % color)
            return None  # Don't proceed if the color isn't found
    ansicolor = ''
    for color in colorlist:
        ansicolor += ansicolors[color]

    # Modify string, if supplied
    if string is None: ansistring = ansicolor # Just return the color
    else:              ansistring = ansicolor + str(string) + ansicolors['reset'] # Add to start and end of the string
    if not ansi_support: ansistring = str(string) # To avoid garbling output on unsupported systems

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
    elif output:
        return ansistring  # Return the modified string
    else:
        try:    print(ansistring) # Content, so print with newline
        except: print(string) # If that fails, just go with plain version
        return None
    


        
    
##############################################################################
### TYPE FUNCTIONS
##############################################################################

__all__ += ['flexstr', 'isiterable', 'checktype', 'isnumber', 'isstring', 'promotetoarray', 'promotetolist']

def flexstr(arg, force=True):
    ''' Try converting to a "regular" string (i.e. "str" in both Python 2 or 3), but proceed if it fails '''
    if isstring(arg): # It's a string
        if six.PY2:
            try:    
                output = str(arg) # Try to convert to ASCII string from unicode
            except: 
                output = arg # If anything goes wrong, just return as-is
        else:
            if isinstance(arg, six.binary_type): 
                try:
                    output = arg.decode() # If it's bytes, decode to unicode
                except:
                    if force: output = repr(arg) # If that fails, just print its representation
                    else:     output = arg
            else: 
                output = arg # Otherwise, return as-is
    else:
        if force: output = repr(arg)
        else:     output = arg # Optionally don't do anything for non-strings
    return output


def isiterable(obj):
    '''
    Simply determine whether or not the input is iterable, since it's too hard to remember this otherwise.
    From http://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable
    '''
    try:
        iter(obj)
        return True
    except:
        return False
    

def checktype(obj=None, objtype=None, subtype=None, die=False):
    ''' 
    A convenience function for checking instances. If objtype is a type,
    then this function works exactly like isinstance(). But, it can also
    be a string, e.g. 'array'.
    
    If subtype is not None, then checktype will iterate over obj and check
    recursively that each element matches the subtype.
    
    Special types are "listlike", which will check for lists, tuples, and
    arrays; and "arraylike", which is the same as "listlike" but will also
    check that elements are numeric.
    
    Arguments:
        obj     = the object to check the type of
        objtype = the type to confirm the object belongs to
        subtype = optionally check the subtype if the object is iterable
        die     = whether or not to raise an exception if the object is the wrong type.
    
    Examples:
        checktype(rand(10), 'array', 'number') # Returns True
        checktype(['a','b','c'], 'listlike') # Returns True
        checktype(['a','b','c'], 'arraylike') # Returns False
        checktype([{'a':3}], list, dict) # Returns True
    '''
    
    # Handle "objtype" input
    if   objtype in ['str','string']:          objinstance = _stringtypes
    elif objtype in ['num', 'number']:         objinstance = _numtype
    elif objtype in ['arr', 'array']:          objinstance = type(np.array([]))
    elif objtype in ['listlike', 'arraylike']: objinstance = (list, tuple, type(np.array([]))) # Anything suitable as a numerical array
    elif type(objtype)==type:                  objinstance = objtype  # Don't need to do anything
    elif objtype is None:                      return None # If not supplied, exit
    else:
        errormsg = 'Could not understand what type you want to check: should be either a string or a type, not "%s"' % objtype
        raise Exception(errormsg)
    
    # Do first-round checking
    result = isinstance(obj, objinstance)
    
    # Do second round checking
    if result and objtype in ['listlike', 'arraylike']: # Special case for handling arrays which may be multi-dimensional
        obj = promotetoarray(obj).flatten() # Flatten all elements
        if objtype == 'arraylike' and subtype is None: subtype = 'number'
    if isiterable(obj) and subtype is not None:
        for item in obj:
            result = result and checktype(item, subtype)

    # Decide what to do with the information thus gleaned
    if die: # Either raise an exception or do nothing if die is True
        if not result: # It's not an instance
            errormsg = 'Incorrect type: object is %s, but %s is required' % (type(obj), objtype)
            raise Exception(errormsg)
        else:
            return None # It's fine, do nothing
    else: # Return the result of the comparison
        return result
   
         
def isnumber(obj, isnan=None):
    ''' Simply determine whether or not the input is a number, since it's too hard to remember this otherwise '''
    output = checktype(obj, 'number')
    if output and isnan is not None: # It is a number, so can check for nan
        output = (np.isnan(obj) == isnan) # See if they match
    return output
    
    
def isstring(obj):
    ''' Simply determine whether or not the input is a string, since it's too hard to remember this otherwise '''
    return checktype(obj, 'string')


def promotetoarray(x):
    ''' Small function to ensure consistent format for things that should be arrays '''
    if isnumber(x):
        return np.array([x]) # e.g. 3
    elif isinstance(x, (list, tuple)):
        return np.array(x) # e.g. [3]
    elif isinstance(x, np.ndarray): 
        if np.shape(x):
            return x # e.g. array([3])
        else: 
            return np.array([x]) # e.g. array(3)
    else: # e.g. 'foo'
        raise Exception("Expecting a number/list/tuple/ndarray; got: %s" % flexstr(x))


def promotetolist(obj=None, objtype=None, keepnone=False):
    '''
    Make sure object is iterable -- used so functions can handle inputs like 'a' or ['a', 'b'].
    
    If keepnone is false, then None is converted to an empty list. Otherwise, it's converted to
    [None].
    
    Version: 2018aug24
    '''
    isnone = False
    if not isinstance(obj, list):
        if obj is None and not keepnone:
            obj = []
            isnone = True
        else:
            obj = [obj] # Main usage case -- listify it
    if objtype is not None and not isnone:  # Check that the types match -- now that we know it's a list, we can iterate over it
        for item in obj:
            checktype(obj=item, objtype=objtype, die=True)
    return obj











##############################################################################
### MISC. FUNCTIONS
##############################################################################

__all__ += ['now', 'tic', 'toc', 'fixedpause', 'percentcomplete', 'checkmem', 'runcommand', 'gitinfo', 'compareversions', 'uniquename', 'importbyname']

def now(utc=False, die=False, tostring=False, fmt=None):
    ''' Get the current time, optionally in UTC time '''
    if utc: tzinfo = dateutil.tz.tzutc()
    else:   tzinfo = None
    timenow = datetime.datetime.now(tzinfo)
    if tostring: output = getdate(timenow)
    else:        output = timenow
    return output
    
    

def getdate(obj=None, fmt='str', dateformat=None):
        ''' Return either the date created or modified ("which") as either a str or int ("fmt") '''
        if obj is None:
            obj = now()
        
        if dateformat is None:
            dateformat = '%Y-%b-%d %H:%M:%S'
        
        try:
            if isstring(obj): return obj # Return directly if it's a string
            obj.timetuple() # Try something that will only work if it's a date object
            dateobj = obj # Test passed: it's a date object
        except Exception as E: # It's not a date object
            raise Exception('Getting date failed; date must be a string or a date object: %s' % repr(E))
        
        if fmt=='str':
            dateformat = '%Y-%m-%d %H:%M:%S'
            output = dateobj.strftime(dateformat)
            return output
        elif fmt=='int': 
            output = time.mktime(dateobj.timetuple()) # So ugly!! But it works -- return integer representation of time
            return output
        else:
            errormsg = '"fmt=%s" not understood; must be "str" or "int"' % fmt
            raise Exception(errormsg)
        return None # Should not be possible to get to this point



def tic():
    '''
    A little pair of functions to calculate a time difference, sort of like Matlab:
    tic() [but you can also use the form t = tic()]
    toc() [but you can also use the form toc(t) where to is the output of tic()]
    '''
    global _tictime  # The saved time is stored in this global.    
    _tictime = time.time()  # Store the present time in the global.
    return _tictime    # Return the same stored number.



def toc(start=None, output=False, label=None, sigfigs=None, filename=None):
    '''
    A little pair of functions to calculate a time difference, sort of like Matlab:
    tic() [but you can also use the form t = tic()]
    toc() [but you can also use the form toc(t) where to is the output of tic()]
    '''   
    # Set defaults
    if label   is None: label = ''
    if sigfigs is None: sigfigs = 3
    
    # If no start value is passed in, try to grab the global _tictime.
    if start is None:
        try:    start = _tictime
        except: start = 0 # This doesn't exist, so just leave start at 0.
            
    # Get the elapsed time in seconds.
    elapsed = time.time() - start
    
    # Create the message giving the elapsed time.
    if label=='': base = 'Elapsed time: '
    else:         base = 'Elapsed time for %s: ' % label
    logmessage = base + '%s s' % sigfig(elapsed, sigfigs=sigfigs)
    
    if output:
        return elapsed
    else:
        if filename is not None: printtologfile(logmessage, filename) # If we passed in a filename, append the message to that file.
        else: print(logmessage) # Otherwise, print the message.
        return None


def fixedpause(delay=None, verbose=True):
    '''
    Delay for a certain amount of time, to ensure accurate timing. Example:

    for i in range(10):
        sc.fixedpause('start') # Initialize
        for j in range(int(1e6)):
            tmp = pl.rand()
        sc.fixedpause(1) # Wait for one second including computation time
    '''
    global _delaytime
    if delay is None or delay=='start':
        _delaytime = time.time()  # Store the present time in the global.
        return _delaytime         # Return the same stored number.
    else:
        try:    
            import pylab as pl
        except Exception as E: 
            raise Exception('Cannot use fixedpause() since pylab import failed: %s' % str(E))
        try:    start = _delaytime
        except: start = time.time()
        elapsed = time.time() - start
        remaining = delay-elapsed
        if remaining>0:
            if verbose:
                print('Pausing for %0.1f s' % remaining)
            time.sleep(remaining)
        else:
            if verbose:
                print('Warning, delay less than elapsed time (%0.1f vs. %0.1f)' % (delay, elapsed))
    return None


def percentcomplete(step=None, maxsteps=None, indent=1):
    ''' Display progress '''
    onepercent = max(1,round(maxsteps/100)); # Calculate how big a single step is -- not smaller than 1
    if not step%onepercent: # Does this value lie on a percent
        thispercent = round(step/maxsteps*100) # Calculate what percent it is
        print('%s%i%%\n'% (' '*indent, thispercent)) # Display the output
    return None



def checkmem(origvariable, descend=False, order='n', plot=False, verbose=False):
    '''
    Checks how much memory the variable in question uses by dumping it to file.
    
    Example:
        from utils import checkmem
        checkmem(['spiffy',rand(2483,589)],descend=1)
    '''
    from .sc_fileio import saveobj
    filename = os.getcwd()+'/checkmem.tmp'
    
    printnames = []
    printbytes = []
    printsizes = []
    varnames = []
    variables = []
    if not descend:
        varnames = ['']
        variables = [origvariable]
    elif descend and np.iterable(origvariable):
        if hasattr(origvariable,'keys'):
            for key in origvariable.keys():
                varnames.append(key)
                variables.append(origvariable[key])
        else:
            varnames = range(len(origvariable))
            variables = [origvariable[i] for i in varnames]
    elif descend and not np.iterable(origvariable):
        varnames = sorted(origvariable.__dict__.keys())
        variables = [getattr(origvariable, attr) for attr in varnames]
    else:
        raise Exception('Something went wrong; this should be unreachable')
    
    for v,variable in enumerate(variables):
        if verbose: print('Processing variable %i of %i' % (v+1, len(variables)))
        saveobj(filename, variable)
        filesize = os.path.getsize(filename)
        factor = 1
        label = 'B'
        labels = ['KB','MB','GB']
        for i,f in enumerate([3,6,9]):
            if filesize>10**f:
                factor = 10**f
                label = labels[i]
        printnames.append(varnames[v])
        printbytes.append(filesize)
        printsizes.append('%0.3f %s' % (float(filesize/float(factor)), label))
        os.remove(filename)

    if order=='a' or order=='alpha' or order=='alphabetical':
        inds = np.argsort(printnames)
    else:
        inds = np.argsort(printbytes)

    for v in inds:
        print('Variable %s is %s' % (printnames[v], printsizes[v]))

    if plot==True:
        try:    
            import pylab as pl
        except Exception as E: 
            raise Exception('Cannot plot since import failed: %s' % repr(E))
        pl.axes(aspect=1)
        pl.pie(pl.array(printbytes)[inds], labels=pl.array(printnames)[inds], autopct='%0.2f')

    return None



def runcommand(command, printinput=False, printoutput=False):
    '''
    Make it easier to run shell commands.

    Examples:
        myfiles = sc.runcommand('ls').split('\n') # Get a list of files in the current folder
        sc.runcommand('sshpass -f %s scp myfile.txt me@myserver:myfile.txt' % 'pa55w0rd', printinput=True, printoutput=True) # Copy a file remotely
    
    Date: 2019jan15
    '''
    if printinput:
        print(command)
    try:
        p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stderr = p.stdout.read().decode("utf-8") # Somewhat confusingly, send stderr to stdout
        stdout = p.communicate()[0].decode("utf-8") # ...and then stdout to the pipe
        output = stdout + '\n' + stderr if stderr else stdout # Only include the error if it was non-empty
    except Exception as E:
        output = 'runcommand(): shell command failed: %s' % str(E) # This is for a Python error, not a shell error -- those get passed to output
    if printoutput: 
        print(output)
    return output



def gitinfo(filepath=None, die=False, hashlen=7, verbose=True):
    ''' Try to extract git information based on the file structure '''
    if filepath is None: filepath = __file__
    try: # First try importing git-python
        import git
        rootdir = os.path.abspath(filepath) # e.g. /user/username/optima/optima
        repo = git.Repo(path=rootdir, search_parent_directories=True)
        try:
            gitbranch = flexstr(repo.active_branch.name)  # Just make sure it's a string
        except TypeError:
            gitbranch = 'Detached head (no branch)'
        githash = flexstr(repo.head.object.hexsha) # Unicode by default
        gitdate = flexstr(repo.head.object.authored_datetime.isoformat())
    except Exception as E1:
        try: # If that fails, try the command-line method
            rootdir = os.path.abspath(filepath) # e.g. /user/username/optima/optima
            while len(rootdir): # Keep going as long as there's something left to go
                gitdir = rootdir+os.sep+'.git' # look for the git directory in the current directory
                if os.path.isdir(gitdir): break # It's found! terminate
                else: rootdir = os.sep.join(rootdir.split(os.sep)[:-1]) # Remove the last directory and keep looking
            headstrip = 'ref: ref'+os.sep+'heads'+os.sep # Header to strip off...hope this is generalizable!
            with open(gitdir+os.sep+'HEAD') as f: gitbranch = f.read()[len(headstrip)+1:].strip() # Read git branch name
            with open(gitdir+os.sep+'refs'+os.sep+'heads'+os.sep+gitbranch) as f: githash = f.read().strip() # Read git commit
            try:    gitdate = flexstr(runcommand('git -C "%s" show -s --format=%%ci' % gitdir).rstrip()) # Even more likely to fail
            except: gitdate = 'Git date N/A'
        except Exception as E2: # Failure? Give up
            gitbranch = 'Git branch N/A'
            githash = 'Git hash N/A'
            gitdate = 'Git date N/A'
            errormsg = 'Could not extract git info; please check paths or install git-python:\n%s\n%s' % (repr(E1), repr(E2))
            if die: raise Exception(errormsg)
            elif verbose:   print(errormsg)

    if len(githash)>hashlen: githash = githash[:hashlen] # Trim hash to short length
    output = {'branch':gitbranch, 'hash':githash, 'date':gitdate} # Assemble outupt
    return output



def compareversions(version1, version2):
    '''
    Function to compare versions, expecting both arguments to be a string of the 
    format 1.2.3, but numeric works too.
    
    Usage:
        compareversions('1.2.3', '2.3.4') # returns -1
        compareversions(2, '2') # returns 0
        compareversions('3.1', '2.99') # returns 1
    
    '''

    if LooseVersion(str(version1)) > LooseVersion(str(version2)):
        return 1
    elif LooseVersion(str(version1)) == LooseVersion(str(version2)):
        return 0
    else:
        return -1


def uniquename(name=None, namelist=None, style=None):
    """
    Given a name and a list of other names, find a replacement to the name 
    that doesn't conflict with the other names, and pass it back.
    """
    if style is None: style = ' (%d)'
    namelist = promotetolist(namelist)
    unique_name = str(name) # Start with the passed in name.
    i = 0 # Reset the counter
    while unique_name in namelist: # Try adding an index (i) to the name until we find one that's unique
        i += 1
        unique_name = str(name) + style%i
    return unique_name # Return the found name.


def importbyname(name=None, output=False, die=True):
    '''
    A little function to try loading optional imports.
    
    Example:
        np = importbyname('numpy')
    '''
    import importlib
    try:
        module = importlib.import_module(name)
        globals()[name] = module
    except Exception as E:
        errormsg = 'Cannot use "%s" since %s is not installed.\nPlease install %s and try again.' % (name,)*3
        print(errormsg)
        if die: raise E
        else:   return False
    if output: return module
    else:      return True



##############################################################################
### NESTED DICTIONARY FUNCTIONS
##############################################################################

__all__ += ['getnested', 'setnested', 'makenested', 'iternested', 'flattendict']

docstring = '''
Four little functions to get and set data from nested dictionaries. The first two were adapted from:
    http://stackoverflow.com/questions/14692690/access-python-nested-dictionary-items-via-a-list-of-keys

"getnested" will get the value for the given list of keys:
    getnested(foo, ['a','b'])

"setnested" will set the value for the given list of keys:
    setnested(foo, ['a','b'], 3)

"makenested" will recursively update a dictionary with the given list of keys:
    makenested(foo, ['a','b'])

"iternested" will return a list of all the twigs in the current dictionary:
    twigs = iternested(foo)

Example 1:
    from nested import makenested, getnested, setnested
    foo = {}
    makenested(foo, ['a','b'])
    foo['a']['b'] = 3
    print getnested(foo, ['a','b'])    # 3
    setnested(foo, ['a','b'], 7)
    print getnested(foo, ['a','b'])    # 7
    makenested(foo, ['yerevan','parcels'])
    setnested(foo, ['yerevan','parcels'], 'were tasty')
    print foo['yerevan']  # {'parcels': 'were tasty'}

Example 2:
    from nested import makenested, iternested, setnested
    foo = {}
    makenested(foo, ['a','x'])
    makenested(foo, ['a','y'])
    makenested(foo, ['a','z'])
    makenested(foo, ['b','a','x'])
    makenested(foo, ['b','a','y'])
    count = 0
    for twig in iternested(foo):
        count += 1
        setnested(foo, twig, count)   # {'a': {'y': 1, 'x': 2, 'z': 3}, 'b': {'a': {'y': 4, 'x': 5}}}

Version: 2014nov29 
'''

def getnested(nesteddict, keylist, safe=False): 
    output = reduce(lambda d, k: d.get(k) if d else None if safe else d[k], keylist, nesteddict)
    return output

def setnested(nesteddict, keylist, value): 
    getnested(nesteddict, keylist[:-1])[keylist[-1]] = value
    return None # Modify nesteddict in place

def makenested(nesteddict, keylist,item=None):
    currentlevel = nesteddict
    for i,key in enumerate(keylist[:-1]):
        if not(key in currentlevel):
            currentlevel[key] = {}
        currentlevel = currentlevel[key]
    currentlevel[keylist[-1]] = item

def iternested(nesteddict,previous = []):
    output = []
    for k in nesteddict.items():
        if isinstance(k[1],dict):
            output += iternested(k[1],previous+[k[0]]) # Need to add these at the first level
        else:
            output.append(previous+[k[0]])
    return output

# Set the docstrings for these functions
for func in [getnested, setnested, makenested, iternested]:
    func.__doc__ = docstring


def flattendict(inputdict=None, basekey=None, subkeys=None, complist=None, keylist=None, limit=100):
    '''
    A function for flattening out a recursive dictionary, with an optional list of sub-keys (ignored if non-existent).
    The flattened out structure is returned as complist. Values can be an object or a list of objects.
    All keys (including basekey) within the recursion are returned as keylist.
    
    Specifically, this function is intended for dictionaries of the form...
        inputdict[key1][sub_key[0]] = [a, key2, b]
        inputdict[key1][sub_key[1]] = [c, d]
        inputdict[key2][sub_key[0]] = e
        inputdict[key2][sub_key[1]] = [e, f, g]
    ...which, for this specific example, will output list...
        [a, e, e, f, g, h, b, c, d]
        
    There is a max-depth of limit for the recursion.
    '''
    
    if limit<1:
        errormsg = 'ERROR: A recursion limit has been reached when flattening a dictionary, stopping at key "%s".' % basekey
        raise Exception(errormsg)    
    
    if complist is None: complist = []
    if keylist is None: keylist = []
    keylist.append(basekey)

    if subkeys is None: inputlist = inputdict[basekey]
    else:
        inputlist = []
        for sub_key in subkeys:
            if sub_key in inputdict[basekey]:
                val = inputdict[basekey][sub_key]
                if isinstance(val, list):
                    inputlist += val
                else:
                    inputlist.append(val)      # Handle unlisted objects.
    
    for comp in inputlist:
        if comp in inputdict.keys():
            flattendict(inputdict=inputdict, basekey=comp, subkeys=subkeys, complist=complist, keylist=keylist, limit=limit-1)
        else:
            complist.append(comp)
    return complist, keylist





##############################################################################
### CLASSES
##############################################################################

__all__ += ['prettyobj', 'LinkException', 'Link', 'Timer']

class prettyobj(object):
    def __repr__(self):
        ''' Use pretty repr for objects '''
        output  = prepr(self)
        return output
    

class LinkException(Exception):
        ''' An exception to raise when links are broken -- note, can't define classes inside classes :( '''
        def __init(self, *args, **kwargs):
            Exception.__init__(self, *args, **kwargs)


class Link(object):
    '''
    A class to differentiate between an object and a link to an object. Not very
    useful at the moment, but the idea eventually is that this object would be
    parsed differently from other objects -- most notably, a recursive method
    (such as a pickle) would skip over Link objects, and then would fix them up
    after the other objects had been reinstated.
    
    Version: 2017jan31
    '''
    
    def __init__(self, obj=None):
        ''' Store the reference to the object being referred to '''
        self.obj = obj # Store the object -- or rather a reference to it, if it's mutable
        try:    self.uid = obj.uid # If the object has a UID, store it separately 
        except: self.uid = None # If not, just use None
    
    
    def __repr__(self):
        ''' Just use default '''
        output  = prepr(self)
        return output
    
    def __call__(self, obj=None):
        ''' If called with no argument, return the stored object; if called with argument, update object '''
        if obj is None:
            if type(self.obj)==LinkException: # If the link is broken, raise it now
                raise self.obj 
            return self.obj
        else:
            self.__init__(obj)
            return None
    
    def __copy__(self, *args, **kwargs):
        ''' Do NOT automatically copy link objects!! '''
        return Link(LinkException('Link object copied but not yet repaired'))
    
    def __deepcopy__(self, *args, **kwargs):
        ''' Same as copy '''
        return self.__copy__(*args, **kwargs)


class Timer(object):
    '''
    Simple timer class

    This wraps `tic` and `toc` with the formatting arguments and
    the start time (at construction)
    Use this in a ``with...as``` block to automatically print
    elapsed time when the block finishes.

    Implementation based on https://preshing.com/20110924/timing-your-code-using-pythons-with-statement/

    Example making repeated calls to the same Timer:

    >>> timer = Timer()
    >>> timer.toc()
    Elapsed time: 2.63 s
    >>> timer.toc()
    Elapsed time: 5.00 s

    Example wrapping code using with-as:

    >>> with Timer(label='mylabel') as t:
    >>>     foo()

    '''

    def __init__(self,**kwargs):
        self.tic()
        self.kwargs = kwargs #: Store kwargs to pass to :func:`toc` at the end of the block

    def __enter__(self):
        '''
        Reset start time when entering with-as block
        '''

        self.tic()
        return self

    def __exit__(self, *args):
        '''
        Print elapsed time when leaving a with-as block
        '''

        self.toc()

    def tic(self):
        '''
        Set start time
        '''

        self.start = tic()

    def toc(self):
        '''
        Print elapsed time
        '''

        toc(self.start,**self.kwargs)
