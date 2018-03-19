##############################################################################
### PRINTING/NOTIFICATION FUNCTIONS
##############################################################################

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
    from numpy import ceil
    nrow = int(ceil(float(len(oldkeys))/ncol))
    newkeys = []
    for x in xrange(nrow):
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
    output = createcollist(oldkeys, 'Attributes', strlen = 18, ncol = 3)
    return output


def objmeth(obj, strlen=18, ncol=3):
    ''' Return a sorted string of object methods for the Python __repr__ method '''
    oldkeys = sorted([method + '()' for method in dir(obj) if callable(getattr(obj, method)) and not method.startswith('__')])
    output = createcollist(oldkeys, 'Methods', strlen=strlen, ncol=ncol)
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


def defaultrepr(obj, maxlen=None):
    ''' Prints out the default representation of an object -- all attributes, plust methods and ID '''
    if maxlen is None: maxlen = 300
    keys = sorted(obj.__dict__.keys()) # Get the attribute keys
    maxkeylen = max([len(key) for key in keys]) # Find the maximum length of the attribute keys
    if maxkeylen<maxlen: maxlen = maxlen - maxkeylen # Shorten the amount of data shown if the keys are long
    formatstr = '%'+ '%i'%maxkeylen + 's' # Assemble the format string for the keys, e.g. '%21s'
    output  = objrepr(obj, showatt=False) # Get the methods
    for key in keys: # Loop over each attribute
        thisattr = flexstr(getattr(obj, key)) # Get the string representation of the attribute
        if len(thisattr)>maxlen: thisattr = thisattr[:maxlen] + ' [...]' # Shorten it
        prefix = formatstr%key + ': ' # The format key
        output += indent(prefix, thisattr)
    output += '============================================================\n'

    return output


def printdr(obj, maxlen=None):
    ''' Shortcut for printing the default repr for an object '''
    print(defaultrepr(obj, maxlen=maxlen))
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
    # Imports
    from textwrap import fill
    from pprint import pformat
    
    # Handle no prefix
    if prefix is None: prefix = ' '*n
    
    # Get text in the right format -- i.e. a string
    if pretty: text = pformat(text)
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
    


def sigfig(X, sigfigs=5, SI=False, sep=False):
    '''
    Return a string representation of variable x with sigfigs number of significant figures -- 
    copied from asd.py.
    
    If SI=True,  then will return e.g. 32433 as 32.433K
    If sep=True, then will return e.g. 32433 as 32,433
    '''
    from numpy import log10, floor
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
            else:
                magnitude = floor(log10(abs(x)))
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
    from numpy import ndim
    if ndim(arr)==1:
        string = ''
        for i in range(len(arr)):
            string += arrformat % arr[i]
        print(string)
    elif ndim(arr)==2:
        for i in range(len(arr)):
            printarr(arr[i], arrformat)
    elif ndim(arr)==3:
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
        from numpy import shape, ndarray
        if datatype==dict: string = ('dict with %i keys' % len(data.keys()))
        elif datatype==list: string = ('list of length %i' % len(data))
        elif datatype==tuple: string = ('tuple of length %i' % len(data))
        elif datatype==ndarray: string = ('array of shape %s' % flexstr(shape(data)))
        elif datatype.__name__=='module': string = ('module with %i components' % len(dir(data)))
        elif datatype.__name__=='class': string = ('class with %i components' % len(dir(data)))
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


def today(timezone='utc', die=False):
    ''' Get the current time, in UTC time '''
    import datetime # today = datetime.today
    try:
        import dateutil
        if timezone=='utc':                           tzinfo = dateutil.tz.tzutc()
        elif timezone is None or timezone=='current': tzinfo = None
        else:                                         raise Exception('Timezone "%s" not understood' % timezone)
    except:
        errormsg = 'Timezone information not available'
        if die: raise Exception(errormsg)
        tzinfo = None
    now = datetime.datetime.now(tzinfo)
    return now


def getdate(obj, which='modified', fmt='str'):
        ''' Return either the date created or modified ("which") as either a str or int ("fmt") '''
        from time import mktime
        
        dateformat = '%Y-%b-%d %H:%M:%S'
        
        try:
            if isinstance(obj, basestring): return obj # Return directly if it's a string
            obj.timetuple() # Try something that will only work if it's a date object
            dateobj = obj # Test passed: it's a date object
        except: # It's not a date object
            if which=='created': dateobj = obj.created
            elif which=='modified': dateobj = obj.modified
            elif which=='spreadsheet': dateobj = obj.spreadsheetdate
            else: raise Exception('Getting date for "which=%s" not understood; must be "created", "modified", or "spreadsheet"' % which)
        
        if fmt=='str':
            try:
                return dateobj.strftime(dateformat).encode('ascii', 'ignore') # Return string representation of time
            except UnicodeDecodeError:
                dateformat = '%Y-%m-%d %H:%M:%S'
                return dateobj.strftime(dateformat)
        elif fmt=='int': return mktime(dateobj.timetuple()) # So ugly!! But it works -- return integer representation of time
        else: raise Exception('"fmt=%s" not understood; must be "str" or "int"' % fmt)




def slacknotification(to=None, message=None, fromuser=None, token=None, verbose=2, die=False):
    ''' 
    Send a Slack notification when something is finished.
    
    Arguments:
        to:
            The Slack channel or user to post to. Note that channels begin with #, while users begin with @.
        message:
            The message to be posted.
        fromuser:
            The pseudo-user the message will appear from.
        token:
            This must be a plain text file containing a single line which is the Slack API URL token.
            Tokens are effectively passwords and must be kept secure. If you need one, contact me.
        verbose:
            How much detail to display.
        die:
            If false, prints warnings. If true, raises exceptions.
    
    Example usage:
        slacknotification('#athena', 'Long process is finished')
        slacknotification(token='/.slackurl', channel='@username', message='Hi, how are you going?')
    
    What's the point? Add this to the end of a very long-running script to notify
    your loved ones that the script has finished.
        
    Version: 2017feb09
    '''
    
    # Imports
    from requests import post # Simple way of posting data to a URL
    from json import dumps # For sanitizing the message
    from getpass import getuser # In case username is left blank
    
    # Validate input arguments
    printv('Sending Slack message...', 2, verbose)
    if token is None: token = '/.slackurl'
    if to is None: to = '#athena'
    if fromuser is None: fromuser = getuser()+'-bot'
    if message is None: message = 'This is an automated notification: your notifier is notifying you.'
    printv('Channel: %s | User: %s | Message: %s' % (to, fromuser, message), 3, verbose) # Print details of what's being sent
    
    # Try opening token file    
    try:
        with open(token) as f: slackurl = f.read()
    except:
        print('Could not open Slack URL/token file "%s"' % token)
        if die: raise
        else: return None
    
    # Package and post payload
    payload = '{"text": %s, "channel": %s, "username": %s}' % (dumps(message), dumps(to), dumps(fromuser))
    printv('Full payload: %s' % payload, 4, verbose)
    response = post(url=slackurl, data=payload)
    printv(response, 3, verbose) # Optionally print response
    printv('Message sent.', 1, verbose) # We're done
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
    

def colorize(color=None, string=None, output=False):
    '''
    Colorize output text. Arguments:
        color = the color you want (use 'bg' with background colors, e.g. 'bgblue')
        string = the text to be colored
        output = whether to return the modified version of the string
    
    Examples:
        colorize('green', 'hi') # Simple example
        colorize(['yellow', 'bgblack']); print('Hello world'); print('Goodbye world'); colorize() # Colorize all output in between
        bluearray = colorize(color='blue', string=str(range(5)), output=True); print("c'est bleu: " + bluearray)
        colorize('magenta') # Now type in magenta for a while
        colorize() # Stop typing in magenta
    
    To get available colors, type colorize('help').
    
    Note: although implemented as a class (to allow the "with" syntax),
    this actually functions more like a function.
    
    Version: 2017oct27
    '''
    
    # Define ANSI colors
    ansicolors = odict([
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
    for key,val in ansicolors.items(): ansicolors[key] = '\033['+val+'m'
    
    # Determine what color to use
    if color is None: color = 'reset' # By default, reset
    colorlist = promotetolist(color) # Make sure it's a list
    for color in colorlist:
        if color not in ansicolors.keys(): 
            if color!='help': print('Color "%s" is not available.' % color)
            print('Available colors are:  \n%s' % '\n  '.join(ansicolors.keys()))
            return None # Don't proceed if no color supplied
    ansicolor = ''
    for color in colorlist:
        ansicolor += ansicolors[color]
    
    # Modify string, if supplied
    if string is None: ansistring = ansicolor # Just return the color
    else:              ansistring = ansicolor + str(string) + ansicolors['reset'] # Add to start and end of the string

    if output: 
        return ansistring # Return the modified string
    else:
        print(ansistring) # Content, so print with newline
        return None
    


        
    
##############################################################################
### TYPE FUNCTIONS
##############################################################################

def flexstr(arg):
    ''' Try converting to a regular string, but try unicode if it fails '''
    try:    output = str(arg)
    except: output = unicode(arg)
    return  output


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
    
    Arguments:
        obj     = the object to check the type of
        objtype = the type to confirm the object belongs to
        subtype = optionally check the subtype if the object is iterable
        die     = whether or not to raise an exception if the object is the wrong type.
    
    Examples:
        checktype(rand(10), 'array', 'number') # Returns true
        checktype(['a','b','c'], 'arraylike') # Returns false
        checktype([{'a':3}], list, dict) # Returns True
    '''
    from numbers import Number
    from numpy import array
    
    # Handle "objtype" input
    if   objtype in ['str','string']:  objinstance = basestring
    elif objtype in ['num', 'number']: objinstance = Number
    elif objtype in ['arr', 'array']:  objinstance = type(array([]))
    elif objtype=='arraylike':         objinstance = (list, tuple, type(array([]))) # Anything suitable as a numerical array
    elif type(objtype)==type:          objinstance = objtype  # Don't need to do anything
    elif objtype is None:              return None # If not supplied, exit
    else:
        errormsg = 'Could not understand what type you want to check: should be either a string or a type, not "%s"' % objtype
        raise Exception(errormsg)
    
    # Do first-round checking
    result = isinstance(obj, objinstance)
    
    # Do second round checking
    if result and objtype=='arraylike': # Special case for handling arrays which may be multi-dimensional
        obj = promotetoarray(obj).flatten() # Flatten all elements
        if subtype is None: subtype = 'number' # This is the default
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
   
         
def isnumber(obj):
    ''' Simply determine whether or not the input is a number, since it's too hard to remember this otherwise '''
    return checktype(obj, 'number')
    

def promotetoarray(x):
    ''' Small function to ensure consistent format for things that should be arrays '''
    from numpy import ndarray, shape
    if isnumber(x):
        return array([x]) # e.g. 3
    elif isinstance(x, (list, tuple)):
        return array(x) # e.g. [3]
    elif isinstance(x, ndarray): 
        if shape(x):
            return x # e.g. array([3])
        else: 
            return array([x]) # e.g. array(3)
    else: # e.g. 'foo'
        raise Exception("Expecting a number/list/tuple/ndarray; got: %s" % flexstr(x))


def promotetolist(obj=None, objtype=None):
    ''' Make sure object is iterable -- used so functions can handle inputs like 'FSW' or ['FSW', 'MSM'] '''
    if type(obj)!=list:
        obj = [obj] # Listify it
    if objtype is not None:  # Check that the types match -- now that we know it's a list, we can iterate over it
        for item in obj:
            checktype(obj=item, objtype=objtype, die=True)
    if obj is None:
        raise Exception('This is mathematically impossible')
    return obj


def promotetoodict(obj=None):
    ''' Like promotetolist, but for odicts -- WARNING, could be made into a method for odicts '''
    if isinstance(obj, odict):
        return obj # Don't need to do anything
    elif isinstance(obj, dict):
        return odict(obj)
    elif isinstance(obj, list):
        newobj = odict()
        for i,val in enumerate(obj):
            newobj['Key %i'%i] = val
        return newobj
    else:
        return odict({'Key':obj})






##############################################################################
### MATHEMATICAL FUNCTIONS
##############################################################################


def quantile(data, quantiles=[0.5, 0.25, 0.75]):
    '''
    Custom function for calculating quantiles most efficiently for a given dataset.
        data = a list of arrays, or an array where he first dimension is to be sorted
        quantiles = a list of floats >=0 and <=1
    
    Version: 2014nov23
    '''
    from numpy import array
    nsamples = len(data) # Number of samples in the dataset
    indices = (array(quantiles)*(nsamples-1)).round().astype(int) # Calculate the indices to pull out
    output = array(data)
    output.sort(axis=0) # Do the actual sorting along the 
    output = output[indices] # Trim down to the desired quantiles
    
    return output



def sanitize(data=None, returninds=False, replacenans=None, die=True):
        '''
        Sanitize input to remove NaNs. Warning, does not work on multidimensional data!!
        
        Examples:
            sanitized,inds = sanitize(array([3,4,nan,8,2,nan,nan,nan,8]), returninds=True)
            sanitized = sanitize(array([3,4,nan,8,2,nan,nan,nan,8]), replacenans=True)
            sanitized = sanitize(array([3,4,nan,8,2,nan,nan,nan,8]), replacenans=0)
        '''
        from numpy import array, isnan, nonzero
        try:
            data = array(data,dtype=float) # Make sure it's an array of float type
            inds = nonzero(~isnan(data))[0] # WARNING, nonzero returns tuple :(
            sanitized = data[inds] # Trim data
            if replacenans is not None:
                newx = range(len(data)) # Create a new x array the size of the original array
                if replacenans==True: replacenans = 'nearest'
                if replacenans in ['nearest','linear']:
                    sanitized = smoothinterp(newx, inds, sanitized, method=replacenans, smoothness=0) # Replace nans with interpolated values
                else:
                    naninds = inds = nonzero(isnan(data))[0]
                    sanitized = dcp(data)
                    sanitized[naninds] = replacenans
            if len(sanitized)==0:
                sanitized = 0.0
                print('                WARNING, no data entered for this parameter, assuming 0')
        except Exception as E:
            if die: 
                raise Exception('Sanitization failed on array: "%s":\n %s' % (repr(E), data))
            else:
                sanitized = data # Give up and just return an empty array
                inds = []
        if returninds: return sanitized, inds
        else:          return sanitized


def getvaliddata(data=None, filterdata=None, defaultind=0):
    '''
    Return the years that are valid based on the validity of the input data.
    
    Example:
        getvaliddata(array([3,5,8,13]), array([2000, nan, nan, 2004])) # Returns array([3,13])
    '''
    from numpy import array, isnan
    data = array(data)
    if filterdata is None: filterdata = data # So it can work on a single input -- more or less replicates sanitize() then
    filterdata = array(filterdata)
    if filterdata.dtype=='bool': validindices = filterdata # It's already boolean, so leave it as is
    else:                        validindices = ~isnan(filterdata) # Else, assume it's nans that need to be removed
    if validindices.any(): # There's at least one data point entered
        if len(data)==len(validindices): # They're the same length: use for logical indexing
            validdata = array(array(data)[validindices]) # Store each year
        elif len(validindices)==1: # They're different lengths and it has length 1: it's an assumption
            validdata = array([array(data)[defaultind]]) # Use the default index; usually either 0 (start) or -1 (end)
        else:
            raise Exception('Array sizes are mismatched: %i vs. %i' % (len(data), len(validindices)))    
    else: 
        validdata = array([]) # No valid data, return an empty array
    return validdata



def getvalidinds(data=None, filterdata=None):
    '''
    Return the years that are valid based on the validity of the input data from an arbitrary number
    of 1-D vector inputs. Warning, closely related to getvaliddata()!
    
    Example:
        getvalidinds([3,5,8,13], [2000, nan, nan, 2004]) # Returns array([0,3])
    '''
    from numpy import isnan, intersect1d
    data = promotetoarray(data)
    if filterdata is None: filterdata = data # So it can work on a single input -- more or less replicates sanitize() then
    filterdata = promotetoarray(filterdata)
    if filterdata.dtype=='bool': filterindices = filterdata # It's already boolean, so leave it as is
    else:                        filterindices = findinds(~isnan(filterdata)) # Else, assume it's nans that need to be removed
    dataindices = findinds(~isnan(data)) # Also check validity of data
    validindices = intersect1d(dataindices, filterindices)
    return validindices # Only return indices -- WARNING, not consistent with sanitize()



def findinds(val1, val2=None, eps=1e-6):
    '''
    Little function to find matches even if two things aren't eactly equal (eg. 
    due to floats vs. ints). If one argument, find nonzero values. With two arguments,
    check for equality using eps. Returns a tuple of arrays if val1 is multidimensional,
    else returns an array.
    
    Examples:
        findinds(rand(10)<0.5) # e.g. array([2, 4, 5, 9])
        findinds([2,3,6,3], 6) # e.g. array([2])
    
    Version: 2016jun06 
    '''
    from numpy import nonzero, array, ndim
    if val2==None: # Check for equality
        output = nonzero(val1) # If not, just check the truth condition
    else:
        if isinstance(val2, basestring):
            output = nonzero(array(val1)==val2)
        else:
            output = nonzero(abs(array(val1)-val2)<eps) # If absolute difference between the two values is less than a certain amount
    if ndim(val1)==1: # Uni-dimensional
        output = output[0] # Return an array rather than a tuple of arrays if one-dimensional
    return output



def findnearest(series=None, value=None):
    '''
    Return the index of the nearest match in series to value -- like findinds, but
    always returns an object with the same type as value (i.e. findnearest with
    a number returns a number, findnearest with an array returns an array).
    
    Examples:
        findnearest(rand(10), 0.5) # returns whichever index is closest to 0.5
        findnearest([2,3,6,3], 6) # returns 2
        findnearest([2,3,6,3], 6) # returns 2
        findnearest([0,2,4,6,8,10], [3, 4, 5]) # returns array([1, 2, 2])
    
    Version: 2017jan07
    '''
    from numpy import argmin
    series = promotetoarray(series)
    if isnumber(value):
        output = argmin(abs(series-value))
    else:
        output = []
        for val in value: output.append(findnearest(series, val))
        output = promotetoarray(output)
    return output
    
    
def dataindex(dataarray, index):        
    ''' Take an array of data and return either the first or last (or some other) non-NaN entry. '''
    from numpy import zeros, shape
    
    nrows = shape(dataarray)[0] # See how many rows need to be filled (either npops, nprogs, or 1).
    output = zeros(nrows)       # Create structure
    for r in range(nrows): 
        output[r] = sanitize(dataarray[r])[index] # Return the specified index -- usually either the first [0] or last [-1]
    
    return output


def smoothinterp(newx=None, origx=None, origy=None, smoothness=None, growth=None, ensurefinite=False, keepends=True, method='linear'):
    '''
    Smoothly interpolate over values and keep end points. Same format as numpy.interp.
    
    Example:
        from utils import smoothinterp
        origy = array([0,0.1,0.3,0.8,0.7,0.9,0.95,1])
        origx = linspace(0,1,len(origy))
        newx = linspace(0,1,5*len(origy))
        newy = smoothinterp(newx, origx, origy, smoothness=5)
        plot(newx,newy)
        hold(True)
        scatter(origx,origy)
    
    Version: 2018jan24
    '''
    from numpy import array, interp, convolve, linspace, concatenate, ones, exp, nan, inf, isnan, isfinite, argsort, ceil, arange
    
    # Ensure arrays and remove NaNs
    if isnumber(newx):  newx = [newx] # Make sure it has dimension
    if isnumber(origx): origx = [origx] # Make sure it has dimension
    if isnumber(origy): origy = [origy] # Make sure it has dimension
    newx = array(newx)
    origx = array(origx)
    origy = array(origy)
    
    # If only a single element, just return it, without checking everything else
    if len(origy)==1: 
        newy = zeros(newx.shape)+origy[0]
        return newy
    
    if not(newx.shape): raise Exception('To interpolate, must have at least one new x value to interpolate to')
    if not(origx.shape): raise Exception('To interpolate, must have at least one original x value to interpolate to')
    if not(origy.shape): raise Exception('To interpolate, must have at least one original y value to interpolate to')
    if not(origx.shape==origy.shape): 
        errormsg = 'To interpolate, original x and y vectors must be same length (x=%i, y=%i)' % (len(origx), len(origy))
        raise Exception(errormsg)
    
    # Make sure it's in the correct order
    correctorder = argsort(origx)
    origx = origx[correctorder]
    origy = origy[correctorder]
    newx = newx[argsort(newx)] # And sort newx just in case
    
    # Only keep finite elements
    finitey = isfinite(origy) # Boolean for whether it's finite
    if finitey.any() and not finitey.all(): # If some but not all is finite, pull out indices that are
        finiteorigy = origy[finitey]
        finiteorigx = origx[finitey]
    else: # Otherwise, just copy the original
        finiteorigy = origy.copy()
        finiteorigx = origx.copy()
        
    # Perform actual interpolation
    if method=='linear':
        newy = interp(newx, finiteorigx, finiteorigy) # Perform standard interpolation without infinities
    elif method=='nearest':
        newy = zeros(newx.shape) # Create the new array of the right size
        for i,x in enumerate(newx): # Iterate over each point
            xind = argmin(abs(finiteorigx-x)) # Find the nearest neighbor
            newy[i] = finiteorigy[xind] # Copy it
    else:
        raise Exception('Method "%s" not found; methods are "linear" or "nearest"' % method)

    # Perform smoothing
    if smoothness is None: smoothness = ceil(len(newx)/len(origx)) # Calculate smoothness: this is consistent smoothing regardless of the size of the arrays
    smoothness = int(smoothness) # Make sure it's an appropriate number
    
    if smoothness:
        kernel = exp(-linspace(-2,2,2*smoothness+1)**2)
        kernel /= kernel.sum()
        validinds = findinds(~isnan(newy)) # Remove nans since these don't exactly smooth well
        if len(validinds): # No point doing these steps if no non-nan values
            validy = newy[validinds]
            prepend = validy[0]*ones(smoothness)
            postpend = validy[-1]*ones(smoothness)
            if not keepends:
                try: # Try to compute slope, but use original prepend if it doesn't work
                    dyinitial = (validy[0]-validy[1])
                    prepend = validy[0]*ones(smoothness) + dyinitial*arange(smoothness,0,-1)
                except:
                    pass
                try: # Try to compute slope, but use original postpend if it doesn't work
                    dyfinal = (validy[-1]-validy[-2])
                    postpend = validy[-1]*ones(smoothness) + dyfinal*arange(1,smoothness+1,1)
                except:
                    pass
            validy = concatenate([prepend, validy, postpend])
            validy = convolve(validy, kernel, 'valid') # Smooth it out a bit
            newy[validinds] = validy # Copy back into full vector
    
    # Apply growth if required
    if growth is not None:
        pastindices = findinds(newx<origx[0])
        futureindices = findinds(newx>origx[-1])
        if len(pastindices): # If there are past data points
            firstpoint = pastindices[-1]+1
            newy[pastindices] = newy[firstpoint] * exp((newx[pastindices]-newx[firstpoint])*growth) # Get last 'good' data point and apply inverse growth
        if len(futureindices): # If there are past data points
            lastpoint = futureindices[0]-1
            newy[futureindices] = newy[lastpoint] * exp((newx[futureindices]-newx[lastpoint])*growth) # Get last 'good' data point and apply growth
    
    # Add infinities back in, if they exist
    if any(~isfinite(origy)): # Infinities exist, need to add them back in manually since interp can only handle nan
        if not ensurefinite: # If not ensuring all entries are finite, put nonfinite entries back in
            orignan      = zeros(len(origy)) # Start from scratch
            origplusinf  = zeros(len(origy)) # Start from scratch
            origminusinf = zeros(len(origy)) # Start from scratch
            orignan[isnan(origy)]     = nan  # Replace nan entries with nan
            origplusinf[origy==inf]   = nan  # Replace plus infinite entries with nan
            origminusinf[origy==-inf] = nan  # Replace minus infinite entries with nan
            newnan      = interp(newx, origx, orignan) # Interpolate the nans
            newplusinf  = interp(newx, origx, origplusinf) # ...again, for positive
            newminusinf = interp(newx, origx, origminusinf) # ...and again, for negative
            newy[isnan(newminusinf)] = -inf # Add minus infinity back in first
            newy[isnan(newplusinf)]  = inf # Then, plus infinity
            newy[isnan(newnan)]  = nan # Finally, the nans
            
    
    return newy
    

def perturb(n=1, span=0.5, randseed=None):
    ''' Define an array of numbers uniformly perturbed with a mean of 1. n = number of points; span = width of distribution on either side of 1.'''
    from numpy.random import rand, seed
    if randseed is not None: seed(int(randseed)) # Optionally reset random seed
    output = 1. + 2*span*(rand(n)-0.5)
    return output
    
    
def scaleratio(inarray,total):
    ''' Multiply a list or array by some factor so that its sum is equal to the total. '''
    from numpy import array
    origtotal = float(sum(inarray))
    ratio = total/origtotal
    outarray = array(inarray)*ratio
    if type(inarray)==list: outarray = outarray.tolist() # Preserve type
    return outarray


def vec2obj(orig=None, newvec=None, inds=None):
    ''' 
    Function to convert an e.g. budget/coverage vector into an e.g. budget/coverage odict ...or anything, really
    
    WARNING: is all the error checking really necessary?
    
    inds can be a list of indexes, or a list of keys, but newvec must be a list, array, or odict.
    
    Version: 2016feb04
    '''
    from copy import deepcopy as dcp
    
    # Validate input
    if orig is None: raise Exception('vec2obj() requires an original object to update')
    if newvec is None: raise Exception('vec2obj() requires a vector as input')
    lenorig = len(orig)
    lennew = len(newvec)
    if lennew!=lenorig and inds is None: raise Exception('vec2obj(): if inds is not supplied, lengths must match (orig=%i, new=%i)' % (lenorig, lennew))
    if inds is not None and max(inds)>=len(orig): 
        raise Exception('vec2obj(): maximum index is greater than the length of the object (%i, %i)' % (max(inds), len(orig)))
    if inds is None: inds = range(lennew)

    # The actual meat of the function
    new = dcp(orig)    
    for i,ind in enumerate(inds):
        new[ind] = newvec[i]
    
    return new


def inclusiverange(*args, **kwargs):
    '''
    Like arange/linspace, but includes the start and stop points. 
    Accepts 0-3 args, or the kwargs start, stop, step. Examples:
    
    x = inclusiverange(3,5,0.2)
    x = inclusiverange(stop=5)
    x = inclusiverange(6, step=2)
    '''
    
    from numpy import linspace
    
    # Handle args
    if len(args)==0:
        start, stop, step = None, None, None
    elif len(args)==1:
        stop = args[0]
        start, step = None
    elif len(args)==2:
        start = args[0]
        stop   = args[1]
        step = None
    elif len(args)==3:
        start = args[0]
        stop = args[1]
        step = args[2]
    else:
        raise Exception('Too many arguments supplied: inclusiverange() accepts 0-3 arguments')
    
    # Handle kwargs
    start = kwargs.get('start', start)
    stop  = kwargs.get('stop',  stop)
    step  = kwargs.get('step',  step)
    
    # Finalize defaults
    if start is None: start = 0
    if stop  is None: stop  = 1
    if step  is None: step  = 1
    
    # OK, actually generate
    x = linspace(start, stop, int(round((stop-start)/float(step))+1)) # Can't use arange since handles floating point arithmetic badly, e.g. compare arange(2000, 2020, 0.2) with arange(2000, 2020.2, 0.2)
    
    return x


##############################################################################
### FILE/MISC. FUNCTIONS
##############################################################################


def tic():
    '''
    A little pair of functions to calculate a time difference, sort of like Matlab:
    tic() [but you can also use the form t = tic()]
    toc() [but you can also use the form toc(t) where to is the output of tic()]
    '''
    global tictime  # The saved time is stored in this global.    
    from time import time
    tictime = time()  # Store the present time in the global.
    return tictime    # Return the same stored number.



def toc(start=None, label=None, sigfigs=None, filename=None, output=False):
    '''
    A little pair of functions to calculate a time difference, sort of like Matlab:
    tic() [but you can also use the form t = tic()]
    toc() [but you can also use the form toc(t) where to is the output of tic()]
    '''   
    from time import time
    
    # Set defaults
    if label   is None: label = ''
    if sigfigs is None: sigfigs = 3
    
    # If no start value is passed in, try to grab the global tictime.
    if start is None:
        try:    start = tictime
        except: start = 0 # This doesn't exist, so just leave start at 0.
            
    # Get the elapsed time in seconds.
    elapsed = time() - start
    
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
    


def percentcomplete(step=None, maxsteps=None, indent=1):
    ''' Display progress '''
    onepercent = max(1,round(maxsteps/100)); # Calculate how big a single step is -- not smaller than 1
    if not step%onepercent: # Does this value lie on a percent
        thispercent = round(step/maxsteps*100) # Calculate what percent it is
        print('%s%i%%\n'% (' '*indent, thispercent)) # Display the output
    return None



def checkmem(origvariable, descend=0, order='n', plot=False, verbose=0):
    '''
    Checks how much memory the variable in question uses by dumping it to file.
    
    Example:
        from utils import checkmem
        checkmem(['spiffy',rand(2483,589)],descend=1)
    '''
    from os import getcwd, remove
    from os.path import getsize
    from cPickle import dump
    from numpy import iterable, argsort
    
    filename = getcwd()+'/checkmem.tmp'
    
    def dumpfile(variable):
        wfid = open(filename,'wb')
        dump(variable, wfid)
        return None
    
    printnames = []
    printbytes = []
    printsizes = []
    varnames = []
    variables = []
    if descend==False or not(iterable(origvariable)):
        varnames = ['']
        variables = [origvariable]
    elif descend==1 and iterable(origvariable):
        if hasattr(origvariable,'keys'):
            for key in origvariable.keys():
                varnames.append(key)
                variables.append(origvariable[key])
        else:
            varnames = range(len(origvariable))
            variables = origvariable
    
    for v,variable in enumerate(variables):
        if verbose: print('Processing variable %i of %i' % (v+1, len(variables)))
        dumpfile(variable)
        filesize = getsize(filename)
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
        remove(filename)

    if order=='a' or order=='alpha' or order=='alphabetical':
        inds = argsort(printnames)
    else:
        inds = argsort(printbytes)

    for v in inds:
        print('Variable %s is %s' % (printnames[v], printsizes[v]))

    if plot==True:
        from pylab import pie, array, axes
        axes(aspect=1)
        pie(array(printbytes)[inds], labels=array(printnames)[inds], autopct='%0.2f')

    return None



def getfilelist(folder=None, ext=None, pattern=None):
    ''' A short-hand since glob is annoying '''
    from glob import glob
    import os
    if folder is None: folder = os.getcwd()
    if pattern is None:
        if ext is None: ext = '*'
        pattern = '*.'+ext
    filelist = sorted(glob(os.path.join(folder, pattern)))
    return filelist



def sanitizefilename(rawfilename):
    '''
    Takes a potentially Linux- and Windows-unfriendly candidate file name, and 
    returns a "sanitized" version that is more usable.
    '''
    import re # Import regular expression package.
    filtername = re.sub('[\!\?\"\'<>]', '', rawfilename) # Erase certain characters we don't want at all: !, ?, ", ', <, >
    filtername = re.sub('[:/\\\*\|,]', '_', filtername) # Change certain characters that might be being used as separators from what they were to underscores: space, :, /, \, *, |, comma
    return filtername # Return the sanitized file name.



def makefilepath(filename=None, folder=None, ext=None, default=None, split=False, abspath=True, makedirs=True, verbose=False, sanitize=False):
    '''
    Utility for taking a filename and folder -- or not -- and generating a valid path from them.
    
    Inputs:
        filename = the filename, or full file path, to save to -- in which case this utility does nothing
        folder = the name of the folder to be prepended to the filename
        ext = the extension to ensure the file has
        default = a name or list of names to use if filename is None
        split = whether to return the path and filename separately
        makedirs = whether or not to make the folders to save into if they don't exist
        verbose = how much detail to print
    
    Example:
        makefilepath(filename=None, folder='./congee', ext='prj', default=[project.filename, project.name], split=True, abspath=True, makedirs=True)
    
    Assuming project.filename is None and project.name is "soggyrice" and ./congee doesn't exist:
        * Makes folder ./congee
        * Returns e.g. ('/home/optima/congee', 'soggyrice.prj')
    
    Actual code example from project.py:
        fullpath = makefilepath(filename=filename, folder=folder, default=[self.filename, self.name], ext='prj')
    
    Version: 2017apr04    
    '''
    
    # Initialize
    import os
    filefolder = '' # The folder the file will be located in
    filebasename = '' # The filename
    
    # Process filename
    if filename is None:
        defaultnames = promotetolist(default) # Loop over list of default names
        for defaultname in defaultnames:
            if not filename and defaultname: filename = defaultname # Replace empty name with default name
    if filename is not None: # If filename exists by now, use it
        filebasename = os.path.basename(filename)
        filefolder = os.path.dirname(filename)
    if not filebasename: filebasename = 'default' # If all else fails
    
    # Add extension if it's defined but missing from the filebasename
    if ext and not filebasename.endswith(ext): 
        filebasename += '.'+ext
    if verbose:
        print('From filename="%s", default="%s", extension="%s", made basename "%s"' % (filename, default, ext, filebasename))
    
    # Sanitize base filename
    if sanitize: filebasename = sanitizefilename(filebasename)
    
    # Process folder
    if folder is not None: # Replace with specified folder, if defined
        filefolder = folder 
    if abspath: # Convert to absolute path
        filefolder = os.path.abspath(filefolder) 
    if makedirs: # Make sure folder exists
        try: os.makedirs(filefolder)
        except: pass
    if verbose:
        print('From filename="%s", folder="%s", abspath="%s", makedirs="%s", made folder name "%s"' % (filename, folder, abspath, makedirs, filefolder))
    
    fullfile = os.path.join(filefolder, filebasename) # And the full thing
    
    if split: return filefolder, filebasename
    else:     return fullfile # Or can do os.path.split() on output


def loadbalancer(maxload=None, index=None, interval=None, maxtime=None, label=None, verbose=True):
    '''
    A little function to delay execution while CPU load is too high -- a very simple load balancer.

    Arguments:
        maxload:  the maximum load to allow for the task to still start (default 0.5)
        index:    the index of the task -- used to start processes asynchronously (default None)
        interval: the time delay to poll to see if CPU load is OK (default 5 seconds)
        maxtime:  maximum amount of time to wait to start the task (default 36000 seconds (10 hours))
        label:    the label to print out when outputting information about task delay or start (default None)
        verbose:  whether or not to print information about task delay or start (default True)

    Usage examples:
        loadbalancer() # Simplest usage -- delay while load is >50%
        for nproc in processlist: loadbalancer(maxload=0.9, index=nproc) # Use a maximum load of 90%, and stagger the start by process number

    Version: 2017oct25
     '''
    from psutil import cpu_percent
    from time import sleep
    from numpy.random import random
    
    # Set up processes to start asynchronously
    if maxload  is None: maxload = 0.8
    if interval is None: interval = 5.0
    if maxtime  is None: maxtime = 36000
    if label    is None: label = ''
    else: label += ': '
    if index is None:  
        pause = random()*interval
        index = ''
    else:              
        pause = index*interval
    if maxload>1: maxload/100. # If it's >1, assume it was given as a percent
    sleep(pause) # Give it time to asynchronize
    
    # Loop until load is OK
    toohigh = True # Assume too high
    count = 0
    maxcount = maxtime/float(interval)
    while toohigh and count<maxcount:
        count += 1
        currentload = cpu_percent(interval=0.1)/100. # If interval is too small, can give very inaccurate readings
        if currentload>maxload:
            if verbose: print(label+'CPU load too high (%0.2f/%0.2f); process %s queued %i times' % (currentload, maxload, index, count))
            sleep(interval*2*random()) # Sleeps for an average of refresh seconds, but do it randomly so you don't get locking
        else: 
            toohigh = False 
            if verbose: print(label+'CPU load fine (%0.2f/%0.2f), starting process %s after %i tries' % (currentload, maxload, index, count))
    return None
    
    

def runcommand(command, printinput=False, printoutput=False):
   ''' Make it easier to run bash commands. Version: 1.1 Date: 2015sep03 '''
   from subprocess import Popen, PIPE
   if printinput: print(command)
   try: output = Popen(command, shell=True, stdout=PIPE).communicate()[0].decode("utf-8")
   except: output = 'Shell command failed'
   if printoutput: print(output)
   return output



def gitinfo(die=False):
    ''' Try to extract git information based on the file structure '''
    try: # This whole thing could fail, you know!
        from os import path, sep # Path and file separator
        rootdir = path.abspath(path.dirname(__file__)) # e.g. /user/username/optima/optima
        while len(rootdir): # Keep going as long as there's something left to go
            gitdir = rootdir+sep+'.git' # look for the git directory in the current directory
            if path.isdir(gitdir): break # It's found! terminate
            else: rootdir = sep.join(rootdir.split(sep)[:-1]) # Remove the last directory and keep looking
        headstrip = 'ref: ref'+sep+'heads'+sep # Header to strip off...hope this is generalizable!
        with open(gitdir+sep+'HEAD') as f: gitbranch = f.read()[len(headstrip)+1:].strip() # Read git branch name
        with open(gitdir+sep+'refs'+sep+'heads'+sep+gitbranch) as f: gitversion = f.read().strip() # Read git commit
    except: 
        try: # Try using git-python instead -- most users probably won't have
            import git
            repo = git.Repo(path=rootdir, search_parent_directories=True)
            gitbranch = flexstr(repo.active_branch.name) # Just make sure it's a string
            gitversion = flexstr(repo.head.object.hexsha) # Unicode by default
        except: # Failure? Give up
            gitbranch = 'Git branch information not retrivable'
            gitversion = 'Git version information not retrivable'
            if die:
                errormsg = 'Could not extract git info; please check paths or install git-python'
                raise Exception(errormsg)
    return gitbranch, gitversion



def compareversions(version1=None, version2=None):
    '''
    Function to compare versions, expecting both arguments to be a string of the 
    format 1.2.3, but numeric works too.
    
    Usage:
        compareversions('1.2.3', '2.3.4') # returns -1
        compareversions(2, '2.0.0.0') # returns 0
        compareversions('3.1', '2.99') # returns 1
    
    '''
    if version1 is None or version2 is None: 
        raise Exception('Must supply both versions as strings')
    versions = [version1, version2]
    for i in range(2):
        versions[i] = array(flexstr(versions[i]).split('.'), dtype=float) # Convert to array of numbers
    maxlen = max(len(versions[0]), len(versions[1]))
    versionsarr = zeros((2,maxlen))
    for i in range(2):
        versionsarr[i,:len(versions[i])] = versions[i]
    for j in range(maxlen):
        if versionsarr[0,j]<versionsarr[1,j]: return -1
        if versionsarr[0,j]>versionsarr[1,j]: return 1
    if (versionsarr[0,:]==versionsarr[1,:]).all(): return 0
    else:
        raise Exception('Failed to compare %s and %s' % (version1, version2))



def boxoff(ax=None, removeticks=True, flipticks=True):
    '''
    I don't know why there isn't already a Matplotlib command for this.
    
    Removes the top and right borders of a plot. Also optionally removes
    the tick marks, and flips the remaining ones outside.

    Version: 2017may22    
    '''
    from pylab import gca
    if ax is None: ax = gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if removeticks:
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    if flipticks:
        ax.tick_params(direction='out', pad=5)
    return ax

##############################################################################
### NESTED DICTIONARY FUNCTIONS
##############################################################################

'''
Four little functions to get and set data from nested dictionaries. The first two were stolen from:
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
    ''' Get a value from a nested dictionary'''
    from functools import reduce
    output = reduce(lambda d, k: d.get(k) if d else None if safe else d[k], keylist, nesteddict)
    return output

def setnested(nesteddict, keylist, value): 
    ''' Set a value in a nested dictionary '''
    getnested(nesteddict, keylist[:-1])[keylist[-1]] = value
    return None # Modify nesteddict in place

def makenested(nesteddict, keylist,item=None):
    ''' Insert item into nested dictionary, creating keys if required '''
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






##############################################################################
### PLOTTING FUNCTIONS
##############################################################################

def setylim(data=None, ax=None):
    '''
    A small script to determine how the y limits should be set. Looks
    at all data (a list of arrays) and computes the lower limit to
    use, e.g.
    
        setylim([array([-3,4]), array([6,4,6])], ax)
    
    will keep Matplotlib's lower limit, since at least one data value
    is below 0.
    
    Note, if you just want to set the lower limit, you can do that 
    with this function via:
        setylim(0, ax)
    '''
    # Get current limits
    currlower, currupper = ax.get_ylim()
    
    # Calculate the lower limit based on all the data
    lowerlim = 0
    upperlim = 0
    data = promotetolist(data) # Make sure it'siterable
    for ydata in data:
        lowerlim = min(lowerlim, promotetoarray(ydata).min())
        upperlim = max(upperlim, promotetoarray(ydata).max())
    
    # Set the new y limits
    if lowerlim<0: lowerlim = currlower # If and only if the data lower limit is negative, use the plotting lower limit
    upperlim = max(upperlim, currupper) # Shouldn't be an issue, but just in case...
    
    # Specify the new limits and return
    ax.set_ylim((lowerlim, upperlim))
    return lowerlim,upperlim


def SItickformatter(x, pos, sigfigs=2, SI=True, *args, **kwargs):  # formatter function takes tick label and tick position
    ''' Formats axis ticks so that e.g. 34,243 becomes 34K '''
    output = sigfig(x, sigfigs=sigfigs, SI=SI) # Pretty simple since sigfig() does all the work
    return output


def SIticks(fig=None, ax=None, axis='y'):
    ''' Apply SI tick formatting to one axis of a figure '''
    from matplotlib import ticker
    if  fig is not None: axlist = fig.axes
    elif ax is not None: axlist = promotetolist(ax)
    else: raise Exception('Must supply either figure or axes')
    for ax in axlist:
        if   axis=='x': thisaxis = ax.xaxis
        elif axis=='y': thisaxis = ax.yaxis
        elif axis=='z': thisaxis = ax.zaxis
        else: raise Exception('Axis must be x, y, or z')
        thisaxis.set_major_formatter(ticker.FuncFormatter(SItickformatter))
    return None


def commaticks(fig=None, ax=None, axis='y'):
    ''' Use commas in formatting the y axis of a figure -- see http://stackoverflow.com/questions/25973581/how-to-format-axis-number-format-to-thousands-with-a-comma-in-matplotlib '''
    from matplotlib import ticker
    if   ax  is not None: axlist = promotetolist(ax)
    elif fig is not None: axlist = fig.axes
    else: raise Exception('Must supply either figure or axes')
    for ax in axlist:
        if   axis=='x': thisaxis = ax.xaxis
        elif axis=='y': thisaxis = ax.yaxis
        elif axis=='z': thisaxis = ax.zaxis
        else: raise Exception('Axis must be x, y, or z')
        thisaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    return None


##############################################################################
### ODICT CLASS
##############################################################################

from collections import OrderedDict
from numpy import array
from numbers import Number
from copy import deepcopy as dcp

class odict(OrderedDict):
    '''
    An ordered dictionary, like the OrderedDict class, but supporting list methods like integer 
    referencing, slicing, and appending.
    
    Version: 2017oct28
    '''
    
    def __init__(self, *args, **kwargs):
        ''' See collections.py '''
        if len(args)==1 and args[0] is None: args = [] # Remove a None argument
        OrderedDict.__init__(self, *args, **kwargs) # Standard init
        return None

    def __slicekey(self, key, slice_end):
        shift = int(slice_end=='stop')
        if isinstance(key, Number): return key
        elif type(key) is str: return self.index(key)+shift # +1 since otherwise confusing with names (CK)
        elif key is None: return (len(self) if shift else 0)
        else: raise Exception('To use a slice, %s must be either int or str (%s)' % (slice_end, key))
        return None


    def __is_odict_iterable(self, key):
        ''' Check to see whether the "key" is actually an iterable '''
        output = type(key)==list or type(key)==type(array([])) # Do *not* include dict, since that would be recursive
        return output
        
        
    def __sanitize_items(self, items):
        ''' Try to convert the output of a slice to an array, but give up easily and return a list '''
        try: 
            output = array(items) # Try standard Numpy array...
            if 'S' in str(output.dtype): # ...but instead of converting to string, convert to object array
                output = array(items, dtype=object)
        except:
            output = items # If that fails, just give up and return the list
        return output
        


    def __getitem__(self, key):
        ''' Allows getitem to support strings, integers, slices, lists, or arrays '''
        if isinstance(key, (str,tuple)):
            try:
                output = OrderedDict.__getitem__(self, key)
                return output
            except Exception as E: # WARNING, should be KeyError, but this can't print newlines!!!
                if len(self.keys()): errormsg = '%s\nodict key "%s" not found; available keys are:\n%s' % (repr(E), flexstr(key), '\n'.join([flexstr(k) for k in self.keys()]))
                else:                errormsg = 'Key "%s" not found since odict is empty'% key
                raise Exception(errormsg)
        elif isinstance(key, Number): # Convert automatically from float...dangerous?
            thiskey = self.keys()[int(key)]
            return OrderedDict.__getitem__(self,thiskey)
        elif type(key)==slice: # Handle a slice -- complicated
            try:
                startind = self.__slicekey(key.start, 'start')
                stopind = self.__slicekey(key.stop, 'stop')
                if stopind<startind:
                    print('Stop index must be >= start index (start=%i, stop=%i)' % (startind, stopind))
                    raise Exception
                slicevals = [self.__getitem__(i) for i in range(startind,stopind)]
                output = self.__sanitize_items(slicevals)
                return output
            except:
                print('Invalid odict slice... returning empty list...')
                return []
        elif self.__is_odict_iterable(key): # Iterate over items
            listvals = [self.__getitem__(item) for item in key]
            output = self.__sanitize_items(listvals)
            return output
        else: # Handle everything else
            return OrderedDict.__getitem__(self,key)
        
        
    def __setitem__(self, key, value):
        ''' Allows setitem to support strings, integers, slices, lists, or arrays '''
        if isinstance(key, (str,tuple)):
            OrderedDict.__setitem__(self, key, value)
        elif isinstance(key, Number): # Convert automatically from float...dangerous?
            thiskey = self.keys()[int(key)]
            OrderedDict.__setitem__(self, thiskey, value)
        elif type(key)==slice:
            startind = self.__slicekey(key.start, 'start')
            stopind = self.__slicekey(key.stop, 'stop')
            if stopind<startind:
                errormsg = 'Stop index must be >= start index (start=%i, stop=%i)' % (startind, stopind)
                raise Exception(errormsg)
            slicerange = range(startind,stopind)
            enumerator = enumerate(slicerange)
            slicelen = len(slicerange)
            if hasattr(value, '__len__'):
                if len(value)==slicelen:
                    for valind,index in enumerator:
                        self.__setitem__(index, value[valind])  # e.g. odict[:] = arr[:]
                else:
                    errormsg = 'Slice "%s" and values "%s" have different lengths! (%i, %i)' % (slicerange, value, slicelen, len(value))
                    raise Exception(errormsg)
            else: 
                for valind,index in enumerator:
                    self.__setitem__(index, value) # e.g. odict[:] = 4
        elif self.__is_odict_iterable(key) and hasattr(value, '__len__'): # Iterate over items
            if len(key)==len(value):
                for valind,thiskey in enumerate(key): 
                    self.__setitem__(thiskey, value[valind])
            else:
                errormsg = 'Keys "%s" and values "%s" have different lengths! (%i, %i)' % (key, value, len(key), len(value))
                raise Exception(errormsg)
        else:
            OrderedDict.__setitem__(self, key, value)
        return None
    
     
    def __repr__(self, maxlen=None, showmultilines=True, divider=False, dividerthresh=10, numindents=0, recurselevel=0, sigfigs=None, numformat=None):
        ''' Print a meaningful representation of the odict '''
        
        # Set primitives for display.
        toolong = ' [...]' # String to display at end of line when maximum value character length is overrun.
        dividerstr = '*'*40+'\n' # String to use as an inter-item divider.
        indentstr = '    ' # Create string to use to indent.
        
        # Only if we are in the root call, indent by the number of indents.
        if recurselevel == 0: 
            theprefix = indentstr * numindents
        else: # Otherwise (if we are in a recursive call), indent only 1 indent.
            theprefix = indentstr 
                            
        # If the odict is empty, make the string just indicate it's an odict.
        if len(self.keys())==0:
            output = 'odict()'
        else:                   
            output = '' # Initialize the output to nothing.
            keystrs = [] # Start with an empty list which we'll save key strings in.
            valstrs = [] # Start with an empty list which we'll save value strings in.
            vallinecounts = [] # Start with an empty list which we'll save line counts in.
            for i in range(len(self)): # Loop over the dictionary values
                thiskeystr = flexstr(self.keys()[i]) # Grab a str representation of the current key.  
                thisval = self.values()[i] # Grab the current value.
                                
                # If it's another odict, make a call increasing the recurselevel and passing the same parameters we received.
                if isinstance(thisval, odict):
                    thisvalstr = flexstr(thisval.__repr__(maxlen=maxlen, showmultilines=showmultilines, divider=divider, 
                        dividerthresh=dividerthresh, numindents=numindents, recurselevel=recurselevel+1, sigfigs=sigfigs, numformat=numformat))
                elif isnumber(thisval): # Flexibly print out numbers, since they're largely why we're here
                    if numformat is not None:
                        thisvalstr = numformat % thisval
                    elif sigfigs is not None:
                        thisvalstr = sigfig(thisval, sigfigs=sigfigs)
                    else:
                        thisvalstr = str(thisval) # To avoid numpy's stupid 0.4999999999945
                else: # Otherwise, do the normal repr() read.
                    thisvalstr = repr(thisval)

                # Add information to the lists to retrace afterwards.
                keystrs.append(thiskeystr)
                valstrs.append(thisvalstr)
                vallinecounts.append(thisvalstr.count('\n') + 1) # Count the number of lines in the value.
            maxvallinecounts = max(vallinecounts)   # Grab the maximum count of lines in the dict values.                    
            
            maxkeylen = max([len(keystr) for keystr in keystrs])
            for i in range(len(keystrs)): # Loop over the lists
                keystr = keystrs[i]
                valstr = valstrs[i]
                vallinecount = vallinecounts[i]
                
                if (divider or (maxvallinecounts>dividerthresh)) and \
                    showmultilines and recurselevel==0 and i!=0: # Add a divider line if we should.
                    newoutput = indent(prefix=theprefix, text=dividerstr, width=80)
                    if newoutput[-1] == '\n':
                        newoutput = newoutput[:-1]
                    output += newoutput        
                            
                # Trim the length of the entry if we need to.
                if not showmultilines:                    
                    valstr = valstr.replace('\n','\\n') # Replace line breaks with characters
                
                # Trim long entries
                if maxlen and len(valstr) > maxlen: 
                    valstr = valstr[:maxlen-len(toolong)] + toolong 
                    
                # Create the the text to add, apply the indent, and add to the output
                spacer = ' '*(maxkeylen-len(keystr))
                if vallinecount == 1 or not showmultilines:
                    rawoutput = '#%i: "%s":%s %s\n' % (i, keystr, spacer, valstr)
                else:
                    rawoutput = '#%i: "%s":%s \n%s\n' % (i, keystr, spacer, valstr)
                    
                # Perform the indentation.
                newoutput = indent(prefix=theprefix, text=rawoutput, width=80)
                
                # Strip ot any terminal newline.
                if newoutput[-1] == '\n':
                    newoutput = newoutput[:-1] 
                    
                # Add the new output to the full output.              
                output += newoutput                    
                    
        # Trim off any terminal '\n'.
        if output[-1] == '\n':
            output = output[:-1]
                
        # Return the formatted string.
        return output
    
        
    def _repr_pretty_(self, p, cycle):
        ''' Function to fix __repr__ in IPython'''
        print(self.__repr__())
    
    
    def disp(self, maxlen=None, showmultilines=True, divider=False, dividerthresh=10, numindents=0, sigfigs=5, numformat=None):
        '''
        Print out flexible representation, short by default.
        
        Example:
            import optima as op
            import pylab as pl
            z = op.odict().make(keys=['a','b','c'], vals=(10*pl.rand(3)).tolist())
            z.disp(sigfigs=3)
            z.disp(numformat='%0.6f')
        '''
        print(self.__repr__(maxlen=maxlen, showmultilines=showmultilines, 
            divider=divider, dividerthresh=dividerthresh, 
            numindents=numindents, recurselevel=0, sigfigs=sigfigs, numformat=None))
    
    
    def export(self, doprint=True):
        ''' Export the odict in a form that is valid Python code '''
        start = 'odict(['
        end = '])'
        output = start
        
        for key in self.keys():
            output += '('+repr(key)
            output += ', '
            child = self.get(key)
            if isinstance(child, odict): output += child.export(doprint=False) # Handle nested odicts -- WARNING, can't doesn't work for e.g. lists of odicts!
            else:                        output += repr(child)
            output += '), '
        
        output += end
        if doprint:
            print(output)
            return None
        else:
            return output
    
    
    def pop(self, key, *args, **kwargs):
        ''' Allows pop to support strings, integers, slices, lists, or arrays '''
        if isinstance(key, basestring):
            return OrderedDict.pop(self, key, *args, **kwargs)
        elif isinstance(key, Number): # Convert automatically from float...dangerous?
            thiskey = self.keys()[int(key)]
            return OrderedDict.pop(self, thiskey, *args, **kwargs)
        elif type(key)==slice: # Handle a slice -- complicated
            try:
                startind = self.__slicekey(key.start, 'start')
                stopind = self.__slicekey(key.stop, 'stop')
                if stopind<startind:
                    print('Stop index must be >= start index (start=%i, stop=%i)' % (startind, stopind))
                    raise Exception
                slicevals = [self.pop(i, *args, **kwargs) for i in range(startind,stopind)] # WARNING, not tested
                try: return array(slicevals) # Try to convert to an array
                except: return slicevals
            except:
                print('Invalid odict slice... returning empty list...')
                return []
        elif self.__is_odict_iterable(key): # Iterate over items
            listvals = [self.pop(item, *args, **kwargs) for item in key]
            try: return array(listvals)
            except: return listvals
        else: # Handle string but also everything else
            try:
                return OrderedDict.pop(self, key, *args, **kwargs)
            except: # WARNING, should be KeyError, but this can't print newlines!!!
                if len(self.keys()): 
                    errormsg = 'odict key "%s" not found; available keys are:\n%s' % (flexstr(key), 
                        '\n'.join([flexstr(k) for k in self.keys()]))
                else: errormsg = 'Key "%s" not found since odict is empty'% key
                raise Exception(errormsg)
    
    
    def index(self, value):
        ''' Return the index of a given key '''
        return self.keys().index(value)
    
    
    def valind(self, value):
        ''' Return the index of a given value '''
        return self.items().index(value)
    
    
    def append(self, key=None, value=None):
        ''' Support an append method, like a list '''
        needkey = False
        if value is None: # Assume called with a single argument
            value = key
            needkey = True
        if key is None or needkey:
            keyname = 'key'+flexstr(len(self))  # Define the key just to be the current index
        else:
            keyname = key
        self.__setitem__(keyname, value)
        return None
    
    
    def insert(self, pos=None, key=None, value=None):
        '''
        Stupid, slow function to do insert -- WARNING, should be able to use approach more like rename...
        
        Usage:
            z = odict()
            z['foo'] = 1492
            z.insert(1604)
            z.insert(0, 'ganges', 1444)
            z.insert(2, 'midway', 1234)
        '''
        
        # Handle inputs
        realpos, realkey, realvalue = pos, key, value
        if key is None and value is None: # Assume it's called like odict.insert(666)
            realvalue = pos
            realkey = 'key'+flexstr(len(self))
            realpos = 0
        elif value is None: # Assume it's called like odict.insert('devil', 666)
            realvalue = key
            realkey = pos
            realpos = 0
        if pos is None:
            realpos = 0
        if realpos>len(self):
            errormsg = 'Cannot insert %s at position %i since length of odict is %i ' % (key, pos, len(self))
            raise Exception(errormsg)
        
        # Create a temporary dictionary to hold all of the items after the insertion point
        tmpdict = odict()
        origkeys = self.keys()
        originds = range(len(origkeys))
        if not len(originds) or realpos==len(originds): # It's empty or in the final position, just append
            self.__setitem__(realkey, realvalue)
        else: # Main usage case, it's not empty
            try: insertind = originds.index(realpos) # Figure out which index we're inseting at
            except:
                errormsg = 'Could not insert item at position %i in odict with %i items' % (realpos, len(originds))
                raise Exception(errormsg)
            keystopop = origkeys[insertind:] # Pop these keys until we get far enough back
            for keytopop in keystopop:
                tmpdict.__setitem__(keytopop, self.pop(keytopop))
            self.__setitem__(realkey, realvalue) # Insert the new item at the right location
            for keytopop in keystopop: # Insert popped items back in
                self.__setitem__(keytopop, tmpdict.pop(keytopop))

        return None
        
        
    def rename(self, oldkey, newkey):
        ''' Change a key name -- WARNING, very inefficient! '''
        nkeys = len(self)
        if isinstance(oldkey, Number): 
            index = oldkey
            keystr = self.keys()[index]
        else: # Forge ahead for strings and anything else!
            index = self.keys().index(oldkey)
            keystr = oldkey
        self.__setitem__(newkey, self.pop(keystr))
        if index<nkeys-1:
            for i in range(index+1, nkeys):
                key = self.keys()[index]
                value = self.pop(key)
                self.__setitem__(key, value)
        return None
    
    
    def sort(self, sortby=None, reverse=False, copy=False):
        '''
        Create a sorted version of the odict. Sorts by order of sortby, if provided, otherwise alphabetical.
        If copy is True, then returns a copy (like sorted()).
        
        Note that you can also use this to do filtering.
        
        Note: very slow, do not use for serious computations!!
        '''
        origkeys = self.keys()
        if sortby is None: allkeys = sorted(origkeys)
        else:
            if not isiterable(sortby): raise Exception('Please provide a list to determine the sort order.')
            if all([isinstance(x,basestring) for x in sortby]): # Going to sort by keys
                allkeys = sortby # Assume the user knows what s/he is doing
            elif all([isinstance(x,bool) for x in sortby]): # Using Boolean values
                allkeys = []
                for i,x in enumerate(sortby):
                     if x: allkeys.append(origkeys[i])
            elif all([isinstance(x,Number) for x in sortby]): # Going to sort by numbers
                if not set(sortby)==set(range(len(self))):
                    errormsg = 'List to sort by "%s" is not compatible with length of odict "%i"' % (sortby, len(self))
                    raise Exception(errormsg)
                else: allkeys = [y for (x,y) in sorted(zip(sortby,origkeys))]
            else: 
                raise Exception('Cannot figure out how to sort by "%s"' % sortby)
        tmpdict = odict()
        if reverse: allkeys.reverse() # If requested, reverse order
        if copy:
            for key in allkeys: tmpdict[key] = self[key]
            return tmpdict
        else:
            for key in allkeys: tmpdict.__setitem__(key, self.pop(key))
            for key in allkeys: self.__setitem__(key, tmpdict.pop(key))
            return None
    
    def sorted(self, sortby=None, reverse=False):
        ''' Shortcut for making a copy of the sorted odict '''
        return self.sort(sortby=sortby, copy=True, reverse=reverse)


    def reverse(self, copy=False):
        ''' Reverse the order of an odict '''
        reversedkeys = self.keys()
        reversedkeys.reverse()
        output = self.sort(sortby=reversedkeys, copy=copy)
        return output
    
    
    def reversed(self):
        ''' Shortcut for making a copy of the sorted odict '''
        return self.reverse(copy=True)
    
    
    def make(self, keys=None, vals=None, keys2=None, keys3=None):
        '''
        An alternate way of making or adding to an odict. Examples:
            a = odict().make(5) # Make an odict of length 5, populated with Nones and default key names
            b = odict().make('foo',34) # Make an odict with a single key 'foo' of value 34
            c = odict().make(['a','b']) # Make an odict with keys 'a' and 'b'
            d = odict().make(['a','b'],0) # Make an odict with keys 'a' and 'b', initialized to 0
            e = odict().make(keys=['a','b'], vals=[1,2]) # Make an odict with 'a':1 and 'b':2
            f = odict({'a':34, 'b':58}).make(['c','d'],[99,45]) # Add extra keys to an exising odict
            g = odict().make(keys=['a','b','c'], keys2=['A','B','C'], keys3=['x','y','z'], vals=0) # Make a triply nested odict
        '''
        # Handle keys
        keylist = []
        if keys is None and vals is None:
            return None # Nothing to do if nothing supplied
        if keys is None and vals is not None:
            keys = len(promotetolist(vals)) # Values are supplied but keys aren't: use default keys
        if isinstance(keys, Number): # It's a single number: pre-generate
            keylist = ['%i'%i for i in range(keys)] # Generate keylist
        elif isinstance(keys, basestring): # It's a single string
            keylist = [flexstr(keys)]
        elif isinstance(keys, list): # It's a list: use directly
            keylist = keys
        else:
            errormsg = 'Could not understand keys "%s": must be number, string, or list' % keys
            raise Exception(errormsg)
        nkeys = len(keylist)
        
        # Handle values
        vals = promotetolist(vals)
        nvals = len(vals)
        if nvals==0: # Special case: it's an empty list
            vallist = [dcp(vals) for _ in range(nkeys)]
        elif nvals==1: # Only a single value: duplicate it
            vallist = [dcp(vals[0]) for _ in range(nkeys)]
        elif nvals==nkeys: # Lengths match, can use directly
            vallist = vals 
        else:
            errormsg = 'Must supply either a single value or a list of same length as the keys (%i keys, %i values supplied)' % (nkeys, nvals)
            raise Exception(errormsg)
        
        # Handle nested keys -- warning, would be better to not hard-code this, but does the brain in as it is!
        if keys2 is not None and keys3 is not None: # Doubly nested
            self.make(keys=keys, vals=odict().make(keys=keys2, vals=odict().make(keys=keys3, vals=vals)))
        elif keys2 is not None: # Singly nested
            self.make(keys=keys, vals=odict().make(keys=keys2, vals=vals))
        else: # Not nested -- normal case of making an odict
            for key,val in zip(keylist,vallist): # Update odict
                self.__setitem__(key, val)
        
        return self # A bit weird, but usually would use this return an odict
    
    
    def makefrom(self, source=None, keys=None, keynames=None, *args, **kwargs):
        '''
        Create an odict from entries in another dictionary. If keys is None, then
        use all keys from the current dictionary.
        
        Examples:
            a = 'cat'; b = 'dog'; o = odict().makefrom(source=locals(), keys=['a','b']) # Make use of fact that variables are stored in a dictionary
            d = {'a':'cat', 'b':'dog'}; o = odict().makefrom(d) # Same as odict(d)
            l = ['cat', 'monkey', 'dog']; o = odict().makefrom(source=l, keys=[0,2], keynames=['a','b'])
        '''
        
        # Make sure it's iterable
        if source is not None: # Don't do anything if there's nothing there
            if not(isiterable(source)): # Make sure it's iterable
                source = promotetolist(source)
            elif isinstance(source, basestring):
                source = [source] # Special case -- strings are iterable, but we don't want to
            
            if len(source)==0:
                return self # Nothing to do here
            else:
                # Handle cases where keys or keynames are not supplied
                if keys is None:
                    if isinstance(source, (list, tuple)):   keys = range(len(source))
                    elif isinstance(source, dict): keys = source.keys()
                    else:                          raise Exception('Unable to guess keys for object of type %s' % type(source))
                keys = promotetolist(keys) # Make sure it's a list
                if keynames is None: keynames = keys # Use key names
                
                # Loop over supplied keys
                for key,keyname in zip(keys,keynames):
                    try: 
                        self.__setitem__(str(keyname), source[key])
                    except Exception as E: 
                        raise Exception('Key "%s" not found: %s' % (key, repr(E)))
                
        return self # As with make()
    
    
    def map(self, func=None):
        '''
        Apply a function to each element of the odict, returning
        a new odict with the same keys.
        
        Example:
            cat = odict({'a':[1,2], 'b':[3,4]})
            def myfunc(mylist): return [i**2 for i in mylist]
            dog = cat.map(myfunc) # Returns odict({'a':[1,4], 'b':[9,16]})
        '''
        output = odict()
        for key in self.keys():
            output[key] = func(self.__getitem__(key))
        return output
    
    
    def fromeach(self, ind=None, asdict=True):
        '''
        Take a "slice" across all the keys of an odict, applying the same
        operation to entry. The simplest usage is just to pick an index.
        However, you can also use it to apply a function to each key.
        
        Example:
            z = odict({'a':array([1,2,3,4]), 'b':array([5,6,7,8])})
            z.fromeach(2) # Returns array([3,7])
            z.fromeach(ind=[1,3], asdict=True) # Returns odict({'a':array([2,4]), 'b':array([6,8])})
        '''
        output = odict()
        for key in self.keys():
            output[key] = self.__getitem__(key)[ind]
        if asdict: return output # Output as a slimmed-down odict
        else:      return output[:] # Output as just the entries
        
    
    def toeach(self, ind=None, val=None):
        '''
        The inverse of fromeach: partially reset elements within
        each odict key.
        
        Example:
            z = odict({'a':[1,2,3,4], 'b':[5,6,7,8]})
            z.toeach(2, [10,20])    # z is now odict({'a':[1,2,10,4], 'b':[5,6,20,8]})
            z.toeach(ind=3,val=666) #  z is now odict({'a':[1,2,10,666], 'b':[5,6,20,666]})
        '''
        nkeys = len(self.keys())
        if not(isiterable(val)): # Assume it's meant to be populated in each
            val = [val]*nkeys # Duplicated
        if len(val)!=nkeys:
            errormsg = 'To map values onto each key, they must be the same length (%i vs. %i)' % (len(val), nkeys)
            raise Exception(errormsg)
        for k,key in self.enumkeys():
            self.__getitem__(key)[ind] = val[k]
        return None
        
    
    def enumkeys(self):
        ''' Shortcut for enumerate(odict.keys()) '''
        iterator = enumerate(self.keys())
        return iterator
    
    
    def enumvals(self):
        ''' Shortcut for enumerate(odict.values()) '''
        iterator = enumerate(self.values())
        return iterator
    
    
    def enumitems(self):
        ''' Returns tuple of 3 things: index, key, value '''
        iterator = [] # Would be better to not pre-allocate but what can you do...
        for ind,item in enumerate(self.items()):
            thistuple = (ind,)+item # Combine into one tuple
            iterator.append(thistuple)
        return iterator
        
        
        
        
        
        
        

        




##############################################################################
### DATA FRAME CLASS
##############################################################################

# Some of these are repeated to make this frationally more self-contained
from numpy import array, zeros, empty, vstack, hstack, matrix, argsort, argmin, floor, log10 # analysis:ignore
from numbers import Number # analysis:ignore

class dataframe(object):
    '''
    A simple data frame, based on simple lists, for simply storing simple data.
    
    Example usage:
        a = dataframe(cols=['x','y'],data=[[1238,2],[384,5],[666,7]) # Create data frame
        print a['x'] # Print out a column
        print a[0] # Print out a row
        print a[0,'x'] # Print out an element
        a[0] = [123,6]; print a # Set values for a whole row
        a['y'] = [8,5,0]; print a # Set values for a whole column
        a['z'] = [14,14,14]; print a # Add new column
        a.addcol('z', [14,14,14]); print a # Alternate way to add new column
        a.rmcol('z'); print a # Remove a column
        a.pop(1); print a # Remove a row
        a.append([555,2,14]); print a # Append a new row
        a.insert(1,[555,2,14]); print a # Insert a new row
        a.sort(); print a # Sort by the first column
        a.sort('y'); print a # Sort by the second column
        a.addrow([555,2,14]); print a # Replace the previous row and sort
        a.getrow(1) # Return the row starting with value '1'
        a.rmrow(); print a # Remove last row
        a.rmrow(1238); print a # Remove the row starting with element '3'
    
    Works for both numeric and non-numeric data.
    
    Version: 2018mar17
    '''

    def __init__(self, cols=None, data=None):
        if cols is None: cols = list()
        if data is None: 
            data = zeros((0,len(cols)), dtype=object) # Object allows more than just numbers to be stored
        else:
            data = array(data, dtype=object)
            if data.ndim != 2:
                errormsg = 'Dimension of data must be 2, not %s' % data.ndim
                raise Exception(errormsg)
            if data.shape[1]==len(cols):
                pass
            elif data.shape[0]==len(cols):
                data = data.transpose()
            else:
                errormsg = 'Number of columns (%s) does not match array shape (%s)' % (len(cols), data.shape)
                raise Exception(errormsg)
        self.cols = cols
        self.data = data
        return None
    
    def __repr__(self, spacing=2):
        ''' spacing = space between columns '''
        if not self.cols: # No keys, give up
            return ''
        
        else: # Go for it
            outputlist = dict()
            outputformats = dict()
            
            # Gather data
            nrows = self.nrows()
            for c,col in enumerate(self.cols):
                outputlist[col] = list()
                maxlen = len(col) # Start with length of column name
                if nrows:
                    for val in self.data[:,c]:
                        output = flexstr(val)
                        maxlen = max(maxlen, len(output))
                        outputlist[col].append(output)
                outputformats[col] = '%'+'%i'%(maxlen+spacing)+'s'
            
            indformat = '%%%is' % (floor(log10(nrows))+1) # Choose the right number of digits to print
            
            # Assemble output
            output = indformat % '' # Empty column for index
            for col in self.cols: # Print out header
                output += outputformats[col] % col
            output += '\n'
            
            for ind in range(nrows): # Loop over rows to print out
                output += indformat % flexstr(ind)
                for col in self.cols: # Print out data
                    output += outputformats[col] % outputlist[col][ind]
                output += '\n'
            
            return output
    
    def _val2row(self, value=None):
        ''' Convert a list, array, or dictionary to the right format for appending to a dataframe '''
        if isinstance(value, dict):
            output = zeros(self.ncols(), dtype=object)
            for c,col in enumerate(self.cols):
                try: 
                    output[c] = value[col]
                except: 
                    errormsg = 'Entry for column %s not found; keys you supplied are: %s' % (col, value.keys())
                    raise Exception(errormsg)
            output = array(output, dtype=object)
        elif value is None:
            output = empty(self.ncols(),dtype=object)
        else: # Not sure what it is, just make it an array
            if len(value)==self.ncols():
                output = array(value, dtype=object)
            else:
                errormsg = 'Row has wrong length (%s supplied, %s expected)' % (len(value), self.ncols())
        return output
    
    def _sanitizecol(self, col):
        ''' Take None or a string and return the index of the column '''
        if col is None: output = 0 # If not supplied, assume first column is control
        elif isinstance(col, basestring): output = self.cols.index(col) # Convert to index
        else: output = col
        return output
        
    def __getitem__(self, key):
        if isinstance(key, basestring):
            colindex = self.cols.index(key)
            output = self.data[:,colindex]
        elif isinstance(key, Number):
            rowindex = int(key)
            output = self.data[rowindex,:]
        elif isinstance(key, tuple):
            colindex = self.cols.index(key[0])
            rowindex = int(key[1])
            output = self.data[rowindex,colindex]
        elif isinstance(key, slice):
            rowslice = key
            slicedata = self.data[rowslice,:]
            output = dataframe(cols=self.cols, data=slicedata)
        else:
            raise Exception('Unrecognized dataframe key "%s"' % key)
        return output
        
    def __setitem__(self, key, value):
        if isinstance(key, basestring): # Add column
            if len(value) != self.nrows(): 
                errormsg = 'Vector has incorrect length (%i vs. %i)' % (len(value), self.nrows())
                raise Exception(errormsg)
            try:
                colindex = self.cols.index(key)
                self.data[:,colindex] = value
            except:
                self.cols.append(key)
                colindex = self.cols.index(key)
                self.data = hstack((self.data, array(value, dtype=object)))
        elif isinstance(key, Number):
            value = self._val2row(value) # Make sure it's in the correct format
            if len(value) != self.ncols(): 
                errormsg = 'Vector has incorrect length (%i vs. %i)' % (len(value), self.ncols())
                raise Exception(errormsg)
            rowindex = int(key)
            try:
                self.data[rowindex,:] = value
            except:
                self.data = vstack((self.data, array(value, dtype=object)))
        elif isinstance(key, tuple):
            try:
                colindex = self.cols.index(key[0])
                rowindex = int(key[1])
                self.data[rowindex,colindex] = value
            except:
                errormsg = 'Could not insert element (%s,%s) in dataframe of shape %' % (colindex, rowindex, self.data.shape)
                raise Exception(errormsg)
        return None
    
    def pop(self, key, returnval=True):
        ''' Remove a row from the data frame '''
        rowindex = int(key)
        thisrow = self.data[rowindex,:]
        self.data = vstack((self.data[:rowindex,:], self.data[rowindex+1:,:]))
        if returnval: return thisrow
        else:         return None
    
    def append(self, value):
        ''' Add a row to the end of the data frame '''
        value = self._val2row(value) # Make sure it's in the correct format
        self.data = vstack((self.data, array(value, dtype=object)))
        return None
    
    def ncols(self):
        ''' Get the number of columns in the data frame '''
        ncols = len(self.cols)
        ncols2 = self.data.shape[1]
        if ncols != ncols2:
            errormsg = 'Dataframe corrupted: %s columns specified but %s in data' % (ncols, ncols2)
            raise Exception(errormsg)
        return ncols

    def nrows(self):
        ''' Get the number of rows in the data frame '''
        try:    return self.data.shape[0]
        except: return 0 # If it didn't work, probably because it's empty
    
    def addcol(self, key, value):
        ''' Add a new column to the data frame -- for consistency only '''
        self.__setitem__(key, value)
    
    def rmcol(self, key):
        ''' Remove a column from the data frame '''
        colindex = self.cols.index(key)
        self.cols.pop(colindex) # Remove from list of columns
        self.data = hstack((self.data[:,:colindex], self.data[:,colindex+1:])) # Remove from data
        return None
    
    def addrow(self, value=None, overwrite=True, col=None, reverse=False):
        ''' Like append, but removes duplicates in the first column and resorts '''
        value = self._val2row(value) # Make sure it's in the correct format
        col   = self._sanitizecol(col)
        index = self._rowindex(key=value[col], col=col, die=False) # Return None if not found
        if index is None or not overwrite: self.append(value)
        else: self.data[index,:] = value # If it exists already, just replace it
        self.sort(col=col, reverse=reverse) # Sort
        return None
    
    def _rowindex(self, key=None, col=None, die=False):
        ''' Get the sanitized row index for a given key and column '''
        col = self._sanitizecol(col)
        coldata = self.data[:,col] # Get data for this column
        if key is None: key = coldata[-1] # If not supplied, pick the last element
        try:    index = coldata.tolist().index(key) # Try to find duplicates
        except: 
            if die: raise Exception('Item %s not found; choices are: %s' % (key, coldata))
            else:   return None
        return index
        
    def rmrow(self, key=None, col=None, returnval=False, die=True):
        ''' Like pop, but removes by matching the first column instead of the index '''
        index = self._rowindex(key=key, col=col, die=die)
        if index is not None: self.pop(index)
        return None
    
    def _todict(self, row):
        ''' Return row as a dict rather than as an array '''
        if len(row)!=len(self.cols): 
            errormsg = 'Length mismatch between "%s" and "%s"' % (row, self.cols)
            raise Exception(errormsg)
        rowdict = dict(zip(self.cols, row))
        return rowdict
    
    def getrow(self, key=None, col=None, default=None, closest=False, die=False, asdict=False):
        '''
        Get a row by value.
        
        Arguments:
            key = the value to look for
            col = the column to look for this value in
            default = the value to return if key is not found (overrides die)
            closest = whether or not to return the closest row (overrides default and die)
            die = whether to raise an exception if the value is not found
            asdict = whether to return results as dict rather than list
        
        Example:
            df = dataframe(cols=['year','val'],data=[[2016,0.3],[2017,0.5]])
            df.getrow(2016) # returns array([2016, 0.3], dtype=object)
            df.getrow(2013) # returns None, or exception if die is True
            df.getrow(2013, closest=True) # returns array([2016, 0.3], dtype=object)
            df.getrow(2016, asdict=True) # returns {'year':2016, 'val':0.3}
        '''
        if not closest: # Usual case, get 
            index = self._rowindex(key=key, col=col, die=(die and default is None))
        else:
            col = self._sanitizecol(col)
            coldata = self.data[:,col] # Get data for this column
            index = argmin(abs(coldata-key)) # Find the closest match to the key
        if index is not None:
            thisrow = self.data[index,:]
            if asdict:
                thisrow = self._todict(thisrow)
        else:
            thisrow = default # If not found, return as default
        return thisrow
        
    def insert(self, row=0, value=None):
        ''' Insert a row at the specified location '''
        rowindex = int(row)
        value = self._val2row(value) # Make sure it's in the correct format
        self.data = vstack((self.data[:rowindex,:], value, self.data[rowindex:,:]))
        return None
    
    def sort(self, col=None, reverse=False):
        ''' Sort the data frame by the specified column '''
        col = self._sanitizecol(col)
        sortorder = argsort(self.data[:,col])
        if reverse: sortorder = array(list(reversed(sortorder)))
        self.data = self.data[sortorder,:]
        return None
        










##############################################################################
### OTHER CLASSES
##############################################################################


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
        output  = defaultrepr(self)
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
        return self.__copy__(self, *args, **kwargs)
        
        
