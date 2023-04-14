'''
This is the graveyard for old Sciris functions that are no longer used. It is preserved
for backwards compatibility, but is not imported into Sciris by default.
'''

import types
import warnings
import textwrap
import traceback
import gzip as gz
import numpy as np
import pickle as pkl
import datetime as dt
import copyreg as cpreg
import multiprocess as mp
from io import BytesIO as IO
from functools import partial
from contextlib import closing
from .. import sc_utils as scu
from .. import sc_odict as sco
from .. import sc_fileio as scf
from .. import sc_profiling as scp


##############################################################################
#%% Python 2 legacy support
##############################################################################

not_string_pickleable = ['datetime', 'BytesIO']
byte_objects = ['datetime', 'BytesIO', 'odict', 'spreadsheet', 'blobject']


def loadobj2or3(filename=None, filestring=None, recursionlimit=None, **kwargs):  # pragma: no cover
    '''
    Try to load as a (Sciris-saved) Python 3 pickle; if that fails, try to load
    as a Python 2 pickle. For legacy support only.

    For available keyword arguments, see sc.load().

    Args:
        filename (str): the name of the file to load
        filestring (str): alternatively, specify an already-loaded bytestring
        recursionlimit (int): how deeply to parse objects before failing (default 1000)
    '''
    try:
        output = scf.loadobj(filename=filename, **kwargs)
    except:
        output = _loadobj2to3(filename=filename, filestring=filestring, recursionlimit=recursionlimit)
    return output


def _loadobj2to3(filename=None, filestring=None, recursionlimit=None): # pragma: no cover
    '''
    Used by loadobj2or3() to load Python2 objects in Python3 if all other
    loading methods fail. Uses a recursive approach, so can set a recursion limit.
    '''

    class Placeholder():
        ''' Replace these corrupted classes with properly loaded ones '''
        def __init__(*args):
            return

        def __setstate__(self, state):
            if isinstance(state,dict):
                self.__dict__ = state
            else:
                self.state = state
            return

    class StringUnpickler(pkl.Unpickler):
        def find_class(self, module, name, verbose=False):
            if verbose: print('Unpickling string module %s , name %s' % (module, name))
            if name in not_string_pickleable:
                return scf.Empty
            else:
                try:
                    output = pkl.Unpickler.find_class(self,module,name)
                except Exception as E:
                    print('Warning, string unpickling could not find module %s, name %s: %s' % (module, name, str(E)))
                    output = scf.Empty
                return output

    class BytesUnpickler(pkl.Unpickler):
        def find_class(self, module, name, verbose=False):
            if verbose: print('Unpickling bytes module %s , name %s' % (module, name))
            if name in byte_objects:
                try:
                    output = pkl.Unpickler.find_class(self,module,name)
                except Exception as E:
                    print('Warning, bytes unpickling could not find module %s, name %s: %s' % (module, name, str(E)))
                    output = Placeholder
                return output
            else:
                return Placeholder

    def recursive_substitute(obj1, obj2, track=None, recursionlevel=0, recursionlimit=None):
        if recursionlimit is None: # Recursion limit
            recursionlimit = 1000 # Better to die here than hit Python's recursion limit

        def recursion_warning(count, obj1, obj2):
            output = 'Warning, internal recursion depth exceeded, aborting: depth=%s, %s -> %s' % (count, type(obj1), type(obj2))
            return output

        recursionlevel += 1

        if track is None:
            track = []

        if isinstance(obj1, scf.Blobject): # Handle blobjects (usually spreadsheets)
            obj1.blob  = obj2.__dict__[b'blob']
            obj1.bytes = obj2.__dict__[b'bytes']

        if isinstance(obj2, dict): # Handle dictionaries
            for k,v in obj2.items():
                if isinstance(v, dt.datetime):
                    setattr(obj1, k.decode('latin1'), v)
                elif isinstance(v, dict) or hasattr(v,'__dict__'):
                    if isinstance(k, (bytes, bytearray)):
                        k = k.decode('latin1')
                    track2 = track.copy()
                    track2.append(k)
                    if recursionlevel<=recursionlimit:
                        recursionlevel = recursive_substitute(obj1[k], v, track2, recursionlevel, recursionlimit)
                    else:
                        print(recursion_warning(recursionlevel, obj1, obj2))
        else:
            for k,v in obj2.__dict__.items():
                if isinstance(v, dt.datetime):
                    setattr(obj1,k.decode('latin1'), v)
                elif isinstance(v,dict) or hasattr(v,'__dict__'):
                    if isinstance(k, (bytes, bytearray)):
                        k = k.decode('latin1')
                    track2 = track.copy()
                    track2.append(k)
                    if recursionlevel<=recursionlimit:
                        recursionlevel = recursive_substitute(getattr(obj1,k), v, track2, recursionlevel, recursionlimit)
                    else:
                        print(recursion_warning(recursionlevel, obj1, obj2))
        return recursionlevel

    def loadintostring(fileobj):
        unpickler1 = StringUnpickler(fileobj, encoding='latin1')
        try:
            stringout = unpickler1.load()
        except Exception as E:
            print('Warning, string pickle loading failed: %s' % str(E))
            exception = traceback.format_exc() # Grab the trackback stack
            stringout = scf.makefailed(module_name='String unpickler failed', name='n/a', error=E, exception=exception)
        return stringout

    def loadintobytes(fileobj):
        unpickler2 = BytesUnpickler(fileobj,  encoding='bytes')
        try:
            bytesout  = unpickler2.load()
        except Exception as E:
            print('Warning, bytes pickle loading failed: %s' % str(E))
            exception = traceback.format_exc() # Grab the trackback stack
            bytesout = scf.makefailed(module_name='Bytes unpickler failed', name='n/a', error=E, exception=exception)
        return bytesout

    # Load either from file or from string
    if filename:
        with gz.GzipFile(filename) as fileobj:
            stringout = loadintostring(fileobj)
        with gz.GzipFile(filename) as fileobj:
            bytesout = loadintobytes(fileobj)

    elif filestring:
        with closing(IO(filestring)) as output:
            with gz.GzipFile(fileobj=output, mode='rb') as fileobj:
                stringout = loadintostring(fileobj)
        with closing(IO(filestring)) as output:
            with gz.GzipFile(fileobj=output, mode='rb') as fileobj:
                bytesout = loadintobytes(fileobj)
    else:
        errormsg = 'You must supply either a filename or a filestring for loadobj() or loadstr(), respectively'
        raise Exception(errormsg)

    # Actually do the load, with correct substitution
    recursive_substitute(stringout, bytesout, recursionlevel=0, recursionlimit=recursionlimit)
    return stringout



##############################################################################
#%% Twisted pickling methods
##############################################################################

# NOTE: The code below is part of the Twisted package, and is included
# here to allow functools.partial() objects (among other things) to be
# pickled; they are not for public consumption. --CK

# From: twisted/persisted/styles.py
# -*- test-case-name: twisted.test.test_persisted -*-
# Copyright (c) Twisted Matrix Laboratories.
# See LICENSE for details.

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

_UniversalPicklingError = pkl.PicklingError

def _pickleMethod(method):
    return (_unpickleMethod, (method.__name__,         method.__self__, method.__self__.__class__))

def _methodFunction(classObject, methodName):
    methodObject = getattr(classObject, methodName)
    return methodObject

def _unpickleMethod(im_name, im_self, im_class):
    if im_self is None:
        return getattr(im_class, im_name)
    try:
        methodFunction = _methodFunction(im_class, im_name)
    except AttributeError: # pragma: no cover
        assert im_self is not None, "No recourse: no instance to guess from."
        if im_self.__class__ is im_class:
            raise
        return _unpickleMethod(im_name, im_self, im_self.__class__)
    else:
        maybeClass = ()
        bound = types.MethodType(methodFunction, im_self, *maybeClass)
        return bound

cpreg.pickle(types.MethodType, _pickleMethod, _unpickleMethod)

# Legacy support for loading Sciris <1.0 objects; may be removed in future
pickleMethod = _pickleMethod
unpickleMethod = _unpickleMethod



##############################################################################
#%% Legacy data frame class
##############################################################################

class legacy_dataframe(object): # pragma: no cover
    '''
    This legacy dataframe is maintained solely to allow loading old files.

    **Example**::

        import sciris as sc
        from sciris.extras import legacy as scl
        remapping = {'sciris.sc_dataframe.dataframe':scl.legacy_dataframe}
        old = sc.load('my-old-file.obj', remapping=remapping)

    | Version: 2020nov29
    | Migrated to ``sc_legacy`` in version 2.0.0.
    '''

    def __init__(self, cols=None, data=None, nrows=None):
        self.cols = None
        self.data = None
        self.make(cols=cols, data=data, nrows=nrows)
        return


    def __repr__(self, spacing=2):
        ''' spacing = space between columns '''
        if not self.cols: # No keys, give up
            return '<empty dataframe>'

        else: # Go for it
            outputlist = sco.odict()
            outputformats = sco.odict()

            # Gather data
            nrows = self.nrows
            for c,col in enumerate(self.cols):
                outputlist[col] = list()
                maxlen = len(col) # Start with length of column name
                if nrows:
                    for val in self.data[:,c]:
                        output = scu.flexstr(val)
                        maxlen = max(maxlen, len(output))
                        outputlist[col].append(output)
                outputformats[col] = '%'+'%i'%(maxlen+spacing)+'s'

            ndigits = (np.floor(np.log10(max(1,nrows)))+1) # Don't allow 0 rows
            indformat = '%%%is' % ndigits # Choose the right number of digits to print

            # Assemble output
            output = indformat % '' # Empty column for index
            for col in self.cols: # Print out header
                output += outputformats[col] % col
            output += '\n'

            for ind in range(nrows): # Loop over rows to print out
                output += indformat % scu.flexstr(ind)
                for col in self.cols: # Print out data
                    output += outputformats[col] % outputlist[col][ind]
                if ind<nrows-1: output += '\n'

            return output


    @property
    def ncols(self):
        ''' Get the number of columns in the data frame '''
        ncols = len(self.cols)
        ncols2 = self.data.shape[1]
        if ncols != ncols2:
            errormsg = 'Dataframe corrupted: %s columns specified but %s in data' % (ncols, ncols2)
            raise Exception(errormsg)
        return ncols


    @property
    def nrows(self):
        ''' Get the number of rows in the data frame '''
        try:    return self.data.shape[0]
        except: return 0 # If it didn't work, probably because it's empty


    @property
    def shape(self):
        ''' Equivalent to the shape of the data array, minus the headers '''
        return (self.nrows, self.ncols)


    def make(self, cols=None, data=None, nrows=None):
        '''
        Creates a dataframe from the supplied input data.

        **Usage examples**::

            df = sc.dataframe()
            df = sc.dataframe(['a','b','c'])
            df = sc.dataframe(['a','b','c'], nrows=2)
            df = sc.dataframe([['a','b','c'],[1,2,3],[4,5,6]])
            df = sc.dataframe(['a','b','c'], [[1,2,3],[4,5,6]])
            df = sc.dataframe(cols=['a','b','c'], data=[[1,2,3],[4,5,6]])
        '''
        import pandas as pd # Optional import

        # Handle columns
        if nrows is None:
            nrows = 0
        if cols is None and data is None:
            cols = list()
            data = np.zeros((int(nrows), 0), dtype=object) # Object allows more than just numbers to be stored
        elif cols is None and data is not None: # Shouldn't happen, but if it does, swap inputs
            cols = data
            data = None

        if isinstance(cols, pd.DataFrame): # It's actually a Pandas dataframe
            self.pandas(df=cols)
            return # We're done

        # A dictionary is supplied: assume keys are columns, and the rest is the data
        if isinstance(cols, dict):
            data = [col for col in cols.values()]
            cols = list(cols.keys())

        elif not scu.checktype(cols, 'listlike'):
            errormsg = 'Inputs to dataframe must be list, tuple, or array, not %s' % (type(cols))
            raise Exception(errormsg)

        # Handle data
        if data is None:
            if np.ndim(cols)==2 and np.shape(cols)[0]>1: # It's a 2D array with more than one row: treat first as header
                data = scu.dcp(cols[1:])
                cols = scu.dcp(cols[0])
            else:
                data = np.zeros((int(nrows),len(cols)), dtype=object) # Just use default
        data = np.array(data, dtype=object)
        if data.ndim != 2:
            if data.ndim == 1:
                if len(cols)==1: # A single column, use the data to populate the rows
                    data = np.reshape(data, (len(data),1))
                elif len(data)==len(cols): # A single row, use the data to populate the columns
                    data = np.reshape(data, (1,len(data)))
                else:
                    errormsg = 'Dimension of data can only be 1 if there is 1 column, not %s' % len(cols)
                    raise Exception(errormsg)
            else:
                errormsg = 'Dimension of data must be 1 or 2, not %s' % data.ndim
                raise Exception(errormsg)
        if data.shape[1]==len(cols):
            pass
        elif data.shape[0]==len(cols):
            data = data.transpose()
        else:
            errormsg = 'Number of columns (%s) does not match array shape (%s)' % (len(cols), data.shape)
            raise Exception(errormsg)

        # Store it
        self.cols = list(cols)
        self.data = data
        return
    


##############################################################################
#%% Parallelization
##############################################################################


def parallelcmd(cmd=None, parfor=None, returnval=None, maxcpu=None, maxmem=None, interval=None, die=True, **kwargs):
    '''
    A function to parallelize any block of code. Note: this is intended for quick
    prototyping only; since it uses exec(), it is not recommended for use in production
    code.

    Args:
        cmd       (str):   a string representation of the code to be run in parallel
        parfor    (dict):  a dictionary of lists of the variables to loop over
        returnval (str):   the name of the output variable
        maxcpu    (float): maximum CPU load; used by ``sc.loadbalancer()``
        maxmem    (float): maximum fraction of virtual memory (RAM); used by ``sc.loadbalancer()``
        interval  (float): the time delay to poll to see if load is OK,  used in ``sc.loadbalancer()``
        die       (bool):  whether to stop immediately if an exception is encountered (otherwise, store the exception as the result)
        kwargs    (dict):  variables to pass into the code

    **Example**::

        const = 4
        parfor = {'val':[3,5,9]}
        returnval = 'result'
        cmd = """
        newval = val+const
        result = newval**2
        """
        results = sc.parallelcmd(cmd=cmd, parfor=parfor, returnval=returnval, const=const)

    | New in version 2.0.0: replaced ``maxload`` with ``maxcpu``/``maxmem``; automatically de-indent the command
    | Migrated to ``sc_legacy`` in version 2.2.0.
    '''

    # Handle maxload
    maxload = kwargs.pop('maxload', None)
    if maxload is not None: # pragma: no cover
        maxcpu = maxload
        warnmsg = 'sc.loadbalancer() argument "maxload" has been renamed "maxcpu" as of v2.0.0'
        warnings.warn(warnmsg, category=FutureWarning, stacklevel=2)

    # Deindent the command
    cmd = textwrap.dedent(cmd)

    # Create queue
    nfor = len(list(parfor.values())[0])
    outputqueue = mp.Queue()
    outputlist = np.empty(nfor, dtype=object)
    processes = []
    for i in range(nfor):
        args = (cmd, parfor, returnval, i, outputqueue, maxcpu, maxmem, interval, die, kwargs)
        prc = mp.Process(target=_parallelcmd_task, args=args)
        prc.start()
        processes.append(prc)
    for i in range(nfor):
        _i,returnval = outputqueue.get()
        outputlist[_i] = returnval
    for prc in processes:
        prc.join() # Wait for them to finish

    outputlist = outputlist.tolist()

    return outputlist


def parallel_progress(fcn, inputs, num_workers=None, show_progress=True, initializer=None): # pragma: no cover
    """
    Run a function in parallel with a optional single progress bar

    The result is essentially equivalent to::

        >>> list(map(fcn, inputs))

    But with execution in parallel and with a single progress bar being shown.

    Args:
        fcn (function): Function object to call, accepting one argument, OR a function with zero arguments in which case inputs should be an integer
        inputs (list): A collection of inputs that will each be passed to the function OR a number, if the fcn() has no input arguments
        num_workers (int): Number of processes, defaults to the number of CPUs
        show_progress (bool): Whether to show a progress bar
        initializer (func): A function that each worker process will call when it starts

    Returns:
        A list of outputs

    | New in version 1.0.0.
    | Migrated to ``sc_legacy`` in version 2.2.0.
    """
    try:
        from tqdm import tqdm
    except ModuleNotFoundError as E:
        errormsg = 'Module tqdm not found; please install with "pip install tqdm"'
        raise ModuleNotFoundError(errormsg) from E

    pool = mp.Pool(num_workers, initializer=initializer)

    results = [None]
    if scu.isnumber(inputs):
        results *= inputs
        pbar = tqdm(total=inputs) if show_progress else None
    else:
        results *= len(inputs)
        pbar = tqdm(total=len(inputs)) if show_progress else None

    def callback(result, idx):
        results[idx] = result
        if show_progress:
            pbar.update(1)

    if scu.isnumber(inputs):
        for i in range(inputs):
            pool.apply_async(fcn, callback=partial(callback, idx=i))
    else:
        for i, x in enumerate(inputs):
            pool.apply_async(fcn, args=(x,), callback=partial(callback, idx=i))

    pool.close()
    pool.join()

    if show_progress:
        pbar.close()

    return results


def _parallelcmd_task(_cmd, _parfor, _returnval, _i, _outputqueue, _maxcpu, _maxmem, _interval, _die, _kwargs): # pragma: no cover # No coverage since pickled
    '''
    The task to be executed by ``sc.parallelcmd()``. All internal variables start with
    underscores to avoid possible collisions in the ``exec()`` statements. Not to be called
    directly.
    
    Migrated to ``sc_legacy`` in version 2.2.0.
    '''
    if _maxcpu or _maxmem:
        scp.loadbalancer(maxcpu=_maxcpu, maxmem=_maxmem, index=_i, interval=_interval)

    # Set the loop variables
    for _key in _parfor.keys():
        _thisval = _parfor[_key][_i] # analysis:ignore
        exec(f'{_key} = _thisval') # Set the value of this variable

    # Set the keyword arguments
    for _key in _kwargs.keys():
        _thisval = _kwargs[_key] # analysis:ignore
        exec(f'{_key} = _thisval') # Set the value of this variable

    # Run the command
    try:
        exec(_cmd) # The meat of the matter!
    except Exception:
        if _die:
            raise Exception
        else:
            warnmsg = f'sc.parallelcmd(): Task {_i} failed, but die=False so continuing.\n{scu.traceback()}'
            warnings.warn(warnmsg, category=RuntimeWarning, stacklevel=2)
            exec(f'{_returnval} = None')

    # Append results
    _outputqueue.put((_i,eval(_returnval)))

    return