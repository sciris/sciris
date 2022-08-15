'''
Legacy methods for handling old pickles (e.g. Python 2 pickles). Included for backwards
compatibility, but not imported into Sciris by default.
'''

import types
import traceback
import gzip as gz
import pickle as pkl
import datetime as dt
import copyreg as cpreg
from io import BytesIO as IO
from contextlib import closing
from . import sc_fileio as scf


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
