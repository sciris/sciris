'''
Simple alternative to the Pandas DataFrame.

This class is rarely used and not well maintained; in most cases, it is probably
better to just use the Pandas one.
'''

##############################################################################
### DATA FRAME CLASS
##############################################################################

# NB: Pandas is imported later
import numbers # For numeric type
import numpy as np
import pandas as pd
from . import sc_utils as scu
from . import sc_math as scm
from . import sc_odict as sco


__all__ = ['dataframe']

class dataframe(pd.DataFrame): # pragma: no cover
    '''
    An extension of the pandas dataframe with additional convenience methods for
    accessing rows and columns and performing other operations.

    **Example**::

        a = sc.dataframe(cols=['x','y'],data=[[1238,2],[384,5],[666,7]]) # Create data frame
        a['x'] # Print out a column
        a[0] # Print out a row
        a['x',0] # Print out an element
        a[0] = [123,6]; print(a) # Set values for a whole row
        a['y'] = [8,5,0]; print(a) # Set values for a whole column
        a['z'] = [14,14,14]; print(a) # Add new column
        a.addcol('z', [14,14,14]); print(a) # Alternate way to add new column
        a.rmcol('z'); print(a) # Remove a column
        a.pop(1); print(a) # Remove a row
        a.append([555,2,14]); print(a) # Append a new row
        a.insert(1,[555,2,14]); print(a) # Insert a new row
        a.sort(); print(a) # Sort by the first column
        a.sort('y'); print(a) # Sort by the second column
        a.addrow([555,2,14]); print(a) # Replace the previous row and sort
        a.getrow(1) # Return the row starting with value '1'
        a.rmrow(); print(a) # Remove last row
        a.rmrow(1238); print(a) # Remove the row starting with element '3'

    The dataframe can be used for both numeric and non-numeric data.

    | New in version 2.0.0: subclass pandas DataFrame
    '''

    def __init__(self, data=None, columns=None, nrows=None, **kwargs):

        # Handle inputs
        if 'cols' in kwargs:
            columns = kwargs.pop('cols')
        if nrows and data is None:
            ncols = len(columns)
            data = np.zeros((nrows, ncols))

        # Create the dataframe
        super().__init__(data=data, columns=columns, **kwargs)
        return


    @property
    def cols(self):
        ''' Get columns as a list '''
        return self.columns.tolist()


    def _val2row(self, value=None):
        ''' Convert a list, array, or dictionary to the right format for appending to a dataframe '''
        if isinstance(value, dict):
            output = np.zeros(self.ncols, dtype=object)
            for c,col in enumerate(self.cols):
                try:
                    output[c] = value[col]
                except:
                    errormsg = 'Entry for column %s not found; keys you supplied are: %s' % (col, value.keys())
                    raise Exception(errormsg)
        elif value is None:
            output = np.empty(self.ncols)
        else: # Not sure what it is, just make it an array
            try:
                output = np.array(value, dtype=self.values.dtype)
            except: # This is to e.g. not force conversion to strings
                output = np.array(value, dtype=object)

        # Validation
        if output.ndim == 1: # Convert from 1D to 2D
            output = np.array([output])
        if output.shape[1] != self.ncols:
            errormsg = 'Row has wrong length (%s supplied, %s expected)' % (len(value), self.ncols)
            raise Exception(errormsg)

        # Try to convert back to default type, but don't worry if not
        try:
            output = np.array(output, dtype=self.values.dtype)
        except:
            pass

        return output


    def _sanitizecol(self, col, die=True):
        ''' Take None or a string and return the index of the column '''
        if col is None:
            output = 0 # If not supplied, assume first column is control
        elif scu.isstring(col):
            try:
                output = self.cols.index(col) # Convert to index
            except Exception as E:
                errormsg = 'Could not get index of column "%s"; columns are:\n%s\n%s' % (col, '\n'.join(self.cols), str(E))
                if die:
                    raise Exception(errormsg)
                else:
                    print(errormsg)
                    output = None
        elif scu.isnumber(col):
            output = col
        else:
            errormsg = 'Unrecognized column/column type "%s" %s' % (col, type(col))
            if die:
                raise Exception(errormsg)
            else:
                print(errormsg)
                output = None
        return output


    @staticmethod
    def _cast(arr):
        ''' Attempt to cast an array to different data types, to avoid having completely numeric arrays of object type '''
        output = arr
        if isinstance(arr, np.ndarray) and np.can_cast(arr, numbers.Number, casting='same_kind'): # Check that everything is a number before trying to cast to an array
            try: # If it is, try to cast the whole array to the type of the first element
                output = np.array(arr, dtype=type(arr[0]))
            except: # If anything goes wrong, do nothing
                pass
        return output


    def __getitem__(self, key=None, die=True, cast=True):
        ''' Simple method for returning; see self.get() for a version based on col and row '''
        try:
            output = super().__getitem__(key)
        except:
            if scu.isstring(key): # e.g. df['a']
                colindex = self.cols.index(key)
                output = self.iloc[:,colindex].values
                if cast:
                    output = self._cast(output)
            elif scu.isnumber(key): # e.g. df[0]
                rowindex = int(key)
                output = self.iloc[rowindex,:]
                if cast:
                    output = self._cast(output)
            elif scu.checktype(key, 'arraylike'): # e.g. df[[0,2,4]]
                rowindices = key
                rowdata = self.iloc[rowindices,:]
                output = dataframe(cols=self.cols, data=rowdata)
            elif isinstance(key, tuple):
                if scu.isstring(key[0]) and scu.isnumber(key[1]): # e.g. df['a',0]
                    colindex = self.cols.index(key[0])
                    rowindex = int(key[1])
                    output = self.iloc[rowindex,colindex]
                elif scu.isstring(key[1]) and scu.isnumber(key[0]): # e.g. df[0,'a']
                    colindex = self.cols.index(key[1])
                    rowindex = int(key[0])
                    output = self.iloc[rowindex,colindex]
                elif isinstance(key[0], slice) or isinstance(key[1], slice): # e.g. df[0,:] or df[:,0]
                    if not isinstance(key[0], slice):
                        rowindices = key[0]
                    elif not isinstance(key[1], slice):
                        rowindices = key[1]
                    else:
                        errormsg = "When using a tuple with a slice as an index, both values can't be slices"
                        raise Exception(errormsg)
                    rowdata = self.iloc[rowindices,:]
                    output = dataframe(cols=self.cols, data=rowdata)
            elif isinstance(key, slice):
                rowslice = key
                slicedata = self.iloc[rowslice,:]
                output = dataframe(cols=self.cols, data=slicedata)
            else:
                errormsg = 'Unrecognized dataframe key "%s"' % key
                if die:
                    raise Exception(errormsg)
                else:
                    print(errormsg)
                    output = None

        return output


    def __setitem__(self, key, value=None):
        try:
            super().__setitem__(key, value)
        except:
            if value is None:
                value = np.zeros(self.nrows, dtype=object)
            if scu.isstring(key): # Add column
                if not scu.isiterable(value):
                    value = scu.promotetolist(value)
                if len(value) != self.nrows:
                    if len(value) == 1:
                        value = [value]*self.nrows # Get it the right length
                    else:
                        if self.ncols==0:
                            self.iloc[:,:] = np.zeros((len(value),0), dtype=object) # Prepare data for writing
                        else:
                            errormsg = 'Cannot add column %s with value %s: incorrect length (%s vs. %s)' % (key, value, len(value), self.nrows)
                            raise Exception(errormsg)
                try:
                    colindex = self.cols.index(key)
                    val_arr = np.reshape(value, (len(value),))
                    self.iloc[:,colindex] = val_arr
                except:
                    self.cols.append(key)
                    colindex = self.cols.index(key)
                    val_arr = np.reshape(value, (len(value),1))
                    self.iloc[:,:] = np.hstack((self.data, np.array(val_arr, dtype=object)))
            elif scu.isnumber(key):
                value = self._val2row(value) # Make sure it's in the correct format
                if len(value) != self.ncols:
                    errormsg = 'Vector has incorrect length (%i vs. %i)' % (len(value), self.ncols)
                    raise Exception(errormsg)
                rowindex = int(key)
                try:
                    self.iloc[rowindex,:] = value
                except:
                    self.appendrow(value)
            elif isinstance(key, tuple):
                if scu.isstring(key[1]) and scu.isnumber(key[0]): # Keys supplied in opposite order
                    key = (key[1], key[0]) # Swap order
                if key[0] not in self.cols:
                    errormsg = f'Column name "{key[0]}" not found; available columns are:\n{scu.newlinejoin(self.columns)}'
                    raise Exception(errormsg)
                try:
                    colindex = self.cols.index(key[0])
                    rowindex = int(key[1])
                    self.iloc[rowindex,colindex] = value
                except Exception as E:
                    errormsg = 'Could not insert element (%s,%s) in dataframe of shape %s: %s' % (key[0], key[1], self.data.shape, str(E))
                    raise Exception(errormsg)
        return


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


    def get(self, cols=None, rows=None, asarray=True, cast=True, default=None):
        '''
        More complicated way of getting data from a dataframe.

        **Example**::

            df = dataframe(cols=['x','y','z'],data=[[1238,2,-1],[384,5,-2],[666,7,-3]]) # Create data frame
            df.get(cols=['x','z'], rows=[0,2])
        '''
        try:
            super().get(cols, default=default)
        except:
            if cols is None:
                colindices = Ellipsis
            else:
                colindices = []
                for col in scu.promotetolist(cols):
                    colindices.append(self._sanitizecol(col))
            if rows is None:
                rowindices = Ellipsis
            else:
                rowindices = rows

            output = self.iloc[rowindices,colindices] # Split up so can handle non-consecutive entries in either
            if output.size == 1: output = output[0] # If it's a single element, return the value rather than the array
            if asarray:
                if cast:
                    output = self._cast(output)
                return output
            else:
                df = dataframe(cols=np.array(self.cols)[colindices].tolist(), data=output)
                return df

    def poprow(self, key, returnval=True):
        ''' Remove a row from the data frame '''
        rowindex = int(key)
        thisrow = self.iloc[rowindex,:]
        self.drop(rowindex, inplace=True)
        if returnval: return thisrow
        else:         return


    def replacedata(self, newdata=None, newdf=None, reset_index=True, inplace=True):
        '''
        Replace data in the dataframe with other data

        Args:
            newdata (array): replace the dataframe's data with these data
            newdf (dataframe): substitute the current dataframe with this one
            reset_index (bool): update the index
            inplace (bool): whether to modify in-place
        '''
        if newdf is None:
            newdf = dataframe(data=newdata, columns=self.columns)
        if reset_index:
            newdf.reset_index(drop=True, inplace=True)
        if inplace:
            self.__dict__ = newdf.__dict__ # Hack to copy in-place
            return
        else:
            return newdf


    def appendrow(self, value, reset_index=True, inplace=True):
        '''
        Add a row to the end of the dataframe. See also ``concat()`` and ``insertrow()``.

        Args:
            value (array): the row(s) to append
            reset_index (bool): update the index
            inplace (bool): whether to modify in-place
        '''
        newrow = self._val2row(value) # Make sure it's in the correct format
        newdata = np.vstack((self.values, newrow))
        return self.replacedata(newdata=newdata, reset_index=reset_index, inplace=inplace)


    def insertrow(self, row=0, value=None, reset_index=True, inplace=True):
        '''
        Insert a row at the specified location. See also ``concat()`` and ``appendrow()``.

        Args:
            row (int): index at which to insert new row(s)
            value (array): the row(s) to insert
            reset_index (bool): update the index
            inplace (bool): whether to modify in-place
        '''
        rowindex = int(row)
        newrow = self._val2row(value) # Make sure it's in the correct format
        newdata = np.vstack((self.data[:rowindex,:], newrow, self.data[rowindex:,:]))
        return self.replacedata(newdata=newdata, reset_index=reset_index, inplace=inplace)


    def concat(self, data, *args, columns=None, reset_index=True, inplace=True, **kwargs):
        '''
        Concatenate additional data onto the current dataframe. See also ``appendrow()``
        and ``insertrow()``.

        Args:
            data (dataframe/array): the data to concatenate
            *args (dataframe/array): additional data to concatenate
            columns (list): if supplied, columns to go with the data
            reset_index (bool): update the index
            inplace (bool): whether to append in place
            **kwargs (dict): passed to ``pd.concat()``
        '''
        dfs = [self]
        if columns is None:
            columns = self.columns
        for arg in [data] + list(args):
            if isinstance(arg, pd.DataFrame):
                df = arg
            else:
                arg = np.array(arg)
                if arg.shape == (self.ncols,): # It's a single row: make 2D
                    arg = np.array([arg])
                df = dataframe(data=arg, columns=columns)
            dfs.append(df)
        newdf = pd.concat(dfs)
        return self.replacedata(newdf=newdf, reset_index=reset_index, inplace=inplace)


    @property
    def ncols(self):
        ''' Get the number of columns in the dataframe '''
        return len(self.columns)


    @property
    def nrows(self):
        ''' Get the number of rows in the dataframe '''
        return len(self)


    def addcol(self, key=None, value=None):
        ''' Add a new column to the data frame -- for consistency only '''
        self.__setitem__(key, value)


    def rmcol(self, key, die=True):
        ''' Remove a column or columns from the data frame '''
        cols = scu.promotetolist(key)
        for col in cols:
            if col not in self.cols:
                errormsg = 'sc.dataframe(): cannot remove column %s: columns are:\n%s' % (col, '\n'.join(self.cols))
                if die: raise Exception(errormsg)
                else:   print(errormsg)
            else:
                self.pop(col)
        return


    def _rowindex(self, key=None, col=None, die=False):
        ''' Get the sanitized row index for a given key and column '''
        col = self._sanitizecol(col)
        coldata = self.data[:,col] # Get data for this column
        if key is None: key = coldata[-1] # If not supplied, pick the last element
        try:    index = coldata.tolist().index(key) # Try to find duplicates
        except:
            if die: raise Exception('Item %s not found; choices are: %s' % (key, coldata))
            else:   return
        return index


    def rmrow(self, key=None, col=None, returnval=False, die=True):
        ''' Like pop, but removes by matching the first column instead of the index '''
        index = self._rowindex(key=key, col=col, die=die)
        if index is not None: self.pop(index, returnval=returnval)
        return


    def _diffindices(self, indices=None):
        ''' For a given set of indices, get the inverse, in set-speak '''
        if indices is None: indices = []
        ind_set = set(np.array(indices))
        all_set = set(np.arange(self.nrows))
        diff_set = np.array(list(all_set - ind_set))
        return diff_set


    def rmrows(self, indices=None, inplace=True):
        ''' Remove rows by index '''
        keep_set = self._diffindices(indices)
        keep_data = self.data[keep_set,:]
        if not inplace:
            output = dataframe(data=keep_data, cols=self.cols)
            return output
        else:
            self.data = keep_data
            return


    def replace(self, col=None, old=None, new=None):
        ''' Replace all of one value in a column with a new value '''
        col = self._sanitizecol(col)
        coldata = self.data[:,col] # Get data for this column
        inds = scm.findinds(coldata==old)
        self.data[inds,col] = new
        return


    def _todict(self, row):
        ''' Return row as a dict rather than as an array '''
        if len(row)!=len(self.cols):
            errormsg = 'Length mismatch between "%s" and "%s"' % (row, self.cols)
            raise Exception(errormsg)
        rowdict = sco.odict(zip(self.cols, row))
        return rowdict


    def findrow(self, key=None, col=None, default=None, closest=False, die=False, asdict=False):
        '''
        Return a row by searching for a matching value.

        Args:
            key: the value to look for
            col: the column to look for this value in
            default: the value to return if key is not found (overrides die)
            closest: whether or not to return the closest row (overrides default and die)
            die: whether to raise an exception if the value is not found
            asdict: whether to return results as dict rather than list

        **Example**::

            df = dataframe(cols=['year','val'],data=[[2016,0.3],[2017,0.5]])
            df.findrow(2016) # returns array([2016, 0.3], dtype=object)
            df.findrow(2013) # returns None, or exception if die is True
            df.findrow(2013, closest=True) # returns array([2016, 0.3], dtype=object)
            df.findrow(2016, asdict=True) # returns {'year':2016, 'val':0.3}
        '''
        if not closest: # Usual case, get
            index = self._rowindex(key=key, col=col, die=(die and default is None))
        else:
            col = self._sanitizecol(col)
            coldata = self.data[:,col] # Get data for this column
            index = np.argmin(abs(coldata-key)) # Find the closest match to the key
        if index is not None:
            thisrow = self.data[index,:]
            if asdict:
                thisrow = self._todict(thisrow)
        else:
            thisrow = default # If not found, return as default
        return thisrow


    def findrows(self, key=None, col=None, asarray=False):
        ''' A method like get() or indexing, but returns a dataframe by default -- WARNING, redundant? '''
        indices = self.rowindex(key=key, col=col)
        arr = self.get(rows=indices)
        if asarray:
            return arr
        else:
            df = dataframe(cols=self.cols, data=arr)
            return df


    def rowindex(self, key=None, col=None):
        ''' Return the indices of all rows matching the given key in a given column. '''
        col = self._sanitizecol(col)
        coldata = self.data[:,col] # Get data for this column
        indices = scm.findinds(coldata==key)
        return indices


    def _filterrows(self, key=None, col=None, keep=True, verbose=False, copy=None):
        ''' Filter rows and either keep the ones matching, or discard them '''
        indices = self.rowindex(key=key, col=col)
        if keep: indices = self._diffindices(indices)
        if verbose: print('Dataframe filtering: %s rows removed based on key="%s", column="%s"' % (len(indices), key, col))
        output = self.rmrows(indices=indices, copy=copy)
        return output


    def filter_in(self, key=None, col=None, verbose=False, copy=None):
        '''Keep only rows matching a criterion (in place) '''
        return self._filterrows(key=key, col=col, keep=True, verbose=verbose, copy=copy)


    def filter_out(self, key=None, col=None, verbose=False, copy=None):
        '''Remove rows matching a criterion (in place) '''
        return self._filterrows(key=key, col=col, keep=False, verbose=verbose, copy=copy)


    def filtercols(self, cols=None, die=True, copy=None):
        ''' Filter columns keeping only those specified '''
        if copy is None: copy = False
        if cols is None: cols = scu.dcp(self.cols) # By default, do nothing
        cols = scu.promotetolist(cols)
        order = []
        notfound = []
        for col in cols:
            if col in self.cols:
                order.append(self.cols.index(col))
            else:
                cols.remove(col)
                notfound.append(col)
        if len(notfound):
            errormsg = 'sc.dataframe(): could not find the following column(s): %s\nChoices are: %s' % (notfound, self.cols)
            if die: raise Exception(errormsg)
            else:   print(errormsg)
        ordered_data = self.data[:,order] # Resort and filter the data
        if copy:
            output = dataframe(cols=cols, data=ordered_data)
            return output
        else:
            self.cols = cols # These should be in the correct order
            self.data = ordered_data
            return


    def sort(self, col=None, reverse=False):
        ''' Sort the data frame by the specified column(s)'''
        if col is None:
            col = 0 # Sort by first column by default
        cols = list(reversed(scu.promotetolist(col)))
        sortorder = [] # In case there are no columns
        for col in cols:
            col = self._sanitizecol(col)
            sortorder = np.argsort(self.data[:,col], kind='mergesort') # To preserve order
            if reverse: sortorder = np.array(list(reversed(sortorder)))
            self.data = self.data[sortorder,:]
        return sortorder


    def sortcols(self, sortorder=None, reverse=False):
        ''' Like sort, but rows by column instead '''
        if sortorder is None:
            sortorder = np.argsort(self.cols, kind='mergesort')
            if reverse: sortorder = np.array(list(reversed(sortorder)))
        self.cols = list(np.array(self.cols)[sortorder])
        self.data = self.data[:,sortorder]
        return sortorder


    def to_pandas(self, df=None):
        ''' Convert to a plain pandas dataframe '''
        return pd.DataFrame(data=self.values, columns=self.cols)