'''
Extension of the pandas dataframe to be more flexible, especially with filtering
rows/columns and concatenating data.
'''

##############################################################################
#%% Dataframe
##############################################################################

import numbers # For numeric type
import numpy as np
import pandas as pd
from . import sc_utils as scu
from . import sc_math as scm
from . import sc_odict as sco


__all__ = ['dataframe']

class dataframe(pd.DataFrame):
    '''
    An extension of the pandas dataframe with additional convenience methods for
    accessing rows and columns and performing other operations, such as adding rows.
    
    Args:
        data (dict/array/dataframe): the data to use
        columns (list): column labels (if a dict is supplied, the value sets the dtype)
        nrows (int): the number of arrows to preallocate (default 0)
        dtypes (list): optional list of data types to set each column to
        kwargs (dict): passed to ``pd.DataFrame()``

    **Examples**::

        a = sc.dataframe(cols=['x','y'], data=[[1238,2],[384,5],[666,7]]) # Create data frame
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
    | New in version 2.2.0: "dtypes" argument
    '''

    def __init__(self, data=None, columns=None, nrows=None, dtypes=None, **kwargs):

        # Handle inputs
        if 'cols' in kwargs:
            columns = kwargs.pop('cols')
        if nrows and data is None:
            ncols = len(columns)
            data = np.zeros((nrows, ncols))
        
        # Handle columns and dtypes
        if isinstance(columns, dict):
            if dtypes is not None:
                errormsg = 'You can supply dtypes as a separate argument or as part of a columns dict, but not both'
                raise ValueError(errormsg)
            dtypes = columns # Already in the right format
            columns = list(columns.keys())
        if isinstance(data, dict):
            if columns is not None:
                colset = sorted(set(columns))
                dataset = sorted(set(data.keys()))
                if colset != dataset:
                    errormsg = f'Incompatible column names provided:\nColumns: {colset}\n   Data: {dataset}'
                    raise ValueError(errormsg)

        # Create the dataframe
        super().__init__(data=data, columns=columns, **kwargs)
        
        # Optionally set dtypes
        if dtypes is not None:
            self.set_dtypes(dtypes)
        
        return


    @property
    def cols(self):
        ''' Get columns as a list '''
        return self.columns.tolist()


    def set_dtypes(self, dtypes):
        '''
        Set dtypes in-place (see pandas ``df.astype()`` for the user-facing version)
        
        New in version 2.2.0.
        '''
        if not isinstance(dtypes, dict):
            dtypes = {col:dtype for col,dtype in zip(self.columns, dtypes)}
        for col,dtype in dtypes.items(): # NB: "self.astype(dtypes, copy=False)" does not modify in place
            self[col] = self[col].astype(dtype)
        return


    # def _to_array(self, arr):
    #     ''' Try to conver to the current data type, or else use an object '''
    #     try: # Try to use current type
    #         output = np.array(arr, dtype=self.values.dtype)
    #     except: # This is to e.g. not force conversion to strings # pragma: no cover
    #         output = np.array(arr, dtype=object)
    #     return output


    # def _val2row(self, value=None, to2d=True):
    #     ''' Convert a list, array, or dictionary to the right format for appending to a dataframe '''
    #     if isinstance(value, dict):
    #         output = np.zeros(self.ncols, dtype=object)
    #         for c,col in enumerate(self.cols):
    #             try:
    #                 output[c] = value[col]
    #             except:
    #                 errormsg = f'Entry for column {col} not found; keys you supplied are: {scu.strjoin(value.keys())}'
    #                 raise Exception(errormsg)
    #     elif value is None:
    #         output = np.empty(self.ncols)
    #     else: # Not sure what it is, just make it an array
    #         output = self._to_array(value)

    #     # Validation
    #     if output.ndim == 1: # Convert from 1D to 2D
    #         shape = output.shape[0]
    #         if to2d:
    #             output = np.array([output])
    #     else:
    #         shape = output.shape[1]
    #     if shape != self.ncols:
    #         errormsg = f'Row has wrong length ({shape} supplied, {self.ncols} expected)'
    #         raise IndexError(errormsg)

    #     # Try to convert back to default type, but don't worry if not
    #     try:
    #         output = np.array(output, dtype=self.values.dtype)
    #     except: # pragma: no cover
    #         pass

    #     return output


    def col_index(self, col, die=True):
        '''
        Get the index of the specified column.
        
        Args:
            col (str): the column to get the index of (return 0 if None)
            die (bool): whether to raise an exception if the column could not be found (else, return NOne)
        
        New in version 2.2.0: renamed from "_sanitizecols"
        '''
        cols = self.cols
        if col is None:
            output = 0 # If not supplied, assume first column is intended
        elif scu.isstring(col):
            try:
                output = cols.index(col) # Convert to index
            except Exception as E: # pragma: no cover
                errormsg = f'Could not get index of column "{col}"; columns are:\n{scu.newlinejoin(cols)}\n{E}'
                if die:
                    raise scu.KeyNotFoundError(errormsg) from E
                else:
                    print(errormsg)
                    output = None
        elif scu.isnumber(col):
            try:
                cols[col]
            except Exception as E:
                errormsg = f'Column "{col}" is not a valid index'
                raise IndexError(errormsg) from E
            output = col
        else: # pragma: no cover
            errormsg = f'Unrecognized column/column type "{col}" {type(col)}'
            if die:
                raise TypeError(errormsg)
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
            except: # If anything goes wrong, do nothing # pragma: no cover
                pass
        return output


    def __getitem__(self, key=None, die=True, cast=True):
        ''' Simple method for returning; see self.flexget() for a version based on col and row '''
        try: # Default to the pandas version
            output = super().__getitem__(key)
        except: # ...but handle a wider variety of keys
            if scu.isstring(key): # e.g. df['a'] -- usually handled by pandas # pragma: no cover
                rowindex = slice(None)
                try:
                    colindex = self.cols.index(key)
                except ValueError:
                    errormsg = f'Key "{key}" is not a valid column; choices are: {scu.strjoin(self.cols)}'
                    raise scu.KeyNotFoundError(errormsg)
            elif isinstance(key, (numbers.Number, list, np.ndarray, slice)): # e.g. df[0], df[[0,2]], df[:4]
                rowindex = key
                colindex = slice(None)
            elif isinstance(key, tuple):
                rowindex = key[0]
                colindex = key[1]
                if scu.isstring(rowindex) and not scu.isstring(colindex): # Swap order if one's a string and the other isn't
                    rowindex, colindex = colindex, rowindex
                if scu.isstring(colindex): # e.g. df['a',0]
                    colindex = self.cols.index(colindex)
            else: # pragma: no cover
                errormsg = f'Unrecognized dataframe key of {type(key)}: must be str, numeric, or tuple'
                if die:
                    raise scu.KeyNotFoundError(errormsg)
                else:
                    print(errormsg)
                    output = None
            output = self.iloc[rowindex,colindex]

        return output


    def set(self, key, value=None):
        ''' Alias to pandas __setitem__ method '''
        return super().__setitem__(key, value)


    def __setitem__(self, key, value=None):
        try:
            # Don't create new non-text columns, only set existing ones; otherwise use regular pandas
            assert isinstance(key, str) or key in self.columns
            super().__setitem__(key, value)
        except Exception as E:
            if scu.isnumber(key):
                newrow = self._val2row(value, to2d=False) # Make sure it's in the correct format
                if len(newrow) != self.ncols:
                    errormsg = f'Vector has incorrect length ({len(newrow)} vs. {self.ncols})'
                    raise Exception(errormsg)
                rowindex = int(key)
                try:
                    self.iloc[rowindex,:] = newrow
                except:
                    self.appendrow(value)
            elif isinstance(key, tuple):
                rowindex = key[0]
                colindex = key[1]
                if scu.isstring(rowindex) and not scu.isstring(colindex): # Swap order if one's a string and the other isn't
                    rowindex, colindex = colindex, rowindex
                if scu.isstring(colindex): # e.g. df['a',0]
                    colindex = self.cols.index(colindex)
                self.iloc[rowindex,colindex] = value
            else:
                exc = type(E)
                errormsg = f'Could not understand key {key}: {E}'
                raise exc(errormsg) from E
        return


    def flexget(self, cols=None, rows=None, asarray=False, cast=True, default=None):
        '''
        More complicated way of getting data from a dataframe. While getting directly
        by key usually returns the array data directly, this usually returns another
        dataframe.

        Args:
            cols (str/list): the column(s) to get
            rows (int/list): the row(s) to get
            asarray (bool): whether to return an array (otherwise, return a dataframe)
            cast (bool): attempt to cast to an all-numeric array
            default (any): the value to return if the column(s)/row(s) can't be found

        **Example**::

            df = sc.dataframe(cols=['x','y','z'],data=[[1238,2,-1],[384,5,-2],[666,7,-3]]) # Create data frame
            df.flexget(cols=['x','z'], rows=[0,2])
        '''
        if cols is None:
            colindices = Ellipsis
        else:
            colindices = []
            for col in scu.tolist(cols):
                colindices.append(self.col_index(col))
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
            output = dataframe(data=output, columns=np.array(self.cols)[colindices].tolist())

        return output


    def disp(self, nrows=None, ncols=None, width=999, precision=4, options=None):
        '''
        Flexible display of a dataframe, showing all rows/columns by default.
        
        Args:
            nrows (int): maximum number of rows to show (default: all)
            ncols (int): maximum number of columns to show (default: all)
            width (int): maximum screen width (default: 999)
            precision (int): number of decimal places to show (default: 4)
            kwargs (dict): passed to ``pd.option_context()``
        
        **Examples**::
            
            df = sc.dataframe(data=np.random.rand(100,10))
            df.disp()
            df.disp(precision=1, ncols=5, options={'display.colheader_justify': 'left'})
        
        New in version 2.0.1.
        '''
        opts = scu.mergedicts({
            'display.max_rows': nrows,
            'display.max_columns': ncols,
            'display.width': width,
            'display.precision': precision,
            }, options
        )
        optslist = [item for pair in opts.items() for item in pair] # Convert from dict to list
        with pd.option_context(*optslist):
            print(self)
        return


    def poprow(self, key, returnval=True):
        '''
        Remove a row from the data frame.
        
        To pop a column, see ``df.pop()``.        
        '''
        rowindex = int(key)
        thisrow = self.iloc[rowindex,:]
        self.drop(rowindex, inplace=True)
        if returnval: return thisrow
        else:         return


    def replacedata(self, newdata=None, newdf=None, reset_index=True, inplace=True, keep_dtypes=True):
        '''
        Replace data in the dataframe with other data

        Args:
            newdata (array): replace the dataframe's data with these data
            newdf (dataframe): substitute the current dataframe with this one
            reset_index (bool): update the index
            inplace (bool): whether to modify in-place
            keep_dtypes (bool): whether to keep original dtypes
        
        New in version 2.2.0: "keep_dtypes" argument
        '''
        dtypes = self.dtypes if keep_dtypes else None
        if newdf is None:
            newdf = dataframe(data=newdata, columns=self.columns, dtypes=dtypes)
        if reset_index:
            newdf.reset_index(drop=True, inplace=True)
        if inplace:
            self.__dict__ = newdf.__dict__ # Hack to copy in-place
            if keep_dtypes:
                self.set_dtypes(dtypes)
            return self
        else:
            return newdf


    def appendrow(self, row, reset_index=True, inplace=True):
        '''
        Add row(s) to the end of the dataframe. 
        
        See also ``concat()`` and ``insertrow()``. Similar to the pandas operation
        ``df.iloc[-1] = ...``, but faster and provides additional type checking.

        Args:
            value (array): the row(s) to append
            reset_index (bool): update the index
            inplace (bool): whether to modify in-place
        
        Note: "appendrow" and "concat" are equivalent, except appendrow() defaults
        to modifying in-place and "concat" defaults to returning a new dataframe.
        
        Warning: modifying dataframes in-place is quite inefficient. For highest
        performance, construct the data in large chunks and then add to the dataframe
        all at once, rather than adding row by row.
        
        **Example**::
            
            import sciris as sc
            import numpy as np

            df = sc.dataframe(dict(
                a = ['foo','bar'], 
                b = [1,2], 
                c = np.random.rand(2)
            ))
            df.appendrow(['cat', 3, 0.3])           # Append a list
            df.appendrow(dict(a='dog', b=4, c=0.7)) # Append a dict
            
        New in version 2.2.0: renamed "value" to "row"; improved performance
        '''
        return self.concat(row, reset_index=reset_index, inplace=inplace)


    def insertrow(self, row=0, value=None, reset_index=True, inplace=True):
        '''
        Insert row(s) at the specified location. See also ``concat()`` and ``appendrow()``.

        Args:
            row (int): index at which to insert new row(s)
            value (array): the row(s) to insert
            reset_index (bool): update the index
            inplace (bool): whether to modify in-place
        
        Warning: modifying dataframes in-place is quite inefficient. For highest
        performance, construct the data in large chunks and then add to the dataframe
        all at once, rather than adding row by row.
        
        **Example**::
            
            import sciris as sc
            import numpy as np

            df = sc.dataframe(dict(
                a = ['foo','cat'], 
                b = [1,3], 
                c = np.random.rand(2)
            ))
            df.insertrow(1, ['bar', 2, 0.2])           # Insert a list
            df.insertrow(0, dict(a='rat', b=0, c=0.7)) # Insert a dict
        '''
        rowindex = int(row)
        newrow = self._val2row(value) # Make sure it's in the correct format
        newdata = np.vstack((self.iloc[:rowindex,:], newrow, self.iloc[rowindex:,:]))
        return self.replacedata(newdata=newdata, reset_index=reset_index, inplace=inplace)


    def concat(self, data, *args, columns=None, reset_index=True, inplace=False, dfargs=None, keep_dtypes=True, **kwargs):
        '''
        Concatenate additional data onto the current dataframe. See also ```appendrow()``
        and ``insertrow()``.

        Args:
            data (dataframe/array): the data to concatenate
            *args (dataframe/array): additional data to concatenate
            columns (list): if supplied, columns to go with the data
            reset_index (bool): update the index
            inplace (bool): whether to append in place
            keep_dtypes (bool): whether to preserve original dtypes
            dfargs (dict): arguments passed to construct each dataframe
            **kwargs (dict): passed to ``pd.concat()``
        
        | New in version 2.0.2: "inplace" defaults to False
        | New in version 2.2.0: improved type handling
        '''
        dfargs = scu.mergedicts(dfargs)
        dfs = [self]
        if columns is None:
            columns = self.columns
        for arg in [data] + list(args):
            if isinstance(arg, pd.DataFrame):
                df = arg
            else:
                if isinstance(arg, dict):
                    columns = list(arg.keys())
                    arg = list(arg.values())
                argarray = np.array(arg)
                if argarray.shape == (self.ncols,): # It's a single row: make 2D
                    arg = [arg]
                df = dataframe(data=arg, columns=columns, **dfargs)
            dfs.append(df)
        newdf = dataframe(pd.concat(dfs, **kwargs))
        return self.replacedata(newdf=newdf, reset_index=reset_index, inplace=inplace, keep_dtypes=keep_dtypes)


    @staticmethod
    def cat(data, *args, dfargs=None, **kwargs):
        '''
        Convenience method for concatenating multiple dataframes.
        
        Args:
            data (dataframe/array): the dataframe/data to use as the basis of the new dataframe
            args (list): additional dataframes (or object that can be converted to dataframes) to concatenate
            dfargs (dict): arguments passed to construct each dataframe
            kwargs (dict): passed to ``sc.dataframe.concat()``
        
        **Example**::
            
            arr1 = np.random.rand(6,3)
            df2 = pd.DataFrame(np.random.rand(4,3))
            df3 = sc.dataframe.cat(arr1, df2)
        
        New in version 2.0.2.
        '''
        dfargs = scu.mergedicts(dfargs)
        df = dataframe(data, **dfargs)
        if len(args):
            df = df.concat(*args, dfargs=dfargs, **kwargs)
        return df
            

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
        return self.__setitem__(key, value)


    def rmcol(self, key, die=True):
        ''' Remove a column or columns from the data frame '''
        cols = scu.tolist(key)
        for col in cols:
            if col not in self.columns: # pragma: no cover
                errormsg = f'sc.dataframe(): cannot remove column {col}: columns are:\n{scu.newlinejoin(self.cols)}'
                if die: raise Exception(errormsg)
                else:   print(errormsg)
            else:
                self.pop(col)
        return self


    def row_index(self, value, col=None, closest=False, die=True):
        '''
        Get the row index for a given value and column.
        
        See ``df.findrow()`` for the equivalent to return the row itself
        rather than the index of the row.
        
        Args:
            value (any): the value to look for
            col (str): the column to look in (default: first)
            closest (bool): if true, return the closest match if an exact match is not found
            die (bool): whether to raise an exception if the value is not found (otherwise, return None)
        
        **Example**::
            
            df = dataframe(data=[[2016,0.3],[2017,0.5]], columns=['year','val'])
            df.row_index(2016) # returns 0
            df.findrow(2013) # returns None, or exception if die is True
            df.findrow(2013, closest=True) # returns 0
        
        New in version 2.2.0: renamed from "_rowindex"
        '''
        col = self.col_index(col)
        coldata = self.iloc[:,col].values # Get data for this column
        if closest:
            index = np.argmin(abs(coldata-value)) # Find the closest match to the key
        else:
            try:
                index = coldata.tolist().index(value) # Try to find duplicates
            except: # pragma: no cover
                if die:
                    errormsg = f'Item {value} not found; choices are: {coldata}'
                    raise IndexError(errormsg)
                else:
                    return
        return index


    def rmrow(self, value=None, col=None, returnval=False, die=True):
        ''' Like pop, but removes by matching the value in the given column instead of the index '''
        index = self.row_index(value=value, col=col, die=die)
        if index is not None: self.poprow(index, returnval=returnval)
        return self


    def _diffinds(self, inds=None):
        ''' For a given set of indices, get the inverse, in set-speak '''
        if inds is None: inds = []
        ind_set = set(np.array(inds))
        all_set = set(np.arange(self.nrows))
        diff_set = np.array(list(all_set - ind_set))
        return diff_set


    def rmrows(self, inds=None, reset_index=True, inplace=True):
        ''' Remove multiple rows by index '''
        keep_set = self._diffinds(inds)
        keep_data = self.iloc[keep_set,:]
        newdf = dataframe(data=keep_data, cols=self.cols)
        return self.replacedata(newdf=newdf, reset_index=reset_index, inplace=inplace)


    def replacecol(self, col=None, old=None, new=None):
        ''' Replace all of one value in a column with a new value '''
        col = self.col_index(col)
        coldata = self.iloc[:,col] # Get data for this column
        inds = scm.findinds(arr=coldata, val=old)
        self.iloc[inds,col] = new
        return self


    def to_odict(self, row=None):
        '''
        Convert dataframe to a dict of columns, optionally specifying certain rows.

        Args:
            row (int/list): the rows to include
        '''
        if row is None:
            row = slice(None)
        data = self.iloc[row,:].values
        datadict = {col:data[:,c] for c,col in enumerate(self.cols)}
        output = sco.odict(datadict)
        return output


    def findrow(self, value=None, col=None, default=None, closest=False, asdict=False, die=False):
        '''
        Return a row by searching for a matching value.
        
        See ``df.row_index()`` for the equivalent to return the index of the row
        rather than the row itself.

        Args:
            value (any): the value to look for
            col (str): the column to look for this value in
            default (any): the value to return if key is not found (overrides die)
            closest (bool): whether or not to return the closest row (overrides default and die)
            returnind (bool): return the index of the row rather than the row itself
            asdict (bool): whether to return results as dict rather than list
            die (bool): whether to raise an exception if the value is not found

        **Examples**::

            df = dataframe(cols=['year','val'],data=[[2016,0.3],[2017,0.5]])
            df.findrow(2016) # returns array([2016, 0.3], dtype=object)
            df.findrow(2013) # returns None, or exception if die is True
            df.findrow(2013, closest=True) # returns array([2016, 0.3], dtype=object)
            df.findrow(2016, asdict=True) # returns {'year':2016, 'val':0.3}
        '''
        index = self.row_index(value=value, col=col, die=(die and default is None), closest=closest)
        if index is not None:
            thisrow = self.iloc[index,:].values
            if asdict:
                thisrow = self.to_odict(thisrow)
        else:
            thisrow = default # If not found, return as default
        return thisrow


    def findinds(self, value=None, col=None):
        ''' Return the indices of all rows matching the given key in a given column. '''
        col = self.col_index(col)
        coldata = self.iloc[:,col].values # Get data for this column
        inds = scm.findinds(arr=coldata, val=value)
        return inds


    def _filterrows(self, inds=None, value=None, col=None, keep=True, verbose=False, reset_index=True, inplace=False):
        ''' Filter rows and either keep the ones matching, or discard them '''
        if inds is None:
            inds = self.findinds(value=value, col=col)
        if keep: inds = self._diffinds(inds)
        if verbose: print(f'Dataframe filtering: {len(inds)} rows removed based on key="{inds}", column="{col}"')
        output = self.rmrows(inds=inds, reset_index=reset_index, inplace=inplace)
        return output


    def filterin(self, inds=None, value=None, col=None, verbose=False, reset_index=True, inplace=False):
        '''Keep only rows matching a criterion '''
        return self._filterrows(inds=inds, value=value, col=col, keep=True, verbose=verbose, reset_index=reset_index, inplace=inplace)


    def filterout(self, inds=None, value=None, col=None, verbose=False, reset_index=True, inplace=False):
        '''Remove rows matching a criterion (in place) '''
        return self._filterrows(inds=inds, value=value, col=col, keep=False, verbose=verbose, reset_index=reset_index, inplace=inplace)


    def filtercols(self, cols=None, die=True, reset_index=True, inplace=False):
        ''' Filter columns keeping only those specified -- note, by default, do not perform in place '''
        if cols is None: cols = scu.dcp(self.cols) # By default, do nothing
        cols = scu.tolist(cols)
        order = []
        notfound = []
        for col in cols:
            if col in self.cols:
                order.append(self.cols.index(col))
            else:
                cols.remove(col)
                notfound.append(col)
        if len(notfound): # pragma: no cover
            errormsg = 'sc.dataframe(): could not find the following column(s): %s\nChoices are: %s' % (notfound, self.cols)
            if die: raise Exception(errormsg)
            else:   print(errormsg)
        ordered_data = self.iloc[:,order] # Resort and filter the data
        newdf = dataframe(cols=cols, data=ordered_data)
        return self.replacedata(newdf=newdf, reset_index=reset_index, inplace=inplace)


    def sortrows(self, by=None, reverse=False, returninds=False, inplace=True, **kwargs):
        '''
        Sort the dataframe rows in place by the specified column(s).
        
        Similar to ``df.sort_values()``, except defaults to sorting in place, and
        optionally returns the indices used for sorting (like ``np.argsort()``).
        
        Args:
            col (str or int): column to sort by (default, first column)
            reverse (bool): whether to reverse the sort order (i.e., ascending=False)
            returninds (bool): whether to return the indices used to sort instead of the dataframe
            inplace (bool): whether to modify the dataframe in-place
            kwargs (dict): passed to ``df.sort_values()``
        
        New in version 2.2.0: "inplace" argument; "col" argument renamed "by"
        '''
        by = kwargs.pop('col', by) # Handle deprecation
        ascending = kwargs.pop('ascending', not(reverse))
        if by is None:
            by = 0 # Sort by first column by default
        if isinstance(by, int):
            by = self.columns[by]
        if returninds:
            sortorder = np.argsort(self[by].values, kind='mergesort') # To preserve order
        df = self.sort_values(by=by, ascending=ascending, inplace=inplace, **kwargs)
        if returninds:
            return sortorder
        else:
            if inplace:
                return self
            else:
                return df


    def sortcols(self, sortorder=None, reverse=False, inplace=True):
        '''
        Like sortrows(), but change column order (usually in place) instead.
        
        Args:
            sortorder (list): the list of indices to resort the columns by (if none, then alphabetical)
            reverse (bool): whether to reverse the order
            inplace (bool): whether to modify the dataframe in-place
        
        New in version 2.2.0: Ensure dtypes are preserved; "inplace" argument; "returninds" argument removed
        '''
        if sortorder is None:
            sortorder = np.argsort(self.cols, kind='mergesort')
            if reverse:
                sortorder = sortorder[::-1]
        newcols = list(np.array(self.cols)[sortorder])
        newdf = dataframe({k:self[k] for k in newcols})
        return self.replacedata(newdf=newdf, inplace=inplace)


    @staticmethod
    def from_dict(*args, **kwargs):
        return dataframe(super().from_dict(*args, **kwargs))

    @staticmethod
    def from_records(*args, **kwargs):
        return dataframe(super().from_records(*args, **kwargs))

    def to_pandas(self, **kwargs):
        ''' Convert to a plain pandas dataframe '''
        return pd.DataFrame(data=self.values, columns=self.cols, **kwargs)