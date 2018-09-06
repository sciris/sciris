##############################################################################
### DATA FRAME CLASS
##############################################################################

import numpy as np
from . import sc_utils as ut # Note, sc_fileio is also used, but is only imported when required to avoid a circular import
from .sc_odict import odict

__all__ = ['dataframe']

class dataframe(object):
    '''
    A simple data frame, based on simple lists, for simply storing simple data.
    
    Example usage:
        a = dataframe(cols=['x','y'],data=[[1238,2],[384,5],[666,7]]) # Create data frame
        print(a)['x'] # Print out a column
        print(a)[0] # Print out a row
        print(a)['x',0] # Print out an element
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
    
    Works for both numeric and non-numeric data.
    
    Version: 2018mar27
    '''

    def __init__(self, cols=None, data=None):
        
        # Handle columns        
        if cols is None: cols = list()
        else:            cols = ut.promotetolist(cols)
        
        # Handle data
        if data is None: 
            data = np.zeros((0,len(cols)), dtype=object) # Object allows more than just numbers to be stored
        else:
            data = np.array(data, dtype=object)
            if data.ndim != 2:
                if data.ndim == 1:
                    if len(cols)==1:
                        data = np.reshape(data, (len(data),1))
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
        self.cols = cols
        self.data = data
        self.shape = (self.nrows(), self.ncols())
        return None
    
    def __repr__(self, spacing=2):
        ''' spacing = space between columns '''
        if not self.cols: # No keys, give up
            return ''
        
        else: # Go for it
            outputlist = odict()
            outputformats = odict()
            
            # Gather data
            nrows = self.nrows()
            for c,col in enumerate(self.cols):
                outputlist[col] = list()
                maxlen = len(col) # Start with length of column name
                if nrows:
                    for val in self.data[:,c]:
                        output = ut.flexstr(val)
                        maxlen = max(maxlen, len(output))
                        outputlist[col].append(output)
                outputformats[col] = '%'+'%i'%(maxlen+spacing)+'s'
            
            indformat = '%%%is' % (np.floor(np.log10(nrows))+1) # Choose the right number of digits to print
            
            # Assemble output
            output = indformat % '' # Empty column for index
            for col in self.cols: # Print out header
                output += outputformats[col] % col
            output += '\n'
            
            for ind in range(nrows): # Loop over rows to print out
                output += indformat % ut.flexstr(ind)
                for col in self.cols: # Print out data
                    output += outputformats[col] % outputlist[col][ind]
                if ind<nrows-1: output += '\n'
            
            return output
    
    def _val2row(self, value=None):
        ''' Convert a list, array, or dictionary to the right format for appending to a dataframe '''
        if isinstance(value, dict):
            output = np.zeros(self.ncols(), dtype=object)
            for c,col in enumerate(self.cols):
                try: 
                    output[c] = value[col]
                except: 
                    errormsg = 'Entry for column %s not found; keys you supplied are: %s' % (col, value.keys())
                    raise Exception(errormsg)
            output = np.array(output, dtype=object)
        elif value is None:
            output = np.empty(self.ncols(),dtype=object)
        else: # Not sure what it is, just make it an array
            if len(value)==self.ncols():
                output = np.array(value, dtype=object)
            else:
                errormsg = 'Row has wrong length (%s supplied, %s expected)' % (len(value), self.ncols())
                raise Exception(errormsg)
        return output
    
    def _sanitizecol(self, col):
        ''' Take None or a string and return the index of the column '''
        if col is None: output = 0 # If not supplied, assume first column is control
        elif isinstance(col, ut._stringtype): output = self.cols.index(col) # Convert to index
        else: output = col
        return output
    
    def __getitem__(self, key=None):
        ''' Simple method for returning; see self.get() for a version based on col and row '''
        if isinstance(key, ut._stringtype):
            colindex = self.cols.index(key)
            output = self.data[:,colindex]
        elif isinstance(key, ut._numtype):
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
        
    def __setitem__(self, key, value=None):
        if value is None:
            value = np.zeros(self.nrows(), dtype=object)
        if isinstance(key, ut._stringtype): # Add column
            if len(value) != self.nrows(): 
                errormsg = 'Vector has incorrect length (%i vs. %i)' % (len(value), self.nrows())
                raise Exception(errormsg)
            try:
                colindex = self.cols.index(key)
                val_arr = np.reshape(value, (len(value),))
                self.data[:,colindex] = val_arr
            except:
                self.cols.append(key)
                colindex = self.cols.index(key)
                val_arr = np.reshape(value, (len(value),1))
                self.data = np.hstack((self.data, np.array(val_arr, dtype=object)))
        elif isinstance(key, ut._numtype):
            value = self._val2row(value) # Make sure it's in the correct format
            if len(value) != self.ncols(): 
                errormsg = 'Vector has incorrect length (%i vs. %i)' % (len(value), self.ncols())
                raise Exception(errormsg)
            rowindex = int(key)
            try:
                self.data[rowindex,:] = value
            except:
                self.data = np.vstack((self.data, np.array(value, dtype=object)))
        elif isinstance(key, tuple):
            try:
                colindex = self.cols.index(key[0])
                rowindex = int(key[1])
                self.data[rowindex,colindex] = value
            except:
                errormsg = 'Could not insert element (%s,%s) in dataframe of shape %' % (key[0], key[1], self.data.shape)
                raise Exception(errormsg)
        self.shape = (self.nrows(), self.ncols())
        return None
    
    def get(self, cols=None, rows=None):
        '''
        More complicated way of getting data from a dataframe; example:
        df = dataframe(cols=['x','y','z'],data=[[1238,2,-1],[384,5,-2],[666,7,-3]]) # Create data frame
        df.get(cols=['x','z'], rows=[0,2])
        '''
        if cols is None:
            colindices = Ellipsis
        else:
            colindices = []
            for col in ut.promotetolist(cols):
                colindices.append(self._sanitizecol(col))
        if rows is None:
            rowindices = Ellipsis
        else:
            rowindices = rows
        
        output = self.data[:,colindices][rowindices,:] # Split up so can handle non-consecutive entries in either
        if output.size == 1: output = output[0] # If it's a single element, return the value rather than the array
        return output
   
    def pop(self, key, returnval=True):
        ''' Remove a row from the data frame '''
        rowindex = int(key)
        thisrow = self.data[rowindex,:]
        self.data = np.vstack((self.data[:rowindex,:], self.data[rowindex+1:,:]))
        if returnval: return thisrow
        else:         return None
    
    def append(self, value):
        ''' Add a row to the end of the data frame '''
        value = self._val2row(value) # Make sure it's in the correct format
        self.data = np.vstack((self.data, np.array(value, dtype=object)))
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
    
    def addcol(self, key=None, value=None):
        ''' Add a new column to the data frame -- for consistency only '''
        self.__setitem__(key, value)
    
    def rmcol(self, key):
        ''' Remove a column from the data frame '''
        colindex = self.cols.index(key)
        self.cols.pop(colindex) # Remove from list of columns
        self.data = np.hstack((self.data[:,:colindex], self.data[:,colindex+1:])) # Remove from data
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
        ''' Like pop, but removes by matching the first column instead of the index -- WARNING, messy '''
        index = self._rowindex(key=key, col=col, die=die)
        if index is not None: self.pop(index)
        return None
    
    def _diffindices(self, indices=None):
        ''' For a given set of indices, get the inverse, in set-speak '''
        if indices is None: indices = []
        ind_set = set(np.array(indices))
        all_set = set(np.arange(self.nrows()))
        diff_set = np.array(list(all_set - ind_set))
        return diff_set
    
    def rmrows(self, indices=None):
        ''' Remove rows by index -- WARNING, messy '''
        keep_set = self._diffindices(indices)
        self.data = self.data[keep_set,:]
        return None
    
    def replace(self, col=None, old=None, new=None):
        ''' Replace all of one value in a column with a new value '''
        col = self._sanitizecol(col)
        coldata = self.data[:,col] # Get data for this column
        inds = ut.findinds(coldata==old)
        self.data[inds,col] = new
        return None
        
    
    def _todict(self, row):
        ''' Return row as a dict rather than as an array '''
        if len(row)!=len(self.cols): 
            errormsg = 'Length mismatch between "%s" and "%s"' % (row, self.cols)
            raise Exception(errormsg)
        rowdict = odict(zip(self.cols, row))
        return rowdict
    
    def findrow(self, key=None, col=None, default=None, closest=False, die=False, asdict=False):
        '''
        Return a row by searching for a matching value.
        
        Arguments:
            key = the value to look for
            col = the column to look for this value in
            default = the value to return if key is not found (overrides die)
            closest = whether or not to return the closest row (overrides default and die)
            die = whether to raise an exception if the value is not found
            asdict = whether to return results as dict rather than list
        
        Example:
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
    
    def rowindex(self, key=None, col=None):
        ''' Return the indices of all rows matching the given key in a given column. '''
        col = self._sanitizecol(col)
        coldata = self.data[:,col] # Get data for this column
        indices = ut.findinds(coldata==key)
        return indices
        
    def _filterrows(self, key=None, col=None, keep=True, verbose=False):
        ''' Filter rows and either keep the ones matching, or discard them '''
        indices = self.rowindex(key=key, col=col)
        if keep: indices = self._diffindices(indices)
        self.rmrows(indices=indices)
        if verbose: print('Dataframe filtering: %s rows removed based on key="%s", column="%s"' % (len(indices), key, col))
        return None
    
    def filter_in(self, key=None, col=None, verbose=False):
        self._filterrows(key=key, col=col, keep=True, verbose=verbose)
        return None
    
    def filter_out(self, key=None, col=None, verbose=False):
        self._filterrows(key=key, col=col, keep=False, verbose=verbose)
        return None
        
    def insert(self, row=0, value=None):
        ''' Insert a row at the specified location '''
        rowindex = int(row)
        value = self._val2row(value) # Make sure it's in the correct format
        self.data = np.vstack((self.data[:rowindex,:], value, self.data[rowindex:,:]))
        return None
    
    def sort(self, col=None, reverse=False):
        ''' Sort the data frame by the specified column '''
        col = self._sanitizecol(col)
        sortorder = np.argsort(self.data[:,col])
        if reverse: sortorder = np.array(list(reversed(sortorder)))
        self.data = self.data[sortorder,:]
        return None
        
    def jsonify(self, cols=None, rows=None, header=None):
        ''' Export the dataframe to a JSON-compatible format '''
        
        # Handle input arguments
        if cols   is None: cols   = self.cols # Use all columns by default
        if rows   is None: rows   = list(range(self.nrows())) # Use all rows by default
        if header is None: header = True # Include headers
        
        # Handle output
        output = []
        if header: output.append(cols)
        for r in rows:
            thisrow = []
            for col in cols:
                datum = self.get(cols=col,rows=r)
                thisrow.append(datum)
            output.append(thisrow)
        return output
    
    def pandas(self, df=None):
        ''' Function to export to pandas (if no argument) or import from pandas (with an argument) '''
        import pandas as pd # Optional import
        if df is None: # Convert
            output = pd.DataFrame(data=self.data, columns=self.cols)
            return output
        else:
            if type(df) != pd.DataFrame:
                errormsg = 'Can only read pandas dataframes, not %s' % type(df)
                raise Exception(errormsg)
            self.cols = list(df.columns)
            self.data = np.array(df, dtype=object)
            return None
    
    def export(self, filename=None, sheetname=None, close=True):
        from . import sc_fileio as io # Optional import
        for_export = ut.dcp(self.data)
        for_export = np.vstack((self.cols, self.data))
        io.savespreadsheet(filename=filename, data=for_export, sheetname=sheetname, close=close)
        return