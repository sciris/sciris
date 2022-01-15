'''
Time/date utilities.

Highlights:
    - ``sc.tic()/sc.toc()/sc.timer()``: simple methods for timing durations
    - ``sc.readdate()``: convert strings to dates using common formats
    - ``sc.daterange()``: create a list of dates
    - ``sc.datedelta()``: perform calculations on date strings
'''

import time
import warnings
import pylab as pl
import datetime as dt
import dateutil as du
from . import sc_utils as scu


###############################################################################
#%% Date functions
###############################################################################

__all__ = ['now', 'getdate', 'readdate', 'date', 'day', 'daydiff', 'daterange', 'datedelta', 'datetoyear']


def now(astype='dateobj', timezone=None, utc=False, dateformat=None):
    '''
    Get the current time as a datetime object, optionally in UTC time.

    ``sc.now()`` is similar to ``sc.getdate()``, but ``sc.now()`` returns a datetime
    object by default, while ``sc.getdate()`` returns a string by default.

    Args:
        astype (str): what to return; choices are "dateobj", "str", "float"; see ``sc.getdate()`` for more
        timezone (str): the timezone to set the itme to
        utc (bool): whether the time is specified in UTC time
        dateformat (str): if ``astype`` is ``'str'``, use this output format

    **Examples**::

        sc.now() # Return current local time, e.g. 2019-03-14 15:09:26
        sc.now(timezone='US/Pacific') # Return the time now in a specific timezone
        sc.now(utc=True) # Return the time in UTC
        sc.now(astype='str') # Return the current time as a string instead of a date object; use 'int' for seconds
        sc.now(tostring=True) # Backwards-compatible alias for astype='str'
        sc.now(dateformat='%Y-%b-%d') # Return a different date format

    New in version 1.3.0: made "astype" the first argument; removed "tostring" argument
    '''
    if isinstance(utc, str): timezone = utc # Assume it's a timezone
    if timezone is not None: tzinfo = du.tz.gettz(timezone) # Timezone is a string
    elif utc:                tzinfo = du.tz.tzutc() # UTC has been specified
    else:                    tzinfo = None # Otherwise, do nothing
    timenow = dt.datetime.now(tzinfo)
    output = getdate(timenow, astype=astype, dateformat=dateformat)
    return output


def getdate(obj=None, astype='str', dateformat=None):
        '''
        Alias for converting a date object to a formatted string.

        See also ``sc.now()``.

        Args:
            obj (datetime): the datetime object to convert
            astype (str): what to return; choices are "str" (default), "dateobj", "float" (full timestamp), "int" (timestamp to second precision)
            dateformat (str): if ``astype`` is ``'str'``, use this output format

        **Examples**::

            sc.getdate() # Returns a string for the current date
            sc.getdate(astype='float') # Convert today's time to a timestamp
        '''
        if obj is None:
            obj = now()

        if dateformat is None:
            dateformat = '%Y-%b-%d %H:%M:%S'
        else:
            astype = 'str' # If dateformat is specified, assume type is a string

        try:
            if scu.isstring(obj):
                return obj # Return directly if it's a string
            obj.timetuple() # Try something that will only work if it's a date object
            dateobj = obj # Test passed: it's a date object
        except Exception as E: # pragma: no cover # It's not a date object
            errormsg = f'Getting date failed; date must be a string or a date object: {repr(E)}'
            raise TypeError(errormsg)

        timestamp = obj.timestamp()
        if   astype == 'str':     output = dateobj.strftime(dateformat)
        elif astype == 'int':     output = int(timestamp)
        elif astype == 'dateobj': output = dateobj
        elif astype in  ['float', 'number', 'timestamp']:
            output = timestamp
        else: # pragma: no cover
            errormsg = f'"astype={astype}" not understood; must be "str" or "int"'
            raise ValueError(errormsg)
        return output


def readdate(datestr=None, *args, dateformat=None, return_defaults=False):
    '''
    Convenience function for loading a date from a string. If dateformat is None,
    this function tries a list of standard date types.

    By default, a numeric date is treated as a POSIX (Unix) timestamp. This can be changed
    with the ``dateformat`` argument, specifically:

    - 'posix'/None: treat as a POSIX timestamp, in seconds from 1970
    - 'ordinal'/'matplotlib': treat as an ordinal number of days from 1970 (Matplotlib default)

    Args:
        datestr (int, float, str or list): the string containing the date, or the timestamp (in seconds), or a list of either
        args (list): additional dates to convert
        dateformat (str or list): the format for the date, if known; if 'dmy' or 'mdy', try as day-month-year or month-day-year formats; can also be a list of options
        return_defaults (bool): don't convert the date, just return the defaults

    Returns:
        dateobj (date): a datetime object

    **Examples**::

        dateobj  = sc.readdate('2020-03-03') # Standard format, so works
        dateobj  = sc.readdate('04-03-2020', dateformat='dmy') # Date is ambiguous, so need to specify day-month-year order
        dateobj  = sc.readdate(1611661666) # Can read timestamps as well
        dateobj  = sc.readdate(16166, dateformat='ordinal') # Or ordinal numbers of days, as used by Matplotlib
        dateobjs = sc.readdate(['2020-06', '2020-07'], dateformat='%Y-%m') # Can read custom date formats
        dateobjs = sc.readdate('20200321', 1611661666) # Can mix and match formats
    '''

    # Define default formats
    formats_to_try = {
        'date':           '%Y-%m-%d', # 2020-03-21
        'date-slash':     '%Y/%m/%d', # 2020/03/21
        'date-dot':       '%Y.%m.%d', # 2020.03.21
        'date-space':     '%Y %m %d', # 2020 03 21
        'date-alpha':     '%Y-%b-%d', # 2020-Mar-21
        'date-alpha-rev': '%d-%b-%Y', # 21-Mar-2020
        'date-alpha-sp':  '%d %b %Y', # 21 Mar 2020
        'date-Alpha':     '%Y-%B-%d', # 2020-March-21
        'date-Alpha-rev': '%d-%B-%Y', # 21-March-2020
        'date-Alpha-sp':  '%d %B %Y', # 21 March 2020
        'date-numeric':   '%Y%m%d',   # 20200321
        'datetime':       '%Y-%m-%d %H:%M:%S',    # 2020-03-21 14:35:21
        'datetime-alpha': '%Y-%b-%d %H:%M:%S',    # 2020-Mar-21 14:35:21
        'default':        '%Y-%m-%d %H:%M:%S.%f', # 2020-03-21 14:35:21.23483
        'ctime':          '%a %b %d %H:%M:%S %Y', # Sat Mar 21 23:09:29 2020
        }

    # Define day-month-year formats
    dmy_formats = {
        'date':           '%d-%m-%Y', # 21-03-2020
        'date-slash':     '%d/%m/%Y', # 21/03/2020
        'date-dot':       '%d.%m.%Y', # 21.03.2020
        'date-space':     '%d %m %Y', # 21 03 2020
    }

    # Define month-day-year formats
    mdy_formats = {
        'date':           '%m-%d-%Y', # 03-21-2020
        'date-slash':     '%m/%d/%Y', # 03/21/2020
        'date-dot':       '%m.%d.%Y', # 03.21.2020
        'date-space':     '%m %d %Y', # 03 21 2020
    }

    # To get the available formats
    if return_defaults:
        return formats_to_try

    # Handle date formats
    format_list = scu.promotetolist(dateformat, keepnone=True) # Keep none which signifies default
    if dateformat is not None:
        if dateformat == 'dmy':
            formats_to_try = dmy_formats
        elif dateformat == 'mdy':
            formats_to_try = mdy_formats
        else:
            formats_to_try = {}
            for f,fmt in enumerate(format_list):
                formats_to_try[f'User supplied {f}'] = fmt

    # Ensure everything is in a consistent format
    datestrs, is_list, is_array = scu._sanitize_iterables(datestr, *args)

    # Actually process the dates
    dateobjs = []
    for datestr in datestrs: # Iterate over them
        dateobj = None
        exceptions = {}
        if isinstance(datestr, dt.datetime):
            dateobj = datestr # Nothing to do
        elif scu.isnumber(datestr):
            if 'posix' in format_list or None in format_list:
                dateobj = dt.datetime.fromtimestamp(datestr)
            elif 'ordinal' in format_list or 'matplotlib' in format_list:
                dateobj = pl.num2date(datestr)
            else:
                errormsg = f'Could not convert numeric date {datestr} using available formats {scu.strjoin(format_list)}; must be "posix" or "ordinal"'
                raise ValueError(errormsg)
        else:
            for key,fmt in formats_to_try.items():
                try:
                    dateobj = dt.datetime.strptime(datestr, fmt)
                    break # If we find one that works, we can stop
                except Exception as E:
                    exceptions[key] = str(E)
            if dateobj is None:
                formatstr = scu.newlinejoin([f'{item[1]}' for item in formats_to_try.items()])
                errormsg = f'Was unable to convert "{datestr}" to a date using the formats:\n{formatstr}'
                if dateformat not in ['dmy', 'mdy']:
                    errormsg += '\n\nNote: to read day-month-year or month-day-year dates, use dateformat="dmy" or "mdy" respectively.'
                raise ValueError(errormsg)
        dateobjs.append(dateobj)

    # If only a single date was supplied, return just that; else return the list/array
    output = scu._sanitize_output(dateobjs, is_list, is_array, dtype=object)
    return output


def date(obj, *args, start_date=None, readformat=None, outformat=None, as_date=True, **kwargs):
    '''
    Convert any reasonable object -- a string, integer, or datetime object, or
    list/array of any of those -- to a date object. To convert an integer to a
    date, you must supply a start date.

    Caution: while this function and ``sc.readdate()`` are similar, and indeed this function
    calls ``sc.readdate()`` if the input is a string, in this function an integer is treated
    as a number of days from start_date, while for ``sc.readdate()`` it is treated as a
    timestamp in seconds.

    Note: in this and other date functions, arguments work either with or without
    underscores (e.g. ``start_date`` or ``startdate``)

    Args:
        obj (str/int/date/datetime/list/array): the object to convert
        args (str/int/date/datetime): additional objects to convert
        start_date (str/date/datetime): the starting date, if an integer is supplied
        readformat (str/list): the format to read the date in; passed to ``sc.readdate()``
        outformat (str): the format to output the date in, if returning a string
        as_date (bool): whether to return as a datetime date instead of a string

    Returns:
        dates (date or list): either a single date object, or a list of them (matching input data type where possible)

    **Examples**::

        sc.date('2020-04-05') # Returns datetime.date(2020, 4, 5)
        sc.date([35,36,37], start_date='2020-01-01', as_date=False) # Returns ['2020-02-05', '2020-02-06', '2020-02-07']
        sc.date(1923288822, readformat='posix') # Interpret as a POSIX timestamp

    | New in version 1.0.0.
    | New in version 1.2.2: "readformat" argument; renamed "dateformat" to "outformat"
    '''
    # Handle deprecation
    start_date = kwargs.pop('startdate', start_date) # Handle with or without underscore
    as_date    = kwargs.pop('asdate', as_date) # Handle with or without underscore
    dateformat = kwargs.pop('dateformat', None)
    if dateformat is not None: # pragma: no cover
        outformat = dateformat
        warnmsg = 'sc.date() argument "dateformat" has been deprecated as of v1.2.2; use "outformat" instead'
        warnings.warn(warnmsg, category=DeprecationWarning, stacklevel=2)

    # Convert to list and handle other inputs
    if obj is None:
        return
    if outformat is None:
        outformat = '%Y-%m-%d'
    obj, is_list, is_array = scu._sanitize_iterables(obj, *args)

    dates = []
    for d in obj:
        if d is None:
            dates.append(d)
            continue
        try:
            if type(d) == dt.date: # Do not use isinstance, since must be the exact type
                pass
            elif isinstance(d, dt.datetime):
                d = d.date()
            elif scu.isstring(d):
                d = readdate(d, dateformat=readformat).date()
            elif scu.isnumber(d):
                if readformat is not None:
                    d = readdate(d, dateformat=readformat).date()
                else:
                    if start_date is None:
                        errormsg = f'To convert the number {d} to a date, you must either specify "posix" or "ordinal" read format, or supply start_date'
                        raise ValueError(errormsg)
                    d = date(start_date) + dt.timedelta(days=int(d))
            else: # pragma: no cover
                errormsg = f'Cannot interpret {type(d)} as a date, must be date, datetime, or string'
                raise TypeError(errormsg)
            if as_date:
                dates.append(d)
            else:
                dates.append(d.strftime(outformat))
        except Exception as E:
            errormsg = f'Conversion of "{d}" to a date failed: {str(E)}'
            raise ValueError(errormsg)

    # Return an integer rather than a list if only one provided
    output = scu._sanitize_output(dates, is_list, is_array, dtype=object)
    return output


def day(obj, *args, start_date=None, **kwargs):
    '''
    Convert a string, date/datetime object, or int to a day (int), the number of
    days since the start day. See also ``sc.date()`` and ``sc.daydiff()``. If a start day
    is not supplied, it returns the number of days into the current year.

    Args:
        obj (str, date, int, list, array): convert any of these objects to a day relative to the start day
        args (list): additional days
        start_date (str or date): the start day; if none is supplied, return days since (supplied year)-01-01.

    Returns:
        days (int or list): the day(s) in simulation time (matching input data type where possible)

    **Examples**::

        sc.day(sc.now()) # Returns how many days into the year we are
        sc.day(['2021-01-21', '2024-04-04'], start_date='2022-02-22') # Days can be positive or negative

    | New in version 1.0.0.
    | New in version 1.2.2: renamed "start_day" to "start_date"
    '''

    # Handle deprecation
    start_date = kwargs.pop('startdate', start_date) # Handle with or without underscore
    start_day = kwargs.pop('start_day', None)
    if start_day is not None: # pragma: no cover
        start_date = start_day
        warnmsg = 'sc.day() argument "start_day" has been deprecated as of v1.2.2; use "start_date" instead'
        warnings.warn(warnmsg, category=DeprecationWarning, stacklevel=2)

    # Do not process a day if it's not supplied, and ensure it's a list
    if obj is None:
        return
    obj, is_list, is_array = scu._sanitize_iterables(obj, *args)

    days = []
    for d in obj:
        if d is None:
            days.append(d)
        elif scu.isnumber(d):
            days.append(int(d)) # Just convert to an integer
        else:
            try:
                if scu.isstring(d):
                    d = readdate(d).date()
                elif isinstance(d, dt.datetime):
                    d = d.date()
                if start_date:
                    start_date = date(start_date)
                else:
                    start_date = date(f'{d.year}-01-01')
                d_day = (d - start_date).days # Heavy lifting -- actually compute the day
                days.append(d_day)
            except Exception as E: # pragma: no cover
                errormsg = f'Could not interpret "{d}" as a date: {str(E)}'
                raise ValueError(errormsg)

    # Return an integer rather than a list if only one provided
    output = scu._sanitize_output(days, is_list, is_array)
    return output


def daydiff(*args):
    '''
    Convenience function to find the difference between two or more days. With
    only one argument, calculate days since 2020-01-01.

    **Examples**::

        diff  = sc.daydiff('2020-03-20', '2020-04-05') # Returns 16
        diffs = sc.daydiff('2020-03-20', '2020-04-05', '2020-05-01') # Returns [16, 26]

    New in version 1.0.0.
    '''
    days = [date(day) for day in args]
    if len(days) == 1:
        days.insert(0, date(f'{now().year}-01-01')) # With one date, return days since Jan. 1st

    output = []
    for i in range(len(days)-1):
        diff = (days[i+1] - days[i]).days
        output.append(diff)

    if len(output) == 1:
        output = output[0]

    return output


def daterange(start_date=None, end_date=None, interval=None, inclusive=True, as_date=False, readformat=None, outformat=None, **kwargs):
    '''
    Return a list of dates from the start date to the end date. To convert a list
    of days (as integers) to dates, use ``sc.date()`` instead.

    Args:
        start_date (int/str/date): the starting date, in any format
        end_date (int/str/date): the end date, in any format
        interval (int/str/dict): if an int, the number of days; if 'week', 'month', or 'year', one of those; if a dict, passed to ``dt.relativedelta()``
        inclusive (bool): if True (default), return to end_date inclusive; otherwise, stop the day before
        as_date (bool): if True, return a list of datetime.date objects instead of strings (note: you can also use "asdate" instead of "as_date")
        readformat (str): passed to date()
        outformat (str): passed to date()

    **Examples**::

        dates1 = sc.daterange('2020-03-01', '2020-04-04')
        dates2 = sc.daterange('2020-03-01', '2022-05-01', interval=dict(months=2), asdate=True)

    | New in version 1.0.0.
    | New in version 1.3.0: "interval" argument
    '''

    # Handle inputs
    start_date = kwargs.pop('startdate', start_date) # Handle with or without underscore
    end_date   = kwargs.pop('enddate',   end_date) # Handle with or without underscore
    as_date    = kwargs.pop('asdate', as_date) # Handle with or without underscore
    if len(kwargs):
        errormsg = f'Unexpected arguments: {scu.strjoin(kwargs.keys())}'
        raise ValueError(errormsg)
    start_date = date(start_date, readformat=readformat)
    end_date   = date(end_date,   readformat=readformat)
    if   interval in [None, 'day']: interval = dict(days=1)
    elif interval == 'week':        interval = dict(weeks=1)
    elif interval == 'month':       interval = dict(months=1)
    elif interval == 'year':        interval = dict(years=1)
    if inclusive:
        end_date += datedelta(days=1)

    # Calculate dates
    dates = []
    curr_date = start_date
    delta = datedelta(**interval)
    while curr_date < end_date:
        dates.append(curr_date)
        curr_date += delta

    # Convert to final format
    dates = date(dates, start_date=start_date, as_date=as_date, outformat=outformat)
    return dates


def datedelta(datestr=None, days=0, months=0, years=0, weeks=0, dt1=None, dt2=None, as_date=None, **kwargs):
    '''
    Perform calculations on a date string (or date object), returning a string (or a date).
    Wrapper to ``dateutil.relativedelta.relativedelta()``.

    If ``datestr`` is ``None``, then return the delta object rather than the new date.

    Args:
        datestr (None/str/date): the starting date (typically a string); if None, return the relative delta
        days (int): the number of days (positive or negative) to increment
        months (int): as above
        years (int): as above
        weeks (int): as above
        dt1, dt2 (dates): if both provided, compute the difference between them
        as_date (bool): if True, return a date object; otherwise, return as input type
        kwargs (dict): passed to ``sc.readdate()``

    **Examples**::

        sc.datedelta('2021-07-07', 3) # Add 3 days
        sc.datedelta('2021-07-07', days=-4) # Subtract 4 days
        sc.datedelta('2021-07-07', weeks=4, months=-1, as_date=True) # Add 4 weeks but subtract a month, and return a dateobj
        sc.datedelta(days=3) # Alias to du.relativedelta.relativedelta(days=3)
    '''
    as_date = kwargs.pop('asdate', as_date) # Handle with or without underscore

    # Calculate the time delta, and return immediately if no date is provided
    delta = du.relativedelta.relativedelta(days=days, months=months, years=years, weeks=weeks, dt1=dt1, dt2=dt2)
    if datestr is None:
        return delta
    else:
        if as_date is None: # Typical case, return the same format as the input
            as_date = False if isinstance(datestr, str) else True
        dateobj = readdate(datestr, **kwargs)
        newdate = dateobj + delta
        newdate = date(newdate, as_date=as_date)
        return newdate


def datetoyear(dateobj, dateformat=None):
    """
    Convert a DateTime instance to decimal year.

    Args:
        dateobj (date, str):  The datetime instance to convert
        dateformat (str): If dateobj is a string, the optional date conversion format to use

    Returns:
        Equivalent decimal year

    **Example**::

        sc.datetoyear('2010-07-01') # Returns approximately 2010.5

    By Luke Davis from https://stackoverflow.com/a/42424261, adapted by Romesh Abeysuriya.

    New in version 1.0.0.
    """
    if scu.isstring(dateobj):
        dateobj = readdate(dateobj, dateformat=dateformat)
    year_part = dateobj - dt.datetime(year=dateobj.year, month=1, day=1)
    year_length = dt.datetime(year=dateobj.year + 1, month=1, day=1) - dt.datetime(year=dateobj.year, month=1, day=1)
    output = dateobj.year + year_part / year_length
    return output


###############################################################################
#%% Timing functions
###############################################################################

__all__+= ['tic', 'toc', 'toctic', 'timer', 'Timer']


def tic():
    '''
    With ``sc.toc()``, a little pair of functions to calculate a time difference:

    **Examples**::

        sc.tic()
        slow_func()
        sc.toc()

        T = sc.tic()
        slow_func2()
        sc.toc(T, label='slow_func2')

    See also ``sc.timer()``.
    '''
    global _tictime  # The saved time is stored in this global
    _tictime = time.time()  # Store the present time in the global
    return _tictime    # Return the same stored number



def toc(start=None, label=None, baselabel=None, sigfigs=None, reset=False, output=False, doprint=None, elapsed=None):
    '''
    With ``sc.tic()``, a little pair of functions to calculate a time difference. See
    also ``sc.timer()``.

    Args:
        start     (float): the starting time, as returned by e.g. ``sc.tic()``
        label     (str): optional label to add
        baselabel (str): optional base label; default is "Elapsed time: "
        sigfigs   (int): number of significant figures for time estimate
        reset     (bool): reset the time; like calling ``sc.toctic()`` or ``sc.tic()`` again
        output    (bool): whether to return the output (otherwise print); if output='message', then return the message string; if output='both', then return both
        doprint   (bool): whether to print (true by default)
        elapsed   (float): use a pre-calculated elapsed time instead of recalculating (not recommneded)

    **Examples**::

        sc.tic()
        slow_func()
        sc.toc()

        T = sc.tic()
        slow_func2()
        sc.toc(T, label='slow_func2')

    New in version 1.3.0: new arguments.
    '''
    now = time.time() # Get the time as quickly as possible

    from . import sc_printing as scp # To avoid circular import
    global _tictime  # The saved time is stored in this global

    # Set defaults
    if sigfigs is None: sigfigs = 3

    # If no start value is passed in, try to grab the global _tictime
    if isinstance(start, str): # Start and label are probably swapped
        start,label = label,start
    if start is None:
        try:    start = _tictime
        except: start = 0 # This doesn't exist, so just leave start at 0.

    # Calculate the elapsed time in seconds
    if elapsed is None:
        elapsed = now - start

    # Create the message giving the elapsed time
    if label is None:
        if baselabel is None:
            base = 'Elapsed time: '
        else:
            base = baselabel
    else:
        if baselabel is None:
            if label:
                base = f'Elapsed time for {label}: '
            else: # Handles case toc(label='')
                base = ''
        else:
            base = f'{baselabel}{label}: '
    logmessage = f'{base}{scp.sigfig(elapsed, sigfigs=sigfigs)} s'

    # Print if asked, or if no other output
    if doprint or ((doprint is None) and (not output)):
        print(logmessage)

    # Optionally reset the counter
    if reset:
        _tictime = time.time()  # Store the present time in the global

    # Return elapsed if desired
    if output:
        if output == 'message':
            return logmessage
        elif output == 'both':
            return (elapsed, logmessage)
        else:
            return elapsed
    else:
        return


def toctic(returntic=False, returntoc=False, *args, **kwargs):
    '''
    A convenience fu`ction for multiple timings. Can return the default output of
    either ``sc.tic()`` or ``sc.toc()`` (default neither). Arguments are passed to ``sc.toc()``.
    Equivalent to ``sc.toc(reset=True)``.

    **Example**::

        sc.tic()
        slow_operation_1()
        sc.toctic()
        slow_operation_2()
        sc.toc()

    New in version 1.0.0.
    '''
    tocout = toc(*args, **kwargs)
    ticout = tic()
    if   returntic: return ticout
    elif returntoc: return tocout
    else:           return


class timer(scu.prettyobj):
    '''
    Simple timer class. Note: ``sc.timer()`` and ``sc.Timer()`` are aliases.

    This wraps ``tic`` and ``toc`` with the formatting arguments and the start time
    (at construction).

    Use this in a ``with`` block to automatically print elapsed time when
    the block finishes.

    Args:
        label (str): label identifying this timer
        auto (bool): whether to automatically increment the label


    Example making repeated calls to the same timer, using ``auto`` to keep track::

        >>> T = sc.timer(auto=True)
        >>> T.toc()
        Elapsed time for (0): 2.63 s
        >>> T.toc()
        Elapsed time for (1): 5.00 s

    Example wrapping code using with-as::

        >>> with sc.timer('mylabel'):
        >>>     sc.timedsleep(0.5)

    Example using a timer to collect data, using ``tt()`` as an alias for ``toctic()``
    to reset the time::

        T = sc.timer(doprint=False)
        for key in 'abcde':
            sc.timedsleep(pl.rand())
            T.tt(key)
        print(T.timings)

    Implementation based on https://preshing.com/20110924/timing-your-code-using-pythons-with-statement/

    | New in version 1.3.0: ``sc.timer()`` alias, and allowing the label as first argument.
    | New in version 1.3.2: ``toc()`` passes label correctly; ``tt()`` method; ``auto`` argument
    '''
    def __init__(self, label=None, auto=False, **kwargs):
        from . import sc_odict as sco # Here to avoid circular import
        self.tic()
        self.kwargs = kwargs # Store kwargs to pass to toc() at the end of the block
        self.kwargs['label'] = label
        self.auto = auto
        self._start = None
        self.elapsed = None
        self.message = None
        self.count = 0
        self.timings = sco.odict()
        self.tic() # Start counting
        return

    def __enter__(self):
        ''' Reset start time when entering with-as block '''
        self.tic()
        return self

    def __exit__(self, *args):
        ''' Print elapsed time when leaving a with-as block '''
        self.toc()
        return

    def tic(self):
        ''' Set start time '''
        self._start = time.time()  # Store the present time locally
        return

    def toc(self, label=None, **kwargs):
        ''' Print elapsed time; see ``sc.toc()`` for keyword arguments '''

        # Get the time
        self.elapsed, self.message = toc(start=self._start, output='both', doprint=False) # Get time as quickly as possible

        # Update the kwargs, including the label
        if label is not None:
            kwargs['label'] = label
        for k,v in self.kwargs.items():
            if k not in kwargs:
                kwargs[k] = v

        # Handle the count and labels
        countstr= f'({self.count:d})'
        if kwargs['label']:
            labelstr = kwargs['label']
            sep = ' '
        else:
            labelstr = ''
            sep = ''
        countlabel = f'{countstr}{sep}{labelstr}'
        timingslabel = countlabel if (self.auto or not(labelstr) or (labelstr in self.timings)) else labelstr # Use labelstr if it's a valid key, else include count information
        self.timings[timingslabel] = self.elapsed
        self.count += 1
        if self.auto:
            kwargs['label'] = countlabel

        # Call again to get the correct output
        output = toc(elapsed=self.elapsed, **kwargs)

        # If reset was used, apply it
        if kwargs.get('reset'):
            self.tic()

        return output

    def start(self):
        ''' Alias for tic() '''
        return self.tic()

    def stop(self, *args, **kwargs):
        ''' Alias for toc() '''
        return self.toc(*args, **kwargs)

    def toctic(self, *args, **kwargs):
        ''' Reset time between timings '''
        kwargs['reset'] = True
        return self.toc(*args, **kwargs)

    def tt(self, *args, **kwargs):
        ''' Alias for toctic() '''
        return self.toctic(*args, **kwargs)




Timer = timer # Alias


###############################################################################
#%% Other functions
###############################################################################

__all__ += ['elapsedtimestr', 'timedsleep']


def elapsedtimestr(pasttime, maxdays=5, minseconds=10, shortmonths=True):
    """
    Accepts a datetime object or a string in ISO 8601 format and returns a
    human-readable string explaining when this time was.

    The rules are as follows:

    * If a time is within the last hour, return 'XX minutes'
    * If a time is within the last 24 hours, return 'XX hours'
    * If within the last 5 days, return 'XX days'
    * If in the same year, print the date without the year
    * If in a different year, print the date with the whole year

    These can be configured as options.

    **Examples**::

        yesterday = sc.datedelta(sc.now(), days=-1)
        sc.elapsedtimestr(yesterday)
    """

    # Elapsed time function by Alex Chan: https://gist.github.com/alexwlchan/73933442112f5ae431cc
    def print_date(date, includeyear=True, shortmonths=True):
        """
        Prints a datetime object as a full date, stripping off any leading
        zeroes from the day (strftime() gives the day of the month as a zero-padded
        decimal number).
        """
        # %b/%B are the tokens for abbreviated/full names of months to strftime()
        if shortmonths:
            month_token = '%b'
        else:
            month_token = '%B'

        # Get a string from strftime()
        if includeyear:
            date_str = date.strftime('%d ' + month_token + ' %Y')
        else:
            date_str = date.strftime('%d ' + month_token)

        # There will only ever be at most one leading zero, so check for this and
        # remove if necessary
        if date_str[0] == '0':
            date_str = date_str[1:]

        return date_str

    now_time = dt.datetime.now()

    # If the user passes in a string, try to turn it into a datetime object before continuing
    if isinstance(pasttime, str):
        try:
            pasttime = readdate(pasttime)
        except ValueError as E: # pragma: no cover
            errormsg = f"User supplied string {pasttime} is not in a readable format."
            raise ValueError(errormsg) from E
    elif isinstance(pasttime, dt.datetime):
        pass
    else: # pragma: no cover
        errormsg = f"User-supplied value {pasttime} is neither a datetime object nor an ISO 8601 string."
        raise TypeError(errormsg)

    # It doesn't make sense to measure time elapsed between now and a future date, so we'll just print the date
    if pasttime > now_time:
        includeyear = (pasttime.year != now_time.year)
        time_str = print_date(pasttime, includeyear=includeyear, shortmonths=shortmonths)

    # Otherwise, start by getting the elapsed time as a datetime object
    else:
        elapsed_time = now_time - pasttime

        # Check if the time is within the last minute
        if elapsed_time < dt.timedelta(seconds=60):
            if elapsed_time.seconds <= minseconds:
                time_str = "just now"
            else:
                time_str = f"{elapsed_time.seconds} secs ago"

        # Check if the time is within the last hour
        elif elapsed_time < dt.timedelta(seconds=60 * 60):

            # We know that seconds > 60, so we can safely round down
            minutes = int(elapsed_time.seconds / 60)
            if minutes == 1:
                time_str = "a minute ago"
            else:
                time_str = f"{minutes} mins ago"

        # Check if the time is within the last day
        elif elapsed_time < dt.timedelta(seconds=60 * 60 * 24 - 1):

            # We know that it's at least an hour, so we can safely round down
            hours = int(elapsed_time.seconds / (60 * 60))
            if hours == 1:
                time_str = "1 hour ago"
            else:
                time_str = f"{hours} hours ago"

        # Check if it's within the last N days, where N is a user-supplied argument
        elif elapsed_time < dt.timedelta(days=maxdays):
            if elapsed_time.days == 1:
                time_str = "yesterday"
            else:
                time_str = f"{elapsed_time.days} days ago"

        # If it's not within the last N days, then we're just going to print the date
        else:
            includeyear = (pasttime.year != now_time.year)
            time_str = print_date(pasttime, includeyear=includeyear, shortmonths=shortmonths)

    return time_str


def timedsleep(delay=None, start=None, verbose=True):
    '''
    Delay for a certain amount of time, to ensure accurate timing.

    Args:
        delay (float): time, in seconds, to wait for
        start (float): if provided, the start time
        verbose (bool): whether to print activity

    **Example**::

        for i in range(10):
            sc.timedsleep('start') # Initialize
            for j in range(int(1e6)):
                tmp = pl.rand()
            sc.timedsleep(1) # Wait for one second including computation time
    '''
    self_time = 0.00012 # Roughly how long this function itself takes to run -- slightly underestimate
    global _delaytime
    if delay is None or delay=='start':
        _delaytime = time.time()  # Store the present time in the global.
        return _delaytime         # Return the same stored number.
    else:
        if start is None:
            try:    start = _delaytime
            except: start = time.time()
        elapsed = time.time() - start
        remaining = delay - elapsed
        if remaining>0:
            if verbose:
                print(f'Pausing for {remaining:0.1f} s')
            time.sleep(remaining - self_time)
            try:    del _delaytime # After it's been used, we can't use it again
            except: pass
        else:
            if verbose:
                print(f'Warning, delay less than elapsed time ({delay:0.1f} vs. {elapsed:0.1f})')
    return
