"""
Time/date utilities.

Highlights:
    - :func:`sc.tic() <tic>` / :func:`sc.toc() <toc>` / :class:`sc.timer() <timer>`: simple methods for timing durations
    - :func:`sc.readdate() <readdate>`: convert strings to dates using common formats
    - :func:`sc.daterange() <daterange>`: create a list of dates
    - :func:`sc.datedelta() <datedelta>`: perform calculations on date strings
"""

import time as pytime
import warnings
import functools
import numpy as np
import pandas as pd
import datetime as dt
import dateutil as du
import matplotlib as mpl
import matplotlib.pyplot as plt
import sciris as sc
import sciris.sc_utils as scu


###############################################################################
#%% Date functions
###############################################################################

__all__ = ['time', 'now', 'getdate', 'readdate', 'date', 'day', 'daydiff', 'daterange', 'datedelta', 'datetoyear','yeartodate']


def time():
    """
    Get current time in seconds -- alias to time.time()

    See also :func:`sc.now() <now>` to return a datetime object, and :func:`sc.getdate() <getdate>` to
    return a string.

    *New in version 3.0.0.*
    """
    return pytime.time()


def now(astype='dateobj', timezone=None, utc=False, tostring=False, dateformat=None):
    """
    Get the current time as a datetime object, optionally in UTC time.

    :func:`sc.now() <now>` is similar to :func:`sc.getdate() <getdate>`, but :func:`sc.now() <now>` returns a datetime
    object by default, while :func:`sc.getdate() <getdate>` returns a string by default.

    Args:
        astype     (str)  : what to return; choices are "dateobj", "str", "float"; see :func:`sc.getdate() <getdate>` for more
        timezone   (str)  : the timezone to set the itme to
        utc        (bool) : whether the time is specified in UTC time
        dateformat (str)  : if ``astype`` is ``'str'``, use this output format

    **Examples**::

        sc.now() # Return current local time, e.g. 2019-03-14 15:09:26
        sc.now(timezone='US/Pacific') # Return the time now in a specific timezone
        sc.now(utc=True) # Return the time in UTC
        sc.now(astype='str') # Return the current time as a string instead of a date object; use 'int' for seconds
        sc.now(tostring=True) # Backwards-compatible alias for astype='str'
        sc.now(dateformat='%Y-%b-%d') # Return a different date format

    *New in version 1.3.0:* made "astype" the first argument; removed "tostring" argument
    """
    if isinstance(utc, str): timezone = utc # Assume it's a timezone
    if timezone is not None: tzinfo = du.tz.gettz(timezone) # Timezone is a string
    elif utc:                tzinfo = du.tz.tzutc() # UTC has been specified
    else:                    tzinfo = None # Otherwise, do nothing
    if tostring: # pragma: no cover
        warnmsg = 'sc.now() argument "tostring" is deprecated; use astype="str" instead'
        warnings.warn(warnmsg, category=FutureWarning, stacklevel=2)
        astype='str'
    timenow = dt.datetime.now(tzinfo)
    output = getdate(timenow, astype=astype, dateformat=dateformat)
    return output


def getdate(obj=None, astype='str', dateformat=None):
        """
        Alias for converting a date object to a formatted string.

        See also :func:`sc.now() <now>`.

        Args:
            obj (datetime): the datetime object to convert
            astype (str): what to return; choices are "str" (default), "dateobj", "float" (full timestamp), "int" (timestamp to second precision)
            dateformat (str): if ``astype`` is ``'str'``, use this output format

        **Examples**::

            sc.getdate() # Returns a string for the current date
            sc.getdate(astype='float') # Convert today's time to a timestamp
        """
        if obj is None:
            obj = now()

        if dateformat is None:
            dateformat = '%Y-%b-%d %H:%M:%S'
        else:
            astype = 'str' # If dateformat is specified, assume type is a string

        try:
            if sc.isstring(obj): # pragma: no cover
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
        elif astype in  ['float', 'number', 'timestamp']: # pragma: no cover
            output = timestamp
        else: # pragma: no cover
            errormsg = f'"astype={astype}" not understood; must be "str" or "int"'
            raise ValueError(errormsg)
        return output


def readdate(datestr=None, *args, dateformat=None, return_defaults=False, verbose=False):
    """
    Convenience function for loading a date from a string. If dateformat is None,
    this function tries a list of standard date types. Note: in most cases :func:`sc.date() <date>`
    should be used instead.

    By default, a numeric date is treated as a POSIX (Unix) timestamp. This can be changed
    with the ``dateformat`` argument, specifically:

    - 'posix'/None: treat as a POSIX timestamp, in seconds from 1970
    - 'ordinal'/'matplotlib': treat as an ordinal number of days from 1970 (Matplotlib default)

    Args:
        datestr (int, float, str or list): the string containing the date, or the timestamp (in seconds), or a list of either
        args (list): additional dates to convert
        dateformat (str or list): the format for the date, if known; if 'dmy' or 'mdy', try as day-month-year or month-day-year formats; can also be a list of options
        return_defaults (bool): don't convert the date, just return the defaults
        verbose (bool): return detailed error messages

    Returns:
        dateobj (datetime): a datetime object

    **Examples**::

        dateobj  = sc.readdate('2020-03-03') # Standard format, so works
        dateobj  = sc.readdate('04-03-2020', dateformat='dmy') # Date is ambiguous, so need to specify day-month-year order
        dateobj  = sc.readdate(1611661666) # Can read timestamps as well
        dateobj  = sc.readdate(16166, dateformat='ordinal') # Or ordinal numbers of days, as used by Matplotlib
        dateobjs = sc.readdate(['2020-06', '2020-07'], dateformat='%Y-%m') # Can read custom date formats
        dateobjs = sc.readdate('20200321', 1611661666) # Can mix and match formats
    """

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
        'default2':       '%Y-%m-%dT%H:%M:%S.%f', # 2020-03-21T14:35:21.23483
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
    format_list = sc.tolist(dateformat, keepnone=True) # Keep none which signifies default
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
        elif sc.isnumber(datestr): # pragma: no cover
            if 'posix' in format_list or None in format_list:
                dateobj = dt.datetime.fromtimestamp(datestr)
            elif 'ordinal' in format_list or 'matplotlib' in format_list:
                dateobj = mpl.dates.num2date(datestr)
            else:
                errormsg = f'Could not convert numeric date {datestr} using available formats {sc.strjoin(format_list)}; must be "posix" or "ordinal"'
                raise ValueError(errormsg)
        else:
            for key,fmt in formats_to_try.items():
                try:
                    dateobj = dt.datetime.strptime(datestr, fmt)
                    break # If we find one that works, we can stop
                except Exception as E:
                    exceptions[key] = str(E)
            if dateobj is None:
                formatstr = sc.newlinejoin([f'{item[1]}' for item in formats_to_try.items()])
                errormsg = f'Was unable to convert "{datestr}" to a date using the formats:\n{formatstr}'
                if dateformat not in ['dmy', 'mdy']:
                    errormsg += '\n\nNote: to read day-month-year or month-day-year dates, use dateformat="dmy" or "mdy" respectively.'
                    if verbose: # pragma: no cover
                        for key,val in exceptions.items():
                            errormsg += f'\n {key}: {val}'
                raise ValueError(errormsg)
        dateobjs.append(dateobj)

    # If only a single date was supplied, return just that; else return the list/array
    output = scu._sanitize_output(dateobjs, is_list, is_array, dtype=object)
    return output


def date(obj=None, *args, start_date=None, readformat=None, to='date', as_date=None, outformat=None, **kwargs):
    """
    Convert any reasonable object -- a string, integer, or datetime object, or
    list/array of any of those -- to a date object (or string, pandas, or numpy
    date).

    If the object is an integer, this is interpreted as follows:

    - With readformat='posix': treat as a POSIX timestamp, in seconds from 1970
    - With readformat='ordinal'/'matplotlib': treat as an ordinal number of days from 1970 (Matplotlib default)
    - With start_date provided: treat as a number of days from this date

    Note: in this and other date functions, arguments work either with or without
    underscores (e.g. ``start_date`` or ``startdate``)

    Args:
        obj (str/int/date/datetime/list/array): the object to convert; if None, return current date
        args (str/int/date/datetime): additional objects to convert
        start_date (str/date/datetime): the starting date, if an integer is supplied
        readformat (str/list): the format to read the date in; passed to :func:`sc.readdate() <readdate>` (NB: can also use "format" instead of "readformat")
        to (str): the output format: 'date' (default), 'datetime', 'str' (or 'string'), 'pandas', or 'numpy'
        as_date (bool): alternate method of choosing between  output format of 'date' (True) or 'str' (False); if None, use "to" instead
        outformat (str): the format to output the date in, if returning a string
        kwargs (dict): only used for deprecated argument aliases

    Returns:
        dates (date or list): either a single date object, or a list of them (matching input data type where possible)

    **Examples**::

        sc.date('2020-04-05') # Returns datetime.date(2020, 4, 5)
        sc.date([35,36,37], start_date='2020-01-01', to='str') # Returns ['2020-02-05', '2020-02-06', '2020-02-07']
        sc.date(1923288822, readformat='posix') # Interpret as a POSIX timestamp

    | *New in version 1.0.0.*
    | *New in version 1.2.2:* "readformat" argument; renamed "dateformat" to "outformat"
    | *New in version 2.0.0:* support for :obj:`np.datetime64 <numpy.datetime64>` objects
    | *New in version 3.0.0:* added "to" argument, and support for :obj:`pd.Timestamp <pandas.Timestamp>` and :obj:`np.datetime64 <numpy.datetime64>` output; allow None
    | *New in version 3.1.0:* allow "datetime" output
    """

    # Handle deprecation and nonstandard usage
    if len(kwargs):
        start_date = kwargs.pop('startdate', start_date) # Handle with or without underscore
        as_date    = kwargs.pop('asdate', as_date) # Handle with or without underscore
        readformat = kwargs.pop('format', readformat) # Handle either name
        dateformat = kwargs.pop('dateformat', None)
        if dateformat is not None: # pragma: no cover
            outformat = dateformat
            warnmsg = 'sc.date() argument "dateformat" has been deprecated as of v1.2.2; use "outformat" instead'
            warnings.warn(warnmsg, category=FutureWarning, stacklevel=2)

        # Handle dt.date()-like behavior with keywords
        year  = kwargs.pop('year', None)
        month = kwargs.pop('month', None)
        day   = kwargs.pop('day', None)
        ymd = [year, month, day]
        valid = sum([v is not None for v in ymd])
        if valid == 3:
            obj = dt.date(year, month, day)
        elif valid > 0:
            errormsg = f'Cannot construct a date with year={year}, month={month}, day={day}; please ensure all arguments are supplied'
            raise ValueError(errormsg)
        if len(kwargs):
            errormsg = f'Unrecognized arguments to sc.date():\n{kwargs}'
            raise TypeError(errormsg)

    # More initialization
    if as_date is not None: # pragma: no cover
        to = 'date' if as_date else 'str' # Legacy support for as_date boolean

    # Handle e.g. sc.date(1986, 4, 4)
    if len(args) == 2 and isinstance(obj, int):
        obj = dt.date(obj, args[0], args[1])
        args = []

    def dateify(obj):
        """ Handle dates vs datetimes """
        if to == 'date' and hasattr(obj, 'date'):
            return obj.date()
        else:
            return obj

    # Convert to list and handle other inputs
    if obj is None:
        obj = dt.datetime.now()
    if outformat is None:
        outformat = '%Y-%m-%d'
    obj, is_list, is_array = scu._sanitize_iterables(obj, *args)

    dates = []
    for d in obj:
        if d is None: # pragma: no cover
            dates.append(d)
            continue
        try:
            if type(d) == dt.date:
                if to == 'datetime': # Do not use isinstance, since must be the exact type
                    d = dt.datetime(d.year, d.month, d.day)
                else:
                    pass
            elif isinstance(d, dt.datetime): # This includes pd.Timestamp
                pass
            elif sc.isstring(d):
                d = readdate(d, dateformat=readformat)
            elif isinstance(d, np.datetime64):
                d = pd.Timestamp(d)
            elif sc.isnumber(d):
                if readformat is not None:
                    d = readdate(d, dateformat=readformat)
                else:
                    if start_date is None:
                        errormsg = f'To convert the number {d} to a date, you must either specify "posix" or "ordinal" read format, or supply start_date'
                        raise ValueError(errormsg)
                    d = date(start_date) + dt.timedelta(days=int(d))
                    if to == 'datetime':
                        d = dt.datetime(d.year, d.month, d.day)
            else: # pragma: no cover
                errormsg = f'Cannot interpret {type(d)} as a date, must be date, datetime, or string'
                raise TypeError(errormsg)

            # Handle output
            if to == 'date': # Convert from datetime to a date
                out = dateify(d)
            elif to in [str, 'str', 'string']:
                out = d.strftime(outformat)
            elif to == 'pandas':
                out = pd.Timestamp(d)
            elif to == 'numpy':
                out = np.datetime64(d)
            else:
                errormsg = f'Could not understand to="{to}": must be "date", "str", "pandas", or "numpy"'
                raise ValueError(errormsg)
            dates.append(out)
        except Exception as E:
            errormsg = f'Conversion of "{d}" to a date failed'
            raise ValueError(errormsg) from E

    # Return a scalar rather than a list if only one provided
    output = scu._sanitize_output(dates, is_list, is_array, dtype=object)
    return output


def day(obj, *args, start_date=None, **kwargs):
    """
    Convert a string, date/datetime object, or int to a day (int), the number of
    days since the start day. See also :func:`sc.date() <date>` and :func:`sc.daydiff() <daydiff>``. If a start day
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

    | *New in version 1.0.0.*
    | *New in version 1.2.2:* renamed "start_day" to "start_date"
    """

    # Handle deprecation
    start_date = kwargs.pop('startdate', start_date) # Handle with or without underscore
    start_day = kwargs.pop('start_day', None)
    if start_day is not None: # pragma: no cover
        start_date = start_day
        warnmsg = 'sc.day() argument "start_day" has been deprecated as of v1.2.2; use "start_date" instead'
        warnings.warn(warnmsg, category=FutureWarning, stacklevel=2)

    # Do not process a day if it's not supplied, and ensure it's a list
    if obj is None:
        return
    obj, is_list, is_array = scu._sanitize_iterables(obj, *args)

    days = []
    for d in obj:
        if d is None:
            days.append(d)
        elif sc.isnumber(d):
            days.append(int(d)) # Just convert to an integer
        else:
            try:
                if sc.isstring(d):
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
    """
    Convenience function to find the difference between two or more days. With
    only one argument, calculate days since Jan. 1st.

    **Examples**::

        diff  = sc.daydiff('2020-03-20', '2020-04-05') # Returns 16
        diffs = sc.daydiff('2020-03-20', '2020-04-05', '2020-05-01') # Returns [16, 26]

        doy = sc.daydiff('2022-03-20') # Returns 79, the number of days since 2022-01-01

    | *New in version 1.0.0.*
    | *New in version 3.0.0:* Calculated relative days with one argument
    """
    days = [date(day) for day in args]
    if len(days) == 1:
        days.insert(0, date(f'{days[0].year}-01-01')) # With one date, return days since Jan. 1st

    output = []
    for i in range(len(days)-1):
        diff = (days[i+1] - days[i]).days
        output.append(diff)

    if len(output) == 1:
        output = output[0]

    return output


def daterange(start_date=None, end_date=None, interval=None, inclusive=True, as_date=None,
              readformat=None, outformat=None, **kwargs):
    """
    Return a list of dates from the start date to the end date. To convert a list
    of days (as integers) to dates, use :func:`sc.date() <date>` instead.

    Note: instead of an end date, can also pass one or more of days, months, weeks,
    or years, which will be added on to the start date via :func:`sc.datedelta() <datedelta>`.

    Args:
        start_date (int/str/date) : the starting date, in any format
        end_date   (int/str/date) : the end date, in any format (see also kwargs below)
        interval   (int/str/dict) : if an int, the number of days; if 'week', 'month', or 'year', one of those; if a dict, passed to ``dt.relativedelta()``
        inclusive  (bool)         : if True (default), return to end_date inclusive; otherwise, stop the day before
        as_date    (bool)         : if True, return a list of ``datetime.date`` objects; else, as input type (e.g. strings; note: you can also use "asdate" instead of "as_date")
        readformat (str)          : passed to :func:`sc.date() <date>`
        outformat  (str)          : passed to :func:`sc.date() <date>`
        kwargs     (dict)         : optionally, use any valid argument to :func:`sc.datedelta() <datedelta>` to create the end_date

    **Examples**::

        dates1 = sc.daterange('2020-03-01', '2020-04-04')
        dates2 = sc.daterange('2020-03-01', '2022-05-01', interval=dict(months=2), asdate=True)
        dates3 = sc.daterange('2020-03-01', weeks=5)

    | *New in version 1.0.0.*
    | *New in version 1.3.0:* "interval" argument
    | *New in version 2.0.0:* :func:`sc.datedelta() <datedelta>` arguments
    | *New in version 3.0.0:* preserve input type
    """

    # Handle inputs
    start_date = kwargs.pop('startdate', start_date) # Handle with or without underscore
    end_date   = kwargs.pop('enddate',   end_date) # Handle with or without underscore
    as_date    = kwargs.pop('asdate', as_date) # Handle with or without underscore
    if as_date is None: # Typical case, return the same format as the input
        as_date = False if isinstance(start_date, str) else True
    if len(kwargs):
        end_date = datedelta(start_date, **kwargs)
    start_date = date(start_date, readformat=readformat)
    end_date = date(end_date, readformat=readformat)

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

def _get_year_length(year):
        """ Get the length of the year: 365 or 366 days """
        return dt.date(year=year+1, month=1, day=1) - dt.date(year=year, month=1, day=1)

def datedelta(datestr=None, days=0, months=0, years=0, weeks=0, dt1=None, dt2=None, as_date=None, **kwargs):
    """
    Perform calculations on a date string (or date object), returning a string (or a date).
    Wrapper to ``dateutil.relativedelta.relativedelta()``.

    If ``datestr`` is ``None``, then return the delta object rather than the new date.

    Args:
        datestr (None/str/date/list): the starting date (typically a string); if None, return the relative delta
        days (int): the number of days (positive or negative) to increment
        months (int): as above
        years (int/float): as above; if a float, converted to days (NB: fractional months and weeks are not supported)
        weeks (int): as above
        dt1, dt2 (dates): if both provided, compute the difference between them
        as_date (bool): if True, return a date object; otherwise, return as input type
        kwargs (dict): passed to :func:`sc.date() <readdate>`

    **Examples**::

        sc.datedelta('2021-07-07', 3) # Add 3 days
        sc.datedelta('2021-07-07', days=-4) # Subtract 4 days
        sc.datedelta('2021-07-07', weeks=4, months=-1, as_date=True) # Add 4 weeks but subtract a month, and return a dateobj
        sc.datedelta(days=3) # Alias to du.relativedelta.relativedelta(days=3)
        sc.datedelta(['2021-07-07', '2022-07-07'], months=1) # Increment multiple dates
        sc.datedelta('2020-06-01', years=0.25) # Use a fractional number of years (to the nearest day)

    | *New in version 3.0.0:* operate on list of dates
    | *New in version 3.1.0:* handle all date input formats
    | *New in version 3.2.0:* handle fractional years
    """
    # Handle keywords
    as_date = kwargs.pop('asdate', as_date) # Handle with or without underscore
    kw = dict(days=days, months=months, years=years, weeks=weeks, dt1=dt1, dt2=dt2)

    # Check if the year is fractional
    fractional_year = not float(years).is_integer()

    def years_to_days(days, years, start_year=None):
        """ Convert fractional years to days """
        int_years = int(years)
        frac_year = years - int_years
        if start_year is None:
            days_per_year = 365
        else:
            days_per_year = _get_year_length(start_year + int_years).days
        days = int(round(frac_year*days_per_year))

        # Modify keywords in place; the function arguments remain the ground truth
        kw['days'], kw['years'] = days, int_years
        return

    # If we're not using a fractional year, we can precompute this
    if not fractional_year:
        delta = du.relativedelta.relativedelta(**kw)

    # Calculate the time delta, and return immediately if no date is provided
    if datestr is None:
        if fractional_year:
            years_to_days(days, years) # Approximate since we don't know the start year, so may be a day off in leap years
            delta = du.relativedelta.relativedelta(**kw)
        return delta

    # Otherwise, process each argument
    else:
        datelist = sc.tolist(datestr)
        newdates = []
        for datestr in datelist:
            if as_date is None: # Typical case, return the same format as the input
                as_date = False if isinstance(datestr, str) else True
            dateobj = date(datestr, **kwargs)
            if fractional_year:
                years_to_days(days, years, start_year=dateobj.year) # We do know the start year, so can calculate exactly
                delta = du.relativedelta.relativedelta(**kw)
            newdate = dateobj + delta
            newdate = date(newdate, as_date=as_date)
            newdates.append(newdate)
        if not isinstance(datestr, list) and len(newdates) == 1: # Convert back to string/date
            newdates = newdates[0]
        return newdates


def yeartodate(year, as_date=True, **kwargs):
    """
    Convert a decimal year to a date

    Args:
        year (int, float):  The numerical year to convert to a DateTime
        as_date (bool): If True (default), return an ``sc.date`` object, otherwise return a string

    Returns:
        An ``sc.date`` object (default) or string, depending on the ``as_date`` argument

    **Example**::

        sc.yeartodate('2010-07-01') # Returns approximately 2010.5

    | *New in version 3.2.1.*
    """
    as_date = kwargs.pop('asdate', as_date) # Handle with or without underscore
    full_years = int(year)
    remainder = year - full_years
    year_days = _get_year_length(full_years).days
    days = int(np.round(remainder*year_days))
    base = dt.date(year=full_years, month=1, day=1)
    out = datedelta(base, days=days)
    if not as_date:
        out = str(out)
    return out


def datetoyear(dateobj, dateformat=None, **kwargs):
    """
    Convert a date to decimal year.

    Args:
        dateobj (date, str, pd.TimeStamp):  The datetime instance to convert
        dateformat (str): If dateobj is a string, the optional date conversion format to use

    Returns:
        Equivalent decimal year from date, or date from decial year

    **Example**::

        sc.datetoyear('2010-07-01') # Returns approximately 2010.5
        sc.datetoyear(2010.5) # Returns datetime.date(2010, 7, 2)

    By Luke Davis from https://stackoverflow.com/a/42424261, adapted by Romesh Abeysuriya.

    | *New in version 1.0.0.*
    | *New in version 3.2.0:* "reverse" argument
    | *New in version 3.2.1:* "reverse" argument replaced by sc.yeartodate()
    """
    # Handle deprecation
    if kwargs.pop('reverse', None): # pragma: no cover
        warnmsg = 'sc.datetoyear() argument "reverse" has been deprecated as of v3.2.1; use sc.yeartodate() instead'
        warnings.warn(warnmsg, category=FutureWarning, stacklevel=2)
        return yeartodate(dateobj, **kwargs)

    # Handle strings and numbers
    if sc.isstring(dateobj) or isinstance(dateobj, pd.Timestamp):
        dateobj = date(dateobj, dateformat=dateformat)
    year_part = dateobj - dt.date(year=dateobj.year, month=1, day=1)
    year_length = _get_year_length(dateobj.year)
    return dateobj.year + year_part / year_length


###############################################################################
#%% Timing functions
###############################################################################

__all__+= ['tic', 'toc', 'toctic', 'timer', 'Timer']


def tic():
    """
    With :func:`sc.toc() <toc>`, a little pair of functions to calculate a time difference:

    **Examples**::

        sc.tic()
        slow_func()
        sc.toc()

        T = sc.tic()
        slow_func2()
        sc.toc(T, label='slow_func2')

    See also :class:`sc.timer() <timer>`.
    """
    global _tictime  # The saved time is stored in this global
    _tictime = pytime.time()  # Store the present time in the global
    return _tictime    # Return the same stored number


def _convert_time_unit(unit, elapsed=None):
    """ Convert between different units of time; not for the user """

    # Shortcut for speed
    if unit == 's':
        return 1, 's'

    # Standard use case
    else:

        # Define the mapping -- in order of expected usage frequency for speed
        mapping = {
            's'  : dict(factor=   1, aliases=[None, 'default', 's', 'sec', 'secs', 'second', 'seconds']),
            'ms' : dict(factor=1e-3, aliases=['ms', 'milisecond', 'miliseconds']),
            'μs' : dict(factor=1e-6, aliases=['us', 'μs', 'microsecond', 'microseconds']),
            'ns' : dict(factor=1e-9, aliases=['ns', 'nanosecond', 'nanoseconds']),
            'min': dict(factor=  60, aliases=['m', 'min', 'mins', 'minute', 'minutes']),
            'hr' : dict(factor=3600, aliases=['h', 'hr', 'hrs', 'hour', 'hours']),
        }

        # Handle 'auto'
        if unit == 'auto':
            if elapsed is None:  unit = 's'
            elif elapsed < 1e-7: unit = 'ns'
            elif elapsed < 1e-4: unit = 'μs'
            elif elapsed < 1e-1: unit = 'ms'
            else:                unit = 's'

        # Perform the mapping
        factor = None
        for label,entry in mapping.items():
            if unit in [label, entry['factor']] + entry['aliases']:
                factor = entry['factor']
                break
        if factor is None:
            errormsg = f'Could not understand "{unit}"; all possible values are:\n{mapping}'
            raise ValueError(errormsg)

    return factor, label


def toc(start=None, label=None, baselabel=None, sigfigs=None, reset=False, unit='s',
        output=False, verbose=None, elapsed=None, **kwargs):
    """
    With :func:`sc.tic() <tic>`, a little pair of functions to calculate a time difference. See
    also :class:`sc.timer() <timer>`.

    By default, output is displayed in seconds. You can change this with the ``unit``
    argument, which can be a string or a float:

        - 'hr' or 3600
        - 'min' or 60
        - 's' or 1 (default)
        - 'ms' or 1e-3
        - 'us' or 1e-6
        - 'ns' or 1e-9
        - 'auto' to choose an appropriate unit

    Args:
        start    (float): the starting time, as returned by e.g. :func:`sc.tic() <tic>`
        label      (str): optional label to add
        baselabel  (str): optional base label; default is "Elapsed time: "
        sigfigs    (int): number of significant figures for time estimate
        reset     (bool): reset the time; like calling :func:`sc.toctic() <toctic>` or :func:`sc.tic() <tic>` again
        unit (str/float): the unit of time to display; see options above
        output    (bool): whether to return the output (otherwise print); if output='message', then return the message string; if output='both', then return both
        verbose   (bool): whether to print (true by default)
        elapsed  (float): use a pre-calculated elapsed time instead of recalculating (not recommneded)
        kwargs    (dict): not used; only for handling deprecations

    **Examples**::

        sc.tic()
        slow_func()
        sc.toc()

        T = sc.tic()
        slow_func2()
        sc.toc(T, label='slow_func2')

    | *New in version 1.3.0:* new arguments
    | *New in version 3.0.0:* "unit" argument
    | *New in version 3.2.1:* renamed "doprint" to "verbose"
    """
    now = pytime.time() # Get the time as quickly as possible
    global _tictime  # The saved time is stored in this global

    # Handle deprecation
    if len(kwargs):
        warnmsg = 'sc.toc() argument "doprint" is deprecated; use "verbose" instead'
        warnings.warn(warnmsg, category=FutureWarning, stacklevel=2)
        verbose = kwargs.pop('doprint', verbose)
        if len(kwargs):
            errormsg = f'Unrecognized sc.toc() arguments:\n{kwargs}'
            raise TypeError(errormsg)

    # Set defaults
    if sigfigs is None: sigfigs = 3

    # If no start value is passed in, try to grab the global _tictime
    if isinstance(start, str): # Start and label are probably swapped # pragma: no cover
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
        else: # pragma: no cover
            base = baselabel
    else:
        if baselabel is None:
            if label:
                base = f'{label}: '
            else: # Handles case toc(label='') # pragma: no cover
                base = ''
        else:
            base = f'{baselabel}{label}: '
    factor, unitlabel = _convert_time_unit(unit, elapsed=elapsed)
    logmessage = f'{base}{sc.sigfig(elapsed/factor, sigfigs=sigfigs)} {unitlabel}'

    # Print if asked, or if no other output
    if verbose or ((verbose is None) and (not output)):
        print(logmessage)

    # Optionally reset the counter
    if reset:
        _tictime = pytime.time()  # Store the present time in the global

    # Return elapsed if desired
    if output: # pragma: no cover
        if output == 'message':
            return logmessage
        elif output == 'both':
            return (elapsed, logmessage)
        else:
            return elapsed
    else:
        return


def toctic(returntic=False, returntoc=False, *args, **kwargs):
    """
    A convenience fuction for multiple timings. Can return the default output of
    either :func:`sc.tic() <tic>` or :func:`sc.toc() <toc>` (default neither). Arguments are passed to :func:`sc.toc() <toc>`.
    Equivalent to :func:`sc.toc(reset=True) <toc>`.

    **Example**::

        sc.tic()
        slow_operation_1()
        sc.toctic()
        slow_operation_2()
        sc.toc()

    *New in version 1.0.0.*
    """
    tocout = toc(*args, **kwargs)
    ticout = tic()
    if   returntic: return ticout
    elif returntoc: return tocout
    else:           return


class timer:
    """
    Simple timer class. Note: :class:`sc.timer() <timer>` and :class:`sc.Timer() <Timer>` are aliases.

    This wraps :func:`sc.tic() <tic>` and :func:`sc.toc() <toc>` with the formatting arguments and the start time
    (at construction).

    Use this in a ``with`` block to automatically print elapsed time when
    the block finishes.

    By default, output is displayed in seconds. You can change this with the ``unit``
    argument, which can be a string or a float:

        - 'hr' or 3600
        - 'min' or 60
        - 's' or 1 (default)
        - 'ms' or 1e-3
        - 'us' or 1e-6
        - 'ns' or 1e-9
        - 'auto' to choose an appropriate unit

    Args:
        label (str): label identifying this timer
        auto (bool): whether to automatically increment the label
        start (bool): whether to start timing from object creation (else, call :meth:`timer.tic()` explicitly)
        unit (str/float): the unit of time to display; see options above
        verbose (bool): whether to print output on each timing
        kwargs (dict): passed to :func:`sc.toc() <toc>` when invoked


    Example making repeated calls to the same timer, using ``auto`` to keep track::

        >>> T = sc.timer(auto=True)
        >>> T.toc()
        (0): 2.63 s
        >>> T.toc()
        (1): 5.00 s

    Example wrapping code using with-as::

        >>> with sc.timer('mylabel'):
        >>>     sc.timedsleep(0.5)

    Example using a timer to collect data, using :meth:`timer.tt() <timer.tt>` as an alias for :func:`sc.toctic() <toctic>`
    to reset the time::

        T = sc.timer(verbose=False)
        for key in 'abcde':
            sc.timedsleep(np.random.rand())
            T.tt(key)
        print(T.timings)

    Implementation based on https://preshing.com/20110924/timing-your-code-using-pythons-with-statement/

    | *New in version 1.3.0:* :class:`sc.timer() <timer>` alias, and allowing the label as first argument
    | *New in version 1.3.2:* ``toc()`` passes label correctly; ``tt()`` method; ``auto`` argument
    | *New in version 2.0.0:* ``plot()`` method; ``total()`` method; ``indivtimings`` and ``cumtimings`` properties
    | *New in version 2.1.0:* ``total`` as property instead of method; updated repr; added disp() method
    | *New in version 3.0.0:* ``unit`` argument; ``verbose`` argument; ``sum, min, max, mean, std`` methods; ``rawtimings`` property
    | *New in version 3.1.0:* Timers can be combined by addition, including ``sum()``
    | *New in version 3.1.5:* ``T.timings`` is now an :class:`sc.objdict() <sc_odict.objdict>` instead of an :class:`sc.odict() <sc_odict.odict>`
    | *New in version 3.2.2:* ``sc.timer()`` can be used as a function decorator
    """
    def __init__(self, label=None, auto=False, start=True, unit='auto', verbose=None, **kwargs):
        self.kwargs = kwargs # Store kwargs to pass to toc() at the end of the block
        self.kwargs['label'] = str(label) if label is not None else None
        self.auto = auto
        self.unit = unit
        self.verbose = verbose
        self._start = None
        self._tics = []
        self._tocs = []
        self.elapsed = None
        self.message = None
        self.count = 0
        self.timings = sc.objdict()
        if start:
            self.tic() # Start counting
        return


    def __enter__(self):
        """ Reset start time when entering with-as block """
        self.tic()
        return self


    def __exit__(self, *args):
        """ Print elapsed time when leaving a with-as block """
        self.toc()
        return


    def __call__(self, func):
        """ Allow being used as a decorator """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self: # Use as a context block
                return func(*args, **kwargs)
        return wrapper


    def __repr__(self):
        """ Display a brief representation of the object """
        string = sc.objectid(self)
        string += 'Timings:\n'
        string += str(self.timings)
        string += f'\nTotal time: {self.total:n} s'
        return string


    def __len__(self):
        """ Count the number of timings """
        return len(self._tocs)


    def __iadd__(self, T2):
        """ Allow multiple timer objects to be combined """
        self._tics += T2._tics
        self._tocs += T2._tocs
        for k,v in T2.timings.items():
            if k in self.timings.keys(): # Add the current position of the key if duplicates are found
                key = f'({len(self.timings)}) ' + k
            else:
                key = k
            self.timings[key] = v
            self.count += 1
        return self


    def __add__(self, T2):
        """ Ditto """
        T1 = sc.dcp(self)
        return T1.__iadd__(T2)


    def __radd__(self, T2):
        """ For sum() """
        if not T2: return self # Skips the 0 in sum(..., start=0)
        else:      return T2.__add__(self)


    def disp(self):
        """ Display the full representation of the object """
        return sc.pr(self)


    def tic(self):
        """ Set start time """
        now = pytime.time()  # Store the present time locally
        self._start = now
        self._tics.append(now) # Store when this tic was invoked
        return


    def toc(self, label=None, **kwargs):
        """ Print elapsed time; see :func:`sc.toc() <toc>` for keyword arguments """
        # Get the time
        self.elapsed, self.message = toc(start=self._start, output='both', verbose=False) # Get time as quickly as possible
        self._tocs.append(pytime.time()) # Store when this toc was invoked

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
        verbose = kwargs.pop('verbose', self.verbose)
        output = toc(elapsed=self.elapsed, unit=self.unit, verbose=verbose, **kwargs)

        # If reset was used, apply it
        if kwargs.get('reset'):
            self.tic()

        return output


    @property
    def total(self):
        """ Calculate total time """
        # If the timer hasn't been started, return 0
        if not len(self._tics): # pragma: no cover
            return 0
        else:
            start = self._tics[0]

        # If the timer hasn't been finished, use the current time; else the latest
        if not len(self._tocs): # pragma: no cover
            end = pytime.time()
        else:
            end = self._tocs[-1]

        elapsed = end - start

        return elapsed


    # Alias/shortcut methods

    def start(self):
        """ Alias for :func:`sc.tic() <tic>` """
        return self.tic()

    def stop(self, *args, verbose=False, **kwargs):
        """ Alias for :func:`sc.toc() <toc>` """
        return self.toc(*args, verbose=verbose, **kwargs)

    def tocout(self, label=None, output=True, **kwargs):
        """ Alias for :func:`sc.toc() <toc>` with output=True """
        return self.toc(label=label, output=output, **kwargs)

    def toctic(self, *args, reset=True, **kwargs):
        """ Like toc, but reset time between timings """
        return self.toc(*args, reset=reset, **kwargs)

    def tt(self, *args, **kwargs):
        """ Alias for :func:`sc.toctic() <toctic>` """
        return self.toctic(*args, **kwargs)

    def tto(self, *args, output=True, **kwargs):
        """ Alias for :func:`sc.toctic() <toctic>` with output=True """
        return self.toctic(*args, output=output, **kwargs)

    @property
    def rawtimings(self):
        """ Return an array of timings """
        return self.timings[:]

    @property
    def indivtimings(self):
        """ Compute the individual time between each timing """
        vals = np.diff(sc.cat(self._tics[0], self._tocs))
        output = sc.odict(zip(self.timings.keys(), vals))
        return output

    @property
    def cumtimings(self):
        """ Compute the cumulative time for each timing """
        vals = np.array(self._tocs) - self._tics[0]
        output = sc.odict(zip(self.timings.keys(), vals))
        return output

    def sum(self):
        """
        Sum of timings; similar to :obj:`timer.total <timer.total>`

        *New in version 3.0.0.*
        """
        return self.rawtimings.sum()

    def min(self):
        """
        Minimum of timings

        *New in version 3.0.0.*
        """
        return self.rawtimings.min()

    def max(self):
        """
        Maximum of timings

        *New in version 3.0.0.*
        """
        return self.rawtimings.max()

    def mean(self):
        """
        Mean of timings

        *New in version 3.0.0.*
        """
        return self.rawtimings.mean()

    def std(self):
        """
        Standard deviation of timings

        *New in version 3.0.0.*
        """
        return self.rawtimings.std()


    def plot(self, fig=None, figkwargs=None, grid=True, **kwargs):
        """
        Create a plot of Timer.timings

        Arguments:
            cumulative (bool): how the timings will be presented, individual or cumulative
            fig (fig): an existing figure to draw the plot in
            figkwargs (dict): passed to :func:`plt.figure() <matplotlib.pyplot.figure>`
            grid (bool): whether to show a grid
            kwargs (dict): passed to :func:`plt.bar() <matplotlib.pyplot.bar>`

        *New in version 2.0.0.*
        """
        figkwargs = sc.mergedicts(figkwargs)

        # Handle the figure
        if fig is None:
            fig = plt.figure(**figkwargs)  # It's necessary to have an open figure or else the commands won't work

        # Plot times
        if len(self.timings) > 0:
            keys = self.timings.keys()
            vals = self.indivtimings[:]

            factor, label = _convert_time_unit(self.unit, elapsed=vals.sum())
            vals /= factor

            ax1 = plt.subplot(2,1,1)
            plt.barh(keys, vals, **kwargs)
            plt.title('Individual timings')
            plt.xlabel(f'Elapsed time ({label})')

            ax2 = plt.subplot(2,1,2)
            plt.barh(keys, np.cumsum(vals), **kwargs)
            plt.title('Cumulative timings')
            plt.xlabel(f'Elapsed time ({label})')

            for ax in [ax1, ax2]:
                ax.invert_yaxis()
                ax.grid(grid)

            sc.figlayout()
        else: # pragma: no cover
            errormsg = "Looks like nothing has been timed. Forgot to do T.start() and T.stop()??'"
            raise RuntimeWarning(errormsg)

        return fig


Timer = timer # Alias


###############################################################################
#%% Other functions
###############################################################################

__all__ += ['elapsedtimestr', 'timedsleep', 'randsleep']


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
        else: # pragma: no cover
            month_token = '%B'

        # Get a string from strftime()
        if includeyear:
            date_str = date.strftime('%d ' + month_token + ' %Y')
        else: # pragma: no cover
            date_str = date.strftime('%d ' + month_token)

        # There will only ever be at most one leading zero, so check for this and
        # remove if necessary
        if date_str[0] == '0':
            date_str = date_str[1:]

        return date_str

    now_time = dt.datetime.now()

    # If the user passes in a string, try to turn it into a datetime object before continuing
    if isinstance(pasttime, str): # pragma: no cover
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
    else: # pragma: no cover
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


# Most robust results just from hard-coding this, despite variability between machines
_sleep_overhead = 5e-5 # Amount of time taken for a "zero" delay

def timedsleep(delay=None, start=None, verbose=False):
    """
    Pause for the specified amount of time, taking into account how long other
    operations take.

    This function is usually used in a loop; it works like ``time.sleep()``, but
    subtracts time taken by the other operations in the loop so that each loop
    iteration takes exactly ``delay`` amount of time. Note: since ``time.sleep()``
    has a minimum overhead (about 2e-4 seconds), below this duration, no pause
    will occur.

    Args:
        delay (float): time, in seconds, to wait for
        start (float): if provided, the start time
        verbose (bool): whether to print details

    **Examples**::

        # Example for a long(ish) computation
        import numpy as np
        for i in range(10):
            sc.timedsleep('start') # Initialize
            n = int(2*np.random.rand()*1e6) # Variable computation time
            for j in range(n):
                tmp = np.random.rand()
            sc.timedsleep(1, verbose=True) # Wait for one second per iteration including computation time


        # Example illustrating more accurate timing
        import time
        n = 1000

        with sc.timer():
            for i in range(n):
                sc.timedsleep(1/n)
        # Elapsed time: 1.01 s

        with sc.timer():
            for i in range(n):
                time.sleep(1/n)
        # Elapsed time: 1.21 s

    *New in version 3.0.0:* "verbose" False by default; more accurate overhead calculation
    """
    global _delaytime
    if delay is None or delay=='start':
        _delaytime = pytime.time()  # Store the present time in the global.
        return _delaytime         # Return the same stored number.
    else:
        if start is None:
            try:    start = _delaytime
            except: start = pytime.time()
        elapsed = pytime.time() - start
        remaining = max(1e-12, delay - elapsed - _sleep_overhead)
        if remaining > 0 and verbose:
            print(f'Pausing for {remaining:n} s')
        elif verbose: # pragma: no cover
            print(f'Warning, delay less than elapsed time ({delay:n} vs. {elapsed:n})')
        pytime.sleep(remaining)
        try:    del _delaytime # After it's been used, we can't use it again
        except: pass
    return


def randsleep(delay=1.0, var=1.0, low=None, high=None, seed=None):
    """
    Sleep for a nondeterminate period of time (useful for desynchronizing tasks)

    Args:
        delay (float/list): average duration in seconds to sleep for; if a pair of values, treat as low and high
        var   (float):      how much variability to have (default, 1.0, i.e. from 0 to 2*interval)
        low   (float):      optionally define lower bound of sleep
        high  (float):      optionally define upper bound of sleep
        seed  (int):        if provided, reset the random seed

    **Examples**::
        sc.randsleep(1) # Sleep for 0-2 s (average 1.0)
        sc.randsleep(2, 0.1) # Sleep for 1.8-2.2 s (average 2.0)
        sc.randsleep([0.5, 1.5]) # Sleep for 0.5-1.5 s
        sc.randsleeep(low=0.5, high=1.5) # Ditto

    *New in version 2.0.0.*
    *New in version 3.0.0:* "seed" argument
    """
    if low is None or high is None:
        if sc.isnumber(delay):
            low  = delay*(1-var)
            high = delay*(1+var)
        else:
            low, high = delay[0], delay[1]

    rng = np.random.default_rng(seed)
    dur = rng.uniform(low, high)
    pytime.sleep(dur)

    return dur
