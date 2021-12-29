'''
Test Sciris miscellaneous utility/helper functions.
'''

import numpy as np
import sciris as sc
import datetime as dt
import pytest


#%% Date/time functions

def test_readdate():
    sc.heading('Test string-to-date conversion')

    # Basic tests
    string0 = '2020-Mar-21'
    string1 = '2020-03-21'
    string2 = 'Sat Mar 21 23:13:56 2020'
    string3 = '2020-04-04'
    dateobj0 = sc.readdate(string0)
    dateobj1 = sc.readdate(string1)
    dateobj2 = sc.readdate(string2)
    assert dateobj0 == dateobj1
    with pytest.raises(ValueError):
        sc.readdate('Not a date')

    # Test different formats
    strlist = [string0, string0, string2]
    strarr = np.array(strlist)
    fromlist = sc.readdate(strlist, string3)
    fromarr = sc.readdate(strarr, string3)
    assert fromlist[2] == dateobj2
    assert sc.readdate(*strlist)[2] == dateobj2
    assert fromarr[2] == dateobj2
    assert len(fromlist) == len(fromarr) == len(strlist) + 1 == len(strarr) + 1
    assert isinstance(fromlist, list)
    assert isinstance(fromarr, np.ndarray)

    # Test timestamps
    now_datetime = sc.now()
    now_timestamp = sc.readdate(sc.tic())
    assert now_timestamp.day == now_datetime.day

    # Format tests
    formats_to_try = sc.readdate(return_defaults=True)
    for key,fmt in formats_to_try.items():
        datestr = sc.getdate(dateformat=fmt)
        dateobj = sc.readdate(datestr, dateformat=fmt)
        print(f'{key:15s} {fmt:22s}: {dateobj}')

    # Basic tests
    assert sc.sc_utils._sanitize_iterables(1)                   == ([1],     False, False)
    assert sc.sc_utils._sanitize_iterables(1, 2, 3)             == ([1,2,3], True, False)
    assert sc.sc_utils._sanitize_iterables([1, 2], 3)           == ([1,2,3], True, False)
    assert sc.sc_utils._sanitize_iterables(np.array([1, 2]), 3) == ([1,2,3], True, True)
    assert sc.sc_utils._sanitize_iterables(np.array([1, 2, 3])) == ([1,2,3], False, True)

    return dateobj0


def test_dates():
    sc.heading('Testing other date functions')
    o = sc.objdict()

    print('\nTesting date')
    o.date1 = sc.date('2020-04-05') # Returns datetime.date(2020, 4, 5)
    o.date2 = sc.date(sc.readdate('2020-04-14'), as_date=False, dateformat='%Y%m') # Returns '202004'
    o.date3 = sc.date([35,36,37], start_date='2020-01-01', as_date=False) # Returns ['2020-02-05', '2020-02-06', '2020-02-07']
    o.date4 = sc.date(1923288822, readformat='posix') # Interpret as a POSIX timestamp
    with pytest.raises(ValueError):
        sc.date([10,20]) # Can't convert an integer without a start date
    assert o.date1.month == 4
    assert o.date2 == '202004'
    assert len(o.date3) == 3
    assert o.date3[0] =='2020-02-05'

    print('\nTesting day')
    o.day = sc.day('2020-04-04') # Returns 94
    assert o.day == 94
    assert sc.day('2020-03-01') > sc.day('2021-03-01') # Because of the leap day
    assert sc.day('2020-03-01', start_date='2020-01-01') < sc.day('2021-03-01', start_date='2020-01-01') # Because years

    print('\nTesting daydiff')
    o.diff  = sc.daydiff('2020-03-20', '2020-04-05') # Returns 16
    o.diffs = sc.daydiff('2020-03-20', '2020-04-05', '2020-05-01') # Returns [16, 26]
    assert len(o.diffs) == 2

    print('\nTesting daterange')
    o.dates = sc.daterange('2020-03-01', '2020-04-04')
    assert len(o.dates) == 35

    print('\nTesting elapsedtimestr')
    now = sc.now()
    dates = sc.objdict()
    dates.future = now.replace(year=now.year+1)
    dates.year =   now.replace(year=now.year-1)
    for key in ['days', 'hours', 'minutes']:
        dates[key] = now - dt.timedelta(**{key:1})
    for key, date in dates.items():
        print(key, sc.elapsedtimestr(date))

    print('\nTesting tictoc and timedsleep')
    sc.tic()
    sc.timedsleep(0.2)
    sc.toctic()
    sc.timedsleep('start')
    with sc.Timer():
        sc.timedsleep(0.1)

    print('\nTesting datetoyear')
    o.year = sc.datetoyear('2010-07-01')

    print('\nTesting datedelta')
    o.dd = sc.datedelta('2021-07-07', 3) # Add 3 days
    assert o.dd == '2021-07-10'

    return o


#%% Nested dictionary functions

def test_nested():
    sc.heading('Testing nested dicts')
    o = sc.objdict()

    foo = {}
    sc.makenested(foo, ['a','b'])
    foo['a']['b'] = 3
    print(sc.getnested(foo, ['a','b']))    # 3
    sc.setnested(foo, ['a','b'], 7)
    print(sc.getnested(foo, ['a','b']))    # 7
    sc.makenested(foo, ['bar','cat'])
    sc.setnested(foo, ['bar','cat'], 'in the hat')
    print(foo['bar'])  # {'cat': 'in the hat'}
    o.foo1 = foo

    foo = {}
    sc.makenested(foo, ['a','x'])
    sc.makenested(foo, ['a','y'])
    sc.makenested(foo, ['a','z'])
    sc.makenested(foo, ['b','a','x'])
    sc.makenested(foo, ['b','a','y'])
    count = 0
    for twig in sc.iternested(foo):
        count += 1
        sc.setnested(foo, twig, count) # Yields {'a': {'y': 1, 'x': 2, 'z': 3}, 'b': {'a': {'y': 4, 'x': 5}}}
    o.foo2 = foo

    return o


def test_dicts():
    sc.heading('Testing dicts')
    o = sc.objdict()

    print('\nTesting flattendict')
    sc.flattendict({'a': {'b': 1, 'c': {'d': 2, 'e': 3}}})
    o.flat = sc.flattendict({'a': {'b': 1, 'c': {'d': 2, 'e': 3}}}, sep='_')

    print('Testing merging dictionaries')
    o.md = sc.mergedicts({'a':1}, {'b':2}) # Returns {'a':1, 'b':2}
    sc.mergedicts({'a':1, 'b':2}, {'b':3, 'c':4}) # Returns {'a':1, 'b':3, 'c':4}
    sc.mergedicts({'b':3, 'c':4}, {'a':1, 'b':2}) # Returns {'a':1, 'b':2, 'c':4}
    with pytest.raises(KeyError):
        sc.mergedicts({'b':3, 'c':4}, {'a':1, 'b':2}, overwrite=False) # Raises exception
    with pytest.raises(TypeError):
        sc.mergedicts({'b':3, 'c':4}, None, strict=True) # Raises exception

    print('\nTesting nested dictionaries')
    dict1 = {'key1':{'a':'A'},  'key2':{'b':'B'}}
    dict2 = {'key1':{'a':'A*'}, 'key2':{'b+':'B+'}, 'key3':{'c':'C'}}
    dict3 = sc.mergenested(dict1, dict2, verbose=True)
    print('↑ Should print warning above')
    print(f'Dict1: {dict1}')
    print(f'Dict2: {dict2}')
    print(f'Dict3: {dict3}')
    assert dict3 == {'key1': {'a': 'A*'}, 'key2': {'b': 'B', 'b+': 'B+'}, 'key3': {'c': 'C'}}
    o.dict3 = dict3

    print('\nTesting search')
    nested = {'a':{'foo':1, 'bar':2}, 'b':{'bar':3, 'cat':4}}
    matches = sc.search(nested, 'bar') # Returns ['["a"]["bar"]', '["b"]["bar"]']
    print(matches)
    o.matches = matches

    return o


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    # Dates
    dateobj   = test_readdate()
    dates     = test_dates()

    # Nested
    nested    = test_nested()
    dicts     = test_dicts()

    sc.toc()
    print('Done.')