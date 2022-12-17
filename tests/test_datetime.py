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
    o.date2 = sc.date(sc.readdate('2020-04-14'), as_date=False, outformat='%Y%m') # Returns '202004'
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


def test_timer():
    sc.heading('Testing tic, toc, and timer')

    t = 0.01
    def nap(n=1):
        sc.timedsleep(t*n, verbose=False)
        return

    print('Testing tic/toc...')

    # Test basic usage
    with sc.capture() as txt1:
        sc.tic(); nap(); sc.toc()
    print(txt1)
    assert 'Elapsed' in txt1

    # Test advanced usage
    with sc.capture() as txt2:
        T = sc.tic()
        nap()
        sc.toc(T, label='slow_func2')
    print(txt2)
    assert 'slow_func2' in txt2

    print('Testing timer...')

    # Check label handling
    with sc.capture() as txt3:
        T = sc.timer(baselabel='mybase: ', label='mylabel')
        T.tic()
        nap()
        T.toc()
    print(txt3)
    assert 'mybase' in txt3
    assert 'mylabel' in txt3

    # Check timer labels
    with sc.capture() as txt4:
        T = sc.timer()
        T.start()
        nap()
        T.stop('newlabel')
    print(txt4)
    assert 'newlabel' in txt4

    # Check relative timings
    T = sc.timer()
    T.start()
    nap(1)
    T.tt()
    nap(5)
    T.tt()
    print(T.timings)
    assert T.timings[0] < T.timings[1]

    # Check toc vs toctic
    T = sc.timer()
    nap(3)
    T.toc('a') # ≈3
    nap()
    T.toctic('b') # ≈4
    nap()
    T.toctic('c') # ≈1
    nap(2)
    T.tt('d') # ≈2
    assert T.timings['c'] < T.timings['d'] < T.timings['a'] < T.timings['b']

    # Check auto naming
    T = sc.timer(auto=True, doprint=False)
    with sc.capture() as txt5:
        n = 5
        for i in range(n):
            nap()
            T.tt()
    assert txt5 == ''
    lbound = n*t/2
    ubound = n*t*2
    assert lbound < T.timings[:].sum() < ubound
    assert '(4)' in T.timings.keys()[4]
    assert T.cumtimings[-1] == T.total

    return T.timings



#%% Run as a script
if __name__ == '__main__':
    T = sc.tic()

    # Dates
    dateobj   = test_readdate()
    dates     = test_dates()
    times     = test_timer()

    sc.blank()
    sc.toc(T)
    print('Done.')