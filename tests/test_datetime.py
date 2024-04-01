'''
Test Sciris miscellaneous utility/helper functions.
'''

import sciris as sc
import numpy as np
import pandas as pd
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
    
    # And more format testing
    dmy = '18-09-2020'
    mdy = '09-18-2020'
    assert sc.readdate(dmy, dateformat='dmy') == sc.readdate(mdy, dateformat='mdy')

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
    o.date0 = sc.date()
    o.date1 = sc.date('2020-04-05') # Returns datetime.date(2020, 4, 5)
    o.date2 = sc.date(sc.readdate('2020-04-14'), as_date=False, outformat='%Y%m') # Returns '202004'
    o.date3 = sc.date([35,36,37], start_date='2020-01-01', as_date=False) # Returns ['2020-02-05', '2020-02-06', '2020-02-07']
    o.date4 = sc.date(1923288822, readformat='posix', to='pandas') # Interpret as a POSIX timestamp
    o.date5 = sc.date(pd.Timestamp(year=2020, month=9, day=18), to='numpy')
    with pytest.raises(ValueError):
        sc.date(to='invalid_format')
    with pytest.raises(ValueError):
        sc.date([10,20]) # Can't convert an integer without a start date
    assert o.date1.month == 4
    assert o.date2 == '202004'
    assert len(o.date3) == 3
    assert o.date3[0] =='2020-02-05'

    print('\nTesting day')
    assert sc.day(None) is None
    o.day = sc.day('2020-04-04') # Returns 94
    assert o.day == 94
    assert sc.day('2020-03-01') > sc.day('2021-03-01') # Because of the leap day
    assert sc.day('2020-03-01', start_date='2020-01-01') < sc.day('2021-03-01', start_date='2020-01-01') # Because years
    out = sc.day([1000, None, sc.now()], start_date='2020-04-04')
    assert out[2] > out[0] # More than 1000 days from April 2020

    print('\nTesting daydiff')
    o.diff  = sc.daydiff('2020-03-20', '2020-04-05') # Returns 16
    o.diffs = sc.daydiff('2020-03-20', '2020-04-05', '2020-05-01') # Returns [16, 26]
    assert len(o.diffs) == 2

    print('\nTesting daterange')
    o.dates = sc.daterange('2020-03-01', '2020-04-04')
    assert len(o.dates) == 35
    ndays = 40
    r2 = sc.daterange('2020-03-01', days=ndays)
    assert len(r2) == ndays + 1 # Since inclusive
    
    dr = sc.objdict()
    for interval in ['day', 'week', 'month', 'year']:
        dr[interval] = sc.daterange('2020-01-01', '2023-01-01', interval=interval)
    assert len(dr.day) > len(dr.week) > len(dr.month) > len(dr.year)

    print('\nTesting elapsedtimestr')
    now = sc.now()
    dates = sc.objdict()
    dates.future = now.replace(year=2023) # Avoid leap years
    dates.year =   now.replace(year=2021)
    for key in ['days', 'hours', 'minutes']:
        dates[key] = now - dt.timedelta(**{key:1})
    for key, date in dates.items():
        print(f'For unit {key}: {sc.elapsedtimestr(date)}')

    print('\nTesting datetoyear')
    o.year = sc.datetoyear('2010-07-01')

    print('\nTesting datedelta')
    o.dd = sc.datedelta('2021-07-07', 3) # Add 3 days
    assert o.dd == '2021-07-10'

    return o

delay = 0.05 # Shorter creates timing errors
def nap(n=1, t=delay):
    ''' Little nap to test timers '''
    return sc.timedsleep(t*n)


def test_timing():
    sc.heading('Testing tic, toc, and timedsleep')
    
    print('\nTesting tictoc and timedsleep')
    sc.tic()
    sc.timedsleep(0.1)
    sc.toctic()
    sc.timedsleep('start')
    with sc.Timer():
        nap()
    
    print('Testing randsleep')
    sc.randsleep(0.1)
    sc.randsleep([0.05, 0.15])
    
    print('Testing tic/toc again...')

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
    
    return T


def test_timer():
    sc.heading('Testing timer')
    o = sc.objdict()

    print('Check label handling')
    with sc.capture() as txt3:
        T = sc.timer(baselabel='mybase: ', label='mylabel')
        T.tic()
        nap()
        T.toc()
        print(T)
        T.disp()
    print(txt3)
    assert 'mybase' in txt3
    assert 'mylabel' in txt3
    o.t1 = T

    print('Check timer labels')
    with sc.capture() as txt4:
        T = sc.timer()
        T.start()
        nap()
        T.stop('newlabel')
    print(txt4)
    assert 'newlabel' in txt4
    o.t2 = T

    print('Check relative timings')
    T = sc.timer()
    T.start()
    nap(1)
    T.tt()
    nap(5)
    T.tt()
    print(T.timings)
    assert T.timings[0] < T.timings[1]
    o.t3 = T

    print('Check toc vs toctic')
    T = sc.timer()
    nap(5)
    T.toc('a') # ≈5
    nap()
    T.toctic('b') # ≈6
    nap()
    T.toctic('c') # ≈1
    nap(2)
    T.tt('d') # ≈2
    assert T.timings['c'] < T.timings['b'] # Should be c < d < a < b, but too stringent
    o.t4 = T

    print('Check auto naming')
    T = sc.timer(auto=True, doprint=False)
    with sc.capture() as txt5:
        n = 5
        for i in range(n):
            nap()
            T.tt()
    assert txt5 == ''
    lbound = n*delay/3 # Very generous margin, but needed unfortunately ...
    ubound = n*delay*3
    assert lbound < T.timings[:].sum() < ubound
    assert '(4)' in T.timings.keys()[4]
    assert T.cumtimings[-1] == T.total
    
    print('Check other things')
    T.tocout()
    T.tto()
    T.sum()
    T.mean()
    T.std()
    T.min()
    T.max()
    print(T.indivtimings)
    
    print('Check plotting')
    T.plot()
    o.t5 = T
    
    print('Check addition')
    o.t6 = o.t1 + o.t2
    o.t7 = o.t3
    o.t7 += o.t4
    o.t8 = sum([o.t5, o.t6])

    return o



#%% Run as a script
if __name__ == '__main__':
    T = sc.tic()

    # Dates
    dateobj   = test_readdate()
    dates     = test_dates()
    timing    = test_timing()
    times     = test_timer()

    sc.blank()
    sc.toc(T)
    print('Done.')