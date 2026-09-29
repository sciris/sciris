'''
Test Sciris printing functions.
'''

import random
from unittest import mock
import numpy as np
import pytest
import sciris as sc


def test_colorize():
    sc.heading('Test text colorization')
    sc.colorize(showhelp=True)
    print('Simple example:')
    sc.colorize('green', 'hi')
    print('More complicated example:')
    sc.colorize(['yellow', 'bgblack'])
    print('Hello world')
    print('Goodbye world')
    sc.colorize('reset') # Colorize all output in between
    bluearray = sc.colorize(color='blue', string=str(range(5)), output=True)
    print("This should be blue: " + bluearray)

    print('This should be a rainbow:')
    sc.printred(    'This should be red')
    sc.printyellow( 'This should be yellow')
    sc.printgreen(  'This should be green')
    sc.printcyan(   'This should be cyan')
    sc.printblue(   'This should be blue')
    sc.printmagenta('This should be magenta')
    sc.printbold(   'This should be bold')

    print('Testing ANSI stripping')
    assert sc.strip_ansi(bluearray) == str(range(5))
    assert sc.strip_ansi('\x1b[31mred\x1b[0m then \x1b[2K more text') == 'red then  more text' # Non-color codes don't remove text

    print('Testing colorize options')
    assert sc.colorize(output=True) == '\x1b[0m' # No arguments resets the color
    assert 'message' in sc.colorize('notacolor', 'message', output=True) # Invalid color still shows the string
    assert sc.colorize('red', 'hi', output=True, enable=False) == 'hi'
    assert sc.heading('Hi', fg='red', output=True).startswith('\x1b[31m') # Alternate usage via kwargs

    print('Testing tight headings')
    normal = sc.heading('Normal heading', output=True)
    tight = sc.heading('Tight heading', tight=True, output=True)
    assert normal.count('\n') == tight.count('\n') + 2 # 1 fewer newline before, and 1 fewer after
    return


def test_printing(test_slack=False):
    sc.heading('Test printing functions')

    example = sc.prettyobj()
    example.data = sc.vectocolor(10)

    print('sc.pp():')
    sc.pp(example.data)
    sc.pp(example.data, jsonify=True)
    string = sc.pp(example.data, doprint=False)

    print('sc.printdata():')
    sc.printdata(example.data)

    o = sc.objdict()

    print('\nTesting print')
    sc.printv('test print', 1, 2)

    print('\nTesting objatt')
    o.att = sc.objatt(o)

    print('\nTesting prepr')
    sc.prepr(o)

    print('\nTesting printarr')
    sc.printarr(np.random.rand(3))
    sc.printarr(np.random.rand(3,4))
    sc.printarr(np.random.rand(3,4,5))
    assert sc.printarr(np.array([1.5, 2.25], dtype=np.float32), doprint=False) == '1.50  2.25  \n' # Not just float64
    assert sc.printarr(np.array([[-100.5, 1.0], [3.25, 4.0]]), doprint=False) == '-100.50     1.00  \n\n   3.25     4.00  \n\n' # Columns aligned
    assert sc.printarr(np.array([1, 20, 300]), doprint=False) == '  1   20  300  \n' # Integers
    sc.printarr(np.array([['cat', 'nudibranch'], [23, 2423482]], dtype=object)) # Objects
    sc.printarr(np.array([])) # Empty

    print('\nTesting printvars')
    a = range(5)
    b = 'example'
    sc.printvars(locals(), ['a','b'], color='green')

    if test_slack:
        print('\nTesting slacknotification')
        sc.slacknotification(webhook='http://invalid.hooks.slack.com.test', message='Test notification to nowhere')
        print('↑↑↑ Will raise a error since not a valid webhook, this is OK')

    print('\nTesting printtologfile')
    sc.printtologfile('Test message')

    print('\nTesting sigfig')
    sc.sigfig(np.random.rand(10)*1e5, SI=True, sep='.')
    sc.sigfig(np.random.rand(), sigfigs=None) # Testing no sigfigs
    assert sc.sigfig([4.958, 23432.23], sigfigs=3, keepints=True) == ['4.96', '23432'] # Testing keepints
    assert sc.sigfig((0.23456, 28847.9), sep=',') == ('0.2346', '28,850') # Testing tuple and separator
    assert sc.sigfig(-23432.23, sigfigs=3, keepints=True) == '-23432' # Negative ints
    assert sc.sigfig(1234567.0, sep='.') == '1.235.000' # Custom separator
    assert sc.sigfig(3432.3842, SI=True, formats='kmb') == '3.432k' # Formats as string
    assert sc.sigfig([9.9999, 999999], SI=True) == ['10.00', '1.000M'] # Rounding up to the next power of ten

    print('\nTesting sigfiground')
    assert sc.sigfiground([1234, 5678], 2) == [1200, 5700]
    assert sc.sigfiground(1e19) == 1e19 and sc.sigfiground(np.inf) == np.inf # Too large to convert to int
    assert sc.arraymean([5, 5, 5], tostring=False) == (5, 0) # Zero standard deviation


    print('\nTesting printmean and printmedian')
    data = [1210, 1072, 1722, 1229, 1902, 1753, 1223, 1024, 1884, 1525, 1449]
    sc.printmean(data)
    o.printmean = sc.printmean(data, doprint=False)
    assert o.printmean == '1450 ± 620'
    assert sc.printmean(data, mean_sf=2, doprint=False) == '1500 ± 600'
    assert sc.printmean(data, err_sf=1, doprint=False) == '1500 ± 600'
    assert sc.printmean([[10, 20], [12, 21]], axis=0, doprint=False) == '11.0 ± 2.0, 20.5 ± 1.0' # Multiple values

    sc.printmedian(data)
    assert sc.printmedian(data, doprint=False, ci='iqr')      == '1450 (IQR: 1220, 1740)'      # Test IQR
    assert sc.printmedian(data, doprint=False, ci='range')    == '1450 (min, max: 1020, 1900)' # Test range
    assert sc.printmedian(data, doprint=False, ci=80)         == '1450 (80% CI: 1070, 1880)'   # Test int
    assert sc.printmedian(data, doprint=False, ci=0.8)        == '1450 (80% CI: 1070, 1880)'   # Test float
    assert sc.printmedian(data, doprint=False, ci=[0, 10])    == '1450 (0%, 10%: 1020, 1070)'  # Test pair of ints
    assert sc.printmedian(data, doprint=False, ci=[0.0, 0.1]) == '1450 (0%, 10%: 1020, 1070)'  # Test pair of floats
    ci = [0, 10]
    assert sc.printmedian(data, doprint=False, ci=tuple(ci)) == sc.printmedian(data, doprint=False, ci=ci) # Test tuple
    assert ci == [0, 10] # Input is not modified
    assert sc.printmedian([-1.234, 0, 1.234], doprint=False) == '0 (95% CI: -1.17, 1.17)' # Median of zero


    print('\nTesting capture')
    str1 = 'I am string 1'
    str2 = 'I am string 2'
    with sc.capture() as txt1:
        print(str1)

    txt2 = sc.capture().start()
    print(str2)
    txt2.stop()

    # print() appends a newline character which we have to remove for the comparison
    assert txt1.rstrip() == str1
    assert txt2.rstrip() == str2
    assert '%s' % txt1 == str1 + '\n'

    txt2.start() # Reuse the same object
    print(str1)
    txt2.stop()
    assert txt2 == str2 + '\n' + str1 + '\n'

    print('\nTesting slacknotification without sending')
    import requests
    response = requests.models.Response()
    response.status_code = 404
    with mock.patch('requests.post', return_value=response) as post:
        with sc.capture():
            sc.slacknotification(message='hi', webhook='https://hooks.slack.com/test\n', verbose=0)
        assert post.call_args.kwargs['url'] == 'https://hooks.slack.com/test' # Newline removed
        with pytest.raises(RuntimeError): # Failure status is caught
            sc.slacknotification(message='hi', webhook='https://hooks.slack.com/test', verbose=0, die=True)

    return o


def test_prepr():
    sc.heading('Test pretty representation of an object')

    n_attrs = 500
    pobj = sc.prettyobj()
    qobj = sc.quickobj()
    for i in range(n_attrs):
        key = f'attr{i:03d}'
        setattr(pobj, key, i**2)
        setattr(qobj, key, i**2)

    print('sc.prettyobj:')
    print(pobj)

    print('sc.quickobj:')
    print(qobj)

    print('Testing pretty representation of an object using slots')
    class Foo:
        __slots__ = ['bar']
        def __init__(self):
            self.bar = 1

    x = Foo()
    print(sc.prepr(x))
    print(sc.prepr(x, maxtime=0))
    assert '20 entries not shown' in sc.prepr(sc.prettyobj({f'k{i}':i for i in range(20)}), maxtime=0)

    class Bar:
        def skip(self): pass

    print(sc.prepr(Bar()))

    print('Testing that properties are not evaluated')
    class Prop(sc.prettyobj):
        @property
        def prop(self):
            raise RuntimeError('Property should not be evaluated')
        def meth(self): pass
    assert 'prop' in sc.prepr(Prop(), die=True)
    assert sc.objmeth(Prop(), return_keys=True) == ['meth']
    assert sc.createcollist(list('abcd'), strlen=1, ncol=3) == '\n  a  c  \n  b  d  \n' # Columns are read downwards

    for tf in [True, False]:
        sc.objrepr(x, showid=tf, showmeth=tf, showprop=tf, showatt=tf)

    print('Testing sc.pr():')
    class Obj: pass
    obj = Obj()
    obj.a = np.arange(10)
    obj.b = dict(a=dict(b=np.random.rand(5)), c=['a', 'b'])
    sc.pr(obj)

    return pobj


def test_recursion():
    sc.heading('Test that prepr is safe against recursion')

    class BranchingRecursion:
        """ Create a dastardly object that references itself twice """
        def __init__(self, x):
            self.x = x
            self.obj1 = self
            self.obj2 = self

        def __repr__(self):
            return sc.prepr(self, maxlen=1000)

    recurse = BranchingRecursion(5)

    with sc.timer() as T:
        with sc.capture() as txt:
            print(recurse)

    assert T.total < 0.3 # Should terminate relatively quickly
    assert 'recursion' in txt # Should show termination message

    return recurse


def test_progress_bar():
    sc.heading('Progress bar and percent complete')

    totalsleep = 0.3
    n = 50

    for i in sc.progressbar(range(n)):
        sc.timedsleep(0.3/n)

    for i in range(n):
        sc.progressbar(i+1, n)
        sc.timedsleep(totalsleep/n)

    with sc.capture() as txt:
        for i in range(n):
            sc.percentcomplete(i, n, stepsize=10) # will print on every 5th iteration
    assert txt.split() == [f'{p}%' for p in range(0, 100, 10)]

    assert sc.progressbar(5, 10, every=0.5, length=4, output=True) == '\r ••—— 50%'
    assert sc.progressbar(4, 10, every=0.5, output=True) == '' # Not printed on this iteration

    return i


def test_progress_bars():

    totalsleep = 0.5
    def run_sim(index, ndays, pbs):
        for i in range(ndays):
            val = random.random()
            sc.timedsleep(val*totalsleep/ndays)
            pbs.update(index) # Update this progress bar based on the index
        return

    nsims = 4
    ndays = 365

    # Create progress bars
    pbs = sc.progressbars(nsims, total=ndays, label='Sim')

    # Run tasks
    sc.parallelize(run_sim, iterarg=range(nsims), ndays=ndays, pbs=pbs)

    return pbs


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    bluearray = test_colorize()
    printing  = test_printing()
    myobj     = test_prepr()
    recurse   = test_recursion()
    ind       = test_progress_bar()
    pbs       = test_progress_bars()

    sc.toc()
    print('Done.')