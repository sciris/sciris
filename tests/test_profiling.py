'''
Test profiling functions.
'''
import sys
import time
import signal
import warnings
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import sciris as sc
import pytest


def test_loadbalancer():
    sc.heading('Testing loadbalancer')
    o = sc.objdict()

    # Test basic functions
    o.ncpus = sc.cpu_count()
    o.cpu = sc.cpuload()
    o.mem = sc.memload()

    # Test loadbalancer
    o.load = sc.loadbalancer(interval=0.1)
    return o


def test_memchecks():
    sc.heading('Testing memory checks')

    o = sc.objdict()

    complexobj = sc.prettyobj(
        a = sc.objdict(
            a_dict = {'foo':'bar'},
            a_arr = np.random.rand(243,589),
            ),
        b = [0,1,2,[3,4,'foo']]
    )

    print('\nTesting checkmem')
    o.mem = sc.checkmem(complexobj, descend=2)

    # Arrays are checked as single objects; non-string keys; subtotals; ordering; plotting
    assert len(sc.checkmem(np.zeros(2000))) == 1
    nested = {'x': {0: np.zeros(3), 1: np.zeros(4)}}
    df = sc.checkmem(nested, descend=2, subtotals=False, order='alphabetical')
    assert df.variable.tolist() == ['x→0', 'x→1']
    df = sc.checkmem(nested, descend=2, plot=True)
    assert 'x (total)' in df.variable.values
    assert 'Total' not in [t.get_text() for t in plt.gca().texts] # Totals are not plotted
    plt.close('all')
    with pytest.raises(RuntimeError):
        sc.checkmem(list(range(10)), maxitems=5)

    # Shared objects are counted once in the total; empty and unpicklable objects; compression
    arr = np.random.rand(100,100)
    df = sc.checkmem(dict(x=arr, y=arr))
    assert df.bytesize[df.is_total].iloc[0] < df.bytesize[~df.is_total].sum()
    assert len(sc.checkmem({})) == 1
    assert sc.checkmem(lambda x: x).bytesize.iloc[0] > 0 # Requires dill
    assert sc.checkmem(np.zeros(1000), compresslevel=1).bytesize.iloc[0] < sc.checkmem(np.zeros(1000)).bytesize.iloc[0]

    print('\nTesting checkram')
    o.ram = sc.checkram()
    print(o.ram)
    assert isinstance(sc.checkram(to_string=False), float)

    return o


def test_profile():
    sc.heading('Test profiling functions (profile/mprofile)')

    print('Benchmarking:')
    bm = sc.benchmark()
    print(bm)
    assert bm['numpy'] > bm['python']
    timers = sc.benchmark(repeats=1, scale=0.1, return_timers=True)
    assert isinstance(timers.numpy, sc.timer)

    print('Profiling:')

    def slow_fn():
        n = 2000
        int_list = []
        int_dict = {}
        for i in range(n):
            int_list.append(i)
            int_dict[i] = i
        return

    def big_fn():
        n = 1000
        int_list = []
        int_dict = {}
        for i in range(n):
            int_list.append([i]*n)
            int_dict[i] = [i]*n
        return

    class Foo:
        def __init__(self):
            self.a = 0
            return

        def outer(self):
            for i in range(100):
                self.inner()
            return

        def inner(self):
            for i in range(1000):
                self.a += 1
            return

    class Bar(Foo):
        def extra(self):
            return

    foo = Foo()
    trace = sys.gettrace()
    try:
        sc.mprofile(big_fn) # NB, cannot re-profile the same function at the same time
    except TypeError as E: # This happens when re-running this script
        print(f'Unable to re-profile memory function; this is usually not cause for concern ({E})')
    assert sys.gettrace() == trace # Check the trace hook was removed

    # Run profiling test
    p = sc.profile(run=foo.outer, follow=[foo.outer, foo.inner])
    lp = sc.profile(slow_fn)
    lp.plot()
    plt.close('all')
    assert len(p.to_df(bytime=0, maxentries=1)) == 1
    pm = p + lp
    assert len(pm.output) == 3
    pm.disp(skiprun=True)

    # Inherited methods are followed
    assert {'__init__', 'outer', 'inner', 'extra'} <= {f.__name__ for f in sc.listfuncs(Bar)}

    # An exception doesn't leave the profiler running
    def boom():
        raise ValueError('deliberate')
    with pytest.raises(ValueError):
        sc.profile(boom, verbose=False)
    if hasattr(sys, 'monitoring'):
        assert sys.monitoring.get_tool(sys.monitoring.PROFILER_ID) is None

    return lp


def test_cprofile():
    sc.heading('Testing function profiler (cprofile)')

    class Slow:

        def math(self):
            n = 1_000_000
            self.a = np.arange(n)
            self.b = sum(self.a)

        def plain(self):
            n = 100_000
            self.int_list = []
            self.int_dict = {}
            for i in range(n):
                self.int_list.append(i)
                for j in range(10):
                    self.int_dict[i+j] = i+j

        def run(self):
            self.math()
            self.plain()

    # This sometimes fails on GitHub Actions, so raise a warning rather than exception
    try:
        # Option 1: as a context block
        with sc.cprofile() as cpr:
            slow = Slow()
            slow.run()

        # Option 2: with start and stop
        cpr = sc.cprofile()
        cpr.start()
        slow = Slow()
        slow.run()
        cpr.stop()

        # Tests
        df = cpr.df
        assert len(df) >= 4 # Should be at least this many profiled functions
        assert df[0].cumpct > df[-1].cumpct # Should be in descending order

        # Reuse the profiler, and check options
        total = cpr.total
        cpr.start()
        slow.math()
        cpr.stop()
        assert cpr.total > total # Stats are updated, not cached
        assert len(cpr.to_df(maxitems=2)) == 2
        df = cpr.to_df(columns='full')
        assert np.allclose(df.percall*df.calls, df.cumtime) # Same units

    except ValueError as e:
        string = 'Another profiling tool is already active'
        if string in str(e):
            warnmsg = f'Skipping test_cprofile(): {string}'
            warnings.warn(warnmsg, category=RuntimeWarning, stacklevel=2)
            cpr = None
        else: # Don't catch anything else
            raise e

    return cpr


def test_tracecalls():
    sc.heading('Testing tracecalls()')

    class ComplexCalls:
        """ Class with complex call structure """

        def __init__(self):
            self.count = 0
            self.call1()
            self.call2()

        def call1(self):
            self.call3()

        def call2(self):
            self.call3()
            self.call4()
            self.call5()

        def call3(self):
            self.call6()

        def call4(self):
            self.count += 1

        def call5(self):
            self.count += 10

        def call6(self):
            self.count += 100

        def call_extra(self):
            self.count += 1000


    with sc.tracecalls() as tc:
        cc = ComplexCalls()

    with sc.capture() as text:
        tc.disp()

    for i in range(6):
        assert f'call{i+1}' in text, 'Call not captured' # Check that calls are captured
    assert 'call_extra' not in text, 'Unexpected call' # Check that uncalled functions are not
    sc.printgreen('✓ Calls logged as expected')

    # Check expected
    out = tc.check_expected(cc)
    assert 'ComplexCalls.call1'      in out.called, 'Call not caught'
    assert 'ComplexCalls.call_extra' in out.not_called, 'Call missed'
    out = tc.check_expected(['ComplexCalls.call1', 'nope'])
    assert out.called == {'ComplexCalls.call1'} and out.not_called == {'nope'}
    sc.printgreen('✓ Call checking working as expected')

    # Regexes are searched for in the file path and function name
    with sc.tracecalls('test_profiling', exclude=['^call[1-3]$', 'tracecalls'], regex=True) as tc3:
        ComplexCalls()
    names = {e.name for e in tc3.entries}
    assert 'ComplexCalls.__init__' in names and 'ComplexCalls.call4' in names
    assert 'ComplexCalls.call1' not in names

    # Check that no matches works, and doesn't hide exceptions
    with sc.tracecalls('no_such_module') as tc2:
        cc = ComplexCalls()
    assert len(tc2) == 0
    with pytest.raises(ValueError):
        with sc.tracecalls('no_such_module'):
            raise ValueError('deliberate')

    return tc


def test_resourcemonitor():
    sc.heading('Testing resource monitor')

    o = sc.objdict()
    o.callback = []

    def callback(checkdata, checkstr):
        ''' Small function to test that callbacks work '''
        print('Callback works as intended')
        o.callback.append(checkdata)
        return

    with pytest.raises(sc.LimitExceeded):
        with sc.capture() as text:
            with sc.resourcemonitor(mem=0.001, interval=0.1, die=False) as resmon:
                print('Effectively zero memory limit')
                sc.timedsleep(0.3)
        assert 'Limits exceeded' in text # Printed even with die=False
        raise resmon.exception
    o.resmon_died = resmon

    # As a standalone (don't forget to call stop!)
    resmon = sc.resourcemonitor(mem=0.95, cpu=0.99, time=0.1, interval=0.1, start=False, die=False, callback=callback, verbose=True)
    with sc.capture() as text:
        resmon.start(label='Load checker')
        sc.timedsleep(0.2)
        resmon.stop()
    assert 'Load checker step 1' in text
    print(resmon.to_df())

    # Ctrl-C still works while the monitor is running
    resmon2 = sc.resourcemonitor(mem=1.0, interval=10, die=False, verbose=False)
    with pytest.raises(KeyboardInterrupt):
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
    resmon2.stop()

    # Check that a busy main thread is interrupted promptly, and that the parent (here, a dummy process) is killed
    def busy(t0):
        while time.time() - t0 < 10:
            pass
    proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    resmon3 = sc.resourcemonitor(mem=0.001, interval=0.05, start=False, kill_children=False, kill_parent=True)
    resmon3.parent = proc.pid
    t0 = time.time()
    with pytest.raises(sc.LimitExceeded):
        with resmon3:
            busy(t0)
    assert time.time() - t0 < 5
    assert proc.wait(timeout=5) != 0 # Killed

    o.resmon = resmon

    return o


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    lb  = test_loadbalancer()
    mc  = test_memchecks()
    lp  = test_profile()
    cpr = test_cprofile()
    tc  = test_tracecalls()
    rm  = test_resourcemonitor()

    sc.toc()
    print('Done.')