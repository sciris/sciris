'''
Test Sciris utility/helper functions.
'''

import pytest
import numpy as np
import sciris as sc
import datetime as dt


### Adaptations from other libraries

def test_adaptations():
    sc.heading('Test function adaptations')
    o = sc.objdict()

    print('\nTesting sha')
    o.sha = sc.sha({'a':np.random.rand(5)})

    print('\nTesting cp and dcp')
    o.sha2 = sc.dcp(o.sha.hexdigest())
    with pytest.raises(ValueError):
        o.sha3 = sc.cp(o.sha)
    with pytest.raises(ValueError):
        o.sha3 = sc.dcp(o.sha)

    print('\nTesting wget')
    o.wget = sc.wget('http://sciris.org/')

    print('\nTesting htmlify')
    o.html = sc.htmlify('foo&\nbar') # Returns b'foo&amp;<br>bar'
    o.nothtml = sc.htmlify(o.wget, reverse=True)

    print('\nTesting traceback')
    o.traceback = sc.traceback()

    return o


def test_uuid():
    sc.heading('Test UID generation')
    import uuid

    # Create them
    u = sc.objdict()
    u.u0 = uuid.uuid4()
    u.u1 = sc.uuid()
    u.u2 = sc.uuid()
    u.u3 = sc.uuid(length=4)
    u.u4 = sc.uuid(which='ascii', length=16)
    u.u5 = sc.uuid(n=3)
    u.u6 = sc.uuid(which='hex', length=20)
    u.u7 = sc.uuid(which='numeric', length=10, n=5)

    # Tests
    assert u.u1 != u.u2
    assert isinstance(u.u1, type(u.u0))
    assert isinstance(u.u3, str)
    with pytest.raises(ValueError):
        sc.uuid(length=400) # UUID is only 16 characters long
    with pytest.raises(ValueError):
        sc.uuid(which='numeric', length=2, n=10) # Not enough unique choices

    print('NOTE: This is supposed to print warnings and then raise a (caught) exception\n')
    with pytest.raises(ValueError):
        sc.uuid(which='numeric', length=2, n=99, safety=1, verbose=True) # Not enough unique choices

    # Print results
    print('UIDs:')
    for key,val in u.items():
        print(f'{key}: {val}')


    print('\nTesting fast_uuid')
    u.uuids = sc.fast_uuid(n=100) # Generate 100 UUIDs

    print('\nTesting uuid')
    u.uuid = sc.uuid()

    return u


def test_traceback():
    sc.heading('Test printing traceback text')

    dct = {'a':3}
    try:
        dct['b'] # This will cause a KeyError
    except:
        text = sc.traceback()

    print('NOTE: This is an example traceback, not an actual error!\n')
    print(f'Example traceback text:\n{text}')

    return text


### Printing/notification functions

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
    return


def test_printing():
    sc.heading('Test printing functions')

    example = sc.prettyobj()
    example.data = sc.vectocolor(10)

    print('sc.pr():')
    sc.pr(example)

    print('sc.pp():')
    sc.pp(example.data)
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
    sc.printarr(np.random.rand(3,4,5))

    print('\nTesting printvars')
    a = range(5)
    b = 'example'
    sc.printvars(locals(), ['a','b'], color='green')

    print('\nTesting slacknotification')
    sc.slacknotification(webhook='http://invalid.hooks.slack.com.test', message='Test notification to nowhere')
    print('↑↑↑ Will raise a error since not a valid webhook, this is OK')

    print('\nTesting printtologfile')
    sc.printtologfile('Test message')

    print('\nTesting sigfig')
    sc.sigfig(np.random.rand(10), SI=True, sep='.')

    return o


def test_prepr():
    sc.heading('Test pretty representation of an object')
    n_attrs = 500
    myobj = sc.prettyobj()
    for i in range(n_attrs):
        key = f'attr{i:03d}'
        setattr(myobj, key, i**2)
    print(myobj)

    print('Testing pretty representation of an object using slots')
    class Foo():
        __slots__ = ['bar']
        def __init__(self):
            self.bar = 1

    x = Foo()
    print(sc.prepr(x))
    return myobj


def test_progress_bar():
    sc.heading('Progress bar and percent complete')

    n = 50

    for i in range(n):
        sc.progressbar(i+1, n)
        sc.timedsleep(0.5/n, verbose=False)

    for i in range(n):
        sc.percentcomplete(i, n, stepsize=10) # will print on every 50th iteration

    return i


### Type functions

def test_promotetolist():
    sc.heading('test_promotetolist()')
    ex0 = 1
    ex1 = 'a'
    ex2 = {'a', 'b'}
    ex3 = np.array([0,1,2])
    ex4 = [1,2,3]
    res0 = sc.promotetolist(ex0, int)
    res1 = sc.promotetolist(ex1)
    res2a = sc.promotetolist(ex2)
    res2b = sc.promotetolist(ex2, objtype='str')
    res3a = sc.promotetolist(ex3)
    res3b = sc.promotetolist(ex3, objtype='number')
    with pytest.raises(TypeError):
        sc.promotetolist(ex0, str)
    with pytest.raises(TypeError):
        sc.promotetolist(ex1, int)
    with pytest.raises(TypeError):
        sc.promotetolist(ex3, objtype='str')
    with pytest.raises(TypeError):
        sc.promotetolist(ex4, objtype='str')
    assert res0 == [1]
    assert res1 == ['a']
    assert res2a == [{'a', 'b'}]
    assert sorted(res2b) == ['a', 'b'] # Sets randomize the order...
    assert repr(res3a) == repr([np.array([0,1,2])]) # Direct quality comparison fails due to the array
    assert res3b == [0,1,2]
    print(res1)
    print(res2a)
    print(res2b)
    print(res3a)
    print(res3b)

    # Check that type checking works
    sc.promotetolist(ex2, objtype=str)

    return res3b


def test_types():
    sc.heading('Test type functions')
    o = sc.objdict()

    print('\nTesting isarray')
    assert sc.isarray(np.array([1,2,3]))
    assert not sc.isarray([1,2,3])
    assert not sc.isarray(np.array([1,2,3]), dtype=float)

    print('\nTesting mergedicts')
    o.dicts1 = sc.mergedicts({'a':4, 'b':5}, {'b':8, 'c':8})
    with pytest.raises(TypeError):
        sc.mergedicts({'a':4, 'b':5}, 3, strict=True)
    with pytest.raises(KeyError):
        sc.mergedicts({'a':4, 'b':5}, {'b':8, 'c':8}, overwrite=False)

    print('\nTesting flexstr')
    o.flexstr = sc.flexstr(b'bytestring')

    print('\nTesting promotetoarray')
    assert not len(sc.promotetoarray(None, skipnone=True))
    assert sc.promotetoarray(np.array(3))[0] == 3
    with pytest.raises(TypeError):
        sc.promotetoarray('not convertible')

    return o


### Time/date functions

def test_readdate():
    sc.heading('Test string-to-date conversion')

    string1 = '2020-Mar-21'
    string2 = '2020-03-21'
    string3 = 'Sat Mar 21 23:13:56 2020'
    dateobj1 = sc.readdate(string1)
    dateobj2 = sc.readdate(string2)
    sc.readdate(string3)
    assert dateobj1 == dateobj2
    with pytest.raises(ValueError):
        sc.readdate('Not a date')

    # Automated tests
    formats_to_try = sc.readdate(return_defaults=True)
    for key,fmt in formats_to_try.items():
        datestr = sc.getdate(dateformat=fmt)
        dateobj = sc.readdate(datestr, dateformat=fmt)
        print(f'{key:15s} {fmt:22s}: {dateobj}')

    return dateobj1


def test_dates():
    sc.heading('Testing other date functions')
    o = sc.objdict()

    print('\nTesting date')
    o.date1 = sc.date('2020-04-05') # Returns datetime.date(2020, 4, 5)
    o.date2 = sc.date('2020-04-14', start_date='2020-04-04', as_date=False) # Returns 10
    o.date3 = sc.date([35,36,37], as_date=False) # Returns ['2020-02-05', '2020-02-06', '2020-02-07']

    print('\nTesting day')
    o.day = sc.day('2020-04-04') # Returns 94
    assert o.day == 94

    print('\nTesting daydiff')
    o.diff  = sc.daydiff('2020-03-20', '2020-04-05') # Returns 16
    o.diffs = sc.daydiff('2020-03-20', '2020-04-05', '2020-05-01') # Returns [16, 26]

    print('\nTesting daterange')
    o.dates = sc.daterange('2020-03-01', '2020-04-04')

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

    return o



### Misc. functions

def test_profile():
    sc.heading('Test profiling functions')

    def slow_fn():
        n = 10000
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

    foo = Foo()
    try:
        sc.mprofile(big_fn) # NB, cannot re-profile the same function at the same time
    except TypeError as E: # This happens when re-running this script
        print(f'Unable to re-profile memory function; this is usually not cause for concern ({E})')
    sc.profile(run=foo.outer, follow=[foo.outer, foo.inner])
    lp = sc.profile(slow_fn)

    return lp


def test_suggest():
    sc.heading('test_suggest()')
    string = 'foo'
    ex1 = ['Foo','Bar']
    ex2 = ['FOO','Foo']
    ex3 = ['Foo','boo']
    ex4 = ['asldfkj', 'aosidufasodiu']
    ex5 = ['foo', 'fou', 'fol', 'fal', 'fil']
    res1 = sc.suggest(string, ex1)
    res2 = sc.suggest(string, ex2)
    res3 = sc.suggest(string, ex3)
    res4 = sc.suggest(string, ex4, threshold=4)
    with pytest.raises(Exception):
        sc.suggest(string, ex1, threshold=4, die=True)
    res5a = sc.suggest(string, ex5, n=3)
    res5b = sc.suggest(string, ex5, fulloutput=True)
    assert res1 == 'Foo'
    assert res2 == 'Foo'
    assert res3 == 'Foo'
    assert res4 == None
    assert res5a == ['foo', 'fou', 'fol']
    assert res5b == {'foo': 0.0, 'fou': 1.0, 'fol': 1.0, 'fal': 2.0, 'fil': 2.0}
    print(res1)
    print(res2)
    print(res3)
    print(res4)
    print(res5a)
    print(res5b)
    return res5b


def test_misc():
    sc.heading('Testing miscellaneous functions')
    o = sc.objdict()

    print('\nTesting checkmem')
    sc.checkmem(['spiffy',np.random.rand(243,589)], descend=True)

    print('\nTesting checkram')
    o.ram = sc.checkram()
    print(o.ram)

    print('\nTesting runcommand')
    sc.runcommand('command_probably_not_found')

    print('\nTesting gitinfo functions')
    o.gitinfo = sc.gitinfo()

    print('\nTesting compareversions')
    assert sc.compareversions('1.2.3', '2.3.4') == -1
    assert sc.compareversions(2, '2') == 0
    assert sc.compareversions('3.1', '2.99') == 1

    print('\nTesting uniquename')
    namelist = ['file', 'file (1)', 'file (2)']
    o.unique = sc.uniquename(name='file', namelist=namelist)
    assert o.unique not in namelist

    print('\nTesting importbyname')
    sc.importbyname('numpy')

    print('\nTesting get_caller()')
    o.caller = sc.getcaller()
    print(o.caller)

    print('\nTesting nestedloop')
    o.nested = list(sc.nestedloop([['a','b'],[1,2]],[0,1]))

    return o



### Nested dictionary functions

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


### Classes


def test_links():
    sc.heading('Testing links')
    o = sc.objdict()

    with pytest.raises(KeyError):
        raise sc.KeyNotFoundError('Example')

    obj = sc.objdict()
    obj.uid = sc.uuid()
    obj.data = np.random.rand(5)
    o.obj = obj
    o.link = sc.Link(obj)
    o.o_copy = sc.dcp(o)

    assert np.all(o.link()['data'] == o.obj['data'])

    with pytest.raises(sc.LinkException):
        o.o_copy.link()

    return o



#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    adapt     = test_adaptations()
    uid       = test_uuid()
    traceback = test_traceback()

    bluearray = test_colorize()
    printing  = test_printing()
    myobj     = test_prepr()
    ind       = test_progress_bar()

    plist     = test_promotetolist()
    types     = test_types()

    dateobj   = test_readdate()
    dates     = test_dates()

    lp        = test_profile()
    dists     = test_suggest()
    misc      = test_misc()

    nested    = test_nested()
    dicts     = test_dicts()

    links   = test_links()

    sc.toc()
    print('Done.')