'''
Test Sciris utility/helper functions.
'''

import sciris as sc
import pytest


def test_colorize():
    sc.heading('Test text colorization')
    sc.colorize(showhelp=True)
    sc.colorize('green', 'hi') # Simple example
    sc.colorize(['yellow', 'bgblack']); print('Hello world'); print('Goodbye world'); sc.colorize('reset') # Colorize all output in between
    bluearray = sc.colorize(color='blue', string=str(range(5)), output=True); print("c'est bleu: " + bluearray)
    sc.colorize('magenta') # Now type in magenta for a while
    print('this is magenta')
    sc.colorize('reset') # Stop typing in magenta


def test_printing():
    sc.heading('Test printing functions')
    example = sc.prettyobj()
    example.data = sc.vectocolor(10)
    print('sc.pr():')
    sc.pr(example)
    print('sc.pp():')
    sc.pp(example.data)
    string = sc.pp(example.data, doprint=False)
    return string


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
    sc.profile(run=foo.outer, follow=[foo.outer, foo.inner])
    sc.profile(slow_fn)
    return foo


def test_prepr():
    sc.heading('Test pretty representation of an object')
    n_attrs = 500
    myobj = sc.prettyobj()
    for i in range(n_attrs):
        key = f'attr{i:03d}'
        setattr(myobj, key, i**2)
    print(myobj)
    return myobj


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

    # Print results
    print(f'UIDs:')
    for key,val in u.items():
        print(f'{key}: {val}')

    return u


def test_thisdir():
    sc.heading('Test getting the current file directory')
    import os

    thisdir = sc.thisdir(__file__)
    assert os.path.split(thisdir)[-1] == 'tests'
    print(f'Current folder: {thisdir}')

    return thisdir


def test_traceback():
    sc.heading('Test printing traceback text')

    dct = {'a':3}
    try:
        dct['b'] # This will cause a KeyError
    except:
        text = sc.traceback()

    print(f'Example traceback text:\n{text}')
    print('NB: this is an example, not an actual error!')

    return text


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


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    test_colorize()
    string = test_printing()
    foo = test_profile()
    myobj = test_prepr()
    uid = test_uuid()
    thisdir = test_thisdir()
    traceback = test_traceback()
    dateobj = test_readdate()

    sc.toc()