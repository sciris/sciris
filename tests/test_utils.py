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
    u0 = uuid.uuid4()
    u1 = sc.uuid()
    u2 = sc.uuid()
    u3 = sc.uuid(length=4)
    assert u1 != u2
    assert isinstance(u1, type(u0))
    assert isinstance(u3, str)
    with pytest.raises(ValueError):
        sc.uuid(length=400)
    print(f'UIDs:\n{u0}\n{u1}\n{u2}\n{u3}')
    return u3


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

    return thisdir


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

    sc.toc()