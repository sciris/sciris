'''
Test Sciris printing functions.
'''

import numpy as np
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
    class Foo:
        __slots__ = ['bar']
        def __init__(self):
            self.bar = 1

    x = Foo()
    print(sc.prepr(x))
    print(sc.prepr(x, maxtime=0))

    class Bar:
        def skip(self): pass

    print(sc.prepr(Bar()))

    for tf in [True, False]:
        sc.objrepr(x, showid=tf, showmeth=tf, showprop=tf, showatt=tf)

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


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    bluearray = test_colorize()
    printing  = test_printing()
    myobj     = test_prepr()
    ind       = test_progress_bar()

    sc.toc()
    print('Done.')