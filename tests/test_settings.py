'''
Test Sciris settings/options.
'''

import os
import numpy as np
import sciris as sc
import pytest


#%% Test options

def test_options():
    sc.heading('Test options')

    print('Basic functions')
    sc.options.help()
    sc.options.help(detailed=True)
    sc.options.disp()
    print(sc.options)
    sc.options(dpi=150)
    sc.options('default')
    with sc.options.context(aspath=True):
        pass
    for style in ['default', 'simple', 'fancy', 'fivethirtyeight']:
        with sc.options.with_style(style):
            pass
    with sc.options.with_style({'xtick.alignment':'left'}):
        pass
    with pytest.raises(KeyError):
        with sc.options.with_style(invalid_key=100):
            pass

    print('Save/load')
    fn = 'options.json'
    sc.options.save(fn)
    sc.options.load(fn)
    sc.rmpath(fn)

    print('Printing NumPy types')
    np_print = lambda: print([np.float64(3)])
    with sc.capture() as txt:
        np_print()
    assert 'float64' not in txt, 'NumPy types not successfully disabled'

    sc.options(showtype=True)
    with sc.capture() as txt:
        np_print()
    assert 'float64' in txt, 'NumPy types not successfully re-enabled'

    # Return to default
    sc.options(showtype='default')

    return


def test_parse_env():
    sc.heading('Testing sc.parse_env()')
    mapping = [
        sc.objdict(to='str',   key='TMP_STR',   val='test',  expected='test', nullexpected=''),
        sc.objdict(to='int',   key='TMP_INT',   val='4',     expected=4,      nullexpected=0),
        sc.objdict(to='float', key='TMP_FLOAT', val='2.3',   expected=2.3,    nullexpected=0.0),
        sc.objdict(to='bool',  key='TMP_BOOL',  val='False', expected=False,  nullexpected=False),
    ]
    for e in mapping:
        os.environ[e.key] = e.val
        assert sc.parse_env(e.key, which=e.to) == e.expected
        del os.environ[e.key]
        assert sc.parse_env(e.key, which=e.to) == e.nullexpected

    return


def test_help():
    sc.heading('Testing help')

    sc.help()
    sc.help('smooth')
    sc.help('JSON', ignorecase=False, context=True)
    with sc.capture() as text:
        sc.help('pickle', source=True, context=True)

    assert text.count('pickle') > 10

    return



#%% Run as a script
if __name__ == '__main__':
    T = sc.timer()

    test_options()
    test_parse_env()
    test_help()

    T.toc('Done.')