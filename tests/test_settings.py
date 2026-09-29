'''
Test Sciris settings/options.
'''

import os
import re
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
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


def test_options_state():
    sc.heading('Test that options are applied and restored correctly')

    # Nested contexts restore the previous values
    with sc.options.context(dpi=111):
        with sc.options.context(dpi=222):
            assert plt.rcParams['figure.dpi'] == 222
        assert plt.rcParams['figure.dpi'] == 111
    assert not sc.options.changed('dpi')

    # Invalid values are not stored
    with pytest.raises(ValueError):
        sc.options(dpi='invalid')
    assert not sc.options.changed('dpi')

    # Resetting restores all rcParams changed by a style
    before = dict(plt.rcParams)
    sc.options(style='fivethirtyeight')
    sc.options.reset()
    assert repr(dict(plt.rcParams)) == repr(before)

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

    # Matplotlib options set by environment variables are applied on import (in a separate process to avoid changing this one)
    env = dict(os.environ, MPLBACKEND='svg', SCIRIS_BACKEND='agg') # Set a different Matplotlib default, since 'agg' may already be the default
    code = 'import sciris as sc, matplotlib.pyplot as plt; print(plt.get_backend())'
    out = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, check=True).stdout
    assert out.strip() == 'agg'

    return


def test_help():
    sc.heading('Testing help')

    sc.help()
    sc.help('smooth')
    sc.help('JSON', ignorecase=False, context=True)
    with sc.capture() as text:
        sc.help('pickle', source=True, context=True)

    assert text.count('pickle') > 10

    with sc.capture() as text:
        sc.help('^json', flags=re.M) # Combined with re.I
        sc.help('not a sciris docstring phrase')
    assert 'savejson' in text and 'No matches' in text

    return



#%% Run as a script
if __name__ == '__main__':
    T = sc.timer()

    test_options()
    test_options_state()
    test_parse_env()
    test_help()

    T.toc('Done.')