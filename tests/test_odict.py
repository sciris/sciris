"""
Define a suite of tests for the odict
"""

import sys
import numpy as np
import sciris as sc
import pytest


def printexamples(examples):
    examples = sc.promotetolist(examples)
    for v,val in enumerate(examples):
        print('Example %s:' % (v+1))
        print(val)
        print('')
    return


def test_main():
    sc.heading('Main tests:')
    foo = sc.odict({'ah':3,'boo':4, 'cough':6, 'dill': 8}) # Create odict
    bar = foo.sorted() # Sort the list
    assert(bar['boo'] == 4) # Show get item by value
    assert(sc.objdict(bar).boo == 4) # Show get item by value, method 2
    assert(bar[1] == 4) # Show get item by index
    assert((bar[0:2] == [3,4]).all()) # Show get item by slice
    assert((bar['cough':'dill'] == [6,8]).all()) # Show alternate slice notation
    assert((bar['ah':'dill':2] == [3,6]).all()) # Show slice with step
    assert((bar[np.array([2,1])] == [6,4]).all()) # Show get item by list
    assert((bar[:] == [3,4,6,8]).all()) # Show slice with everything
    assert((bar[2:] == [6,8]).all()) # Show slice without end
    bar[3] = [3,4,5] # Show assignment by item
    bar[0:2] = ['the', 'power'] # Show assignment by slice -- NOTE, inclusive slice!!
    bar[[0,2]] = ['cat', 'trip'] # Show assignment by list
    bar.rename('cough','chill') # Show rename
    bar.rename(2,'mill') # Show rename numerically
    printexamples([foo,bar])
    return


def test_insert():
    sc.heading('Insert:')
    z = sc.odict()
    z['foo'] = 11
    z.insert(1123)
    z.insert('bar', 112358)
    z.insert(0, 'cat', 11235813)
    z.insert(2, 'dog', 1123581321)
    assert len(z) == 5
    z.insert(0, 'dog', 5)
    assert z.keys()[0] == 'dog' and z['dog'] == 5 and len(z) == 5 # Existing key is moved and overwritten
    z.insert(1, 'none', None)
    assert z[1] is None # None is a valid value
    printexamples(z)
    return


def test_add():
    sc.heading('Add:')
    dict1 = sc.odict(a=3, b=4)
    dict2 = sc.odict(c=5, d=7)
    dict3 = dict1 + dict2
    dict4 = sum([dict1, dict2])
    assert dict3.keys() == dict1.keys() + dict2.keys()
    assert dict3.values() == dict1.values() + dict2.values()
    assert dict3 == dict4
    printexamples(dict3)
    return


def test_make():
    sc.heading('Make:')
    a = sc.odict().make(5) # Make an odict of length 5, populated with empty lists and default key names
    b = sc.odict().make('foo',34) # Make an odict with a single key 'foo' of value 34
    c = sc.odict().make(['a','b']) # Make an odict with keys 'a' and 'b'
    d = sc.odict().make(['a','b'], 0) # Make an odict with keys 'a' and 'b', initialized to 0
    e = sc.odict().make(keys=['a','b'], vals=[1,2]) # Make an odict with 'a':1 and 'b':2
    f = sc.odict().make(keys=['a','b'], vals=np.array([1,2])) # As above, since arrays are coerced into lists
    g = sc.odict({'a':34, 'b':58}).make(['c','d'],[99,45]) # Add extra keys to an exising odict
    h = sc.odict().make(keys=['a','b','c'], keys2=['A','B','C'], keys3=['x','y','z'], vals=7) # Make a triply nested odict
    i = sc.odict().make(vals=['x', 'y', 'z'])
    printexamples([a,b,c,d,e,f,g,h,i])
    assert h['a']['A']['x'] == 7

    sc.heading('Make from:')
    a = 'cat'; b = 'dog'; o = sc.odict().makefrom(source=locals(), keys=['a','b']) # Make use of fact that variables are stored in a dictionary
    d = {'a':'cat', 'b':'dog'}; p = sc.odict().makefrom(d) # Same as sc.odict(d)
    l = ['cat', 'monkey', 'dog']; q = sc.odict().makefrom(source=l, keys=[0,2], keynames=['a','b'])
    printexamples([o,p,q])
    return


def test_map():
    sc.heading('Map:')
    cat = sc.odict({'a':[1,2], 'b':[3,4]})
    def myfunc(mylist): return [i**2 for i in mylist]
    dog = cat.map(myfunc) # Returns sc.odict({'a':[1,4], 'b':[9,16]})
    printexamples([cat, dog])
    return


def test_each():
    sc.heading('From each:')
    z = sc.odict({'a':np.array([1,2,3,4]), 'b':np.array([5,6,7,8])})
    f = z.fromeach(2, asdict=False) # Returns array([3,7])
    g = z.fromeach(ind=[1,3], asdict=True) # Returns sc.odict({'a':array([2,4]), 'b':array([6,8])})
    printexamples([f,g])
    assert sc.odict(a=[1,2,3,4]).fromeach([1,3])['a'] == [2,4] # Lists can be indexed by a list

    sc.heading('To each:')
    z = sc.odict({'a':[1,2,3,4], 'b':[5,6,7,8]})
    z.toeach(2, [10,20])    # z is now sc.odict({'a':[1,2,10,4], 'b':[5,6,20,8]})
    z.toeach(ind=3,val=666) #  z is now sc.odict({'a':[1,2,10,666], 'b':[5,6,20,666]})
    printexamples(z)
    return


def test_find():
    sc.heading('Findkeys, findbykey, and findvals:')
    yy = sc.odict({'foo':[1,2,3,4], 'bar':[5,6,7,8], ('cat','dog'):[5,6,7,8]})
    print(yy.findkeys()) # Equivalent to yy.keys()
    print(yy.findkeys('oo'))
    print(yy.findkeys('^oo'))
    print(yy.findkeys('oo', method='endswith'))
    print(yy.findkeys('cat'))
    print(yy.findbykey('ar'))
    print(yy.findbyval([5,6,7,8]))
    print(yy.findbyval([5,6,7,8], first=False))
    a = yy.filter('a')
    b = yy.filter('a', exclude=True)
    c = yy.filter(['foo', 'bar'])
    d = yy.filtervals([1,2,3,4])
    printexamples([a,b,c,d])
    assert sc.odict(a=np.array([1,2]), b=5).findbyval(5) == 'b' # Array values don't raise
    return


def test_repr():
    sc.heading('Testing repr')
    n_entries = 300
    qq = sc.odict()
    print(qq)
    for i in range(n_entries):
        key = f'key{i:03d}'
        qq[key] = i**2
    qq.disp(maxitems=15)
    qq[:] = 4
    assert all([v==4 for v in qq.values()])

    # Test obj repr: short on one line, long with one key per line
    assert repr(sc.obj(a=3, b=sc.obj(c=5))) == 'sc.obj(a=3, b=sc.obj(c=5))'
    assert repr(sc.obj({'my key':1})) == "sc.obj({'my key': 1})"
    long = sc.obj(x=np.arange(20).reshape(2,10), y=sc.obj(z='a string long enough to go over the line width'))
    print(long)
    assert repr(long).splitlines()[0] == 'sc.obj('
    selfref = sc.obj(a=1)
    selfref.b = selfref
    assert repr(selfref) == 'sc.obj(a=1, b=sc.obj(...))'
    return


def test_default():
    sc.heading('Testing default')

    o = sc.odict(foo=[1,2,3], defaultdict=list)
    p = sc.objdict(foo=[1,2,3], defaultdict=list)
    o['bar'].extend([1,2,3])
    p.bar.extend([1,2,3])
    assert o[0] == o[1] == p.foo == p.bar

    lam = lambda: 44
    o = sc.odict(defaultdict=lam)
    assert o['anything'] == lam()

    # Testing nesting
    j = sc.objdict(a=1, b=2, defaultdict='nested')
    j.c.d.e = 12345
    k = sc.odict(a=1, b=2, defaultdict='nested')
    k['c']['d']['e'] = 12345
    assert j['c']['d']['e'] == k['c']['d']['e']
    print(j)

    return


def test_other():
    sc.heading('Testing other')
    o = sc.odict(foo=[1,2,3,4], bar=[5,6,7,8])

    print('Testing enumerate')
    for i,k in o.enumkeys():
        print(i, k)
    for i,v in o.enumvals():
        print(i, v)
    for i,k,v in o.enumitems():
        print(i, k,v)

    print('Testing display')
    o.disp()

    print('Testing export')
    o.export()

    print('Testing pop')
    o1 = sc.dcp(o)
    o2 = sc.dcp(o)
    o3 = sc.dcp(o)
    o4 = sc.dcp(o)
    o1.pop('foo')
    o2.pop(0)
    o3.pop([0,1])
    o4.pop(slice(-1))
    assert o2[0] == [5,6,7,8] # Indexing is correct after popping by index

    print('Testing copy')
    o.copy()
    o.copy(deep=True)

    print('Testing append')
    o.append(239)
    o.valind(239)

    print('Testing sort')
    a = 12
    b = 8
    c = 244
    v = sc.odict(a=12, b=8, c=244)
    assert v[:].tolist() == [a,b,c]
    v.sort(sortby='values', reverse=True)
    assert v[:].tolist() == [c,a,b]
    v.sort(sortby=[1,0,2])
    assert v[:].tolist() == [a,c,b]

    print('Testing reverse')
    w = v.reversed()
    v.reverse()
    assert v[:].tolist() == w[:].tolist() == [b,c,a]
    v.sort(sortby=['a','b'])
    assert v.keys() == ['a','b'] # Unlisted keys are dropped

    print('Testing promote')
    od  = sc.odict.promote(['There','are',4,'keys'])
    sc.odict.promote(od)
    sc.odict.promote(dict(od))

    print('Testing clear')
    od.clear()

    print('Testing inherited dict methods')
    o5 = sc.odict(a=1)
    o5[0]
    o5.setdefault('b', 2)
    assert o5[-1] == 2 # Indexing is correct after setdefault()

    return o


class Locked(sc.obj): # At the top level so it can be pickled
    __slots__ = ('locked',)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.locked = False
    def __setitem__(self, key, value):
        if self.locked:
            raise KeyError(key)
        super().__setitem__(key, value)
class ChildLocked(Locked):
    pass


def test_asobj():
    sc.heading('Testing objdict/asobj')

    # Testing standard usage
    d = dict(foo=1, bar=2)
    d_obj = sc.asobj(d)
    d_obj.foo = 10
    assert d_obj.bar == 2

    print('Testing derived classes')
    class MyObjDict(sc.objdict):
        def __init__(self, x):
            self.x = x
        def square(self):
            return self.x**2
        def __repr__(self):
            output = sc.objrepr(self)
            output += sc.objdict.__repr__(self)
            return output

    class MyObjObj(type(d_obj)):
        def __init__(self, x):
            self.x = x
        def square(self):
            return self.x**2

    myobjdict = MyObjDict(3)
    myobjobj = MyObjObj(4)
    assert myobjdict.x == 3
    assert myobjobj.x == 4
    assert myobjdict.square() == 9
    assert myobjobj.square() == 16
    print(myobjdict)
    print(myobjobj)
    repr(myobjdict)
    del myobjdict.x
    myobjdict.setattribute('x', 10)
    assert myobjdict.x == 10
    assert getattr(myobjdict, 'x') == myobjdict.getattribute('x') ==  10
    myobjdict.setattribute('__test__', 'truly an attribute')
    myobjdict.delattribute('__test__')
    myobjobj.setattribute('my_attr', 5)
    assert myobjobj.getattribute('my_attr') == 5

    with pytest.raises(KeyError):
        myobjdict.x = 'cannot change actual attribute'
    with pytest.raises(AttributeError):
        myobjdict.setattribute('keys', 4)

    assert 'x' in dir(sc.objdict(x=1)) # For tab completion
    with pytest.warns(UserWarning):
        sc.objdict(index=3) # Key can't be accessed as an attribute

    print('Testing obj')
    o = sc.obj(a=1, b=2)
    o.c = 3
    o['d'] = 4
    assert isinstance(o, dict) and isinstance(o, sc.odict)
    assert o.d == o['d'] == o[3] == o[-1] == 4
    assert vars(o) is o
    assert 'a' in dir(o) and 'sort' in dir(o)
    assert o.keys() == ['a', 'b', 'c', 'd']
    assert repr(o) == 'sc.obj(a=1, b=2, c=3, d=4)'
    del o.d
    del o['c']
    o.e = 5
    assert o[2] == 5 # Added and deleted keys are seen by position
    assert not hasattr(o, 'nope')
    with pytest.raises(sc.KeyNotFoundError):
        o['nope']
    with pytest.raises(IndexError):
        o[3]
    with pytest.raises(TypeError):
        sc.obj(defaultdict=list)
    assert isinstance(o | {'f':6}, sc.obj) and isinstance({'f':6} | o, sc.obj)
    assert 'a' in dir(sc.obj({1:2, 'a':3})) # Non-string keys are skipped
    if sys.version_info >= (3, 12):
        assert list(o[:2]) == [1, 2]
    s = o.sorted(reverse=True)
    assert isinstance(s, sc.obj) and s.keys() == ['e', 'b', 'a']
    for copied in [sc.dcp(o), sc.cp(o), o.copy(), sc.loadstr(sc.dumpstr(o))]:
        assert type(copied) is sc.obj and copied == o and copied is not o
        copied.a = 100 # Attributes and items are still the same thing
        assert copied['a'] == 100 and o.a == 1
    assert sc.jsonify(o) == dict(o)
    assert repr(sc.objdict(a=sc.obj(b=1))) == 'sc.objdict(a=sc.obj(b=1))'
    p = sc.obj(clear=1, _repr=2, fromkeys=3) # Keys can't replace methods
    p.a = 4
    assert p['clear'] == 1 and p.keys() == ['clear', '_repr', 'fromkeys', 'a']
    assert repr(p) == 'sc.obj(clear=1, _repr=2, fromkeys=3, a=4)' and dict(p)['clear'] == 1
    assert sc.obj.fromkeys(['x']).keys() == ['x'] and sc.dcp(p) == p
    p.clear()
    assert not len(p)
    with pytest.raises(AttributeError):
        p.keys = 3
    with pytest.raises(AttributeError):
        p.fromkeys = 3

    class MyObj(sc.obj):
        def double(self):
            return self.a*2
        def keys(self):
            return super().keys() + ['extra']
    m = MyObj(a=2, double=5)
    m.b = 3
    assert m.double() == 4 and m['double'] == 5 and m[2] == 3
    assert m.keys() == ['a', 'double', 'b', 'extra']
    assert type(sc.dcp(m)) is MyObj
    with pytest.raises(AttributeError):
        m.double = 3

    for cls in [Locked, ChildLocked]:
        l = cls(a=1)
        l.b = 2
        assert l.locked is False and l.keys() == ['a', 'b']
        l.setattribute('locked', True)
        for func in [lambda: l.__setitem__('c', 3), lambda: setattr(l, 'c', 3)]:
            with pytest.raises(KeyError):
                func()
        for copied in [sc.dcp(l), sc.loadstr(sc.dumpstr(l))]:
            assert type(copied) is cls and copied.locked is True and copied == l
    with pytest.raises(AttributeError):
        sc.obj().setattribute('x', 1) # No slot for it

    print('Testing dictobj')
    obj = sc.dictobj(x=4, y=6)
    obj.a = 5
    obj['b'] = 10
    print(obj)
    fk = sc.dictobj.fromkeys(['foo', 'bar'])
    json = obj.to_json()
    assert 'x' in json
    assert fk.foo is None
    assert sc.dictobj(x=1) != sc.dictobj(x=2) and sc.dictobj(x=1) == {'x':1} # Compares contents
    assert sc.loadjson(string=json) == obj == sc.dcp(obj) == sc.loadstr(sc.dumpstr(obj)) # Keys and attributes are the same thing
    assert isinstance(obj | {'c':1}, sc.dictobj) and isinstance(obj.copy(), sc.dictobj)
    old = sc.dictobj.__new__(sc.dictobj)
    old.__setstate__({'x':4}) # Pickles from before 3.5.0 stored the data as attributes
    assert old.x == old['x'] == 4

    return


def test_aliases():
    sc.heading('Testing aliases')

    o = sc.odict(foo=[1,2,3], bar=[4,5,6])
    d = dict(o)

    print('Testing dict_keys')
    assert o.dict_keys() == d.keys()

    print('Testing dict_values')
    odv = o.dict_values() # Not sure why direct comparison doesn't work
    dv = d.values()
    assert type(odv) == type(dv)
    for iodv,idv in zip(odv, dv):
        assert iodv == idv

    print('Testing dict_items')
    assert o.dict_items() == d.items()

    print('Testing iteritems')
    assert o.iteritems() == o.items()

    print('Testing makenested')
    val = 7
    n = sc.odict()
    n.makenested(['a','b'], value=val)

    print('Testing getnested')
    assert n.getnested(['a','b']) == val

    print('Testing setnested')
    newval = 14
    n.setnested(['a','b'], newval)
    assert n.getnested(['a','b']) == newval

    print('Testing iternested')
    count = 0
    for twig in n.iternested():
        count += 1
    assert count == 1

    return


def test_argparse():
    sc.heading('Testing argparse')

    def parse(cmd, **kwargs):
        orig = sys.argv
        sys.argv = ['script.py'] + cmd.split()
        try:
            return sc.argparse(**kwargs)
        finally:
            sys.argv = orig

    defaults = dict(full=False, provenance=False, verbose=False)
    assert parse('--verbose', **defaults) == dict(full=False, provenance=False, verbose=True) # Flags are matched by name, not position
    assert parse('--full --verbose', **defaults) == dict(full=True, provenance=False, verbose=True)
    assert parse('true verbose=yes', **defaults) == dict(full=True, provenance=False, verbose=True)
    assert parse('--iterations 100 --output-file data.csv', iterations=10, output_file='') == dict(iterations=100, output_file='data.csv')
    with pytest.raises(sc.KeyNotFoundError):
        parse('--bogus', **defaults)
    with pytest.raises(ValueError):
        parse('--iterations', iterations=10) # Missing value
    with pytest.raises(ValueError):
        parse('1 2', iterations=10) # Too many positional arguments
    return


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    test_main()
    test_insert()
    test_add()
    test_make()
    test_map()
    test_each()
    test_find()
    test_repr()
    test_default()
    test_other()
    test_asobj()
    test_aliases()
    test_argparse()

    sc.toc()
    print('Done.')
