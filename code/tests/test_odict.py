"""
Define a suite of tests for the odict
"""

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
    assert((bar[np.array([2,1])] == [6,4]).all()) # Show get item by list
    assert((bar[:] == [3,4,6,8]).all()) # Show slice with everything
    assert((bar[2:] == [6,8]).all()) # Show slice without end
    bar[3] = [3,4,5] # Show assignment by item
    bar[0:2] = ['the', 'power'] # Show assignment by slice -- NOTE, inclusive slice!!
    bar[[0,2]] = ['cat', 'trip'] # Show assignment by list
    bar.rename('cough','chill') # Show rename
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
    printexamples(z)
    return


def test_add():
    sc.heading('Add:')
    dict1 = sc.odict(a=3, b=4)
    dict2 = sc.odict(c=5, d=7)
    dict3 = dict1 + dict2
    assert dict3.keys() == dict1.keys() + dict2.keys()
    assert dict3.values() == dict1.values() + dict2.values()
    printexamples(dict3)
    return


def test_make():
    sc.heading('Make:')
    a = sc.odict().make(5) # Make an odict of length 5, populated with Nones and default key names
    b = sc.odict().make('foo',34) # Make an odict with a single key 'foo' of value 34
    c = sc.odict().make(['a','b']) # Make an odict with keys 'a' and 'b'
    d = sc.odict().make(['a','b'], 0) # Make an odict with keys 'a' and 'b', initialized to 0
    e = sc.odict().make(keys=['a','b'], vals=[1,2]) # Make an odict with 'a':1 and 'b':2
    f = sc.odict().make(keys=['a','b'], vals=np.array([1,2])) # As above, since arrays are coerced into lists
    g = sc.odict({'a':34, 'b':58}).make(['c','d'],[99,45]) # Add extra keys to an exising odict
    h = sc.odict().make(keys=['a','b','c'], keys2=['A','B','C'], keys3=['x','y','z'], vals=0) # Make a triply nested odict
    printexamples([a,b,c,d,e,f,g,h])

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
    f = z.fromeach(2) # Returns array([3,7])
    g = z.fromeach(ind=[1,3], asdict=True) # Returns sc.odict({'a':array([2,4]), 'b':array([6,8])})
    printexamples([f,g])

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
    a = yy.filter('a')
    b = yy.filter('a', exclude=True)
    c = yy.filter(['foo', 'bar'])
    d = yy.filtervals([1,2,3,4])
    printexamples([a,b,c,d])
    return


def test_repr():
    sc.heading('Testing repr')
    n_entries = 300
    qq = sc.odict()
    for i in range(n_entries):
        key = f'key{i:03d}'
        qq[key] = i**2
    qq.disp(maxitems=15)
    qq[:] = 4
    assert all([v==4 for v in qq.values()])
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
    for i,j,k in o.enumitems():
        print(i, j, k)

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

    print('Testing copy')
    o.copy('bar', 'cat')

    print('Testing append')
    o.append(239)
    o.valind(239)

    print('Testing sort')
    v = sc.odict(dog=12, cat=8, hamster=244)
    v.sort(sortby='values')

    print('Testing reverse')
    o.reverse()
    o.reversed()

    print('Testing promote')
    od = sc.odict.promote(['There','are',4,'keys'])

    print('Testing clear')
    od.clear()

    return o


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

    with pytest.raises(ValueError):
        myobjdict.x = 'cannot change actual attribute'
    with pytest.raises(AttributeError):
        myobjdict.setattribute('keys', 4)

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

    sc.toc()
    print('Done.')
