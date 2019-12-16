import sciris as sc

torun = [
'colorize',
'printing',
'profile',
]

# Test colorize
if 'colorize' in torun:
    sc.colorize(showhelp=True)
    sc.colorize('green', 'hi') # Simple example
    sc.colorize(['yellow', 'bgblack']); print('Hello world'); print('Goodbye world'); sc.colorize('reset') # Colorize all output in between
    bluearray = sc.colorize(color='blue', string=str(range(5)), output=True); print("c'est bleu: " + bluearray)
    sc.colorize('magenta') # Now type in magenta for a while
    print('this is magenta')
    sc.colorize('reset') # Stop typing in magenta

# Test printing functions
if 'printing' in torun:
    example = sc.prettyobj()
    example.data = sc.vectocolor(10)
    print('sc.pr():')
    sc.pr(example)
    print('sc.pp():')
    sc.pp(example.data)
    string = sc.pp(example.data, doprint=False)
    
    
# Test profiling functions
if 'profile' in torun:
    
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