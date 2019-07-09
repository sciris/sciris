'''
Version: 2019jul09
'''

import pylab as pl
import sciris as sc

torun = [
'simple',
'complex',
]

if 'doplot' not in locals(): doplot = True

if 'simple' in torun:
    result = sc.asd(pl.linalg.norm, [1, 2, 3])
    print('Result:')
    print(result.x)
    
if 'complex' in torun:
    
    def objective(pars):
        x, y, z = pars
        err = 50*(y - x**2)**2 + (0.5 - x)**2 + 2*abs(z-0.5); # Rosenbrock's valley
        return err
    
    randseed = 239847
    startvals = [-1, -1, -1]
    minvals = [-1, -1, -1]
    maxvals = [1, 1, 1]
    result = sc.asd(objective, startvals, xmin=minvals, xmax=maxvals, randseed=randseed)
        
    if doplot:
        
        # Plot parameter space
        pl.seed(randseed)
        npts = 15
        perturb = 0.05
        xvec = pl.linspace(-1,1,npts)
        yvec = pl.linspace(-1,1,npts)
        zvec = pl.linspace(-1,1,npts)
        ovec = []
        alldata = []
        for x in xvec:
            for y in yvec:
                for z in zvec:
                    xp = x + perturb*pl.randn()
                    yp = y + perturb*pl.randn() 
                    zp = z + perturb*pl.randn() 
                    o = pl.log10(objective([xp, yp, zp]))
                    alldata.append([xp, yp, zp, o])
        alldata = pl.array(alldata)
        X = alldata[:,0]
        Y = alldata[:,1]
        Z = alldata[:,2]
        O = alldata[:,3]
        fig = pl.figure(figsize=(16,12))
        ax = sc.scatter3d(X, Y, Z, O, fig=fig, plotkwargs={'alpha':0.3})
        ax.view_init(elev=50, azim=-45)
        pl.xlabel('x')
        pl.ylabel('y')
        
        # Plot trajectory
        X2 = result.details.xvals[:,0]
        Y2 = result.details.xvals[:,1]
        Z2 = result.details.xvals[:,2]
        O2 = pl.log10(result.details.fvals)
        ax = sc.scatter3d(X2, Y2, Z2, O2, fig=fig, plotkwargs={'alpha':1.0, 'marker':'d'})
        ax = sc.plot3d(X2, Y2, Z2, fig=fig, plotkwargs={'c':(0,0,0), 'lw':3})


