# Imports
import numpy as np
import time
import multiprocessing as mp
import pickle
import gzip
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D # analysis:ignore

# Set parameters and define random wave generator
xmin = 0
xmax = 10
npts = 200
std = 0.1
repeats = 10
noisevals = np.linspace(0,1,21)
x = np.linspace(xmin, xmax, npts)

def randgen(std):
    a = np.cos(x)
    b = np.random.randn(npts)*std
    return a+b

# Start timing
start = time.time()

# Create object in parallel
multipool = mp.Pool(processes=mp.cpu_count())
output = multipool.map(randgen, noisevals)
multipool.close()
multipool.join()

# Save to files
filenames = []
for n,noiseval in enumerate(noisevals):
    filename = 'noise%0.1f.obj' % noiseval
    with gzip.GzipFile(filename, 'wb') as fileobj:
        fileobj.write(pickle.dumps(output[n]))
    filenames.append(filename)

# Create odict from files
data = {}
for filename in filenames:
    with gzip.GzipFile(filename) as fileobj:
        filestring = fileobj.read()
        data[filename] = pickle.loads(filestring)

# Create 3D plot
data_array = np.array([data[filename] for filename in filenames])
fig = pl.figure()
ax = fig.gca(projection='3d')
ax.view_init(elev=45, azim=30)
ny,nx = np.array(data_array).shape
x = np.arange(nx)
y = np.arange(ny)
X, Y = np.meshgrid(x, y)
settings = {'rstride':1, 'cstride':1, 'linewidth':0, 'antialiased':False, 'cmap':'viridis'}
surf = ax.plot_surface(X, Y, data_array, **settings)
fig.colorbar(surf)

# Print elapsed time
elapsed = time.time() - start
print('Elapsed time: %0.1f s' % elapsed)