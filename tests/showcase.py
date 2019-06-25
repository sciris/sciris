# Imports
import numpy as np
import sciris as sc

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
sc.tic() 

# Create object in parallel
output = sc.parallelize(randgen, noisevals)

# Save to files
filenames = []
for n,noiseval in enumerate(noisevals):
    filename = 'noise%0.1f.obj' % noiseval
    sc.saveobj(filename, output[n])
    filenames.append(filename)

# Create odict from files
data = sc.odict()
for filename in filenames:
    data[filename] = sc.loadobj(filename)

# Create 3D plot
sc.surf3d(data[:])

# Print elapsed time
sc.toc()


