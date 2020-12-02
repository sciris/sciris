# Imports
import numpy as np
import sciris as sc

# Set parameters and define random wave generator
xmin = 0
xmax = 10
npts = 50
noisevals = np.linspace(0, 1, 11)
x = np.linspace(xmin, xmax, npts)

def randwave(std):
    np.random.seed()
    a = np.cos(x)
    b = np.random.randn(npts)
    return a + b*std

# Start timing
sc.tic()

# Create object in parallel
output = sc.parallelize(randwave, noisevals)

# Save to files
filenames = []
for n,noiseval in enumerate(noisevals):
    filename = f'noise{noiseval:0.1f}.obj'
    sc.saveobj(filename, output[n])
    filenames.append(filename)

# Create dict from files
data = sc.odict({filename:sc.loadobj(filename) for filename in filenames})

# Create 3D plot
sc.surf3d(data[:])

# Print elapsed time
sc.toc()

