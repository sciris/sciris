# Define random wave generator
import numpy as np

def randwave(std, xmin=0, xmax=10, npts=50):
    np.random.seed(int(100*std)) # Ensure differences between runs
    a = np.cos(np.linspace(xmin, xmax, npts))
    b = np.random.randn(npts)
    return a + b*std

# Other imports
import time
import multiprocessing as mp
import pickle
import gzip
import matplotlib.pyplot as plt

# Start timing
start = time.time()

# Calculate output in parallel
multipool = mp.Pool(processes=mp.cpu_count())
waves = multipool.map(randwave, np.linspace(0, 1, 11))
multipool.close()
multipool.join()

# Save to files
filenames = []
for i,wave in enumerate(waves):
    filename = f'wave{i}.obj'
    with gzip.GzipFile(filename, 'wb') as fileobj:
        fileobj.write(pickle.dumps(wave))
    filenames.append(filename)

# Create dict from files
data_dict = {}
for fname in filenames:
    with gzip.GzipFile(fname) as fileobj:
        filestring = fileobj.read()
        data_dict[fname] = pickle.loads(filestring)

# Create 3D plot
data = np.array([data_dict[fname] for fname in filenames])
fig = plt.figure()
ax = plt.axes(projection='3d')
ny,nx = np.array(data).shape
x = np.arange(nx)
y = np.arange(ny)
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, data, cmap='coolwarm')
fig.colorbar(surf)

# Print elapsed time
elapsed = time.time() - start
print(f'Elapsed time: {elapsed:0.1f} s')
