"""
Script to benchmark mostly speed of saving objects using zstandard compression instead of gzip
TODO: reading/decompression test
TODO: - split gzip/zstd benchmarks to automate estimation of ratios
      - append iteration number to filename, to avoid overwriting the same file
      - log timing (or other via resourcemonitor) information into a pandas dataframe

Benchamrks:
4x type of objects of different levels of complexity:
    - numpy array
    - pandas dataframe
    - covasim sim object (generated from devtests)
       - 2x simulation lengths
    - covasim multisim object (generated from devtests)
       - 2x values of repetitions
    - pregenerated multisim from ICMR project?

2x drives:
    - M.2 PCIe NVMe SSD (Samsung 970 EVO)
    - SATA HD (Seagate Barracuda ST4000DM004)
"""

import os
import numpy as np
import pandas as pd
import sciris as sc

def generate_simple_numpy_array():
    '''
    Define the test data using numpy
    '''
    nrows = 15
    ncols = 3
    testdata   = np.zeros((nrows+1, ncols), dtype=object) # Includes header row
    testdata[0,:] = ['A', 'B', 'C'] # Create header
    testdata[1:,:] = np.random.rand(nrows,ncols) # Create data

    return testdata


def generate_simple_pandas_df():
    '''
    Define the test data using pandas
    '''
    nrows = 15
    ncols = 3
    dates = pd.date_range("20221101", periods=nrows)
    testdata = pd.DataFrame(np.random.randn(nrows, ncols), index=dates, columns=list("ABC"))
    return  testdata


def load_covasim_sim_obj_short():
    '''
     
    '''
    pass


def load_covasim_sim_obj_long():
    pass


def load_covasim_msim_obj_few():
    pass


def load_covasim_msim_obj_many():
    pass


def load_covasim_msim_obj_icmr():
    pass


testdata = generate_simple_numpy_array()
#testdata = generate_simple_pandas_df()



def run_benchmark(benchmark_params, filename, obj):
    for cmp_lib in benchmark_params.compression_lib:
        for clevel in benchmark_params.compression_lvl:
            timings = []
            for ii in range(benchmark_params.n_iterations):
                sc.tic()
                sc.save(filename, obj, compression=cmp_lib, compresslevel=clevel)
                tt = sc.toc(output=True)
                timings.append(tt)
            # Cure estimation of average duration
            avg_duration = sum(timings) / benchmark_params.n_iterations
            print(f'{cmp_lib} | {clevel} : On average (mean | nreps {benchmark_params.n_iterations}) it took {avg_duration} seconds')
    return None

#%% Run as a script
if __name__ == '__main__':
    # Define the parameters for the benchmark
    benchmark_params = sc.prettyobj
    benchmark_params.compression_lib = ['gzip', 'zstd']
    benchmark_params.compression_lvl = [1, 5, 9]
    benchmark_params.n_iterations = 20


    # Files to benchmark
    filedir = 'files' + os.sep
    files = sc.prettyobj()
    files.numpy_array_obj = filedir + 'bmk_numpy_array.obj'
    files.pandas_data_frame = filedir + 'bmk_pandas_df.obj'
    files.covsim_sim_short = filedir + 'bmk_covsim_sim_default.obj'  # Defaults covasim (3.1.4)
    files.covsim_sim_long = filedir + 'bmk_covsim_sim_2x_long.obj'  # Defaults covasim 2x length
    files.covsim_msim_few = filedir + 'bmk_msim_default_nrep_010.obj'  # Defaults covasim 10x reps
    files.covsim_msim_many = filedir + 'bmk_msim_default_nrep_100.obj'  # Defaults covasim 100x reps
    # files.covsim_msim_many  = filedir + 'bmk_msim_icmr.obj'              # Defaults covasim 100x reps

    sc.heading('Benchmarking simple numpy array')
    testdata = generate_simple_numpy_array()
    run_benchmark(benchmark_params, files.numpy_array_obj, testdata)

    sc.heading('Benchmarking simple pandas data frame')
    testdata = generate_simple_pandas_df()
    run_benchmark(benchmark_params, files.pandas_data_frame, testdata)