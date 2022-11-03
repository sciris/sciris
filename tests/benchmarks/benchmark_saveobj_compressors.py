"""
Script to benchmark mostly speed of saving objects using zstandard compression instead of gzip
TODO: reading/decompression benchmark
TODO: - split gzip/zstd benchmarks to automate estimation of ratios
      - append iteration number, compressor, etc to filename, to avoid overwriting the same file
      - check other resources (via resourcemonitor?) to the log pandas dataframe

Benchamrks:
4x type of objects of different levels of complexity:
    - numpy array
    - pandas dataframe
    - covasim sim object (generated from devtests)
       - 2x simulation lengths
    - covasim multisim object (generated from devtests)
       - 2x values of repetitions
    - pregenerated multisim from ICMR project? (gets a bit complicated because of dependencies to load obj)

2x drives:
    - M.2 PCIe NVMe SSD (Samsung 970 EVO)
    - SATA HD (Seagate Barracuda ST4000DM004)
"""

import os
import numpy as np
import pandas as pd
import sciris as sc
import covasim as cv

from sciris import sc_odict as sco
from sciris import sc_nested as scn


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


def generate_covasim_sim_obj_short():
    '''
    Runs covasim 3.1.4 default single simulation
    '''
    sim = cv.Sim(verbose=False)
    sim.run()

    return sim


def generate_covasim_sim_obj_long():
    '''
    Runs covasim 3.1.4 single simulation with defaults
    except sim length is twice as long as the default value.
    '''
    sim = cv.Sim(n_days=120, verbose=False)
    sim.run()

    return sim


def generate_covasim_msim_obj_few():
    '''
    Runs a covasim 3.1.4 multisim, 10x the base sim which is the
    default single sim
    '''
    # Save once with msim.save() to get rid of people
    dummy_name = 'files' + os.sep + 'dummy_sim.obj'
    # Run sim only once otherwise, just load the dummy file
    if not os.path.exists(dummy_name):
        sim = cv.Sim(verbose=False)
        msim = cv.MultiSim(base_sim=sim)
        msim.run(n_runs=10)
        msim.save(dummy_name)
    msim = sc.load(dummy_name)
    return msim


def generate_covasim_msim_obj_many():
    '''
    Runs a covasim 3.1.4 multisim, 10x the base sim which is the
    default single sim
    '''
    dummy_name = 'files' + os.sep + 'dummy_msim.obj'
    if not os.path.exists(dummy_name):
        sim = cv.Sim(verbose=False)
        msim = cv.MultiSim(base_sim=sim)
        msim.run(n_runs=100)
        msim.save(dummy_name)
    msim = sc.load(dummy_name)

    return msim


# def load_covasim_msim_obj_icmr():
#     '''
#     Load object obtained by running vaccine_access_in_India/calibration/dehli.py
#     :return:
#     '''
#     dummy_name = 'files' + os.sep + 'bmk_msim_icmr_dehli.msim'
#     msim = sc.load(dummy_name)
#
#     return msim


def to_df(log):
    ''' Convert the log into a pandas dataframe '''
    entries = []
    for entry in log:
        flat = scn.flattendict(entry, sep='_')
        entries.append(flat)
    df = pd.DataFrame(entries)
    return df


def run_benchmark(benchmark_params, filename, obj, log=None):
    '''
     Quick-ish assessment of running times of sc.save() using different compressors
    :param benchmark_params: pretty object
    :param filename: base filename where to store results
    :param obj: the object to be saved
    :return: None
    '''

    if log is None:
        log = []

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
            # Gather into output form
            benchdata = sco.objdict(
                filename=filename,
                duration=avg_duration,
                nreps=benchmark_params.n_iterations,
                compressor=cmp_lib,
                compression=clevel,
            )
            log.append(benchdata)
    return log

#%% Run as a script
if __name__ == '__main__':
    # Define the parameters for the benchmark
    benchmark_params = sc.prettyobj
    benchmark_params.compression_lib = ['gzip', 'zstd']
    benchmark_params.compression_lvl = [1, 5, 9]
    benchmark_params.n_iterations = 50

    # Files to benchmark
    filedir = 'files' + os.sep
    files = sc.prettyobj()
    files.numpy_array_obj   = filedir + 'bmk_numpy_array.obj'
    files.pandas_data_frame = filedir + 'bmk_pandas_df.obj'
    files.covsim_sim_short  = filedir + 'bmk_covsim_sim_default.obj'     # Defaults covasim (3.1.4)
    files.covsim_sim_long   = filedir + 'bmk_covsim_sim_default_2x.obj'  # Defaults covasim 2x length
    files.covsim_msim_few   = filedir + 'bmk_msim_default_nrep_010.obj'  # Defaults covasim 10x reps
    files.covsim_msim_many  = filedir + 'bmk_msim_default_nrep_100.obj'  # Defaults covasim 100x reps
    #files.covsim_msim_icmr  = filedir + 'bmk_msim_icmr.obj'              #

    sc.heading('Benchmarking simple numpy array')
    testdata = generate_simple_numpy_array()
    log = run_benchmark(benchmark_params, files.numpy_array_obj, testdata)

    sc.heading('Benchmarking simple pandas data frame')
    testdata = generate_simple_pandas_df()
    log = run_benchmark(benchmark_params, files.pandas_data_frame, testdata, log=log)

    sc.heading('Benchmarking covasim default sim')
    testdata = generate_covasim_sim_obj_short()
    log = run_benchmark(benchmark_params, files.covsim_sim_short, testdata, log=log)

    sc.heading('Benchmarking covasim default sim, 2x long')
    testdata = generate_covasim_sim_obj_long()
    log = run_benchmark(benchmark_params, files.covsim_sim_long, testdata, log=log)

    sc.heading('Benchmarking covasim msim, with default base sim, 10 reps')
    testdata = generate_covasim_msim_obj_few()
    log = run_benchmark(benchmark_params, files.covsim_msim_few, testdata, log=log)

    sc.heading('Benchmarking covasim msim, with default base sim, 100 reps')
    testdata = generate_covasim_msim_obj_many()
    log = run_benchmark(benchmark_params, files.covsim_msim_many, testdata, log=log)

    df_log = to_df(log)

    drive = 'hdd'
    benchmark_filepath = f'{filedir}benchmark_{drive}_nreps_{benchmark_params.n_iterations:02d}.csv'
    df_log.to_csv(benchmark_filepath)

