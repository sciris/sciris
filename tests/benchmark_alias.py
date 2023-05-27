import timeit
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as npr
import numpy as np
import sciris as sc

# Define the number of trials and values for n_samples
n_trials = 1
n_repeats = 128
n_probs = 256
n_samples_values = [2 ** i for i in range(0, 20)]

# Use same approach as in JH's example
probs = npr.random(n_probs)
probs /= probs.sum()

# Define the setup code
sampler1 = sc.alias_sampler(probs=probs)
sampler2 = sc.AliasSample(probs)


# Define the benchmark function
def run_benchmark(sampler, n_samples):
    sampler.draw(n_samples)


results_df = pd.DataFrame(
    columns=["Sampler", "num_samples", "num_trials", "mean time per iteration (ms)", "standard deviation (ms)"])

# Perform the benchmark for each value of n_samples
for n_samples in n_samples_values:
    # Run the benchmark for first sampler
    time_alias_numpy = timeit.repeat('run_benchmark(sampler1, n_samples)', globals=globals(),
                                     number=n_trials, repeat=n_repeats) * 1000
    time_per_iteration_numpy = np.median(time_alias_numpy) * 1000
    std_dev_numpy = np.std(time_alias_numpy) * 1000

    results_df = pd.concat([results_df, pd.DataFrame({'Sampler': ['numpy_sampler'],
                                                      'num_samples': [n_samples],
                                                      'num_trials': [n_trials],
                                                      'mean time per iteration (ms)': time_per_iteration_numpy,
                                                      'standard deviation (ms)': std_dev_numpy})],
                           ignore_index=True)

    # Run the benchmark for second sampler
    time_alias_numba = timeit.repeat('run_benchmark(sampler2, n_samples)', globals=globals(),
                                     number=n_trials, repeat=n_repeats) * 1000
    time_per_iteration_numba = np.mean(time_alias_numba) * 1000
    std_dev_numba = np.std(time_alias_numba) * 1000
    results_df = pd.concat([results_df, pd.DataFrame({'Sampler': ['numba_sampler'],
                                                      'num_samples': [n_samples],
                                                      'num_trials': [n_trials],
                                                      'mean time per iteration (ms)': time_per_iteration_numba,
                                                      'standard deviation (ms)': std_dev_numba})],
                           ignore_index=True)

# Save the results to a CSV file
results_df.to_csv(f"benchmark_alias_results_nprobs_{n_probs}.csv", index=False)

# Plot the results and compare performance
fig, ax = plt.subplots()
for sampler in results_df['Sampler'].unique():
    sampler_df = results_df[results_df['Sampler'] == sampler]
    ax.plot(sampler_df['num_samples'][1:], sampler_df['mean time per iteration (ms)'][1:], marker='o', label=sampler)

samplers = results_df['Sampler'].unique()
sampler_df = results_df[results_df['Sampler'] == samplers[0]]
mean_time_numpy = sampler_df['mean time per iteration (ms)']
sampler_df = results_df[results_df['Sampler'] == samplers[1]]
mean_time_numba = sampler_df['mean time per iteration (ms)']
ratio = np.array(mean_time_numpy) / np.array(mean_time_numba)
ax.plot(np.array(sampler_df['num_samples'])[1:], ratio[1:], marker='o', label='ratio numpy/numba')

ax.set_xscale('log', base=2)  # Set logarithmic scale in base 2
ax.set_xlabel("number of samples drawn (log_2)")
ax.set_ylabel("median time per iteration (ms)")
ax.set_title(f"Execution Time Comparison (len(probs)={n_probs})")
ax.legend()
plt.show()
