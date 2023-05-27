'''
Test Sciris alias_sampler class
'''

import pytest
import sciris as sc
import numpy as np
import scipy.stats as stats

@pytest.fixture
def alias_sampler_fixture():
    # Fixture to create an instance of alias_sampler for testing
    probs = [0.1, 0.1, 0.05, 0.25, 0.1, 0.25, 0.15]
    return sc.alias_sampler(probs, randseed=42)


def test_draw_indices(alias_sampler_fixture):
    # Test drawing indices from the distribution
    indices = alias_sampler_fixture.draw(1000)

    # Check if indices are within the valid range
    assert all(0 <= idx < len(alias_sampler_fixture.probs) for idx in indices)


def test_draw_values(alias_sampler_fixture):
    # Test drawing values from the distribution
    values = ['thx', 'omw', 'lol', 'tmi', 'idk', 'btw', 'brb', 'imo']
    sampler = sc.alias_sampler(alias_sampler_fixture.probs, values)
    sampled_values = sampler.draw(1000)

    # Check if sampled values are from the given set of values
    assert all(value in values for value in sampled_values)


def test_invalid_probs_and_vals():
    # Test when probabilities and values have different lengths
    probs = [0.2, 0.3, 0.1, 0.4]
    values = ['thx', 'omw', 'lol']
    with pytest.raises(ValueError):
        sc.alias_sampler(probs, values)


def test_negative_probs():
    # Test when one or more probabilities are negative
    probs = [0.2, 0.3, -0.1, 0.6]
    with pytest.raises(ValueError):
        sc.alias_sampler(probs)


def test_probs_not_add_up_to_one():
    # Test when probabilities don't add up to one
    probs = [0.2, 0.3, 0.1, 0.5]
    sampler = sc.alias_sampler(probs, verbose=False)

    # Check if probabilities are normalized internally
    assert pytest.approx(sum(sampler.probs)) == 1.0


def test_sample_distribution():
    """
    Verify that the generated samples from the draw method of the alias_sampler
    class follow the desired distribution using a statistical test:
    """
    # Define a normal probability distribution -- will have to be normalised to add up to 1
    probs_x = np.array([0.0021, 0.0047, 0.0097, 0.0181, 0.0309, 0.0483, 0.069, 0.0902,
                      0.1078, 0.1179, 0.1179, 0.1078, 0.0902, 0.069, 0.0483, 0.0309,
                      0.0181, 0.0097, 0.0047, 0.0021])

    vals_x = np.arange(-10, 10)

    # Create an instance of alias_sampler
    sampler = sc.alias_sampler(probs_x, vals_x, randseed=21)

    # Generate a larg-ish number of samples
    n_samples = 2 ** 16
    samples = sampler.draw(n_samples)

    # Perform a chi-square goodness-of-fit test
    observed_x, _ = np.histogram(samples, bins=len(vals_x))
    expected_x = n_samples * sampler.probs
    _, p_value = stats.chisquare(observed_x, expected_x)

    # Verify that the p-value is above a certain threshold (e.g., 0.05)
    assert p_value > 0.05


# %% Run as a script
if __name__ == '__main__':
    sc.tic()
    test_negative_probs()
    test_probs_not_add_up_to_one()
    test_sample_distribution()

    sc.toc()
    print('Done.')
