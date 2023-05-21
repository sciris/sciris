'''
Test Sciris alias_sampler class
'''

import sciris as sc
import pytest
import sciris as sc
import numpy.random as npr


@pytest.fixture
def alias_sampler_fixture():
    # Fixture to create an instance of alias_sampler for testing
    probs = [0.1, 0.1, 0.05, 0.25, 0.1, 0.25, 0.15]
    return sc.alias_sampler(probs)


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


# %% Run as a script
if __name__ == '__main__':
    sc.tic()

    # Add something here

    sc.toc()
    print('Done.')
