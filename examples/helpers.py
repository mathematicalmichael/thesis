import bet.sample as sample
import bet.sampling.basicSampling as bsam
from scipy.stats import distributions as dists
import numpy as np

def baseline_discretization(model,
                            num_samples=1000,
                            input_dim=2,
                            param_ref=None,
                            input_cpd=None,
                            n_mc_points=None,
                            min_val=0,
                            max_val=1):
    """
    """
    sampler = bsam.sampler(model)
    if isinstance(min_val, int) or isinstance(min_val, float):
        min_val = [min_val]*int(input_dim)
    if isinstance(max_val, int) or isinstance(max_val, float):
        max_val = [max_val]*int(input_dim)
    assert len(min_val) == len(max_val)
    assert len(min_val) == input_dim

    domain = np.vstack([min_val, max_val]).T
    if isinstance(input_cpd, int):
        input_set = bsam.regular_sample_set(domain, num_samples_per_dim=input_cpd)
        # still needs manual writing of initial probabilities
        input_set.set_distribution(dist=dists.uniform, loc=min_val, scale=max_val)
    else:
        input_set = bsam.random_sample_set('r', domain, num_samples)

    # equip object with reference value
    input_set.set_reference_value(param_ref)

    # Estimate volumes of Voronoi cells associated with the parameter samples
    if isinstance(n_mc_points, int):
        print("Using Monte-Carlo assumption for volumes.")
        input_set.estimate_volume(n_mc_points=n_mc_points)
    else:
        input_set.estimate_volume_mc()

    disc = sampler.compute_QoI_and_create_discretization(input_sample_set=input_set)
    return disc

import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP

def solve_set_based(discretization,
                    rect_size,
                    obs_cpd=1):
    """
    """
    disc = discretization.copy()
    Q_ref = disc.get_output().get_reference_value()
    simpleFunP.regular_partition_uniform_distribution_rectangle_size(
            data_set=disc, Q_ref=Q_ref, rect_size=rect_size,
            cells_per_dimension = obs_cpd)
    calculateP.prob(disc)
    return disc

def solve_sample_based(discretization,
                       rect_size):
    """
    """
    disc = discretization.copy()
    rect_loc = disc.get_output().get_reference_value() - 0.5*rect_size
    disc.set_observed(dist=dists.uniform, loc=rect_loc, scale=rect_size)
    disc.compute_pushforward()
    return disc

def comparison_wrapper(model,
                       num_samples=1000,
                       input_dim=2,
                       param_ref=None,
                       rect_size=0.2,
                       cpd_observed=1,
                       input_cpd=None,
                       n_mc_points=None,
                       min_val=0,
                       max_val=1):

    disc = baseline_discretization(model=model,
                               num_samples=num_samples,
                               input_dim=input_dim,
                               param_ref=param_ref,
                               input_cpd=input_cpd,
                               n_mc_points=n_mc_points,
                               min_val=min_val,
                               max_val=max_val)

    disc_set = solve_set_based(discretization=disc,
                             rect_size=rect_size,
                             obs_cpd=cpd_observed)


    disc_samp = solve_sample_based(discretization=disc,
                             rect_size=rect_size)

    return (disc, disc_set, disc_samp)
