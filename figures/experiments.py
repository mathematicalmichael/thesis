import numpy as np
import bet.sample as samp
from scipy.stats import distributions as dist

def mud_problem(lam, qoi, qoi_true, domain, sd=0.05, num_obs=None):
    try:
        dim_input = lam.shape[1]
    except IndexError:
        dim_input = 1
        lam = lam.reshape(-1,1)

    try:
        dim_output = qoi.shape[1]
    except IndexError:
        dim_output = 1
        qoi = qoi.reshape(-1,1)

    if num_obs is None:
        num_obs = dim_output
    elif num_obs < 1:
        raise ValueError("num_obs must be >= 1")
    elif num_obs > dim_output:
        raise ValueError("num_obs must be <= dim(qoi)")

    i_set = samp.sample_set(dim_input)
    _dom_ = np.array(domain)
    i_set.set_domain(_dom_)
    
    i_set.set_values(lam)
    o_set = samp.sample_set(dim_output)
    o_set.set_values(qoi)
    d = samp.discretization(i_set, o_set)
    data = qoi_true[0:num_obs] + np.random.randn(num_obs)*sd
    
    mn, mx = np.min(_dom_, axis=1), np.max(_dom_, axis=1)
    d.set_initial(dist=dist.uniform(loc=mn,scale=mx-mn), gen=False)
    # needed if changing dimensions around until fix is made in BET
#     d._output_probability_set = None # will throw warning
    d._output_probability_set = samp.sample_set(num_obs)
    d.data_driven(data=data, std=sd, inds=list(range(0,num_obs)))
    return d


def experiment_equipment(fun, num_measure, sd_vals, num_trials, reference_value):
    """
    Fixed number of sensors, varying the quality of equipment.
    """
    sd_err = []
    sd_var = []
    for sd in sd_vals:
        temp_err = []
        for t in range(num_trials):
            d = fun(sd=sd, num_obs=num_measure)
            mud_point = d.mud_point()
            temp_err.append(np.linalg.norm(mud_point - reference_value))
        sd_err.append(np.mean(temp_err))
        sd_var.append(np.var(temp_err))

    return sd_err, sd_var


def experiment_measurements(fun, num_measurements, sd, num_trials, seed=21):
    """
    Fixed sensors, varying how much data is incorporated into the solution.
    """
    experiments = {}
    solutions = {}
    for ns in num_measurements:
        discretizations = []
        mud_solutions = []
        for t in range(num_trials):
            np.random.seed(seed+t)
            _d = fun(sd=sd, num_obs=ns)
            discretizations.append(_d)
            mud_solutions.append(_d.mud_point())
        experiments[ns] = discretizations
        solutions[ns] = mud_solutions
    
    return experiments, solutions


def extract_statistics(solutions, reference_value):
    num_sensors_plot_conv = solutions.keys()
    means = []
    variances = []
    for ns in num_sensors_plot_conv:
        mud_solutions = solutions[ns]
        num_trials = len(mud_solutions)
        err = [np.linalg.norm(m - reference_value) for m in mud_solutions]
        assert len(err) == num_trials
        mean_mud_sol = np.mean(err)
        var_mud_sol = np.var(err)
        means.append(mean_mud_sol)
        variances.append(var_mud_sol)
    return means, variances

def experiment_measurements_index(fun, num_measurements, sd, num_trials, seed=21):
    """
    Fixed sensors, varying how much data is incorporated into the solution.
    """
    experiments = {}
    solutions = {}
    for ns in num_measurements:
        ratios = []
        mud_solutions = []
        for t in range(num_trials):
            np.random.seed(seed+t)
            _r = fun(sd=sd, num_obs=ns)
            ratios.append(_r)
            mud_solutions.append(np.argmax(_r))
        experiments[ns] = ratios
        solutions[ns] = mud_solutions
    
    return experiments, solutions


def extract_statistics_index(x, solutions, reference_value):
    num_sensors_plot_conv = solutions.keys()
    means = []
    variances = []
    for ns in num_sensors_plot_conv:
        mud_solutions = [ x[_index] for _index in solutions[ns] ]
        err = np.linalg.norm(np.array(mud_solutions) - reference_value, axis=1)/np.sqrt(len(reference_value))
        mean_mud_sol = np.mean(err)
        var_mud_sol = np.var(err)
        means.append(mean_mud_sol)
        variances.append(var_mud_sol)
    return means, variances