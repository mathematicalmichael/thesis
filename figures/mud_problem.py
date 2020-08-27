import numpy as np

# from mud import full_functional, norm_input, norm_data, norm_predicted,  
from mud.plot import make_2d_unit_mesh
from mud.util import std_from_equipment

from newpoisson import poisson_sensor_model
# from newpoisson import poisson # function evaluation (full response surface)
from fenics import FunctionSpace, RectangleMesh, Point, Function
import pickle

from plotting import plot_surface, plot_experiment_measurements, plot_experiment_equipment, log_linear_regression, plot_decay_solution
from experiments import mud_problem, experiment_equipment, experiment_measurements, extract_statistics


### PDE ###
def generate_sensors_pde(num_measure, xmin=0.05, ymin=0.05, xmax=0.95, ymax=0.95):
    sensors      = np.random.rand(num_measure,2)
    sensors[:,0] = xmin + (xmax-xmin)*sensors[:,0] # x_0 location
    sensors[:,1] = ymin + (ymax-ymin)*sensors[:,1] # x_1 location
    return sensors


def load_poisson(sensors, l, nx=36, ny=36):
    num_samples = len(l)
    print(f"Loaded {num_samples} evaluations of parameter space.")
    
    mesh = RectangleMesh(Point(0,0), Point(1,1), nx, ny)
    V = FunctionSpace(mesh, 'Lagrange', 1)

    qoi = []
    lam = []
    # go through all the files and load them into an array
    for i in range(len(l)):
        fname = l[i][i]['u']
        u = Function(V, fname)
        q = [u(xi,yi) for xi,yi in sensors] # sensors
        qoi.append(np.array(q))
        lam.append(l[i][i]['gamma'])
    qoi = np.array(qoi)
    lam = np.array(lam)
    print('QoI', qoi.shape,
          'Input', lam.shape,
          'Measurements', sensors.shape,
          'Num Input Samples', num_samples) # check shapes correct
    
    return lam, qoi


def main_pde(model_list, num_trials=5,
             tolerances=[],
             measurements=[],
             fsize=32,
             seed=21,
             lam_true=3.0,
             domain=[[1,5]] ):
    print(f"Will run simulations for S={measurements}")
    res = []
    num_measure = max(measurements)
    
    sd_vals     = [ std_from_equipment(tolerance=tol, probability=0.99) for tol in tolerances ]
    sigma       = sd_vals[-1] # sorted, pick largest
    for _prefix in [ 'pde', 'pde-alt']:
        print(f"Example: {_prefix}")
        if _prefix == 'pde-alt':
            sensors = generate_sensors_pde(num_measure, ymax=0.5, xmax=0.25)
        else:
            sensors = generate_sensors_pde(num_measure, ymax=0.95, xmax=0.95)

        qoi_true = poisson_sensor_model(sensors, gamma=lam_true, nx=36, ny=36) # wrapper around `poisson`
        lam, qoi= load_poisson(sensors, model_list)
        def mud_wrapper(num_obs, sd):
            return mud_problem(domain=domain, lam=lam, qoi=qoi, sd=sd, qoi_true=qoi_true, num_obs=num_obs)

        print("Increasing Measurements Study")
        experiments, solutions = experiment_measurements(num_measurements=measurements,
                                                 sd=sigma,
                                                 num_trials=num_trials,
                                                 seed=seed,
                                                 fun=mud_wrapper)

        means, variances = extract_statistics(solutions, lam_true)
        regression_mean, slope_mean = log_linear_regression(measurements, means)
        regression_vars, slope_vars = log_linear_regression(measurements, variances)

        ##########

        num_sensors = min(100, num_measure)

        print("Increasing Measurement Precision Study")
        sd_means, sd_vars = experiment_equipment(num_trials=num_trials,
                                              num_measure=num_sensors,
                                              sd_vals=sd_vals,
                                              reference_value=lam_true,
                                              fun=mud_wrapper)

        regression_err_mean, slope_err_mean = log_linear_regression(tolerances, sd_means)
        regression_err_vars, slope_err_vars = log_linear_regression(tolerances, sd_vars)

        _in = (lam, qoi, sensors, qoi_true, experiments, solutions)
        _rm = (regression_mean, slope_mean, regression_vars, slope_vars, means, variances)
        _re = (regression_err_mean, slope_err_mean, regression_err_vars, slope_err_vars, sd_means, sd_vars, num_sensors)
        res.append((_prefix, _in, _rm, _re))
    return res


### ODE ###
def generate_sensors_ode(measurement_hertz=100, start_time=1, end_time=3):
    num_measure = measurement_hertz*(end_time-start_time)
    return np.linspace(start_time, end_time, num_measure)


def makeDecayModel(t, lam_true):
    def model(lam = np.array([[lam_true]]) ):
        if isinstance(lam, float) or isinstance(lam, int):
            lam = np.array([[lam]])
        initial_cond = 0.75
        rate = lam[:,0].reshape(-1,1)
        response = initial_cond*np.exp(np.outer(rate, -t))
        if response.shape[0] == 1:
            return response.ravel() # this allows support for simpler 1D plotting.
        else:
            return response
    return model


def main_ode(num_trials,
             fsize=32,
             seed=21,
             lam_true=0.5,
             domain=[[0,1]],
             tolerances=[0.1],
             time_ratios=[1]):
    res = []
    print(f"Will run simulations for %T={time_ratios}")
    sd_vals      = [ std_from_equipment(tolerance=tol, probability=0.99) for tol in tolerances ]
    sigma        = sd_vals[-1] # sorted, pick largest
    t_min, t_max = 1, 3
    for _prefix in [ 'ode', 'ode-alt']:
        print(f"Example: {_prefix}")
        if _prefix == 'ode-alt':
            sensors = generate_sensors_ode(measurement_hertz=200, start_time=t_min, end_time=t_max)
        else:
            sensors = generate_sensors_ode(measurement_hertz=100, start_time=t_min, end_time=t_max)
        
        measurements = [ int(np.floor(len(sensors)*r)) for r in time_ratios ]
        print(f"Measurements: {measurements}")
#         times        = [ sensors[m-1] for m in measurements ]
        num_measure = max(measurements)
        
        model    = makeDecayModel(sensors, lam_true)
        qoi_true = model() # no args evaluates true param
        np.random.seed(seed)
        lam = np.random.rand(int(1E4)).reshape(-1,1)
        qoi = model(lam)
        
        def mud_wrapper(num_obs, sd):
            return mud_problem(domain=domain, lam=lam, qoi=qoi, sd=sd, qoi_true=qoi_true, num_obs=num_obs)

        print("Increasing Measurements Quantity Study")
        experiments, solutions = experiment_measurements(num_measurements=measurements,
                                                 sd=sigma,
                                                 num_trials=num_trials,
                                                 seed=seed,
                                                 fun=mud_wrapper)

        means, variances = extract_statistics(solutions, lam_true)
        regression_mean, slope_mean = log_linear_regression(time_ratios, means)
        regression_vars, slope_vars = log_linear_regression(time_ratios, variances)

        ##########

        num_sensors = num_measure

        print("Increasing Measurement Precision Study")
        sd_means, sd_vars = experiment_equipment(num_trials=num_trials,
                                              num_measure=num_sensors,
                                              sd_vals=sd_vals,
                                              reference_value=lam_true,
                                              fun=mud_wrapper)

        regression_err_mean, slope_err_mean = log_linear_regression(tolerances, sd_means)
        regression_err_vars, slope_err_vars = log_linear_regression(tolerances, sd_vars)

        _in = (lam, qoi, sensors, qoi_true, experiments, solutions)
        _rm = (regression_mean, slope_mean, regression_vars, slope_vars, means, variances)
        _re = (regression_err_mean, slope_err_mean, regression_err_vars, slope_err_vars, sd_means, sd_vars, num_sensors)
        res.append((_prefix, _in, _rm, _re))

        plot_decay_solution(measurements, solutions, makeDecayModel, fsize=fsize,
                            end_time=t_max, lam_true=lam_true, qoi_true=qoi_true,
                            sigma=sigma, time_vector=sensors, prefix=_prefix)

    return res


######


if __name__ == "__main__":
    import argparse
    desc = """
        Examples
        """
    
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', '--prefix',        default='pde', type=str)
    parser.add_argument('-m', '--num-measure',   default=[],  action='append')
    parser.add_argument('-r', '--ratio-measure', default=[],  action='append')
    parser.add_argument('-n', '--num-trials',    default=20,    type=int)
    parser.add_argument('-t', '--tolerance',     default=[], action='append')
    parser.add_argument('-s', '--seed',          default=21)
    parser.add_argument('-lw', '--linewidth',    default=5)
    parser.add_argument('-i', '--input-dim',     default=1, type=int)
    parser.add_argument('--fsize',               default=32, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    prefix       = args.prefix
    num_trials   = args.num_trials
    fsize        = args.fsize
    linewidth    = args.linewidth
    seed         = args.seed
    inputdim     = args.input_dim
    test         = args.test  # testing mode: disables savefig.
    tolerances   = list(np.sort([ float(t) for t in args.tolerance ]))
    if len(tolerances) == 0: tolerances = [0.1]

    if prefix == 'pde':
        measurements = list(np.sort([ int(n) for n in args.num_measure ]))
        if len(measurements) == 0: measurements = [100]
    else:
        time_ratios  = list(np.sort([ float(r) for r in args.ratio_measure ]))
        if len(time_ratios) == 0: time_ratios = [1.0]

    print("Running...")
    if prefix == 'pde':
        lam_true = 3.0
        model_list  = pickle.load(open(f'res{inputdim}u.pkl', 'rb'))
        res = main_pde(model_list, num_trials=num_trials,
                         fsize=fsize,
                         seed=seed,
                         lam_true=lam_true,
                         tolerances=tolerances,
                         measurements=measurements)

        plot_surface(res=res, measurements=measurements,
                     fsize=fsize, prefix=prefix, lam_true=lam_true, test=test)

        if len(measurements) > 1: plot_experiment_measurements(measurements, res,
                                                               prefix, fsize,
                                                               linewidth, test=test)
        if len(tolerances) > 1: plot_experiment_equipment(tolerances, res,
                                                              prefix, fsize,
                                                              linewidth, test=test)
    elif prefix == 'ode':
        lam_true = 0.5
        res = main_ode(num_trials=num_trials,
                         fsize=fsize,
                         seed=seed,
                         lam_true=lam_true,
                         tolerances=tolerances,
                         time_ratios=time_ratios)

        if len(time_ratios) > 1: plot_experiment_measurements(time_ratios, res,
                                                              prefix, fsize, linewidth,
                                                              xlabel='Proportion of Time Elapsed',
                                                              test=test)
        if len(tolerances) > 1: plot_experiment_equipment(tolerances, res,
                                                          prefix, fsize, linewidth,
                                                          title=f"Variance of MUD Error\nfor t={1+2*np.median(time_ratios):1.3f}s",
                                                          test=test)
    ##########
    
    
    if args.save:
        import pickle
        pickle.dump('results.pkl', res)
