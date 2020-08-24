import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from fenics import plot as _plot
from newpoisson import poisson # function evaluation (full response surface)

import numpy as np
plt.rcParams['figure.figsize'] = 10,10
plt.rcParams['font.size'] = 16


def log_linear_regression(input_values, output_values):
    x, y = np.log10(input_values), np.log10(output_values)
    slope, intercept = (np.linalg.pinv(np.vander(x, 2))@np.array(y).reshape(-1,1)).ravel()
    regression_line = 10**(slope*x + intercept)
    return regression_line, slope

        
def plot_surface(res, measurements, prefix, lam_true, fsize=32, test=False):
    
    for _res in res:
        _prefix, _in, _rm, _re = _res
        lam, qoi, sensors, qoi_true, experiments, solutions = _in
        gamma = lam
        plot_num_measure = min(100, max(measurements))
        raveled_input = np.repeat(gamma, qoi.shape[1])
        raveled_output = qoi.reshape(-1)
        x = raveled_input
        y = raveled_output

        fig = plt.figure(figsize=(10,8))
        gs = gridspec.GridSpec(3, 3)
        ax_main = plt.subplot(gs[1:3, :2])
        # ax_xDist = plt.subplot(gs[0, :2],sharex=ax_main)
        ax_yDist = plt.subplot(gs[1:3, 2],sharey=ax_main)

        a = np.argsort(gamma)
        slopes = []
    
        # ax_main.plot(x,y,marker='.')
        for idx in range(plot_num_measure):
            ax_main.plot(gamma[a], qoi[a,idx], c='k', 
                     label=f'sensor {idx}: (%.2f, %.2f)'%(sensors[idx,0], sensors[idx,1]), 
                     lw=1, alpha=0.1)
            slopes.append(qoi[a[-1],idx] - qoi[a[0],idx])
        sa = np.argsort(slopes)
        slopes = np.array(slopes)
        ranked_slopes = slopes[sa]

        xlabel_text = "$\lambda$"
        # ylabel_text = "$u(x_i, \lambda)$"
        ylabel_text = "Measurement\nResponse"
        ax_main.axes.set_xlabel(xlabel_text, fontsize=fsize)
        ax_main.axes.set_ylabel(ylabel_text, fontsize=fsize)
        ax_main.axes.set_ylim((-1.25,0.5))
        # ax_main.axes.set_title('Sensitivity of Measurements', fontsize=1.25*fsize)
        ax_main.axvline(3)

        ax_yDist.hist(qoi_true, bins=np.linspace(-1.25,0.5,35), orientation='horizontal', align='mid')
        # ax_yDist.set(xlabel='count')
        ax_yDist.tick_params(labelleft=False, labelbottom=False)
        plt.savefig(f'{_prefix}_qoi_response.png', bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(10,10))
        # TO-DO fix bins
        plt.title("Sensitivity of\nMeasurement Locations", fontsize=1.25*fsize)
        plt.hist(ranked_slopes, bins=np.linspace(-1.25,0,25), density=True)
        plt.xlabel("Slope", fontsize=fsize)
        plt.savefig(f'{_prefix}_sensitivity_qoi.png', bbox_inches='tight')
        plt.show()

        ##########

        plt.figure(figsize=(10,10))
        print("Most sensitive sensors in first 100:")
        num_sensitive  = 20
        most_sensitive = sa[sa < 100][0:num_sensitive]
        print(most_sensitive)
        _plot(poisson(lam_true))
        for i in range(min(100, max(measurements))):
            plt.scatter(sensors[i,0], sensors[i,1], c='w', s=200)
            if i in most_sensitive:
                plt.scatter(sensors[i,0], sensors[i,1], c='y', s=100)
        #     plt.annotate(f"{i+1:02d}", (sensors[i,0]-0.0125, sensors[i,1]-0.01), alpha=1, fontsize=0.35*fsize)
        # plt.title('Reference solution', fontsize=1.25*fsize)
        plt.xlabel('$x_1$', fontsize=fsize)
        plt.ylabel('$x_2$', fontsize=fsize)
        if not test: plt.savefig(f'{_prefix}_reference_solution.png', bbox_inches='tight')
        plt.show()


def plot_decay_solution(measurements, solutions, model_generator, sigma, prefix, 
                        time_vector, lam_true, qoi_true, end_time=3, fsize=32, test=False):
    alpha_signal = 0.2
    alpha_points = 0.6
#     num_meas_plot_list = [25, 50, 400]

    for num_meas_plot in measurements:
        filename = f'{prefix}_{num_meas_plot}_reference_solution.png'
        plt.rcParams['figure.figsize'] = 25,10
        fig = plt.figure()

        plotting_mesh = np.linspace(0, end_time, 1000*end_time)
        plot_model = model_generator(plotting_mesh, lam_true)
        true_response = plot_model() # no args evaluates true param


        # sample signals
        num_sample_signals  = 500
        alpha_signal_sample = 0.025
        for i in range(num_sample_signals):
            _true_response = plot_model(np.random.rand()) # uniform(0,1) draws from parameter space
            plt.plot(plotting_mesh, _true_response, lw=1, c='k', alpha=alpha_signal_sample)

        # error bars
        sigma_label = f"$\\pm3\\sigma \qquad\qquad \\sigma^2={sigma**2:1.3E}$"
        plt.plot(plotting_mesh[1000:], true_response[1000:]+3*sigma, ls='--', lw=3, c='xkcd:black', alpha=1)
        plt.plot(plotting_mesh[1000:], true_response[1000:]-3*sigma, ls='--', lw=3, c='xkcd:black', alpha=1, label=sigma_label)
        plt.plot(plotting_mesh[:1000], true_response[:1000]+3*sigma, ls='--', lw=3, c='xkcd:black', alpha=alpha_signal)
        plt.plot(plotting_mesh[:1000], true_response[:1000]-3*sigma, ls='--', lw=3, c='xkcd:black', alpha=alpha_signal)

        # solutions
        mud_solutions = solutions[num_meas_plot]
        plt.plot(plotting_mesh, plot_model(mud_solutions[0][0]), lw=1, c='xkcd:red', alpha=20*alpha_signal_sample, label=f'{len(mud_solutions)} Updated Solutions for S={num_meas_plot}')
        for _lam in mud_solutions[1:]:
            _true_response = plot_model(_lam[0])
            plt.plot(plotting_mesh, _true_response, lw=1, c='xkcd:red', alpha=20*alpha_signal_sample)


        # true signal
        plt.plot(plotting_mesh, true_response, lw=5, c='k', alpha=1, label="True Signal, $\\xi \\sim N(0, \\sigma^2)$")


        # observations
        np.random.seed(11)
        annotate_height = 0.82
        u = qoi_true + np.random.randn(len(qoi_true))*sigma
        plot_num_measure = num_meas_plot
        plt.scatter(time_vector[:plot_num_measure], u[:plot_num_measure], color='k', marker='.', s=250, alpha=alpha_points, label=f'{num_meas_plot} Sample Measurements')
        plt.annotate("$ \\downarrow$ Observations begin", (0.95,annotate_height), fontsize=fsize)
    #     plt.annotate("$\\downarrow$ Possible Signals", (0,annotate_height), fontsize=fsize)


        plt.ylim([0,0.9])
        plt.xlim([0,end_time+.05])
        plt.ylabel('Response', fontsize=60)
        plt.xlabel('Time', fontsize=60)
        plt.xticks(fontsize=fsize)
        plt.yticks(fontsize=fsize)
        plt.legend(fontsize=fsize, loc='upper right')
        plt.tight_layout()
        if not test: plt.savefig(filename, bbox_inches='tight')
        plt.show()


def plot_experiment_equipment(tolerances, res, prefix, fsize=32, linewidth=5, title=f"Variance of MUD Error", test=False):
        plt.figure(figsize=(10,10))
        for _res in res:
            _prefix, _in, _rm, _re = _res
            regression_err_mean, slope_err_mean, regression_err_vars, slope_err_vars, sd_means, sd_vars, num_sensors = _re
            plt.plot(tolerances, regression_err_mean, label=f"{_prefix:10s}slope: {slope_err_mean:1.4f}", lw=linewidth)
            plt.scatter(tolerances, sd_means, marker='x', lw=20)

        plt.yscale('log')
        plt.xscale('log')
        plt.Axes.set_aspect(plt.gca(), 1)
#         plt.ylim(1E-4, 5E-2)
        # plt.ylabel("Absolute Error", fontsize=fsize)
        plt.xlabel('Tolerance', fontsize=fsize)
        plt.legend()
        plt.title(f"Mean of MUD Error for S={num_sensors}", fontsize=1.25*fsize)
        if not test: plt.savefig(f'{prefix}_convergence_mud_std_mean.png', bbox_inches='tight')
        plt.show()


        plt.figure(figsize=(10,10))
        for _res in res:
            _prefix, _in, _rm, _re = _res
            regression_err_mean, slope_err_mean, regression_err_vars, slope_err_vars, sd_means, sd_vars, num_sensors = _re
            plt.plot(tolerances, regression_err_vars, label=f"{_prefix:10s}slope: {slope_err_vars:1.4f}", lw=linewidth)
            plt.scatter(tolerances, sd_vars, marker='x', lw=20)
        plt.xscale('log')
        plt.yscale('log')
#         plt.ylim(3E-8, 2E-3)
        plt.Axes.set_aspect(plt.gca(), 1)
        # plt.ylabel("Absolute Error", fontsize=fsize)
        plt.xlabel('Tolerance', fontsize=fsize)
        plt.legend()
        plt.title(title, fontsize=1.25*fsize)
        if not test: plt.savefig(f'{prefix}_convergence_mud_std_var.png', bbox_inches='tight')
        plt.show()

def plot_experiment_measurements(measurements, res, prefix, fsize=32, linewidth=5, xlabel='Number of Measurements', test=False):
        plt.figure(figsize=(10,10))
        for _res in res:
            _prefix, _in, _rm, _re = _res
            regression_mean, slope_mean, regression_vars, slope_vars, means, variances = _rm
            plt.plot(measurements, regression_mean, label=f"{_prefix:10s}slope: {slope_mean:1.4f}", lw=linewidth)
            plt.scatter(measurements, means, marker='x', lw=20)
        plt.xscale('log')
        plt.yscale('log')
        plt.Axes.set_aspect(plt.gca(), 1)
        plt.ylim(0.9*min(means), 1.3*max(means))
#         plt.ylim(5E-3,2E-1)
        plt.xlabel(xlabel, fontsize=fsize)
        plt.legend()
        # plt.ylabel('Absolute Error in MUD', fontsize=fsize)
        plt.title(f"Absolute Error (Mean)", fontsize=1.25*fsize)
        if not test: plt.savefig(f'{prefix}_convergence_mud_obs_mean.png', bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(10,10))
        for _res in res:
            _prefix, _in, _rm, _re = _res
            regression_mean, slope_mean, regression_vars, slope_vars, means, variances = _rm
            plt.plot(measurements, regression_vars, label=f"{_prefix:10s}slope: {slope_vars:1.4f}", lw=linewidth)
            plt.scatter(measurements, variances, marker='x', lw=20)
        plt.xscale('log')
        plt.yscale('log')
        plt.Axes.set_aspect(plt.gca(), 1)
        plt.ylim(0.9*min(variances), 1.3*max(variances))
#         plt.ylim(1E-5, 2E-2)
        plt.xlabel(xlabel, fontsize=fsize)
        plt.legend()
        # plt.ylabel('Absolute Error in MUD', fontsize=fsize)
        plt.title(f"Absolute Error (Variance)", fontsize=1.25*fsize)
        if not test: plt.savefig(f'{prefix}_convergence_mud_obs_var.png', bbox_inches='tight')
        plt.show()