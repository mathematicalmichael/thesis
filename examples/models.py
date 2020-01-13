import numpy as np
import dolfin as df

def makeDecayModel(t):
    """
    Builds decay model by evaluating at times `t`

    Parameters
    ----------
    t : list-like
        Times (s) at which to evaluate the exponential model.

    Returns
    -------
    model : function
        A function that evaluates the exponential model at times `t`.
        It takes as input an (samples, dim) array representing
        rate and initial condition, respectively.

    """
    def model(input_samples):
        """
        Function to evaluate exponential decay model at predefined times.

        Parameters
        ----------
        input_samples : `numpy.ndarray`
            Array of size (num_samples, input_dim) representing the samples.

        Returns
        -------
        model : function
            A function that evaluates the exponential model.
            It takes as input an (num_samples, input_dim) array representing
            rate and initial condition, respectively.

        """
        # support passing different types
        if isinstance(input_samples, list) or isinstance(input_samples, tuple):
            input_samples = np.array(input_samples)
        if isinstance(input_samples, np.ndarray):
            if input_samples.ndim == 1:
                input_samples = input_samples.reshape(1, -1)
        # extract rate from first column
        rate = input_samples[:, 0].reshape(-1, 1)
        # if only 1-D array supplied, assume fixed initial condition
        try:
            initial_cond = input_samples[:, 1].reshape(-1, 1)
        except IndexError:
            initial_cond = 0.5

        # compute response at predefined QoI (times)
        response = initial_cond*np.exp(-np.outer(rate, t))
        if response.shape[0] == 1:  # single-row evaluation => ravel output
            return response.ravel()  # allows support for simpler 1D plotting.
        else:
            return response

    return model


def makeMatrixModel(A):
    """
    Description

    Parameters
    ----------
    A : `numpy.ndarray`
        Matrix of size (output_dim, input_dim) representing the linear map.

    Returns
    -------
    model : function
        A function that evaluates the linear model.
        It takes as input an (num_samples, input_dim) array.

    """

    def model(input_samples):
        """
        Build model given by result of matrix-vector product.

        Parameters
        ----------
        input_samples : `numpy.ndarray`
            Array of size (num_samples, input_dim) representing the samples.

        Returns
        -------
        output_samples : `numpy.ndarray`
            Array of size (num_samples, output_dim) resulting from applying the
            map A to `input_samples`.

        """
        # matrix and input_samples are of incompatible sizes by default
        if input_samples.ndim == 2:
            assert A.shape[1] == input_samples.shape[1]
        else:
            assert A.shape[1] == input_samples.shape[0]
        return np.inner(A, input_samples).T

    return model


def skewmat(skew):
    """
    Build operator with predefined skewness.

    Parameters
    ----------
    skew : scalar, or list-like
        Skewness parameter for 2-dimensional map. If scalar, `output_dim = 2`.
        If list-like, build operator to have `output_dim = len(skew)+1`

    Returns
    -------
    Q_map : `np.ndarray`
        Operator with skewness=`skew` of shape (output_dim, 2)

    """
    if isinstance(skew, list) or isinstance(skew, tuple):
        skewnesses = skew
    else:
        skewnesses = [skew]
    # all map components have the same norm, rect_size to have measures of events equal btwn spaces.
    Q_map = [[1.0, 0.0]]
    for skew in skewnesses:
        # taken with the first component, this leads to a 2-2 map with skewsness 's'
        Q_map.append([np.sqrt(skew**2 - 1), 1])
    Q_map = np.array(Q_map)

    return Q_map


def makeSkewModel(skew):
    """
    Build skew model in two dimensions (2->2).

    Parameters
    ----------
    skew : float
        skewness parameter for 2-dimensional map

    Returns
    -------
    model : function
        A function that evaluates the linear model with skewness=`skew`.
        It takes as input an (num_samples, input_dim) array representing
        rate and initial condition, respectively.

    """
    Q_map = skewmat(skew)
    return makeMatrixModel(Q_map)


def make1DHeatModel(temp_locs_list, end_time = 1.0):
    """
    TODO: put parameters into options
    """
    # heatrod code here - including all settings
    t_stop = end_time

    # Some fixed parameter values
    amp = 50.0  # amplitude of the heat source
    px = 0.5  # location of the heat source
    width = 0.05  # width of the heat source
    T_R = 0  # initial temp of the plate
    cap = 1.5  # heat capacity
    rho = 1.5  # density

    # 'parameters' reserved for FEnICS
    df.parameters['allow_extrapolation'] = True

    # mesh properties
    nx = 50  # this is our h value
    mesh = df.IntervalMesh(nx, 0, 1)
    degree = 1
    r = 1.0  # time stepping ratios - attention to stability
    dt = r/nx

    # turn off the heat halfway through
    t_heatoff = 0.5

    def model(parameter_samples):
        """
        TODO: string-formatting that displays the models parameters.

        Returns model evaluated at `T={end_time}`.
        """.format(end_time=end_time)

        if parameter_samples.ndim == 1:
            assert len(parameter_samples) == 2
            num_samples = 1
        else:
            num_samples = parameter_samples.shape[0]
        QoI_samples = np.zeros((num_samples, len(temp_locs_list)))

        for i in range(num_samples):
            try:
                kappa_0 = parameter_samples[i, 0]
                kappa_1 = parameter_samples[i, 1]
            except IndexError:
                kappa_0 = parameter_samples[0]
                kappa_1 = parameter_samples[1]

            # define the subspace we will solve the problem in
            V = df.FunctionSpace(mesh, 'Lagrange', degree)

            # split the domain down the middle(dif therm cond)
            kappa_str = 'x[0] > 0.5 ?'\
                'kappa_1 : kappa_0'

            # Physical parameters
            kappa = df.Expression(kappa_str, kappa_0=kappa_0,
                               kappa_1=kappa_1, degree=1)
            # Define initial condition(initial temp of plate)
            T_current = df.interpolate(df.Constant(T_R), V)

            # Define variational problem
            T = df.TrialFunction(V)

            # two f's and L's for heat source on and off
            f_heat = df.Expression(
                'amp*exp(-(x[0]-px)*(x[0]-px)/width)', amp=amp, px=px, width=width, degree=1)
            f_cool = df.Constant(0)
            v = df.TestFunction(V)
            a = rho*cap*T*v*df.dx + dt*kappa * \
                df.inner(df.nabla_grad(v), df.nabla_grad(T))*df.dx
            L_heat = (rho*cap*T_current*v + dt*f_heat*v)*df.dx
            L_cool = (rho*cap*T_current*v + dt*f_cool*v)*df.dx

            A = df.assemble(a)
            b = None  # variable used for memory savings in assemble calls

            T = df.Function(V)
            t = dt  # initialize first time step
            print("%d Starting timestepping."%i)
            # time stepping method is BWD Euler. (theta = 1)
            while t <= t_stop:
                if t < t_heatoff:
                    b = df.assemble(L_heat, tensor=b)
                else:
                    b = df.assemble(L_cool, tensor=b)
                df.solve(A, T.vector(), b)

                t += dt
                T_current.assign(T)

            # now that the state variable (Temp at time t_stop) has been computed,
            # we take our point-value measurements
            QoI_samples[i,:] = np.array([T(xi) for xi in temp_locs_list])
        # if QoI_samples.shape[0] == 1:
        #     QoI_samples = QoI_samples.ravel()[0]
        return QoI_samples

    return model



# this is from another file.
# if __name__ == "__main__":
#     desc = 'Make figure for demonstration of stochastic framework.'
#     parser = argparse.ArgumentParser(description=desc)
#     parser.add_argument('--fsize', default=52,
#                         help='Figure font size.')
#     parser.add_argument('--png', action='store_true',
#                         help='Store as png instead of pdf.')
#     parser.add_argument('--nolabel', action='store_false',
#                         help='Strip figures of labels.')
#     parser.add_argument('-p','--preview', action='store_true',
#                         help='Supress plt.show()')
#     args = parser.parse_args()

#     size, save_png, = args.fsize, args.png
#     no_label, show_plot = args.nolabel, args.preview

#     print("Plotting...")
#     stochastic_framework_figure(fsize=size, png=save_png,
#                                 showLabel=no_label, showFig=show_plot)
#     print("Done.")
