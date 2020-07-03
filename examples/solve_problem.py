import numpy as np
from helpers import baseline_discretization, solve_set_based, solve_sample_based, comparison_wrapper
from models import makeMatrixModel, makeSkewModel, makeDecayModel, make1DHeatModel

# define problem dimensions
inputDim, outputDim = 2, 2
# define reference parameter
# refParam = np.array([0.5]*inputDim)

if __name__ == "__main__":
    import argparse

    desc = """
        Make voronoi-cell diagrams with uniform random samples
        in a 2D unit domain.
        """
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-m', '--model', default='random', type=str,
                    help="""
                        Choose model from
                        - 'skew' (linear matrix map)
                        - 'identity' (linear matrix map)
                        - 'random' (linear matrix map)
                        - 'decay' (exponential decay)
                        - 'heatrod' (1-dimensional heat eq)
                        - 'diagonal' (linear matrix map)

                    If unrecognized, it will revert to 'random' (linear map).
                    """)

    parser.add_argument('-n', '--num', default=int(1E2), type=int,
                        help="""
                            Set number of samples (default: 1E2).
                        If given as <=1, it will revert to the default value.
                        """)

    parser.add_argument('-u', '--uncert_rect_size', default=0.2, type=float,
                    help='Set uncertainty (`rect_size`) (default: 0.2)')

    parser.add_argument('-s', '--seed', default=21, type=int,
                        help='Set random seed (default: 21).')

    parser.add_argument('-o', '--observed_cells_per_dim', default=1, type=int,
                        help="""
                            Cells per dimension (default: 1) for regular grid
                            discretizing the `output_probability_set`.
                            If given as <1, it will revert to the default value.
                            """)

    parser.add_argument('-i', '--input_cells_per_dim', default=49, type=int,
                    help="""
                        Cells per dimension (default: 49) for regular grid
                        discretizing the `input_sample_set`.
                        If given as <1, it will revert to the default value.
                        """)

    parser.add_argument('--mc_points', default=0, type=int,
                help="""
                    Number of samples (default: 0) in calculation of
                    volumes using Monte-Carlo emulation (integration).
                    If given as <100, it will revert to None.
                    If None, or not supplied, default to using the Monte-Carlo
                    assumption (volumes = 1/num_samples).
                    """)

    parser.add_argument('--reg', action='store_true',
                    help='Use regular grid sampling for input space.')

    parser.add_argument('--pdf', action='store_true',
                        help='Store as pdf instead of png.')

    parser.add_argument('--show', action='store_true',
                    help='Call `plt.show()` after plotting.')

    parser.add_argument('--noplot', action='store_true',
                        help='Option to not create/save plots.')

    parser.add_argument('--fontsize', default=16, type=float,
                    help='Sets `plt.rcParams[\'font.size\']` (default: 16).')

    parser.add_argument('--figsize', default=5, type=int,
                    help="""
                        Sets `plt.rcParams[\'figure.size\']`(default: 5).
                        Assumes square aspect ratio.
                    """)

    parser.add_argument('-l', '--numlevels', default=10, type=int,
                help="""
                    Number of contours to plot (default=10).
                    If given as <2, it will revert to 2.
                """)

    parser.add_argument('--figlabel', default='', type=str,
                    help='Label in figure saved name.')

    parser.add_argument('-t', '--title', default=None, type=str,
                help='Title for figure. If `None`, use `--model` in title.')

    # which problem type to solve?
    parser.add_argument('--set', action='store_true',
                        help='Only do set-based solution.')

    parser.add_argument('--sample', action='store_true',
                    help='Only do sample-based solution.')

    # model- specific arguments
    parser.add_argument('--skew', default=1.0, type=float,
                    help='Sets skew if `--model=\'skew\'` (default: 1.0).')

    parser.add_argument('--lam1', default=0.5, type=float,
                    help='Sets first default parameter (default: 0.5).')

    parser.add_argument('--lam2', default=0.5, type=float,
                    help='Sets second default parameter (default: 0.5).')

    parser.add_argument('--t0', default=1, type=float,
                    help='Decay model: 1st observation time (default: 1).')

    parser.add_argument('--t1', default=2, type=float,
                    help='Decay model: 2nd observation time (default: 2).')

    parser.add_argument('--noeval', action='store_true',
                    help='Sample based model: plot using original samples, not mesh.')

    #### START OF FUNCTIONALITY ###
    args = parser.parse_args()
    numSamples, r_seed = args.num, args.seed
    eval = not args.noeval # evaluate sample-based model at new samples?

    if numSamples <= 1:
        print("Incompatible number of samples. Using default.")
        numSamples = int(1E2)

    if r_seed > 0:
        np.random.seed(r_seed)

    # define width of sidelengths of support of observed
    uncert_rect_size = args.uncert_rect_size
    # regular-grid discretization of sets (set-based approach):
    # cpd = cells per dimension

    # output_probability_set discretization
    cpd_observed = args.observed_cells_per_dim
    if cpd_observed < 1: cpd_observed = 1
    # input_sample_set discretization (if regular)

    # only pay attention to cpd_input if regular sampling has been specified.
    if args.reg:
        cpd_input = args.input_cells_per_dim
        if cpd_input < 1: cpd_input = 49
    else:
        cpd_input = None

    n_mc_points = int(args.mc_points)
    if n_mc_points < 100:
        n_mc_points = None

    # MODEL SELECTION
    if args.lam1 < 0 or args.lam1 > 1:
        lam1 = 0.5

    if args.lam2 < 0 or args.lam2 > 1:
        lam2 = 0.5

    refParam = np.array([args.lam1, args.lam2])
    model_choice = args.model
    # default domain
    min_val, max_val = 0, 1
    # TODO: add options to set it: need to take in list, potentially
    if model_choice == 'skew':
        # can be list for higher-dimensional outputs.
        skew = args.skew
        if skew < 1:
            raise ValueError("Skewness must be greater than 1.")
        myModel = makeSkewModel(skew)
    elif model_choice == 'decay':
        # times to evaluate define the QoI map

        # error-handling for times.
        if args.t0 > 0:
            t0 = args.t0
        else:
            raise ValueError("t0<0.")

        if args.t1 > 0:
            t1 = args.t1
        else:
            raise ValueError("t1<0.")

        if args.t1 <= t0:
            raise ValueError("t1<t0.")
        else:
            t1 = args.t1

        eval_times = [t0, t1]
        myModel = makeDecayModel(eval_times)
        model_choice += ' T=[%s-%s]'%(t0, t1)
    elif model_choice == 'random':
        A = np.random.randn(outputDim,inputDim)
        myModel = makeMatrixModel(A)
    elif model_choice == 'diagonal':
        print("Using `t0/t1` as diagonal entries for operator.")
        diag = [args.t0, args.t1]
        D = np.diag(diag)
        myModel = makeMatrixModel(D)
    elif model_choice == 'heatrod':
        print("Using `t0/t1` for thermometer locations")
        assert args.t0 < 1 and args.t0 > 0
        assert args.t1 < 1 and args.t1 > 0
        myModel = make1DHeatModel([args.t0, args.t1])
        min_val, max_val = 0.01, 0.2
        if np.min(refParam) < min_val or np.max(refParam) > max_val:
            print("Reference parameter passed out of range. Mapping to interval.")
            refParam = refParam*(max_val - min_val) + min_val
    # ADD NEW MODELS BELOW HERE with `elif`
    else:
        model_choice = 'identity'
        I = np.eye(inputDim)
        myModel = makeMatrixModel(I)

    if not args.set and not args.sample:  # if both false, set both to true.
        args.set, args.sample = True, True
        print("Solving using both methods.")
        disc, disc_set, disc_samp = comparison_wrapper(model=myModel,
                                                   num_samples=numSamples,
                                                   input_dim=inputDim,
                                                   param_ref=refParam,
                                                   rect_size=uncert_rect_size,
                                                   cpd_observed=cpd_observed,
                                                   input_cpd=cpd_input,
                                                   n_mc_points=n_mc_points,
                                                   min_val=min_val,
                                                   max_val=max_val)

    else:  # only do one or the other.
        # Create baseline discretization
        disc = baseline_discretization(model=myModel,
                                       num_samples=numSamples,
                                       input_dim=inputDim,
                                       param_ref=refParam,
                                       input_cpd=cpd_input,
                                       n_mc_points=n_mc_points,
                                       min_val=min_val,
                                       max_val=max_val)

        if args.sample:
            print("Solving only with sample-based approach.")
            # Set up sample-based approach
            disc_samp = solve_sample_based(discretization=disc,
                                         rect_size=uncert_rect_size)
        elif args.set:
            print("Solving only with set-based approach.")
            # Set up set-based approach
            disc_set = solve_set_based(discretization=disc,
                                         rect_size=uncert_rect_size,
                                         obs_cpd=cpd_observed)

    ### STEP 4 ###
    # plot results
    if not args.noplot:
        save_pdf = args.pdf
        figLabel = args.figlabel
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from plot_examples import plot_2d
        plt.rcParams['font.size'] = args.fontsize
        plt.rcParams['figure.figsize'] = args.figsize, args.figsize  # square ratio

        ### MISC ###

        Qref = disc.get_output().get_reference_value()
        print('Reference Value:', refParam, 'maps to', Qref)


        ### ACTUAL PLOTTING CODE ###

        nbins = 50
        # xmn, xmx = 0.25, 0.75
        # ymn, ymx = 0.25, 0.75
        xmn, xmx = min_val, max_val
        ymn, ymx = min_val, max_val
        xi, yi = np.mgrid[xmn:xmx:nbins*1j, ymn:ymx:nbins*1j]

        if args.title is None:
            model_title = model_choice.capitalize() + ' Model'
        else:
            model_title = args.title

        numLevels = args.numlevels
        if numLevels <2: numLevels = 2

        show_prev = args.show
        # label keyword defaults to approx
        if args.set:
            print("\tPlotting set-based.")
            plot_2d(xi, yi, disc_set, num_levels=numLevels,
                    label=figLabel, annotate='set',
                    title=model_title, pdf=save_pdf, preview=show_prev)

        if args.sample:
            print("\tPlotting sample-based.")
            plot_2d(xi, yi, disc_samp, num_levels=numLevels,
                    label=figLabel, annotate='sample',
                    title=model_title, pdf=save_pdf,
                    eval=eval, preview=show_prev)


    print("Done.")
