import numpy as np


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
                input_samples = input_samples.reshape(1,-1)
        # extract rate from first column
        rate = input_samples[:,0].reshape(-1,1)
        # if only 1-D array supplied, assume fixed initial condition
        try:
            initial_cond = input_samples[:,1].reshape(-1,1)
        except IndexError:
            initial_cond = 0.5

        # compute response at predefined QoI (times)
        response = initial_cond*np.exp(-np.outer(rate, t))
        if response.shape[0] == 1: # single-row evaluation => ravel output
            return response.ravel() # allows support for simpler 1D plotting.
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
        return np.inner(A,input_samples).T
    
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
    Q_map = [ [1.0, 0.0] ] # all map components have the same norm, rect_size to have measures of events equal btwn spaces.
    for skew in skewnesses:
        Q_map.append( [np.sqrt(skew**2 - 1), 1] ) # taken with the first component, this leads to a 2-2 map with skewsness 's'
    Q_map = np.array( Q_map )
    
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
