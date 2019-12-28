import numpy as np


lam_true = 0.5
def makeDecayModel(t):
    """
    Description

    Parameters
    ----------
    t : list-like
        Times (s) at which to evaluate the exponential model

    Returns
    -------
    model : function
        A function that evaluates the exponential model at times `t`.
        It takes as input an (samples, dim) array representing
        rate and initial condition, respectively.
        
    """
    def model(lam = np.array([[lam_true]])):
        """
        """
        # support passing different types
        if isinstance(lam, list) or isinstance(lam, tuple):
            lam = np.array(lam)
        if isinstance(lam, np.ndarray):
            if lam.ndim == 1:
                lam = lam.reshape(1,-1)
        # extract rate from first column
        rate = lam[:,0].reshape(-1,1)
        # if only 1-D array supplied, assume fixed initial condition
        try:
            initial_cond = lam[:,1].reshape(-1,1)
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
        It takes as input an (num_samples, input_dim) array representing
        rate and initial condition, respectively.
        
    """
    
    def model(input_samples):
        """
        Description

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
