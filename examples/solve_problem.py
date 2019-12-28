import numpy as np
from helpers import baseline_discretization, solve_set_based, solve_sample_based, comparison_wrapper
from plot_examples import plot_2d

# can choose model from here
from models import makeMatrixModel as make_model
# from models import makeDecayModel as make_model

np.random.seed(11)

### STEP 0 ###
# Define the problem structure/assumptions
inputDim = 2
numSamples = 1E3

# define reference parameter
refParam = np.array([0.5]*inputDim)

# define width of sidelengths of support of observed
uncert_rect_size = 0.2
# regular-grid discretization of set (set-based approach): 
# cpd = cells per dimension
cpd_observed = 1
cpd_input = 49

# MC assumption on volumes
MC_assumption = False
# if true, set n_mc_points, and pass it to baseline

# define matrix for linear model, instantiate sampler
# I = np.eye(inputDim)
outputDim = 2
I = np.random.rand(outputDim,inputDim)*2 - 1
myModel = make_model(I)
# myModel = make_model([1, 2])

### STEP 1 ###
# Create baseline discretization



# disc = baseline_discretization(model=myModel,
#                                num_samples=numSamples,
#                                input_dim=inputDim,
#                                param_ref=refParam,
#                                input_cpd=None,
#                                n_mc_points=None)



# ### STEP 2 ###
# # Set up set-based approach
# disc_set = solve_set_based(discretization=disc,
#                              rect_size=uncert_rect_size,
#                              obs_cpd=cpd_observed)


# ### STEP 3 ### 
# # Set up sample-based approach

# disc_samp = solve_sample_based(discretization=disc,
#                              rect_size=uncert_rect_size)


disc, disc_set, disc_samp = comparison_wrapper(model=myModel,
                                               num_samples=numSamples,
                                               input_dim=inputDim,
                                               param_ref=refParam,
                                               rect_size=uncert_rect_size,
                                               cpd_observed=cpd_observed,
                                               input_cpd=None,
                                               n_mc_points=None)

### STEP 4 ###
# plot results
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.rcParams['font.size'] = 16
plt.rcParams['figure.figsize'] = 5,5



### MISC ###

Qref = disc.get_output().get_reference_value()
print('Reference Value:', refParam, 'maps to', Qref)


### ACTUAL PLOTTING CODE ### 

nbins = 50
# xmn, xmx = 0.25, 0.75
# ymn, ymx = 0.25, 0.75
xmn, xmx = 0, 1
ymn, ymx = 0, 1
xi, yi = np.mgrid[xmn:xmx:nbins*1j, ymn:ymx:nbins*1j]

plot_2d(xi, yi, disc_set, annotate='set')

plot_2d(xi, yi, disc_samp, annotate='sample')

