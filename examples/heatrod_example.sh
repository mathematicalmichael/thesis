#!/bin/sh

### FIXED PARAMS ACROSS EXAMPLES ###

FONTSIZE=16
FIGSIZE=5
MODEL='heatrod'
SEED=21
NUM_MC_SAMPLES=10000

# just paying attention to characertistic-function densities.
M=1

# defaults
NUM_SAMPLES=50
LAM_1=0.15
LAM_2=0.05

# masters work used 3x3 grid in the interior
# of (0.01, 0.2)^2 and showed bottom-left + top-center


# avg skewness 1.08
TLOC0=0.25
TLOC1=0.51

# avg skewness 1.56
# TLOC0=0.67
# TLOC1=0.98
