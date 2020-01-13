#!/bin/sh
FONTSIZE=16
FIGSIZE=5
MODEL='heatrod'
SEED=21
NUM_SAMPLES=500
NUM_MC_SAMPLES=10000
M=1

# [0.25, 0.51, 0.67, 0.98]
LOC_1=0.25
LOC_2=0.51
# LOC_1=0.67
# LOC_2=0.98

LAM_1=0.15
LAM_2=0.05

#python solve_problem.py -m $MODEL --plot --fontsize $FONTSIZE --figsize $FIGSIZE $@

# low number of samples for input space
# sample-based
python solve_problem.py -m $MODEL --fontsize $FONTSIZE --figsize $FIGSIZE \
  --sample --seed $SEED -n $NUM_SAMPLES \
  --t0 $LOC_1 --t1 $LOC_2 \
  --lam1 $LAM_1 --lam2 $LAM_2 $@

# set-based
python solve_problem.py -m $MODEL --fontsize $FONTSIZE --figsize $FIGSIZE \
  --set --seed $SEED -o $M -n $NUM_SAMPLES \
  --t0 $LOC_1 --t1 $LOC_2 \
  --lam1 $LAM_1 --lam2 $LAM_2 $@

# with volume emulation
python solve_problem.py -m $MODEL --fontsize $FONTSIZE --figsize $FIGSIZE \
  --set --seed $SEED -o $M -n $NUM_SAMPLES \
  --mc_points $NUM_MC_SAMPLES  \
  --t0 $LOC_1 --t1 $LOC_2 \
  --lam1 $LAM_1 --lam2 $LAM_2 $@
