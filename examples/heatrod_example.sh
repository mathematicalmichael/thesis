#!/bin/sh
FONTSIZE=16
FIGSIZE=5
MODEL='heatrod'
SEED=21
# [0.25, 0.51, 0.67, 0.98]
LOC_1=0.25
LOC_2=0.51
#python solve_problem.py -m $MODEL --plot --fontsize $FONTSIZE --figsize $FIGSIZE $@

# low number of samples for input space
# sample-based
python solve_problem.py -m $MODEL --fontsize $FONTSIZE --figsize $FIGSIZE --sample --seed $SEED -n 100 --t0 $LOC_1 --t1 $LOC_2 $@
# set-based
python solve_problem.py -m $MODEL --fontsize $FONTSIZE --figsize $FIGSIZE --set --seed $SEED -o 2 -n 100 --t0 $LOC_1 --t1 $LOC_2 $@
python solve_problem.py -m $MODEL --fontsize $FONTSIZE --figsize $FIGSIZE --set --seed $SEED --mc_points 10000 -o 2 -n 100 --t0 $LOC_1 --t1 $LOC_2 $@
