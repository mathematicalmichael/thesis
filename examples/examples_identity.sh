#!/bin/sh
FONTSIZE=16
FIGSIZE=5
MODEL='identity'
SEED=21
#python solve_problem.py -m $MODEL --plot --fontsize $FONTSIZE --figsize $FIGSIZE $@

# low number of samples for input space
python solve_problem.py -m $MODEL --plot --fontsize $FONTSIZE --figsize $FIGSIZE --sample --seed $SEED -n 100
# estimating volumes using high-fidelity discretization of output_probability_set leads to uniformity
python solve_problem.py -m $MODEL --plot --fontsize $FONTSIZE --figsize $FIGSIZE --set --seed $SEED -o 2 -n 100
python solve_problem.py -m $MODEL --plot --fontsize $FONTSIZE --figsize $FIGSIZE --set --seed $SEED --mc_points 10000 -o 2 -n 100

