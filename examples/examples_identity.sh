#!/bin/sh
FONTSIZE=16
FIGSIZE=5
MODEL='identity'
python solve_problem.py -m $MODEL --plot --fontsize $FONTSIZE --figsize $FIGSIZE
