#!/bin/sh
python mud_problem.py --test --prefix pde -n 2 \
	-m 25 -m 50 -m 250 \
	-t 0.1 -t 1E-2 -t 1E-3 
