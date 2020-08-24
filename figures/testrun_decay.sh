#!/bin/sh
python mud_problem.py --test --prefix ode -n 2 \
	-r 0.25 -r 0.5 -r 1 \
	-t 0.1 -t 1E-2 -t 1E-3 
