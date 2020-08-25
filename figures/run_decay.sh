#!/bin/sh
python mud_problem.py --prefix ode -n 20 \
	-r 0.05 -r 0.1 -r 0.125 -r 0.25 -r 0.5 -r 1 \
	-t 0.1 -t 5E-2 -t 1E-2 -t 5E-3
