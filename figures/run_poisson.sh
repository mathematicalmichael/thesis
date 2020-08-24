#!/bin/sh
python mud_problem.py -n 20 \
	-m 5 -m 10 -m 25 -m 50 -m 100 -m 250 -m 500 -m 1000 \
	-t 0.1 -t 5E-2 -t 1E-2 -t 5E-3
