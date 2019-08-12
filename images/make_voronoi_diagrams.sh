#!/bin/bash
echo -e "Creating Voronoi Diagrams \n"
python voronoi_unit_domain.py --num 25 --seed 0
python voronoi_unit_domain.py --num 25 --seed 0 --nolabel
python voronoi_unit_domain.py --num 25 --seed 10
python voronoi_unit_domain.py --num 25 --seed 10 --nolabel
python voronoi_unit_domain.py --num 500 --seed 50
