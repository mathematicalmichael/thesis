#!/bin/sh
echo "Python source and version check:"
which python && python --version
echo -e "Creating Voronoi Diagrams \n"
python voronoi_unit_domain.py --num 25 --seed 0
python voronoi_unit_domain.py --num 25 --seed 0 --nolabel
python voronoi_unit_domain.py --num 25 --seed 10
python voronoi_unit_domain.py --num 25 --seed 10 --nolabel
python voronoi_unit_domain.py --num 500 --seed 50
# move all voronoi images into their own directory
mkdir -p voronoi_diagrams/
mv voronoi_diagram_* voronoi_diagrams/
echo "Done."
