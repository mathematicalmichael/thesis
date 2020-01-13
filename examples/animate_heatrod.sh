#!/bin/sh
mkdir -p heatrod_time/
rm heatrod_time/*.png
python test_heatrod.py
convert heatrod_time/*.png test.gif
imgcat test.gif
rm heatrod_time/*.png
