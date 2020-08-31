#!/bin/sh
NUM_SAMPLES=1000
FILE_PREFIX=res
for dim in 1 2 5 ; do
    echo "Running for Dim=${dim}. Saving to ${FILE_PREFIX}${dim}"
    python pp.py -n $NUM_SAMPLES -d u -i ${dim} -o ${FILE_PREFIX}${dim}u 
done