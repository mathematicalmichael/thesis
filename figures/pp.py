#!/usr/env/bin python
import os
# os.environ['OMP_NUM_THREADS'] = '1'
from newpoisson import poisson
import numpy as np
from fenics import set_log_level, File, RectangleMesh, Point

mesh = RectangleMesh(Point(0,0), Point(1,1), 36, 36)
# comm = mesh.mpi_comm()

set_log_level(40) # ERROR=40
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

if __name__=='__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Poisson Problem")
    parser.add_argument('-n', '--num', default = 10, type=int,
                       help="Number of samples")
    parser.add_argument('-o', '--outfile', default='results',
                       help="Output filename (no extension)")
    parser.add_argument('-i', '--input-dim', default=1, type=int)
    parser.add_argument('-d', '--dist', default='u', help='Distribution. `n` (normal), `u` (uniform, default)')
    args = parser.parse_args()

    num_samples = args.num
    dist = args.dist
    outfile = args.outfile.replace('.pkl','')
    inputdim = args.input_dim
    if inputdim == 1: # U[1,5]
        randsamples = 1 + 4*np.random.rand(num_samples)
    else: # N(0,1)
        if dist == 'n':
            randsamples = np.random.randn(num_samples, inputdim)
        elif dist == 'u':
            randsamples = -4*np.random.rand(num_samples, inputdim)
        else:
            raise ValueError("Improper distribution choice, use `n` (normal), `u` (uniform)")

    sample_seed_list = list(zip(range(num_samples), randsamples))

    def wrapper(sample, outfile):
        g=sample[1]
        u = poisson(gamma=g, mesh=mesh)
        # Save solution 
        fname = f"{outfile}-data/poisson-{int(sample[0]):06d}.xml"
        File(fname, 'w') << u
        return {int(sample[0]): {'u': fname, 'gamma': sample[1]}}


    results = []
    for sample in sample_seed_list:
        r = wrapper(sample, outfile)
        results.append(r)
#         print(results)
    import pickle
    pickle.dump(results, open(f'{outfile}.pkl','wb'))
