#!/usr/env/bin python
from newpoisson import poisson
import numpy as np
import dolfin as d
comm = d.MPI.comm_world
# comm = MPI.COMM_WORLD
import os
os.environ['OMP_NUM_THREADS'] = '1'
import fenics as fin

fin.set_log_level(40) # ERROR=40


mesh = fin.RectangleMesh(fin.Point(0,0), fin.Point(1,1), 32, 32)
comm = mesh.mpi_comm()
rank = comm.rank
num_procs = comm.Get_size()
print(f"Processor {rank}")

if __name__=='__main__':
    
    if rank == 0:
        import argparse
        
        parser = argparse.ArgumentParser(description="Poisson Problem")
        parser.add_argument('-n', '--num', default = 10, type=int,
                           help="Number of samples")
        parser.add_argument('-o', '--outfile', default='results',
                           help="Output filename (no extension)")
        parser.add_argument('-i', '--input-dim', default=1, type=int)
        parser.add_argument('-d', '--dist', default='n')
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
        sample_chunk = np.array_split(sample_seed_list, num_procs)

    else:
        sample_chunk = None
        sample_seed_list = None
        outfile = None
    
    # BROADCAST
    sample_chunk = comm.bcast(sample_chunk, root=0)
    outfile = comm.bcast(outfile, root=0)
    
    def wrapper(sample, outfile):
        g=sample[1]
        u = poisson(gamma=g, mesh=mesh)
        # Save solution
        fname = f"{outfile}-data/poisson-{int(sample[0]):06d}.xml"
        fin.File(fname) << u
#         f = fin.HDF5File(comm, fname, 'w')
#         f.write(u, 'solution')
#         f.close()
        return {int(sample[0]): {'u': fname, 'gamma': sample[1]}}


    results = []
    for sample in sample_chunk[rank]: # only run samples for this procesor
        r = wrapper(sample, outfile)
#         print(sample, r)
        results.append(r)

    # GATHER
    comm.barrier()
    results = comm.gather(results, root=0)#.get()
    
    # SAVE
    if rank==0:
        print(results)
        import pickle
        pickle.dump(results, open('%s.pkl'%outfile,'wb'))
#         qoi = np.array([l[0][i]['qoi'].ravel() for i,l in enumerate(results)])
#         gamma = np.array([l[0][i]['gamma'] for i,l in enumerate(results)])

#         np.save('results_gathered_qoi.npy', qoi)
#         np.save('results_gathered_lam.npy', gamma)
        print("Done")