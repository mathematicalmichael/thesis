from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.rank
num_procs = comm.Get_size()

if rank == 0:
    num_trials = 2
    trial_seed_list = list(zip(range(num_trials),0.03 + 0.04*np.random.rand(num_trials)))
    trial_chunk = np.array_split(trial_seed_list, num_procs)
else:
    trial_chunk = None

trial_chunk = comm.bcast(trial_chunk, root=0)
print({rank: trial_chunk[rank]})


#####################################
import os
# os.environ['OMP_NUM_THREADS'] = '5'
# os.environ['OMP_THREAD_LIMIT'] = '5'
# !echo $OMP_NUM_THREADS
# !export OMP_DYNAMIC=FALSE

# from progressbar import progressbar as progbar
from fenics import *
import mshr
import numpy as np
import dolfin
import numpy as np
# from mpi4py import MPI as pyMPI
# comm = MPI.comm_world
def channel_model(T=3.0,
                  dt=1/2000,
                  mu=0.001,
                  rho=1,
                  center=(0.25, 0.25),
                  radius=0.05,
                  mesh_res=24,
                  circ_seg=24,
                  amp=24.0,
                  num_time_steps_qoi = 300,
                  folder_name='results/navier_stokes_cylinder_mesh'
                 ):
    
#     T = 3.0            # final time
#     dt = 1/1000 # time step size

#     mu = 0.001         # dynamic viscosity
#     rho = 1            # density
#     c_x, c_y, c_r = 0.25, 0.25, 0.05
    c_x, c_y = center
    c_r = radius
    
    # Create mesh
    wall_ht = 0.5
    wall_wd = 2.0

    num_steps = int(T/dt)   # number of time steps
    channel = mshr.Rectangle(Point(0, 0), Point(wall_wd, wall_ht))
    cylinder = mshr.Circle(Point(c_x, c_y), c_r, segments=circ_seg)
    domain = channel - cylinder# - cylinder2
    mesh = mshr.generate_mesh(domain, mesh_res)

    # Define function spaces
    V = VectorFunctionSpace(mesh, 'P', 2)
    Q = FunctionSpace(mesh, 'P', 1)

    # Define boundaries
    inflow   = 'near(x[0], 0)'
    outflow  = 'near(x[0], %f)'%wall_wd
    walls    = 'near(x[1], 0) || near(x[1], %f)'%wall_ht
    cylinder = 'pow(x[0] - %f, 2) + pow(x[1] - %f, 2) <= pow(%f, 2)'%(c_x,
                                                                   c_y,
                                                                   c_r + 1E-4
                                                                   )

    # Define inflow profile
    # quadratic function with roots on top/bottom boundary
#     amp = 24.0
    inflow_profile = ('%f*x[1]*(%f - x[1])'%(amp, wall_ht), '0')

    # Define boundary conditions
    bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
    bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
    bcu_cylinder = DirichletBC(V, Constant((0, 0)), cylinder)
    bcp_outflow = DirichletBC(Q, Constant(0), outflow)

    bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
    bcp = [bcp_outflow]

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)

    # Define functions for solutions at previous and current time steps
    u_n = Function(V)
    u_  = Function(V)
    p_n = Function(Q)
    p_  = Function(Q)

    # Define expressions used in variational forms
    U  = 0.5*(u_n + u)
    n  = FacetNormal(mesh)
    f  = Constant((0, 0))
    k  = Constant(dt)
    mu = Constant(mu)
    rho = Constant(rho)

    # Define symmetric gradient
    def epsilon(u):
        return sym(nabla_grad(u))

    # Define stress tensor
    def sigma(u, p):
        return 2*mu*epsilon(u) - p*Identity(len(u))

    ###########

    # Define variational problem for step 1
    F1 = rho*dot((u - u_n) / k, v)*dx \
       + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
       + inner(sigma(U, p_n), epsilon(v))*dx \
       + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
       - dot(f, v)*dx
    a1 = lhs(F1)
    L1 = rhs(F1)

    # Define variational problem for step 2
    a2 = dot(nabla_grad(p), nabla_grad(q))*dx
    L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

    # Define variational problem for step 3
    a3 = dot(u, v)*dx
    L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

    # Assemble matrices
    A1 = assemble(a1)
    A1_orig = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)

    # Apply boundary conditions to matrices
    [bc.apply(A1) for bc in bcu]
    [bc.apply(A2) for bc in bcp]

    # TODO: HAVE NAMING CONVENTION TO AVOID FILE COLLISIONS
    # Create XDMF files for visualization output
    xdmffile_u = XDMFFile('%s/velocity.xdmf'%folder_name)
    xdmffile_p = XDMFFile('%s/pressure.xdmf'%folder_name)

    # Create time series (for use in reaction_system.py)
    timeseries_u = TimeSeries('%s/velocity_series'%folder_name)
    timeseries_p = TimeSeries('%s/pressure_series'%folder_name)

    # Save mesh to file (for use in reaction_system.py)
#     File('%s/cylinder.xml.gz'%folder_name) << mesh

    # Create progress bar
    # progress = Progress('Time-stepping', num_steps)
    set_log_level(10)

    # Time-stepping
    t = 0
#     pbar = progbar.ProgressBar().start()
    for n in range(num_steps):
        # Update current time
        t += dt

        # Step 1: Tentative velocity step
        b1 = assemble(L1)
        [bc.apply(b1) for bc in bcu]
        solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')

        # Step 2: Pressure correction step
        b2 = assemble(L2)
    #     [bc.apply(b2) for bc in bcp]

        solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')

        # Step 3: Velocity correction step
        b3 = assemble(L3)
        solve(A3, u_.vector(), b3, 'cg', 'sor')

        # Save solution to file (XDMF/HDF5)
        xdmffile_u.write(u_, t)
        xdmffile_p.write(p_, t)

        # Save nodal values to file
        timeseries_u.store(u_.vector(), t)
        timeseries_p.store(p_.vector(), t)

        # Update previous solution
        u_n.assign(u_)
        p_n.assign(p_)

#     num_time_steps_qoi = 300
    coords = p_.function_space().mesh().coordinates()
    locs = [ [2, 0.1], [2, 0.2], [2, 0.3], [2, 0.4] ]
    time_vec = np.linspace(1,T,num_time_steps_qoi)
    num_locs = len(locs)
    locs = np.array(locs)
    locs[:,0] = wall_wd
    p = np.empty((num_locs, num_time_steps_qoi))
    for i, time in enumerate(time_vec):
        timeseries_p.retrieve(p_.vector(), time)
        pv = p_.compute_vertex_values()
        for j in range(num_locs):
            p[j, i-1] = p_(locs[j,0], locs[j,1])
    return (p, (u_, timeseries_u), (p_, timeseries_p))

def wrapper(i):
    # absolute directory
    folder_name = 'results/navier_stokes_cylinder_mesh{}'.format(i[0])
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    try:
        p, uu, pp = channel_model(center=(0.25, 0.25), radius=i[1],
                             folder_name=folder_name)
    except:
        p = None
#     return {i[0]: {'qoi': p, 'u': uu, 'p': pp}}
    return {i[0]: {'qoi': p, 'r': i[1]}}

results = []
for trial in trial_chunk[rank]:
    results.append(wrapper(trial))
    
results = comm.gather('results').get()
print(results)
qoi = np.array([l[i]['qoi'].ravel() for i,l in enumerate(results_gather)])
radii = np.array([l[i]['r'] for i,l in enumerate(results_gather)])
np.save('results_gathered_qoi.npy', qoi)
np.save('results_gathered_lam.npy', radii)
