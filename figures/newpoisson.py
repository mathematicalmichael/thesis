import dolfin as fin
# from fenicstools.Probe import Probes


def pcwExpr(u,n,d=1):
    """
    Takes an Expression `u` (on unit domain) and returns the string
    for another expression based on evaluating a piecewise-linear approximation.
    """
    dx=1/(n+1)
    intervals = [i*dx for i in range(n+2)]
    node_values = [u(i) for i in intervals]
    return pcwInt(intervals, node_values, d)

       
def pcwGFun(u,d=1):
    """
    Takes an iterable `u` with y-values (on interior of equispaced unit domain)
    and returns the string for an expression
    based on evaluating a piecewise-linear approximation through these points.
    """
    n = len(u)
    dx = 1/(n+1)
    intervals = [i*dx for i in range(n+2)]
    node_values = [0] + list(u) + [1]
    return pcwInt(intervals, node_values, d)

def pcwInt(xvals, yvals, d=1):
    s = ''
    for i in range(1,len(xvals)):
        start = xvals[i-1]
        end = xvals[i]
        diff = start-end
        s += f' ((x[{d}] >= {start}) && (x[{d}] < {end}))*'
        s += f'({yvals[i-1]}*((x[{d}]-{end})/{diff}) + (1 - ((x[{d}]-{end})/{diff}))*{yvals[i]} ) +'
    return s[1:-1]

    

def poisson(gamma, mesh=None, nx=36, ny=36, width=1):
    """
    `gamma` is scaling parameter for left boundary condition
    `n_x` and `n_y` are the number of elements for the horizontal/vertical axes of the mesh
    """
    # Create mesh and define function space
    if mesh is None: mesh = fin.RectangleMesh(fin.Point(0,0), fin.Point(width,1), nx, ny)
    V = fin.FunctionSpace(mesh, "Lagrange", 1)
    u = fin.TrialFunction(V)
    v = fin.TestFunction(V)

    # Define Dirichlet boundary (x = 0 or x = 1)
    def boundary(x):
        return x[1] < fin.DOLFIN_EPS or x[1] > 1.0 - fin.DOLFIN_EPS

    class BoundaryX0(fin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fin.near(x[0], 0, 1E-14)

    class BoundaryX1(fin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fin.near(x[0], width, 1E-14)

    class BoundaryY0(fin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fin.near(x[1], 0, 1E-14)

    class BoundaryY1(fin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fin.near(x[1], 1, 1E-14)

    boundary_markers = fin.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

    bx0 = BoundaryX0()
    bx1 = BoundaryX1()
    by0 = BoundaryY0()
    by1 = BoundaryY1()

    by1.mark(boundary_markers, 0)
    bx1.mark(boundary_markers, 1)
    by0.mark(boundary_markers, 2)
    bx0.mark(boundary_markers, 3)


    ds = fin.Measure('ds', domain=mesh, subdomain_data=boundary_markers)
    # Define boundary condition
    u0 = fin.Constant(0.0)
    if isinstance(gamma, int) or isinstance(gamma, float): # 1-D case
        # u_L = fin.Expression("-gamma*sin((2*kappa+1)*pi*x[1])", gamma=gamma, kappa=0.0, degree=2)
        # the function below will have a min at (2/7, -gamma) by design (scaling factor chosen via calculus)
        u_L = fin.Expression(f"pow(x[1], 2) * pow(x[1] - 1, 5) * gamma", gamma=gamma*823543/12500, degree=3)
    elif len(gamma) == 1:
        u_L = fin.Expression("-gamma*sin((2*kappa+1)*pi*x[1])", gamma=gamma[0], kappa=0.0, degree=2)
    else: # Higher-D case
#         if gamma is None:
#             u_L = fin.Expression(highDstr(gamma), gamma=3.0, kappa=0.0, degree=2)
#             u_L = pcwExpr(u,n=10,d=1)
        u_L = fin.Expression(pcwGFun(gamma, d=1), degree=1)
    
    u_R = fin.Constant(0)


    # starting from top, going clockwise
    boundary_conditions = {
    0: {'Dirichlet': u0},
    1: {'Neumann': u_R},
    2: {'Dirichlet': u0},
    3: {'Neumann': u_L}
    }

    bcs = []
    for i in boundary_conditions:
        if 'Dirichlet' in boundary_conditions[i]:
            bc = fin.DirichletBC(V,
                             boundary_conditions[i]['Dirichlet'],
                             boundary_markers, i)
            bcs.append(bc)

    integrals_N = []
    for i in boundary_conditions:
        if 'Neumann' in boundary_conditions[i]:
            if boundary_conditions[i]['Neumann'] != 0:
                g = boundary_conditions[i]['Neumann']
                integrals_N.append(g*v*ds(i))

    bc = fin.DirichletBC(V, u0, boundary)

    # Define variational problem
    f = fin.Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
    # g = Expression("sin(2*pi*x[1])", degree=2)
    a = fin.inner(fin.grad(u), fin.grad(v))*fin.dx
    L = f*v*fin.dx + sum(integrals_N)

    # Compute solution
    u = fin.Function(V)
    fin.solve(a == L, u, bc) 
    return u


def poisson_sensor_model(sensors, gamma, nx, ny, mesh=None):
    """
    Convenience function wrapper to just return a qoi given a parameter.
    """
    assert sensors.shape[1] == 2, "pass with shape (num_sensors, 2)"
    num_sensors = sensors.shape[0]
    
    u = poisson(gamma, mesh, nx=nx, ny=ny)
    q = [u(xi,yi) for xi,yi in sensors]
#     q = []
#     V = u.function_space()
#     probes = Probes(sensors.flatten(), V)
# #     probes(interpolate(u,V))
#     probes(u)
#     q = probes.array()

    return q


if __name__=='__main__':

    u = poisson(gamma=10, width=1)
    fin.plot(u, interactive=True)

    # Save solution in VTK format
    # file = File("poisson.pvd")
    # file << u

    show_plot = True
    if show_plot:
        import numpy as np
#         import matplotlib.pyplot as plt
#         x=np.linspace(0,1,50)
#         y=np.linspace(0,1,50)

#         for k in [0, 0.25, 0.5, 0.75, 1]:
#             plt.plot(x,[u(xi,k) for xi in x], lw=4*k+1, label=k)
#             plt.title('Horizontal Slices')
#         plt.legend()
#         plt.show()

#         for k in width*np.array([0, 0.25, 0.5, 0.75, 1]):
#             plt.plot(y,[u(k,yi) for yi in y], lw=4*k+1, label=k)
#             plt.title('Vertical Slices')
#         plt.legend()
#         plt.show()


        plt.show()

