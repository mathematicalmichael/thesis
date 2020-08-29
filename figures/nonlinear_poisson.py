import fenics as fe

# diffusion term: function of u, or constant, or something nonlinear.
def q(u):
    return fe.Constant(1.0)

n = 32
mesh = fe.UnitSquareMesh(n,n)

V = fe.FunctionSpace(mesh, 'P', 1)
v = fe.TestFunction(V)

class BoundaryX0(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fe.near(x[0], 0, 1E-14)

class BoundaryX1(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fe.near(x[0], 1, 1E-14)

class BoundaryY0(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fe.near(x[1], 0, 1E-14)

class BoundaryY1(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fe.near(x[1], 1, 1E-14)

boundary_markers = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

bx0 = BoundaryX0()
bx1 = BoundaryX1()
by0 = BoundaryY0()
by1 = BoundaryY1()


bx0.mark(boundary_markers, 3)
bx1.mark(boundary_markers, 1)
by0.mark(boundary_markers, 2)
by1.mark(boundary_markers, 0)


ds = fe.Measure('ds', domain=mesh, subdomain_data=boundary_markers)
# domain_markers # used to mark subdomains with CellFunctions
# dx = fe.Measure('dx', domain=mesh, subdomain_data=domain_markers)

u_ = fe.Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
u_sin = fe.Expression('sin(3.0*pi*x[1])', degree=2)
u_T= fe.Constant(1.0)
u_B = fe.Constant(1.0)

# u_L = fe.Expression('1 + 2*x[1]*x[1]', degree=2)
u_R = fe.Constant(0)
u_L = fe.Expression('1.0 + kappa*x[1]*(1-x[1])', kappa=10.0, degree=2)

# 0 is top. 1 is right, 2 is bottom, 3 is left.
boundary_conditions = {
0: {'Dirichlet': u_T},
1: {'Neumann': u_L},
2: {'Dirichlet': u_B},
3: {'Neumann': u_L}
}


bcs = []
for i in boundary_conditions:
    if 'Dirichlet' in boundary_conditions[i]:
        bc = fe.DirichletBC(V,
                         boundary_conditions[i]['Dirichlet'],
                         boundary_markers, i)
        bcs.append(bc)

integrals_N = []
for i in boundary_conditions:
    if 'Neumann' in boundary_conditions[i]:
        if boundary_conditions[i]['Neumann'] != 0:
            g = boundary_conditions[i]['Neumann']
            integrals_N.append(g*v*ds(i))



# Define variational problem
f = fe.Constant(0.0)
# f = fe.Expression('(x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) < pow(0.25,2)', degree=2)



L = f*v*fe.dx - sum(integrals_N)
u = fe.Function(V)
F = q(u)*fe.dot(fe.grad(u), fe.grad(v))*fe.dx - L
# Compute solution
fe.solve(F == 0, u, bcs)


# Plot solution
u.rename('u', 'solution')
fe.plot(u)
# fe.plot(mesh)
import matplotlib.pyplot as plt
plt.show()
# Save solution to file in VTK format
# vtkfile = File('poisson.pvd')
show_plot = True
if show_plot:
    import numpy as np

    x=np.linspace(0,1,n)
    y=np.linspace(0,1,n)

    plt.plot(x,[u(xi,0) for xi in x])
    plt.plot(x,[u(xi,1) for xi in x])
    plt.show()

    plt.plot(x,[u(0,yi) for yi in y])
    plt.plot(x,[u(1,yi) for yi in y])
    plt.show()

    plt.plot(x,[u_L(0,yi) for yi in y])
    plt.plot(x,[u_R(1,yi) for yi in y])
    plt.show()