import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
kde = ss.gaussian_kde

num = 1000
dim = 2
vals1 = ss.distributions.norm(loc=[0,0],scale=[0.05,2]).rvs((num//2,dim))
vals2 = ss.distributions.norm(loc=[0,0],scale=[1, 0.05]).rvs((num//2,dim))
vals2 = (np.array([[1, -1], [-1, 1]])@vals2.T).T
vals = np.concatenate((vals1,vals2),axis=0)
std_approx = vals.std(axis=0)
nx, ny = 60,60
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
xv, yv = np.meshgrid(x, y)

b = num**(-1/(dim+4))

P = np.vstack((xv.ravel(),yv.ravel()))
g = kde(vals.T)
gb = kde(vals.T, bw_method=b)
g_eval = g.pdf(P)
gb_eval = gb.pdf(P)
b - g.scotts_factor()
g.scotts_factor() - gb.scotts_factor()
np.linalg.norm(g_eval - gb_eval)

plt.scatter(P[0,:], P[1,:], c=g_eval)
newvals = gb.resample(num).T
plt.scatter(vals[:,0], vals[:,1],s=10,c='w')
plt.scatter(newvals[:,0], newvals[:,1],s=5,c='r')
plt.axis([-1, 1, -1, 1])