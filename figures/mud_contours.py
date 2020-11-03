from mud.funs import mud_sol, map_sol
from mud.plot import make_2d_unit_mesh
from mud.norm import full_functional, norm_input, norm_data, norm_predicted
from mud.util import transform_setup, transform_linear_map, null_space

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


N, r = 250, 1
X, Y, XX = make_2d_unit_mesh(N, r)
inputs = XX

def plot_full(A=np.array([[1, 1]]), b=np.zeros((1,1)),
              cov_11=0, cov_01=1,
              initial_mean=np.zeros((1,2)), wt=1, wp=1,
              show_full=True, show_data=True, show_num=False,
              obs_std=1, figname='latest_figure.png', fsize=48):
    
    std_of_data = [obs_std]
    obs_cov = np.diag(std_of_data)
    observed_data_mean = np.array([[1]]) + np.random.randn()*obs_std*0
    initial_cov = np.array([[1, cov_01],[cov_01, cov_11]])
    
    #     assert np.linalg.det(initial_cov)>0
    assert np.all(np.linalg.eigvals(initial_cov) > 0)
#     uu,ss,ww = np.linalg.svd(initial_cov)
    # change of basis based on SVD of initial covariance
#     inputs = (np.linalg.inv(ww)@XX.T).T
#     assert np.linalg.norm(uu - np.linalg.inv(uu)) + np.linalg.norm(ww-uu) + np.linalg.norm(uu-uu.T) < 1E-8
    
#     A_, b_ = transform_linear_map(A, data_list, std_of_data)
#     ls = np.linalg.pinv(A_)@-b_.T

    z = full_functional(A, XX, b, initial_mean, initial_cov, observed_data_mean, observed_cov=obs_cov)
    zp = norm_predicted(A, XX, initial_mean, initial_cov)
    zi = norm_input(XX, initial_mean, initial_cov)
    zd = norm_data(A, XX, b, observed_data_mean, observed_cov=obs_cov)
    # sanity check that all arguments passed correctly:
    assert np.linalg.norm(z - zi - zd + zp) < 1E-8

    # plotting contours
    z = (wt*zi + zd - wp*zp)
    mud_a = np.argmin(z)
    map_a = np.argmin(wt*zi + zd)

    inputs = XX
    mud_pt = inputs[mud_a,:]
    map_pt = inputs[map_a,:]
    
    msize = 500
    ls = np.linalg.pinv(A)@observed_data_mean.T
    if show_data: 
        plt.contour(inputs[:,0].reshape(N,N),
                    inputs[:,1].reshape(N,N),
                    (zd).reshape(N,N), 25,
                    cmap=cm.viridis, alpha=0.5,vmin=0, vmax=4)
        plt.axis('equal')

        
        
        s = np.linspace(-2*r,2*r,10)

        if A.shape[0] < A.shape[1]:
            # nullspace through least-squares
            null_line = null_space(A)*s + ls.reshape(-1,1)
            plt.plot(null_line[0,:],null_line[1,:],
                     label='Solution Contour',
                     lw=2, color='xkcd:red')
            if not show_full:
                plt.annotate('Solution Contour', (0.1, 0.9),
                             fontsize=fsize, backgroundcolor="w")

    if show_full:
        plt.contour(inputs[:,0].reshape(N,N),
                    inputs[:,1].reshape(N,N),
                    z.reshape(N,N), 50,
                    cmap=cm.viridis, alpha=1.0)
        plt.axis('equal')
    elif wt+wp>0:
            plt.contour(inputs[:,0].reshape(N,N),
                        inputs[:,1].reshape(N,N),
                        (wt*zi - wp*zp).reshape(N,N), 100,
                        cmap=cm.viridis, alpha=0.25)
            plt.axis('equal')

    
    
    if wt+wp>0:
        
        plt.scatter(initial_mean[0], initial_mean[1],
                    label='Initial Mean',
                    color='k', s=msize) 
        if not show_full: plt.annotate('Initial Mean',
                    (initial_mean[0]+0.001*fsize, initial_mean[1]-0.001*fsize),
                    fontsize=fsize, backgroundcolor="w")

        if show_full: 
            # scatter and line from origin to least squares
            plt.scatter(ls[0], ls[1],
                        label='Least Squares',
                        color='xkcd:blue',
                        marker='d', s=msize, zorder=10)
            plt.plot([0, ls[0]], [0, ls[1]],
                        color='xkcd:blue',
                        marker='d', lw=1, zorder=10)
            plt.annotate('Least Squares',
                        (ls[0]-0.001*fsize, ls[1]+0.001*fsize),
                        fontsize=fsize, backgroundcolor="w")
            
            if show_num: # numerical solutions
                if wp>0:
                    plt.scatter(mud_pt[0], mud_pt[1],
                                label='min: Tk - Un', color='xkcd:sky blue',
                                marker='o', s=3*msize, zorder=10)
                if (wt>0 and wp!=1):
                    plt.scatter(map_pt[0], map_pt[1],
                                label='min: Tk', color='xkcd:blue',
                                marker='o', s=3*msize, zorder=10)
            if (wt>0 and wp!=1): # analytical MAP point
                map_pt_eq = map_sol(A, b, observed_data_mean,
                                    initial_mean, initial_cov, 
                                    data_cov=obs_cov, w=wt)
                plt.scatter(map_pt_eq[0], map_pt_eq[1],
                            label='MAP', color='xkcd:orange',
                            marker='x', s=msize, lw=10, zorder=10)
                plt.annotate('MAP',
                             (map_pt_eq[0]+0.001*fsize, map_pt_eq[1]-0.001*fsize),
                             fontsize=fsize, backgroundcolor="w")
            if wp>0: # analytical MUD point
                mud_pt_eq = mud_sol(A, b, observed_data_mean,
                                    initial_mean, initial_cov)
                plt.scatter(mud_pt_eq[0], mud_pt_eq[1],
                            label='MUD', color='xkcd:brown',
                            marker='*', s=2*msize, lw=5, zorder=10)
                plt.annotate('MUD',
                             (mud_pt_eq[0]+0.001*fsize, mud_pt_eq[1]-0.001*fsize),
                             fontsize=fsize, backgroundcolor="w") 

        if A.shape[0] < A.shape[1]:
            # want orthogonal nullspace, function gives one that is already normalized
            v = null_space(A@initial_cov) 
            v = v[::-1]  # in 2D, we can just swap entries and put a negative sign in front of one
            v[0] = - v[0]
            
            if show_full and wp>0:
                # grid search to find upper/lower bounds of line being drawn.
                # importance is the direction, image is nicer with a proper origin/termination
                s = np.linspace(-1, 1, 1000)
                new_line = (v.reshape(-1,1)*s) + initial_mean.reshape(-1,1)
                mx = np.argmin(np.linalg.norm(new_line - initial_mean.reshape(-1,1), axis=0))
                mn = np.argmin(np.linalg.norm(new_line - mud_pt_eq.reshape(-1,1), axis=0))
                plt.plot(new_line[0,mn:mx], new_line[1,mn:mx], lw=1, label='projection line', c='k')
        elif show_full:
            plt.plot([initial_mean[0], ls[0]],
                     [initial_mean[1], ls[1]],
                     lw=1, label='Projection Line', c='k')

    #     print(p)
    
    plt.axis('square')
    plt.axis([0, r, 0, r])
#     plt.legend(fontsize=fsize)
    plt.xticks(fontsize=0.75*fsize)
    plt.yticks(fontsize=0.75*fsize)
    plt.tight_layout()
    plt.savefig(figname, dpi=300)

#     plt.title('Predicted Covariance: {}'.format((A@initial_cov@A.T).ravel() ))
    plt.show()