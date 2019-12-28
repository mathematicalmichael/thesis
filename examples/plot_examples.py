from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
# plt.rcParams['figure.figsize'] = 10,10
# plt.rcParams['font.size'] = 16
# plt.rcParams['figure.figsize'] = 5,5

def plot_2d(xi, yi, disc, label='Approx', num_levels=10, max_ht=None, annotate=''):
    lambda_mesh = np.vstack([xi.flatten(),yi.flatten()]).T
    if disc.get_input().get_probabilities() is None:
        zi_disc = disc.updated_pdf(lambda_mesh)
    else:
        pdf_disc = disc.get_input().get_probabilities()/disc.get_input().get_volumes()
        zi_disc = pdf_disc[disc.get_input().query(lambda_mesh)[1]]
    Z = zi_disc.reshape(xi.shape)
    if max_ht is None: max_ht = max(Z.ravel())
    fig, axes = plt.subplots(ncols=1, nrows=1)
    # axes[0].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.viridis)
    # axes[1].pcolormesh(xi, yi, zi_a.reshape(xi.shape), shading='gouraud', cmap=plt.cm.viridis)
    

    a00 = axes.contourf(xi, yi, Z, levels=num_levels, vmin=0, vmax=max_ht)
    axes.set_ylabel('$\lambda_2$', fontsize=24)
    axes.set_xlabel('$\lambda_1$', fontsize=24)

    
    # fig.colorbar(a00, ax=axes[0])
    # fig.colorbar(a11, ax=axes[1])
    fig.subplots_adjust(right=0.8, bottom=0.2)
    cbar_ax = fig.add_axes([0.8125, 0.2, 0.075, 0.675])
    m = plt.cm.ScalarMappable(cmap=cm.viridis)
#     m.set_array(Z)
    m.set_clim(0., max_ht)
    plt.colorbar(m, boundaries=np.linspace(0, max_ht, num_levels+1), cax=cbar_ax, format='%2.1f')

    axes.annotate(annotate, (0.1, 0.8), c='w', fontsize=24)
    axes.axis('equal')
    # plt.tight_layout()
    # TODO: better savename
    savename = '%s_N%d.pdf'%(label+'-'+annotate, disc.check_nums())
    savename = savename.replace('$','').replace('=','').replace(',','_').replace(' ','')
    plt.savefig(savename, bbox_inches='tight')
    plt.show()