from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
# plt.rcParams['figure.figsize'] = 10,10
# plt.rcParams['font.size'] = 16
# plt.rcParams['figure.figsize'] = 5,5

def plot_2d(xi, yi, disc, label='approx', num_levels=10, max_ht=None,
            annotate='', title='', pdf=False, eval=False, preview=False):
    lambda_mesh = np.vstack([xi.flatten(),yi.flatten()]).T
    xmin, xmax = np.min(xi), np.max(xi)
    ymin, ymax = np.min(yi), np.max(yi)
    if disc.get_input().get_probabilities() is None:
        if eval:
            zi_disc = disc.updated_pdf(lambda_mesh) # full-eval
        else:
            pdf_disc = disc.updated_pdf()
            zi_disc = pdf_disc[disc.get_input().query(lambda_mesh)[1]]
    else:
        vols = disc.get_input().get_volumes()*(xmax-xmin)*(ymax-ymin)
        # volumes are "proportion of domain": needs to correspond to actual area
        # vols[vols == 0] = np.inf
        pdf_disc = disc.get_input().get_probabilities()
        print('sum probs', np.sum(pdf_disc))
        pdf_disc[vols != 0] = pdf_disc[vols != 0]/vols[vols != 0]
        # pdf_disc = pdf_disc/vols
        zi_disc = pdf_disc[disc.get_input().query(lambda_mesh)[1]]
    Z = zi_disc.reshape(xi.shape)
    if max_ht is None: max_ht = max(Z.ravel())
    fig, axes = plt.subplots(ncols=1, nrows=1)
    # axes[0].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.viridis)
    # axes[1].pcolormesh(xi, yi, zi_a.reshape(xi.shape), shading='gouraud', cmap=plt.cm.viridis)


    C = axes.contourf(xi, yi, Z, levels=num_levels, vmin=0, vmax=max_ht, cmap=cm.viridis)
    axes.set_ylabel('$\lambda_2$', fontsize=24)
    axes.set_xlabel('$\lambda_1$', fontsize=24)
    plt.title(title)
    fig.subplots_adjust(right=0.8, bottom=0.2)

    annotate_loc = [xmin*1.1, ymax*.8]
    axes.annotate(annotate, annotate_loc, color='w', fontsize=24)
    axes.axis('equal')
#
#     plt.colorbar(C, cax=cbar_ax, format='%2.1f')
    colorbar = clippedcolorbar(C, extend='both')


    # plt.tight_layout()
    # TODO: better savename
    if pdf:
        ext = 'pdf'
    else:
        ext = 'png'

    emul = 'mc'
    if len(np.unique(disc.get_input()._volumes_local)) != 1:
        emul = 'em'
    savename = 't%s-%s_N%d_%s.%s'%(title, label+'-'+annotate, disc.check_nums(), emul, ext)
    savename = savename.replace('$','').replace('=','').replace(',','_').replace(' ','')
    plt.savefig(savename, bbox_inches='tight')
    if preview: plt.show()


def clippedcolorbar(CS, **kwargs):
    from matplotlib.cm import ScalarMappable
    from numpy import arange, floor, ceil
    fig = CS.ax.get_figure()
    cbar_ax = fig.add_axes([0.8125, 0.2, 0.075, 0.675])
    vmin = CS.get_clim()[0]
    vmax = CS.get_clim()[1]
    m = ScalarMappable(cmap=CS.get_cmap())
    m.set_array(CS.get_array())
    m.set_clim(CS.get_clim())
    step = CS.levels[1] - CS.levels[0]
    cliplower = CS.zmin<vmin
    clipupper = CS.zmax>vmax
    noextend = 'extend' in kwargs.keys() and kwargs['extend']=='neither'
    # set the colorbar boundaries
    boundaries = arange((floor(vmin/step)-1+1*(cliplower and noextend))*step, (ceil(vmax/step)+1-1*(clipupper and noextend))*step, step)
    # boundaries = [vmin]+CS.levels[ (CS.levels>vmin) & (CS.levels<vmax) ].tolist()+[vmax] # fails when vmin<CS.levels.min()
    kwargs['boundaries'] = boundaries
    # if the z-values are outside the colorbar range, add extend marker(s)
    # This behavior can be disabled by providing extend='neither' to the function call
    if not('extend' in kwargs.keys()) or kwargs['extend'] in ['min','max']:
        extend_min = cliplower or ( 'extend' in kwargs.keys() and kwargs['extend']=='min' )
        extend_max = clipupper or ( 'extend' in kwargs.keys() and kwargs['extend']=='max' )
        if extend_min and extend_max:
            kwargs['extend'] = 'both'
        elif extend_min:
            kwargs['extend'] = 'min'
        elif extend_max:
            kwargs['extend'] = 'max'
    return fig.colorbar(m, cax=cbar_ax, format='%2.1f', **kwargs)
