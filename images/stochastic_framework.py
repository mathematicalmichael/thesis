import numpy as np
from matplotlib import pyplot as plt
import argparse

plt.rcParams.update({'font.size': 42})
plt.rcParams['figure.figsize'] = 20, 20
from mpl_toolkits.mplot3d import Axes3D

def stochastic_framework_figure(fsize=52, png=False,
                                nolabel=False, showfig=False):
    fig3d = plt.figure()
    ax = fig3d.add_subplot(111, projection='3d')

    # xx, zz = np.meshgrid(np.linspace(0,1,10), np.linspace(0,1,10))
    # point = np.array([0.5, 0.5, 0.5])
    # normal = np.array([0, 1, 0])
    # d = -point.dot(normal)
    # y = (-normal[0] * xx - normal[1] * zz - d)*1./normal[1]
    # ax.plot_surface(xx, yy, z, alpha=0.2)

    ax = plt.subplot(projection='3d')
    ax.grid(False)
    ax.axis('off')
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.text(1,0,0.05, '$\\Lambda$', fontsize=fsize)
    ax.text(0.5,0,0.025,'$\\lambda^\dagger$', fontsize=fsize/1.5)

    ax.text(0,1,0.05, '$\\Xi$', fontsize=fsize)
    ax.text(0,0.5,0.025,'$\\xi^\dagger$', fontsize=fsize/1.5)

    ax.text(0,0.05,0.95, '$\\mathcal{D}$', fontsize=fsize)

    x, y, z = np.zeros((3,3))
    u, v, w = np.array([[1,0,0],[0,1,0],[0,0,1]])
    ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.025, color='black')

    xx, zz = np.meshgrid(range(2), range(2))
    ax.plot_surface(xx, zz*0 + 0.5, zz, alpha = 0.10, color='xkcd:blue')
    ax.quiver(0,0.5,0, 1,0,0, arrow_length_ratio=0, color='xkcd:blue')

    if not nolabel:
        ax.text(0.475,0.85,0, "Possible\nObservations", fontsize=fsize/1.5)
        ax.text(1.05,0.65,0, "Computable\nPushforward", fontsize=fsize/1.5)
    ax.quiver(0.5,0,0, 0,1,0, arrow_length_ratio=0, color='xkcd:blood red')

    ax.view_init(25, 45)
    ax.set_xlim3d([0,0.9])
    ax.set_ylim3d([0,0.9])
    ax.set_zlim3d([0,0.85])

    plt.tight_layout(pad=0)
    if png:
        plt.savefig('stochastic_framework.png', dpi=600)
    else:
        plt.savefig('stochastic_framework.pdf')
    if showfig:
        plt.show()


if __name__ == "__main__":
    desc = 'Make figure for demonstration of stochastic framework.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--fsize', default=52,
                        help='Store as png instead of pdf.')
    parser.add_argument('--png', action='store_true',
                        help='Store as png instead of pdf.')
    parser.add_argument('--nolabel', action='store_false',
                        help='Strip figures of labels.')
    parser.add_argument('-p','--preview', action='store_true',
                        help='Supress plt.show()')
    args = parser.parse_args()

    size, save_png, = args.fsize, args.png
    show_label, show_plot = args.nolabel, args.preview
    print("Plotting surface...")
    stochastic_framework_figure(fsize=size, png=save_png,
                                nolabel=show_label, showfig=show_plot)
    print("Done.")