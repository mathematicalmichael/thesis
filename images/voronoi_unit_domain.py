import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spatial
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import colors
import argparse

############################################################

# parameter domain
lam_domain = np.array([[0.0, 1.0],
                       [0.0, 1.0]])

label_fsize = 24
tick_fsize = 16
num_ticks = 5
size_scatter = 2

#############################################################


# Method for finite voronoi diagram
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        if vor.points.shape[0] < 10:
            inflate = 5
        else:
            inflate = 1
        radius = inflate * vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


############################################################

def voronoi_diagram(num_samples=25, r_seed=10,
                    png=False, show_labels=True):
    np.random.seed(r_seed)
    # labels, plots, etc. assume dimension = 2.
    samples = np.random.random([num_samples, 2])
    plt.figure()

    vor = Voronoi(samples)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    for i in range(samples.shape[0]):
        polygon = vertices[regions[i]]
        plt.fill(*zip(*polygon), alpha=0.075, color='k')

    #plt.xlim(vor.min_bound[0], vor.max_bound[0])
    #plt.ylim(vor.min_bound[1], vor.max_bound[1])
    plt.xlim(lam_domain[0, 0], lam_domain[0, 1])
    plt.ylim(lam_domain[1, 0], lam_domain[1, 1])
    plt.gca().set_aspect('equal')
    plt.scatter(samples[:, 0], samples[:, 1],
                s=size_scatter, facecolor='k')

    if show_labels:
        labels = ''
        plt.xlabel(r'$\lambda_1$', fontsize=label_fsize)
        plt.ylabel(r'$\lambda_2$', fontsize=label_fsize)
        plt.xticks(np.linspace(0, 1, num_ticks), fontsize=tick_fsize)
        plt.yticks(np.linspace(0, 1, num_ticks), fontsize=tick_fsize)
    else:
        labels = '_no_label'
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()

    if png:
        savename = 'voronoi_diagram_N%d_r%d%s.png' % (
            num_samples, r_seed, labels)
        plt.savefig(savename, dpi=600, bbox_inches='tight')
    else:
        savename = 'voronoi_diagram_N%d_r%d%s.pdf' % (
            num_samples, r_seed, labels)
        plt.savefig(savename, bbox_inches='tight')


if __name__ == "__main__":
    desc = 'Make voronoi-cell diagrams with uniform random samples '
    desc += 'in a 2D unit domain.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-n', '--num', default=25,
                        help='Set number of cells (default: 25)')
    parser.add_argument('-s', '--seed', default=21,
                        help='Set random seed (default: 21).')
    parser.add_argument('--png', action='store_true',
                        help='Store as png instead of pdf.')
    parser.add_argument('--nolabel', action='store_false',
                        help='Strip figures of labels.')
    args = parser.parse_args()
    num_samples, r_seed = int(args.num), int(args.seed)
    png, show_label = args.png, args.nolabel
    voronoi_diagram(num_samples, r_seed, png, show_label)
