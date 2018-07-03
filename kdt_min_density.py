
# coding: utf-8

# ## Introduce minimum density constraints on the k-d ball
# 

# In[4]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
from astropy.io import fits
from scipy.stats import binned_statistic_2d

import sys
sys.path.insert(0, "../")

import velociraptor


# In[5]:


data = fits.open("data/gaia-sources-for-npm.fits")[1].data

with open("results/rv_single_epoch_scatter.pkl", "rb") as fp:
    rv_results = pickle.load(fp)
    


# In[6]:


def plot_binned_statistic(x, y, z, bins=100, function=np.nanmedian,
                          xlabel=None, ylabel=None, zlabel=None,
                          ax=None, colorbar=False, figsize=(6, 6),
                          zlog10=False,
                          subsample=None, **kwargs):
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure
    
    if function not in ("count", "mean", "median"):
        finite = np.isfinite(x * y * z)
    else:
        finite = np.arange(len(x))

    if subsample is not None:
        idx = np.where(finite)[0]
        if subsample < 1:
            subsample *= idx.size
        finite = np.random.choice(idx, int(subsample), replace=False)
    
    H, xedges, yedges, binnumber = binned_statistic_2d(
        x[finite], y[finite], z[finite],
        statistic=function, bins=bins)
    
    if zlog10:
        H = np.log10(H)
    
    imshow_kwds = dict(
        aspect=np.ptp(xedges)/np.ptp(yedges), 
        extent=(xedges[0], xedges[-1], yedges[-1], yedges[0]),
        cmap="inferno")
    imshow_kwds.update(kwargs)

    image = ax.imshow(H.T, **imshow_kwds)
    if colorbar:
        cbar = plt.colorbar(image, ax=ax)
        if zlabel is not None:
            cbar.set_label(zlabel)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    fig.tight_layout()
    return fig


# In[8]:





# In[10]:




# In[ ]:

# Set up k-d tree

config = dict(
    kdtree_label_names=("bp_rp", "absolute_rp_mag", "phot_rp_mean_mag"),
    kdtree_relative_scales=(0.1, 1.0, 1.0),
    kdtree_minimum_radius=(0.05, 0.50, 0.50),
    kdtree_minimum_points=128,
    kdtree_maximum_points=1024
)


# Set up a KD-tree.
X = np.vstack([data[ln] for ln in config["kdtree_label_names"]]).T

finite = np.all(np.isfinite(X), axis=1)
finite_indices = np.where(finite)[0]

N, D = X.shape
F = finite_indices.size


import npm_utils as npm

kdt, scales, offsets = npm.build_kdtree(X[finite], 
    relative_scales=config["kdtree_relative_scales"])

kdt_kwds = dict(offsets=offsets, scales=scales, full_output=False)
kdt_kwds.update(
    minimum_radius=config["kdtree_minimum_radius"],
    minimum_points=config["kdtree_minimum_points"],
    maximum_points=config["kdtree_maximum_points"],
)




def query_around_point(kdtree, point, offsets=0, scales=1, minimum_radius=None, 
    minimum_points=1, maximum_points=None, minimum_density=None, dualtree=False,
    full_output=False, **kwargs):
    """
    Query around a point in the KD-Tree until certain conditions are met (e.g.,
    the number of points in the ball, and the minimum radius that the ball
    expands out to).
    
    :param kdtree:
        The pre-computed KD-Tree.

    :param point:
        The (unscaled) point to query around.

    :param offsets: [optional]
        The offsets to apply to the query point.

    :param scales: [optional]
        The scaling to apply to the query point, after subtracting the offsets.

    :param minimum_radius: [optional]
        The minimum radius (or radii) that the ball must extend to.

    :param minimum_points: [optional]
        The minimum number of points to return in the ball.

    :param maximum_points: [optional]
        The maximum number of points to return in the ball. If the number of
        points returned exceeds this value, then a random subset of the points
        will be returned.

    :param minimum_density: [optional]
        The minimum average density of points per dimension for the ball. This
        can be useful to ensure that points that are in the edge of the k-d tree
        parameter space will be compared against points that are representative
        of the underlying space, and not just compared against nearest outliers.

    :param dualtree: [optional]
        Use the dual tree formalism for the query: a tree is built for the query
        points, and the pair of trees is  used to efficiently search this space.
        This can lead to better performance as the number of points grows large.

    :param full_output: [optional]
        If `True`, return a two length tuple of the distances to each point and
        the indicies, otherwise just return the indices.
    """

    offsets = np.atleast_1d(offsets)
    scales = np.atleast_1d(scales)

    point_orig = np.atleast_1d(point).reshape(1, -1)
    point = (point_orig - offsets)/scales

    # Simple case.
    if minimum_radius is None and minimum_density is None:
        # We can just query the nearest number of points.
        d, indices = kdtree.query(point, k=minimum_points, 
            sort_results=True, return_distance=True, dualtree=dualtree)

    else:
        # We need to find the minimum radius that meets our constraints.
        if minimum_radius is None: 
            minimum_radius = 0

        if minimum_density is None:
            minimum_density = 0

        minimum_radius = np.atleast_1d(minimum_radius)
        minimum_density = np.atleast_1d(minimum_density)
        
        # Need to scale the minimum radius from the label space to the normalised
        # k-d tree space.
        minimum_radius_norm = np.max(minimum_radius / np.atleast_1d(scales))

        K = kdtree.two_point_correlation(point, minimum_radius_norm)[0]

        # "density" = N/(2*R)
        # if N > 2 * R * density then our density constraint is met
        K_min = np.max(np.hstack([
            minimum_points, 
            2 * minimum_density * minimum_radius
        ]))

        # Check that the minimum radius norm will also meet our minimum number
        # of points constraint. Otherwise, we need to use two point
        # auto-correlation functions to see how far to go out to.
        if K >= K_min:
            # All constraints met.
            radius_norm = minimum_radius_norm

        else:
            # We need to use the k-d tree to step out until our constraints are
            # met.
            maximum_radius_norm = 2 * np.max(np.ptp(kdtree.data, axis=0))

            # This is the initial coarse search.
            N, D = kdtree.data.shape
            left, right = (minimum_radius_norm, maximum_radius_norm)

            Q = kwargs.get("Q", 1000) # MAGIC HACK
 
            # MAGIC HACK
            tolerance = maximum_points if maximum_points is not None \
                                       else 2 * minimum_points

            while True:
                # Shrink it.
                ri = np.logspace(np.log10(left), np.log10(right), Q)

                counts = kdtree.two_point_correlation(point, ri)

                minimum_counts = np.clip(2 * np.max(np.dot(ri.reshape(-1, 1), 
                    (minimum_density * scales).reshape(1, -1)), axis=1),
                    minimum_points, N)

                indices = np.arange(Q)[counts >= minimum_counts]

                left, right = (ri[indices[0] - 1], ri[indices[1]])

                if np.diff(counts[indices]).max() < tolerance:
                    break

            radius_norm = left

        # two_point_correlation(point, minimum_radius_norm)
        #   is eequivalent to
        # query_radius(point, minimum_radius_norm, count_only=True)
        # but in my tests two_point_correlation was a little faster.

        # kdtree.query_radius returns indices, d
        # kdtree.query returns d, indices
        # .... are you serious?

        indices, d = kdtree.query_radius(point, radius_norm, 
            return_distance=True, sort_results=True)

    d, indices = (d[0], indices[0])

    L = len(indices)
    if maximum_points is not None and L > maximum_points:
        if maximum_points < minimum_points:
            raise ValueError("minimum_points must be smaller than maximum_points")

        # Sub-sample a random number.
        sub_idx = np.random.choice(L, maximum_points, replace=False)
        d, indices = (d[sub_idx], indices[sub_idx])

    assert minimum_points is None or indices.size >= minimum_points
    return (d, indices) if full_output else indices





# Now do it with a minimum density constraint.
kdt_kwds["minimum_density"] = 5000


B = 500


# What if we altered the properties of the ball that it had to extend to a
# *dense enough* region of parameter space?
kwds = dict(function="count", cmap="Greys", zlog10=True, zorder=1, 
            z=np.ones(len(data)))

fig, axes_density = plt.subplots(1, 2, figsize=(8, 4))
plot_binned_statistic(x=data["bp_rp"], y=data["absolute_rp_mag"],
                      bins=(np.linspace(0, 4, B), np.linspace(-10, 10, B)),
                      xlabel=r"\textrm{bp - rp}", ylabel=r"\textrm{absolute rp mag}",
                      ax=axes_density[0], **kwds)

plot_binned_statistic(x=data["bp_rp"], y=data["phot_rp_mean_mag"],
                      bins=(np.linspace(0, 4, B), np.linspace(2, 14, B)),
                      xlabel=r"\textrm{bp - rp}", ylabel=r"\textrm{apparent rp mag}",
                      ax=axes_density[1], **kwds)

fig.tight_layout()

for ax in axes_density:
    ax.set_title(r"\textrm{with density constraint}")

# Set up a matplotlib event to wait for clicks, then show the ball, etc.


collections = []

def onclick_min_density(event):

    print(event)
    N = len(collections)
    for i in range(N):
        collections.pop(0).set_visible(False)

    if not event.inaxes:
        return None

    # Need a three dimensional point from a two dimensional figure.
    point = np.array([
        event.xdata,
        event.ydata,
    ])

    if event.inaxes == axes_density[0]:
        slice_idx = np.array([0, 1])
    elif event.inaxes == axes_density[1]:
        slice_idx = np.array([0, 2])

    index = finite_indices[np.sum((X[finite_indices, :][:, slice_idx] - point)**2, axis=1).argmin()]

    indices = finite_indices[query_around_point(kdt, X[index], **kdt_kwds)]


    collections.extend([
        axes_density[0].scatter(X[indices, 0], X[indices, 1], 
                        c="tab:red", alpha=0.3, zorder=5),
        axes_density[0].scatter([X[index, 0]], [X[index, 1]],
                        c="tab:blue", alpha=1, zorder=10),
        axes_density[1].scatter(X[indices, 0], X[indices, 2], 
                        c="tab:red", alpha=0.3, zorder=5),
        axes_density[1].scatter([X[index, 0]], [X[index, 2]],
                        c="tab:blue", alpha=1, zorder=10),
        ])

    plt.draw()


cid = fig.canvas.mpl_connect('button_press_event', onclick_min_density)










