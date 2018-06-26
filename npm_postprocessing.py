"""
Make plots after running npm.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
from astropy.table import Table
from matplotlib.ticker import MaxNLocator

from mpl_utils import mpl_style
plt.style.use(mpl_style)


DATA_PATH = "data/rv-all-subset-1e4.fits"
RESULTS_PATH = "rv-all-subset-1e4-results.pickle"


# Load results and data.
data = Table.read(DATA_PATH)

with open(RESULTS_PATH, "rb") as fp:
    subset, opt_params, predictor_label_names, p_single, rv_excess = pickle.load(fp)

data = data[subset]


def _get_label_index(label_name, single, kind):

    if kind not in ("mu", "sigma"):
        raise ValueError("kind must be mu or sigma")

    index = list(predictor_label_names).index(label_name)
    L = len(predictor_label_names)

    offset = 1 + index
    if kind == "sigma":
        offset += L

    if not single:
        offset += 2 * L

    return offset



# What plots do we want to make after running npm.py?

# H-R diagram:
    # scatter plot coloured by probability of binraity.
    # scatter plot coloured by RV excess
    # scatter plot coloured by the various optimised parameter values
    # hess diagram of binary fraction

# binary fraction
    # equispaced/equidensity as a function of various parameters,....


def scatter_hrd(x, y, z=None, xlabel=None, ylabel=None, zlabel=None, ax=None,
    figsize=(10, 10), zsort=None, vmin=None, vmax=None, **kwargs):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    if z is not None and zsort is not None:
        idx = np.argsort(z)
        if zsort < 0:
            idx = idx[::-1]

        x = np.atleast_1d(x[idx]).flatten()
        y = np.atleast_1d(y[idx]).flatten()
        z = np.atleast_1d(z[idx]).flatten()

    else:
        idx = None

    scatter_kwds = dict(cmap="viridis", vmin=vmin, vmax=vmax, alpha=1, s=1,
                        rasterized=True)
    scatter_kwds.update(kwargs)
    scat = ax.scatter(x, y, c=z, **scatter_kwds)
    if z is not None:
        cbar = plt.colorbar(scat)
        if zlabel is not None:
            cbar.set_label(zlabel)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ax.get_ylim()[::-1])

    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))

    fig.tight_layout()

    return fig



kwds = dict(xlabel=r"\textrm{bp - rp}", ylabel=r"\textrm{absolute rp mag}")


fig = scatter_hrd(data["bp_rp"], data["absolute_rp_mag"], p_single, 
                  zlabel=r"\textrm{p(single star$|$data)}", s=5, zsort=1, **kwds)
fig.savefig("figures/subset/p_single.png", dpi=150)


fig = scatter_hrd(data["bp_rp"], data["absolute_rp_mag"], rv_excess.T[0],
                  zlabel=r"$\textrm{radial velocity excess (clipped) / km\,s}^{-1}$", 
                  zsort=-1, alpha=1, s=5, vmin=0, vmax=10, cmap="viridis_r", **kwds)
fig.savefig("figures/subset/rv_excess.png", dpi=150)


fig = scatter_hrd(data["bp_rp"], data["absolute_rp_mag"], rv_excess.T[1],
                  zlabel=r"\textrm{radial velocity excess significance (clipped)}", 
                  zsort=-1, s=5, alpha=1, cmap="viridis_r", vmin=0, vmax=5, **kwds)
fig.savefig("figures/subset/rv_excess_sig.png", dpi=150)


# Plot as a function of the optimized values.
fig = scatter_hrd(data["bp_rp"], data["absolute_rp_mag"], opt_params.T[0],
                  zlabel=r"$\mathrm{\theta}$", s=5,
                  zsort=1, alpha=1, cmap="viridis_r", **kwds)
fig.savefig("figures/subset/opt_theta.png", dpi=150)


for label_name in predictor_label_names:
    for single in (True, False):
        for kind in ("mu", "sigma"):

            index = _get_label_index(label_name, single=single, kind=kind)
            zlabel = r"\textrm{{{2} star}} $\{0}_\textrm{{{1}}} \textrm{{ / km\,s}}^{{-1}}$".format(
                kind, label_name.replace("_", " "), "single" if single else "multiple")

            fig = scatter_hrd(data["bp_rp"], data["absolute_rp_mag"],
                              opt_params.T[index], zlabel=zlabel,
                              s=5, zsort=1, alpha=1, cmap="viridis_r", 
                              **kwds)
            fig.savefig(
                "figures/subset/opt_{0}_star_{1}_{2}.png".format(
                    "single" if single else "multiple", label_name, kind),
                dpi=150)


# Plot binary fraction histogram as a function of various things?
