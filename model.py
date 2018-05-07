
"""
Model the radial velocity floor in Gaia data.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
from mpl_style import mpl_style
from matplotlib.ticker import MaxNLocator
from astropy.table import Table

import stan_utils as stan

plt.style.use(mpl_style)

def _savefig(fig, basename, **kwargs):
    kwds = dict(dpi=300)
    kwds.update(kwargs)
    fig.savefig("figures/{}.pdf".format(basename), **kwds)
    fig.savefig("figures/{}.png".format(basename), **kwds)

"""
gaia_sources = Table.read("data/rv_floor_cal-result.fits")

# Take a random subset of stars.
subset_indices = np.random.choice(len(gaia_sources), size=1000, replace=False)
subset = gaia_sources[subset_indices]
"""

subset = Table.read("data/rv_floor_cal_subset-result.fits.gz")


rv_variance = np.array(subset["radial_velocity_error"])**2
apparent_magnitude = np.array(subset["phot_g_mean_mag"])
flux = np.array(subset["phot_g_mean_flux"])

scatter_kwds = dict(s=1, facecolor="#000000", alpha=0.05, rasterized=True)



fig, ax = plt.subplots()
ax.scatter(flux, rv_variance, **scatter_kwds)
ax.semilogx()
ax.set_xlabel(r"\textrm{mean g flux}")
ax.set_ylabel(r"\textrm{radial velocity variance} $(\textrm{km\,s}^{-1})$")
ax.set_ylim(-0.5, 10)
ax.yaxis.set_major_locator(MaxNLocator(6))
fig.tight_layout()
_savefig(fig, "flux-vs-rvvar")



fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, band in zip(axes, ("g", "bp", "rp")):
    ax.scatter(subset["phot_{}_mean_mag".format(band)],
               rv_variance,
               **scatter_kwds)

    ax.set_xlabel(r"\textrm{{apparent {0} magnitude}}".format(band))
    ax.set_ylabel(r"\textrm{radial velocity variance} $(\textrm{km\,s}^{-1})$")

    ax.set_xlim(5, 15)
    ax.set_ylim(0, 1)

fig.tight_layout()
_savefig(fig, "mags-vs-rvvar")



fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, band in zip(axes, ("g", "bp", "rp")):
    ax.scatter(subset["phot_{}_mean_flux".format(band)],
               rv_variance,
               **scatter_kwds)

    ax.set_xlabel(r"\textrm{{apparent {0} flux}}".format(band))
    ax.set_ylabel(r"\textrm{radial velocity variance} $(\textrm{km\,s}^{-1})$")
    ax.semilogx()
    ax.set_xlim(1.25e4, 1e8)
    ax.set_ylim(0, 1)

fig.tight_layout()
_savefig(fig, "fluxes-vs-rvvar")



# Do some dumb fitting just to get some "intuition".
keep = ((rv_variance < 1) * (flux < 1e6)) + ((rv_variance < 0.1) * (flux > 1e6))
#keep = np.ones(len(flux), dtype=bool)

x_fit, y_fit = (flux[keep], rv_variance[keep])

p0 = np.array([1, 0])
function = lambda _x, *params: params[0] * (_x)**-2 + params[1] 

p_opt, p_cov = op.curve_fit(function, x_fit, y_fit, p0=p0)

xi = np.logspace(np.log10(x_fit.min()), np.log10(x_fit.max()), 100)


fig, ax = plt.subplots()
ax.scatter(x_fit, y_fit, **scatter_kwds)
ax.plot(xi, function(xi, *p_opt), c='r', lw=2, zorder=1000)
ax.plot(xi, function(xi, p_opt[0], 0.01), c='b', lw=2)
ax.semilogx()
ax.set_xlabel(r"\textrm{mean g flux}")
ax.set_ylabel(r"\textrm{radial velocity variance} $(\textrm{km\,s}^{-1})$")
ax.set_ylim(-0.5, 10)
ax.yaxis.set_major_locator(MaxNLocator(6))
fig.tight_layout()
_savefig(fig, "dumb-initial-fit")


# OK, let's prepare a model.
model = stan.load_stan_model("model.stan")

raise a



