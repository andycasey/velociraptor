
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

def _savefig(fig, basename, dry_run=True, **kwargs):
    kwds = dict(dpi=300)
    kwds.update(kwargs)
    if not dry_run:
        fig.savefig("figures/{}.pdf".format(basename), **kwds)
        fig.savefig("figures/{}.png".format(basename), **kwds)
    else:
        print("Dry-run: not saving figure")

    return None


#gaia_sources = Table.read("data/rv_floor_cal-result.fits")

subset = Table.read("data/rv_floor_cal_subset-result.fits.gz")

# Take a random subset of stars.
subset_indices = np.random.choice(len(subset), size=1000, replace=False)
subset = subset[subset_indices]


rv_variance = np.array(subset["radial_velocity_error"])**2
apparent_magnitude = np.array(subset["phot_g_mean_mag"])
flux = np.array(subset["phot_g_mean_flux"])
rp_flux = np.array(subset["phot_rp_mean_flux"])

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
qc = np.isfinite(rp_flux)
N = sum(qc)
data = dict(N=N, rv_variance=rv_variance[qc], flux=rp_flux[qc])

mu_design_matrix = np.array([
    np.ones(N),
    data["flux"]**-1,
    data["flux"]**-2,
])
sigma_design_matrix = np.array([
    np.ones(N),
    data["flux"]**-1,
    data["flux"]**-2,
])
data.update(
    mu_design_matrix=mu_design_matrix.T, 
    sigma_design_matrix=sigma_design_matrix.T,
    M=mu_design_matrix.shape[0],
    S=sigma_design_matrix.shape[0])

init = dict(outlier_fraction=0.5,
    mu_coefficients=np.array([1e10, 1e5, 1e-2]),
    sigma_coefficients=np.array([1e10, 1e5, 1e-2]))


model = stan.load_stan_model("model.stan")
p_opt = model.optimizing(data=data, init=init)


fig, ax = plt.subplots()
ax.scatter(data["flux"], data["rv_variance"], **scatter_kwds)
ax.plot(xi, np.polyval(p_opt["mu_coefficients"], 1.0/xi), c="r", lw=2, zorder=1000)
ax.semilogx()
ax.set_xlabel(r"\textrm{mean rp flux}")
ax.set_ylabel(r"\textrm{radial velocity variance} $(\textrm{km\,s}^{-1})$")
ax.set_xlim(1.25e4, 1e8)
ax.set_ylim(-0.1, 1.5)

fig.tight_layout()
#_savefig(fig, "stan-initial-fit")



samples = model.sampling(**stan.sampling_kwds(data=data, chains=2, iter=2000,
    init=init))


raise a

#plt.style.use({"text.usetex": False })
#fig = stan.plots.traceplot(samples, ("outlier_fraction", "c0", "s0"))
#fig.tight_layout()
#_savefig(fig, "stan-trace")

c0 = samples.extract("c0")["c0"].mean()
c1 = samples.extract("c1")["c1"].mean()

fig, ax = plt.subplots()
ax.scatter(data["flux"], data["rv_variance"], **scatter_kwds)
ax.plot(xi, c0 + c1 * xi**-2, c="r", lw=2, zorder=1000)
ax.semilogx()
ax.set_xlabel(r"\textrm{mean rp flux}")
ax.set_ylabel(r"\textrm{radial velocity variance} $(\textrm{km\,s}^{-1})$")
#ax.set_xlim(1.25e4, 1e8)
#ax.set_ylim(-0.1, 1.5)

s0 = samples.extract("s0")["s0"].mean()
s1 = samples.extract("s1")["s1"].mean()

fig, ax = plt.subplots()
ax.plot(xi, s0 + s1 * xi**-2, c="r", lw=2)
ax.semilogx()
ax.set_xlabel(r"\textrm{mean rp flux}")
ax.set_ylabel(r"\textrm{scatter}")

raise a



