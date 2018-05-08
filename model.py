
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

np.random.seed(42)

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
subset_indices = np.random.choice(len(subset), size=10000, replace=False)
subset = subset[subset_indices]

# Take the rv_nb_transits into account
subset["radial_velocity_scatter"] = subset["radial_velocity_error"] \
                                  * np.sqrt(subset["rv_nb_transits"]) \
                                  / np.sqrt(np.pi/2.0)


rv_variance = np.array(subset["radial_velocity_scatter"])**2
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


def get_mu_design_matrix(subset):
    return np.array([
        np.ones(len(subset)),
        subset["phot_rp_mean_flux"]**-1,
        subset["phot_rp_mean_flux"]**-2,
    ])

def get_sigma_design_matrix(subset):
    return np.array([
        np.ones(len(subset)),
        subset["phot_rp_mean_flux"]**-1,
        subset["phot_rp_mean_flux"]**-2,
    ])


mu_design_matrix = get_mu_design_matrix(subset[qc])
sigma_design_matrix = get_sigma_design_matrix(subset[qc])


data.update(
    mu_design_matrix=mu_design_matrix.T, 
    sigma_design_matrix=sigma_design_matrix.T,
    M=mu_design_matrix.shape[0],
    S=sigma_design_matrix.shape[0])

init = dict(outlier_fraction=0.1,
    mu_coefficients=np.array([1e-2, 1e5, 1e10]),
    sigma_coefficients=np.array([1e-2, 1e5, 1e10]))


model = stan.load_stan_model("model.stan")
p_opt = model.optimizing(data=data, init=init)


xi = np.logspace(np.log10(rp_flux[qc].min()), np.log10(rp_flux[qc].max()), 100)

fig, ax = plt.subplots()
ax.scatter(data["flux"], data["rv_variance"], **scatter_kwds)
ax.plot(xi, np.polyval(p_opt["mu_coefficients"][::-1], 1.0/xi), c="r", lw=2, zorder=1000)
ax.semilogx()
ax.set_xlabel(r"\textrm{mean rp flux}")
ax.set_ylabel(r"\textrm{radial velocity variance} $(\textrm{km\,s}^{-1})$")
ax.set_xlim(1.25e4, 1e9)
ax.set_ylim(-0.1, 1.5)

fig.tight_layout()
#_savefig(fig, "stan-initial-fit")



samples = model.sampling(**stan.sampling_kwds(data=data, chains=2, iter=2000,
    init=init))


mu_coefficients = np.mean(samples.extract(("mu_coefficients", ))["mu_coefficients"], axis=0)
sigma_coefficients = np.mean(samples.extract(("sigma_coefficients", ))["sigma_coefficients"], axis=0)

mu = np.polyval(mu_coefficients[::-1], 1.0/xi)
sigma = np.polyval(sigma_coefficients[0:3][::-1], 1.0/xi)

fig, ax = plt.subplots()
ax.scatter(data["flux"], data["rv_variance"], **scatter_kwds)
ax.plot(xi, mu, c="b", lw=2, zorder=1000)
ax.fill_between(xi, mu - sigma, mu + sigma, facecolor="b", alpha=0.3, edgecolor="none")

ax.semilogx()
ax.set_xlabel(r"\textrm{mean rp flux}")
ax.set_ylabel(r"\textrm{radial velocity variance} $(\textrm{km\,s}^{-1})$")
#ax.set_xlim(1.25e4, 1e9)
#ax.set_ylim(-0.1, 1.5)

fig.tight_layout()

# Now for every star, calculate the distribution of excess radial velocity
# variance.

# Do this by taking the total variance within 3\sigma of the RV floor, and
# subtracting that from the measured radial velocity variance.

# Since the coefficients can be correlated, let's do this by actually calculating
# what the variance would be.

mu_coefficients = samples.extract(("mu_coefficients", ))["mu_coefficients"]
sigma_coefficients = samples.extract(("mu_coefficients", ))["mu_coefficients"]

rv_floor = np.zeros((mu_coefficients.shape[0], N), dtype=float)
for i in range(mu_coefficients.shape[0]):
    rv_floor[i] = np.dot(mu_coefficients[i], data["mu_design_matrix"].T) \
                + np.dot(sigma_coefficients[i], data["sigma_design_matrix"].T)

# Take percentiles of the floor.
rv_floor_limit = np.percentile(rv_floor, 1 - (1.0-0.997)/2.0, axis=0)

indices = np.argsort(data["flux"])

fig, ax = plt.subplots()
ax.scatter(data["flux"], data["rv_variance"], s=1, facecolor="#000000", 
    alpha=0.25, rasterized=True)
ax.plot(data["flux"][indices], rv_floor_limit[indices], c="r", lw=2)
ax.semilogx()
ax.set_xlabel(r"\textrm{mean rp flux}")
ax.set_ylabel(r"\textrm{radial velocity variance} $(\textrm{km\,s}^{-1})$")

fig.tight_layout()



def plot_histogram_steps(ax, x_bins, y, y_err):

    xx = np.array(x_bins).repeat(2)[1:]
    xstep = np.repeat((x_bins[1:] - x_bins[:-1]), 2)
    xstep = np.concatenate(([xstep[0]], xstep, [xstep[-1]]))
    # Now: add one step at end of row.
    xx = np.append(xx, xx.max() + xstep[-1]) - xstep/2.0

    yy = np.array(y).repeat(2)

    _ = ax.plot(xx, yy, '-')
    ax.errorbar(x_bins, y, y_err, fmt=None, capsize=0, ecolor=_[0].get_color())

    return (xx, yy)

shorthand_parameters = [
    ("phot_rp_mean_flux", "mean rp flux", True),
    ("phot_bp_mean_flux", "mean bp flux", True),
    ("phot_g_mean_flux", "mean g flux", True),
    ("bp_rp", "bp-rp colour", False),
    ("phot_g_mean_mag", "mean g magnitude", False),
    ("phot_rp_mean_mag", "mean rp magnitude", False),
    ("phot_bp_mean_mag", "mean bp magnitude", False),
    ("teff_val", "inferred temperature", False),
    ("ra", "right ascension", False),
    ("dec", "declination", False),
    ("radial_velocity", "radial velocity", False),
    ("rv_nb_transits", "number of radial velocity transits", False),
    ("phot_g_abs_mag", "absolute g magnitude", False)
]

# Calculate absolute magnitudes and use a quality cut.
subset["phot_g_abs_mag"] = subset["phot_g_mean_mag"] + 5 * np.log10(subset["parallax"]/100.0)

good = (subset["parallax"] > 0) * ((subset["parallax"]/subset["parallax_error"]) > 5)
subset["phot_g_abs_mag"][~good] = np.nan

for label_name, description, is_log in shorthand_parameters:

    # Calculate fraction above/below as a function of magnitude.
    N_bins = 15
    has_excess_rv_variance = data["rv_variance"] > rv_floor_limit

    xvals = subset[label_name][qc]
    sqc = np.isfinite(xvals)

    subset_x = xvals[sqc]
    subset_has_excess_rv_variance = has_excess_rv_variance[sqc]


    if is_log:
        equispaced_bins = np.logspace(
            np.log10(xvals[sqc].min()), 
            np.log10(xvals[sqc].max()), 
            N_bins)

    else:
        equispaced_bins = np.linspace(subset_x.min(), subset_x.max(), N_bins)

    equispaced_bin_centers = equispaced_bins[:-1] + np.diff(equispaced_bins)/2.
    equispaced_f_rv_variance = np.zeros(equispaced_bins.size - 1, dtype=float)
    equispaced_f_rv_variance_error = np.zeros_like(equispaced_f_rv_variance)

    for i, left_edge in enumerate(equispaced_bins[:-1]):
        right_edge = equispaced_bins[i + 1]

        in_bin = (right_edge > subset_x) * (subset_x >= left_edge)
        equispaced_f_rv_variance[i] += np.sum(subset_has_excess_rv_variance * in_bin)/np.sum(in_bin)
        equispaced_f_rv_variance_error[i] += np.sqrt(np.sum(subset_has_excess_rv_variance * in_bin)) \
                                           / np.sum(in_bin)

    if is_log:
        equidensity_bins = 10**np.percentile(np.log10(subset_x), np.linspace(0, 100, N_bins))
    else:
        equidensity_bins = np.percentile(subset_x, np.linspace(0, 100, N_bins))

    equidensity_bin_centers = equispaced_bins[:-1] + np.diff(equidensity_bins)/2.
    equidensity_f_rv_variance = np.zeros(equidensity_bins.size - 1, dtype=float)
    equidensity_f_rv_variance_error = np.zeros_like(equidensity_f_rv_variance)

    for i, left_edge in enumerate(equidensity_bins[:-1]):
        right_edge = equidensity_bins[i + 1]

        in_bin = (right_edge > subset_x) * (subset_x >= left_edge)
        equidensity_f_rv_variance[i] += np.sum(has_excess_rv_variance * in_bin) \
                                      / np.sum(in_bin)
        equidensity_f_rv_variance_error[i] += np.sqrt(np.sum(has_excess_rv_variance * in_bin)) \
                                            / np.sum(in_bin)


    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    xx, yy = plot_histogram_steps(axes[0], 
        equispaced_bin_centers, equispaced_f_rv_variance, equispaced_f_rv_variance_error)
    if is_log:
        axes[0].semilogx()
    axes[0].set_xlim(xx[0], xx[-1])
    axes[0].set_title(r"\textrm{equi-spaced bins}")


    xx, yy = plot_histogram_steps(axes[1], equidensity_bin_centers, 
        equidensity_f_rv_variance, equidensity_f_rv_variance_error)
    if is_log:
        axes[1].semilogx()
    axes[1].set_xlim(xx[0], xx[-1])
    axes[1].set_title(r"\textrm{equi-density bins}")


    ylims = np.hstack([ax.get_ylim() for ax in axes])
    for ax in axes:
        ax.set_ylim(np.min(ylims), np.max(ylims))

        ax.set_ylabel(r"\textrm{binary fraction}")
        ax.set_xlabel(r"\textrm{{{0}}}".format(description))

    fig.tight_layout()
    _savefig(fig, "binary-fraction-with-{0}".format(description.replace(" ", "-")))


def plot_model_residuals(samples, subset, label_name, semilogx=False,
    scatter_kwargs=None):

    # Predict the radial velocity floor for the MAP value of the model, and
    # subtract it from the subset, then plot the residuals along some axis.
    scatter_kwds = dict(s=1, facecolor="#000000", alpha=0.25, rasterized=True)
    scatter_kwds.update(scatter_kwargs or {})


    mean_mu_coefficients = np.mean(
        samples.extract(("mu_coefficients", ))["mu_coefficients"], axis=0)

    mean_sigma_coefficients = np.mean(
        samples.extract(("mu_coefficients", ))["mu_coefficients"], axis=0)

    mu_design_matrix = get_mu_design_matrix(subset)
    sigma_design_matrix = get_sigma_design_matrix(subset)

    subset_mu = np.dot(mean_mu_coefficients, mu_design_matrix)
    subset_sigma = np.dot(mean_sigma_coefficients, sigma_design_matrix)

    rv_residual = subset["radial_velocity_error"]**2 - subset_mu
    rv_residual_significance = rv_residual/subset_sigma


    show = rv_residual_significance > 3


    fig, ax = plt.subplots()
    ax.scatter(subset[label_name][show], rv_residual[show], **scatter_kwds)
    if semilogx:
        ax.semilogx()
    else:
        ax.xaxis.set_major_locator(MaxNLocator(6))

    ax.set_xlabel(r"\textrm{{{0}}}".format(label_name.replace("_", " ")))
    ax.set_ylabel(r"\textrm{excess radial velocity significance}")
    ax.yaxis.set_major_locator(MaxNLocator(6))

    fig.tight_layout()

    return fig



def plot_excess_rv_variance(samples, subset, label_name, semilogx=False,
    scatter_kwargs=None):

    # Predict the radial velocity floor for the MAP value of the model, and
    # subtract it from the subset, then plot the residuals along some axis.
    scatter_kwds = dict(s=1, facecolor="#000000", alpha=0.25, rasterized=True)
    scatter_kwds.update(scatter_kwargs or {})


    mean_mu_coefficients = np.mean(
        samples.extract(("mu_coefficients", ))["mu_coefficients"], axis=0)

    mean_sigma_coefficients = np.mean(
        samples.extract(("mu_coefficients", ))["mu_coefficients"], axis=0)

    mu_design_matrix = get_mu_design_matrix(subset)
    sigma_design_matrix = get_sigma_design_matrix(subset)

    subset_mu = np.dot(mean_mu_coefficients, mu_design_matrix)
    subset_sigma = np.dot(mean_sigma_coefficients, sigma_design_matrix)

    rv_residual = subset["radial_velocity_error"]**2 - subset_mu
    rv_residual_significance = rv_residual/subset_sigma

    map_rv_excess = np.clip(
        subset["radial_velocity_error"]**2 - (subset_mu + 3 * subset_sigma),
        0,
        np.inf)

    show = map_rv_excess > 0


    fig, ax = plt.subplots()
    ax.scatter(subset[label_name][show], map_rv_excess[show], **scatter_kwds)
    if semilogx:
        ax.semilogx()
    else:
        ax.xaxis.set_major_locator(MaxNLocator(6))

    ax.set_xlabel(r"\textrm{{{0}}}".format(label_name.replace("_", " ")))
    ax.set_ylabel(r"\textrm{excess radial velocity variance} $(\textrm{km\,s}^{-1})$")
    ax.yaxis.set_major_locator(MaxNLocator(6))

    fig.tight_layout()

    return fig

plot_model_residuals(samples, subset, "bp_rp")
plot_excess_rv_variance(samples, subset, "bp_rp")


